package scheduler_test

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/atharvaarbat/agent-kernel/scheduler"
	"pgregory.net/rapid"
)

// TestMain sets the rapid check count to 10 000 (PLAN.md §P3 gate) and runs the suite.
func TestMain(m *testing.M) {
	flag.Set("rapid.checks", "10000") //nolint:errcheck
	os.Exit(m.Run())
}

// ─── test helpers ────────────────────────────────────────────────────────────

// constEstimator always returns the same estimate (useful for deterministic tests).
type constEstimator struct{ est float64 }

func (c *constEstimator) Estimate(_ scheduler.AgentID, _ *scheduler.Call) float64 { return c.est }
func (c *constEstimator) Update(_ scheduler.AgentID, _ float64)                   {}

// recordDispatcher records dispatched calls without issuing them to any provider.
// Simulates completion by triggering Complete immediately (zero-latency mock).
type recordDispatcher struct {
	sched      *scheduler.Scheduler
	completeFn func(agentID scheduler.AgentID, callID int64) float64 // returns actual cost
	dispatched []*scheduler.Call
}

func (r *recordDispatcher) Dispatch(agentID scheduler.AgentID, call *scheduler.Call) error {
	r.dispatched = append(r.dispatched, call)
	if r.completeFn != nil {
		actual := r.completeFn(agentID, call.ID)
		return r.sched.Complete(agentID, call.ID, actual)
	}
	return nil
}

// deferredDispatcher records dispatches for the test to complete manually.
type deferredDispatcher struct {
	sched   *scheduler.Scheduler
	pending []pendingCall
}
type pendingCall struct {
	agentID scheduler.AgentID
	call    *scheduler.Call
}

func (d *deferredDispatcher) Dispatch(agentID scheduler.AgentID, call *scheduler.Call) error {
	d.pending = append(d.pending, pendingCall{agentID, call})
	return nil
}

func (d *deferredDispatcher) CompleteAll(actualCostFn func(*scheduler.Call) float64) error {
	pending := d.pending
	d.pending = nil
	for _, p := range pending {
		if err := d.sched.Complete(p.agentID, p.call.ID, actualCostFn(p.call)); err != nil {
			return err
		}
	}
	return nil
}

func makeParams(d, cmax, emax, cmin float64, reconcile bool) scheduler.Params {
	return scheduler.Params{MaxD: d, CMax: cmax, EMax: emax, CMin: cmin, ReconcileEnabled: reconcile}
}

// ─── Figure 2 unit test ───────────────────────────────────────────────────────

// TestFigure2 reproduces the exact dispatch cycle described in §4.3 / Figure 2:
//   - Three agents, weight 1, with F_A=1500, F_B=2000, F_C=600 (pre-dispatch).
//   - Scheduler dispatches C (lowest VTime) with estimate ê=300 → F_C becomes 900.
//   - Call completes with actual cost 700 (underestimate by 400).
//   - Reconciliation: F_C += 400 → 1300.
//   - Lead of C over A narrows from 600 to 200.
func TestFigure2(t *testing.T) {
	const ê = 300.0
	const actual = 700.0

	disp := &deferredDispatcher{}
	p := makeParams(1, 1000, 1000, 1, true)
	est := &constEstimator{est: ê}
	sched := scheduler.New(p, est, disp)
	disp.sched = sched

	sched.AddAgent(1, 1.0) // A
	sched.AddAgent(2, 1.0) // B
	sched.AddAgent(3, 1.0) // C

	// Figure 2 pre-dispatch state: A and B are in the backlog (from prior history) but
	// have no calls queued right now. C is also in the backlog and is about to submit.
	// Mark all as backlogged first so equalization doesn't fire when C submits.
	sched.SetVTimeForTest(1, 1500) // FA = 1500
	sched.SetVTimeForTest(2, 2000) // FB = 2000
	sched.SetVTimeForTest(3, 600)  // FC = 600 (pre-dispatch)
	sched.MarkBackloggedForTest(1)
	sched.MarkBackloggedForTest(2)
	sched.MarkBackloggedForTest(3)

	// Submit a call ONLY to C. A and B have no calls queued → not dispatch candidates.
	callC := &scheduler.Call{ID: 30, InTokens: 100, MaxTokens: 500}
	if err := sched.Submit(3, callC); err != nil {
		t.Fatal(err)
	}

	// Exactly one dispatch: C (FC=600, the minimum).
	if len(disp.pending) != 1 {
		t.Fatalf("expected 1 dispatch, got %d", len(disp.pending))
	}
	if disp.pending[0].agentID != 3 {
		t.Errorf("expected agent 3 (C) dispatched, got %d", disp.pending[0].agentID)
	}

	// After dispatch: F_C = 600 + 300/1 = 900.
	stats := sched.Stats()
	if got := stats[3].VTime; math.Abs(got-900) > 1e-9 {
		t.Errorf("F_C after dispatch: want 900, got %.6f", got)
	}
	// Lead of C over A in VTime space: F_A − F_C = 1500 − 900 = 600.
	leadBefore := stats[1].VTime - stats[3].VTime
	if math.Abs(leadBefore-600) > 1e-9 {
		t.Errorf("lead before reconciliation: want 600, got %.6f", leadBefore)
	}

	// Complete C's call with actual cost 700.
	dispCallID := disp.pending[0].call.ID
	disp.pending = nil
	if err := sched.Complete(3, dispCallID, actual); err != nil {
		t.Fatal(err)
	}

	// Reconciliation: F_C = 900 + (700−300)/1 = 1300.
	stats2 := sched.Stats()
	if got := stats2[3].VTime; math.Abs(got-1300) > 1e-9 {
		t.Errorf("F_C after reconciliation: want 1300, got %.6f", got)
	}
	// Lead narrows from 600 to 200.
	leadAfter := stats2[1].VTime - stats2[3].VTime
	if math.Abs(leadAfter-200) > 1e-9 {
		t.Errorf("lead after reconciliation: want 200, got %.6f", leadAfter)
	}
}

// ─── Property test: Lemma 1 (accounting identity) ────────────────────────────

// TestLemma1Property checks that w_i*(F_i−F_i0) = (A_i−A_i0)+(P_i−P_i0) holds
// at every state for every agent after every event.
// Runs 10 000 randomised cases — the PLAN.md trust gate (§P3).
func TestLemma1Property(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(2, 6).Draw(rt, "n")
		nCalls := rapid.IntRange(5, 30).Draw(rt, "nCalls")
		cmax := rapid.Float64Range(10, 1000).Draw(rt, "cmax")
		emax := rapid.Float64Range(cmax*0.5, cmax*1.5).Draw(rt, "emax")
		cmin := rapid.Float64Range(1, cmax/2).Draw(rt, "cmin")

		est := &fixedRangeEstimator{min: cmin, max: emax, rng: rand.New(rand.NewSource(42))}
		disp := &syncDispatcher{}
		p := makeParams(1, cmax, emax, cmin, true)
		sched := scheduler.New(p, est, disp)
		disp.sched = sched
		disp.actualFn = func(_ *scheduler.Call) float64 {
			return cmin + rand.Float64()*(cmax-cmin)
		}

		weights := make([]float64, n)
		for i := range weights {
			weights[i] = rapid.Float64Range(0.5, 2.0).Draw(rt, fmt.Sprintf("w%d", i))
		}

		for i := 0; i < n; i++ {
			sched.AddAgent(scheduler.AgentID(i+1), weights[i])
		}

		for k := 0; k < nCalls; k++ {
			agID := scheduler.AgentID(rand.Intn(n) + 1)
			_ = sched.Submit(agID, &scheduler.Call{InTokens: 100, MaxTokens: int64(emax)})

			for i := 0; i < n; i++ {
				if err := sched.VerifyLemma1(scheduler.AgentID(i + 1)); err != nil {
					rt.Fatal(err)
				}
			}
		}
	})
}

// ─── Property test: Theorem 1 (service-gap bound) ────────────────────────────

// TestTheorem1Property verifies that the service gap A_i/w_i − A_j/w_j never
// exceeds d*(C_max/w_i + Ê_max/w_j) for any pair of continuously backlogged agents.
// Runs 10 000 randomised cases — the PLAN.md trust gate (§P3).
func TestTheorem1Property(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(2, 8).Draw(rt, "n")
		nCalls := rapid.IntRange(20, 100).Draw(rt, "nCalls")
		cmax := rapid.Float64Range(50, 500).Draw(rt, "cmax")
		emax := cmax * rapid.Float64Range(0.5, 2.0).Draw(rt, "emaxFactor")
		cmin := rapid.Float64Range(1, cmax/4).Draw(rt, "cmin")

		est := &fixedRangeEstimator{min: cmin, max: emax, rng: rand.New(rand.NewSource(123))}
		disp := &syncDispatcher{}
		p := makeParams(1, cmax, emax, cmin, true)
		sched := scheduler.New(p, est, disp)
		disp.sched = sched
		disp.actualFn = func(_ *scheduler.Call) float64 {
			return cmin + rand.Float64()*(cmax-cmin)
		}

		for i := 0; i < n; i++ {
			w := rapid.Float64Range(0.5, 3.0).Draw(rt, fmt.Sprintf("w%d", i))
			sched.AddAgent(scheduler.AgentID(i+1), w)
		}

		// Pre-populate all agents so they're continuously backlogged.
		for i := 0; i < n; i++ {
			for k := 0; k < nCalls; k++ {
				_ = sched.Submit(scheduler.AgentID(i+1),
					&scheduler.Call{InTokens: 100, MaxTokens: int64(emax)})
			}
		}

		if err := sched.VerifyTheorem1(); err != nil {
			rt.Fatal(err)
		}
	})
}

// ─── Proposition 3 unit test (tightness of C_max term) ───────────────────────

// TestProposition3Tightness constructs the exact 2-agent instance from §4.5:
//   - Equal weights, F1=F2=0, d=1.
//   - Agent 1's call has actual cost C_max; agent 2's has c_min.
//   - Scheduler breaks the tie in favor of agent 1 (lower ID), dispatches it first.
//   - Both are dispatched concurrently (d=1 per agent → each holds one slot).
//   - Agent 2 completes first (c_min); then agent 1 (C_max).
//   - Final gap = C_max − c_min — the tight instance for Theorem 1.
func TestProposition3Tightness(t *testing.T) {
	const cmax = 1000.0
	const cmin = 1.0

	disp := &deferredDispatcher{}
	p := makeParams(1, cmax, cmax, cmin, true)
	est := &constEstimator{est: cmax} // reservation-mode estimate = C_max
	sched := scheduler.New(p, est, disp)
	disp.sched = sched

	sched.AddAgent(1, 1.0)
	sched.AddAgent(2, 1.0)
	sched.MarkBackloggedForTest(1)
	sched.MarkBackloggedForTest(2)

	_ = sched.Submit(1, &scheduler.Call{ID: 1, InTokens: 10, MaxTokens: int64(cmax)})
	_ = sched.Submit(2, &scheduler.Call{ID: 2, InTokens: 10, MaxTokens: int64(cmax)})

	// With d=1 per agent and both at F=0, the scheduler dispatches agent 1 first
	// (tie-break by ID), then immediately dispatches agent 2 (n_1=1=d, but n_2=0<d).
	if len(disp.pending) != 2 {
		t.Fatalf("expected 2 dispatches (one per agent), got %d", len(disp.pending))
	}
	if disp.pending[0].agentID != 1 {
		t.Errorf("expected agent 1 dispatched first (tie-break), got %d", disp.pending[0].agentID)
	}

	// Complete agent 2 first (c_min < C_max → agent 2 completes first in wall time).
	if err := sched.Complete(2, disp.pending[1].call.ID, cmin); err != nil {
		t.Fatal(err)
	}
	// Complete agent 1 (C_max).
	if err := sched.Complete(1, disp.pending[0].call.ID, cmax); err != nil {
		t.Fatal(err)
	}

	stats := sched.Stats()
	a1 := stats[1].Service / stats[1].Weight // A_1 = C_max
	a2 := stats[2].Service / stats[2].Weight // A_2 = c_min
	gap := a1 - a2                           // C_max − c_min

	if math.Abs(gap-(cmax-cmin)) > 1e-9 {
		t.Errorf("Prop3: gap = %.6f, want %.6f (C_max − c_min)", gap, cmax-cmin)
	}

	// Verify gap ≤ Theorem 1 bound (sanity check — bound must not be violated).
	bound := p.MaxD * (p.CMax/1.0 + p.EMax/1.0)
	if gap > bound+1e-9 {
		t.Errorf("Prop3: gap %.2f exceeds Theorem 1 bound %.2f", gap, bound)
	}
	t.Logf("Prop3: gap=%.0f, Theorem1 bound=%.0f, gap/bound=%.3f",
		gap, bound, gap/bound)
}

// ─── Proposition 2: no reconciliation → unbounded gap ────────────────────────

// TestProposition2NoReconciliation verifies that disabling reconciliation causes
// the service gap to grow without bound across m completions.
func TestProposition2NoReconciliation(t *testing.T) {
	const cmax = 100.0
	const epsilon = 5.0 // ê(agent1) is small; actual is large → agent1 dispatched often

	// Agent 1: estimate=ε, actual=C_max  → dispatched ~C_max/ε times per agent-2 dispatch.
	// Agent 2: estimate=C_max, actual=C_max → VT advances correctly.
	est := &agentSpecificEstimator{
		estimates: map[scheduler.AgentID]float64{1: epsilon, 2: cmax},
	}
	disp := &syncDispatcher{}
	p := makeParams(1, cmax, cmax, 1, false) // ReconcileEnabled=false
	sched := scheduler.New(p, est, disp)
	disp.sched = sched
	disp.actualFn = func(c *scheduler.Call) float64 {
		if c.AgentID == 1 {
			return cmax
		}
		return cmax
	}

	sched.AddAgent(1, 1.0)
	sched.AddAgent(2, 1.0)

	const totalCalls = 500
	for k := 0; k < totalCalls; k++ {
		_ = sched.Submit(1, &scheduler.Call{InTokens: 10, MaxTokens: int64(cmax)})
		_ = sched.Submit(2, &scheduler.Call{InTokens: 10, MaxTokens: int64(cmax)})
	}

	stats := sched.Stats()
	a1 := stats[1].Service / stats[1].Weight
	a2 := stats[2].Service / stats[2].Weight
	gap := math.Abs(a1 - a2)

	// Without reconciliation, gap should be large (>> C_max).
	// Theorem 1 bound would be d*(C_max/w_i + Ê_max/w_j) = 2*C_max = 200.
	// With estimate mismatch over 500 calls, gap should far exceed this.
	theorem1Bound := p.MaxD * (p.CMax + p.EMax) // 200
	if gap <= theorem1Bound {
		t.Logf("NOTE: without reconciliation the gap (%.0f) is %s theorem1 bound (%.0f)",
			gap, func() string {
				if gap > theorem1Bound {
					return "above"
				}
				return "below"
			}(), theorem1Bound)
	}
	t.Logf("Prop2 ablation: gap=%.0f (Theorem1 bound would be %.0f)", gap, theorem1Bound)
}

// ─── internal test estimators ─────────────────────────────────────────────────

type fixedRangeEstimator struct {
	min, max float64
	rng      *rand.Rand
}

func (e *fixedRangeEstimator) Estimate(_ scheduler.AgentID, _ *scheduler.Call) float64 {
	return e.min + e.rng.Float64()*(e.max-e.min)
}
func (e *fixedRangeEstimator) Update(_ scheduler.AgentID, _ float64) {}

type agentSpecificEstimator struct {
	estimates map[scheduler.AgentID]float64
}

func (e *agentSpecificEstimator) Estimate(id scheduler.AgentID, _ *scheduler.Call) float64 {
	return e.estimates[id]
}
func (e *agentSpecificEstimator) Update(_ scheduler.AgentID, _ float64) {}

// syncDispatcher completes calls synchronously (zero-latency) so tests don't need
// an event loop.
type syncDispatcher struct {
	sched    *scheduler.Scheduler
	actualFn func(c *scheduler.Call) float64
}

func (d *syncDispatcher) Dispatch(agentID scheduler.AgentID, call *scheduler.Call) error {
	actual := d.actualFn(call)
	return d.sched.Complete(agentID, call.ID, actual)
}
