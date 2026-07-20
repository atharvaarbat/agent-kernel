// Package scheduler implements Reconciled Virtual Time (RVT), Algorithm 1 from the paper.
package scheduler

import (
	"fmt"
	"math"
	"sync"
)

// AgentID is a stable numeric identifier for an agent process.
type AgentID int64

// Call represents a single infer/invoke syscall submitted by an agent.
type Call struct {
	ID        int64
	AgentID   AgentID
	InTokens  int64
	MaxTokens int64

	// Set at dispatch time by the scheduler.
	EstimateCost float64
}

// AgentStats is the exported snapshot of per-agent scheduler state.
type AgentStats struct {
	ID       AgentID
	Weight   float64
	VTime    float64 // F_i
	Pending  float64 // P_i = Σ ê(c) for calls in flight
	Service  float64 // A_i = Σ a(c) for completed calls
	InFlight int     // n_i
	QueueLen int
	InBacklog bool
}

// Estimator computes and updates the per-agent cost estimate.
// Implementations live in the estimator package; the interface lives here to avoid
// circular imports.
type Estimator interface {
	Estimate(agentID AgentID, call *Call) float64
	Update(agentID AgentID, actualCost float64)
}

// Dispatcher is called (non-blocking) when the scheduler selects a call to issue.
// The implementation records the call and schedules a future Complete callback.
type Dispatcher interface {
	Dispatch(agentID AgentID, call *Call) error
}

// Params holds the scheduling parameters referenced in the paper's assumptions.
type Params struct {
	MaxD float64 // d: max in-flight calls per agent (A2)
	CMax float64 // C_max: upper bound on actual call cost (A4)
	EMax float64 // Ê_max: upper bound on estimate (A4)
	CMin float64 // c_min: lower bound on actual cost (Corollary 2)

	// ReconcileEnabled: if false, the RECONCILE step in COMPLETE is skipped.
	// Setting this to false instantiates the Proposition 2 ablation.
	ReconcileEnabled bool
}

// agentState is the per-agent scheduler record (unexported: modified only under mu).
type agentState struct {
	id     AgentID
	weight float64

	vtime   float64 // F_i: virtual time
	pending float64 // P_i: pending estimate mass
	service float64 // A_i: reconciled service

	inFlight int     // n_i
	queue    []*Call // submitted-but-not-dispatched calls (FIFO)

	inBacklog bool

	// t0 snapshot: values at the most recent entry into the backlog.
	// Used for Lemma 1 verification and Theorem 1 gap computation.
	vtimeAtEntry   float64
	serviceAtEntry float64
	pendingAtEntry float64
}

// LogEvent is an entry appended to the in-memory event log (WAL stub).
type LogEvent struct {
	Type     string  // "SUBMIT", "DISPATCH", "COMPLETE"
	AgentID  AgentID
	CallID   int64
	Estimate float64
	Actual   float64
}

// Scheduler is the RVT state machine. All exported methods are safe for concurrent
// use; the internal dispatch loop runs under a single mutex, satisfying A1.
type Scheduler struct {
	mu sync.Mutex

	agents   map[AgentID]*agentState
	inFlight map[int64]*Call // callID → dispatched call (holds EstimateCost)

	params     Params
	estimator  Estimator
	dispatcher Dispatcher

	nextCallID int64
	log        []LogEvent

	// I5 conservation counters.
	totalCharged float64
	totalActual  float64
}

// New creates a Scheduler with the given parameters, estimator, and dispatcher.
func New(params Params, est Estimator, disp Dispatcher) *Scheduler {
	return &Scheduler{
		agents:     make(map[AgentID]*agentState),
		inFlight:   make(map[int64]*Call),
		params:     params,
		estimator:  est,
		dispatcher: disp,
	}
}

// AddAgent registers agent id with the given weight. Must be called before Submit.
func (s *Scheduler) AddAgent(id AgentID, weight float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.agents[id] = &agentState{id: id, weight: weight}
}

// SetVTimeForTest directly sets an agent's virtual time. For unit tests only.
func (s *Scheduler) SetVTimeForTest(id AgentID, vtime float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ag, ok := s.agents[id]; ok {
		ag.vtime = vtime
		ag.vtimeAtEntry = vtime
	}
}

// MarkBackloggedForTest marks an agent as already in the backlog (as if it previously
// submitted and had calls). This prevents equalization from firing on the next Submit,
// which lets unit tests set arbitrary VTimes and then submit calls without equalization
// disturbing them. For unit tests only.
func (s *Scheduler) MarkBackloggedForTest(id AgentID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ag, ok := s.agents[id]; ok {
		ag.inBacklog = true
		ag.vtimeAtEntry = ag.vtime
		ag.serviceAtEntry = ag.service
		ag.pendingAtEntry = ag.pending
	}
}

// Submit implements Algorithm 1 §SUBMIT.
// It enqueues call c for agent i, applies equalization if i is re-entering the backlog,
// then triggers DISPATCH.
func (s *Scheduler) Submit(agentID AgentID, call *Call) error {
	s.mu.Lock()

	ag, ok := s.agents[agentID]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("unknown agent %d", agentID)
	}

	call.AgentID = agentID
	if call.ID == 0 {
		s.nextCallID++
		call.ID = s.nextCallID
	}

	// Equalization (Algorithm 1, line 2–4): if i ∉ B, F_i ← max(F_i, min_{j∈B} F_j).
	if !ag.inBacklog {
		minVT := s.minVTimeInBacklog()
		if !math.IsInf(minVT, 1) && minVT > ag.vtime {
			ag.vtime = minVT
		}
		ag.inBacklog = true
		// Record t0 snapshot for invariant checks.
		ag.vtimeAtEntry = ag.vtime
		ag.serviceAtEntry = ag.service
		ag.pendingAtEntry = ag.pending
	}

	ag.queue = append(ag.queue, call)
	s.log = append(s.log, LogEvent{Type: "SUBMIT", AgentID: agentID, CallID: call.ID})

	dispatches := s.tryDispatchAll()
	s.mu.Unlock()

	for _, d := range dispatches {
		if err := s.dispatcher.Dispatch(d.AgentID, d); err != nil {
			return fmt.Errorf("dispatch %d/%d: %w", d.AgentID, d.ID, err)
		}
	}
	return nil
}

// Complete implements Algorithm 1 §COMPLETE.
// It reconciles agent i's virtual time with the actual cost a, updates accounting,
// and triggers DISPATCH.
func (s *Scheduler) Complete(agentID AgentID, callID int64, actualCost float64) error {
	s.mu.Lock()

	ag, ok := s.agents[agentID]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("unknown agent %d", agentID)
	}

	call, ok := s.inFlight[callID]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("call %d not in flight for agent %d", callID, agentID)
	}
	delete(s.inFlight, callID)

	ê := call.EstimateCost

	// RECONCILE (Algorithm 1, line 26): F_i ← F_i + (a − ê(c)) / w_i.
	// Skipped when ReconcileEnabled=false to instantiate Proposition 2.
	if s.params.ReconcileEnabled {
		ag.vtime += (actualCost - ê) / ag.weight
	}

	ag.pending -= ê  // P_i ← P_i − ê(c)
	ag.inFlight--    // n_i ← n_i − 1
	ag.service += actualCost // A_i ← A_i + a

	// I5 conservation tracking.
	s.totalActual += actualCost

	// Update estimator.
	s.estimator.Update(agentID, actualCost)

	s.log = append(s.log, LogEvent{Type: "COMPLETE", AgentID: agentID, CallID: callID,
		Estimate: ê, Actual: actualCost})

	// If queue empty and no calls in flight, remove from backlog.
	if len(ag.queue) == 0 && ag.inFlight == 0 {
		ag.inBacklog = false
	}

	dispatches := s.tryDispatchAll()
	s.mu.Unlock()

	for _, d := range dispatches {
		if err := s.dispatcher.Dispatch(d.AgentID, d); err != nil {
			return fmt.Errorf("dispatch %d/%d: %w", d.AgentID, d.ID, err)
		}
	}
	return nil
}

// Stats returns a snapshot of all agents' states.
func (s *Scheduler) Stats() map[AgentID]*AgentStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[AgentID]*AgentStats, len(s.agents))
	for id, ag := range s.agents {
		out[id] = &AgentStats{
			ID:        ag.id,
			Weight:    ag.weight,
			VTime:     ag.vtime,
			Pending:   ag.pending,
			Service:   ag.service,
			InFlight:  ag.inFlight,
			QueueLen:  len(ag.queue),
			InBacklog: ag.inBacklog,
		}
	}
	return out
}

// Log returns the ordered list of logged events (for WAL / replay).
func (s *Scheduler) Log() []LogEvent { return s.log }

// I5Conservation returns (totalCharged, totalActual) for budget conservation checks.
func (s *Scheduler) I5Conservation() (charged, actual float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.totalCharged, s.totalActual
}

// VerifyLemma1 checks the accounting identity for agent id at the current state.
// Returns non-nil if the identity is violated (indicates a scheduler bug).
func (s *Scheduler) VerifyLemma1(id AgentID) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ag, ok := s.agents[id]
	if !ok {
		return fmt.Errorf("unknown agent %d", id)
	}
	lhs := ag.weight * (ag.vtime - ag.vtimeAtEntry)
	rhs := (ag.service - ag.serviceAtEntry) + (ag.pending - ag.pendingAtEntry)
	if math.Abs(lhs-rhs) > 1e-9 {
		return fmt.Errorf("agent %d: w*(F−F0)=%.12f ≠ (A−A0)+(P−P0)=%.12f (delta=%.2e)",
			id, lhs, rhs, lhs-rhs)
	}
	return nil
}

// VerifyTheorem1 checks the service-gap bound for all pairs of continuously backlogged
// agents at the current moment.
// Returns the first violation found, or nil.
func (s *Scheduler) VerifyTheorem1() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var bl []*agentState
	for _, ag := range s.agents {
		if ag.inBacklog {
			bl = append(bl, ag)
		}
	}

	d := s.params.MaxD
	cmax := s.params.CMax
	emax := s.params.EMax

	for ii, agi := range bl {
		for _, agj := range bl[ii+1:] {
			ΔAi := agi.service - agi.serviceAtEntry
			ΔAj := agj.service - agj.serviceAtEntry
			gapIJ := ΔAi/agi.weight - ΔAj/agj.weight
			gapJI := ΔAj/agj.weight - ΔAi/agi.weight
			boundIJ := d * (cmax/agi.weight + emax/agj.weight)
			boundJI := d * (cmax/agj.weight + emax/agi.weight)
			if gapIJ > boundIJ+1e-9 {
				return fmt.Errorf("Thm1 violated: A%d/w%d − A%d/w%d = %.6f > %.6f",
					agi.id, agi.id, agj.id, agj.id, gapIJ, boundIJ)
			}
			if gapJI > boundJI+1e-9 {
				return fmt.Errorf("Thm1 violated: A%d/w%d − A%d/w%d = %.6f > %.6f",
					agj.id, agj.id, agi.id, agi.id, gapJI, boundJI)
			}
		}
	}
	return nil
}

// ─── internal helpers ────────────────────────────────────────────────────────

// minVTimeInBacklog returns the smallest virtual time among backlogged agents.
// Called with s.mu held.
func (s *Scheduler) minVTimeInBacklog() float64 {
	min := math.Inf(1)
	for _, ag := range s.agents {
		if ag.inBacklog && ag.vtime < min {
			min = ag.vtime
		}
	}
	return min
}

// tryDispatchAll repeatedly calls tryDispatchOne until no candidate remains.
// Called with s.mu held; returns calls that must be issued outside the lock.
func (s *Scheduler) tryDispatchAll() []*Call {
	var out []*Call
	for {
		c := s.tryDispatchOne()
		if c == nil {
			break
		}
		out = append(out, c)
	}
	return out
}

// tryDispatchOne implements the DISPATCH subroutine of Algorithm 1 (lines 10–23).
// Called with s.mu held; returns the dispatched call (or nil if no candidate).
func (s *Scheduler) tryDispatchOne() *Call {
	// Cand = { i ∈ B : n_i < d ∧ queue_i ≠ ∅ ∧ admissible(i) }
	var best *agentState
	for _, ag := range s.agents {
		if !ag.inBacklog || ag.inFlight >= int(s.params.MaxD) || len(ag.queue) == 0 {
			continue
		}
		// admissible(i): no provider-side constraints in mock mode; always true.
		if best == nil ||
			ag.vtime < best.vtime ||
			(ag.vtime == best.vtime && ag.id < best.id) { // deterministic tie-break (I4)
			best = ag
		}
	}
	if best == nil {
		return nil
	}

	call := best.queue[0]
	best.queue = best.queue[1:]

	ê := s.estimator.Estimate(best.id, call)
	call.EstimateCost = ê

	best.vtime += ê / best.weight // F_i ← F_i + ê/w_i
	best.pending += ê             // P_i ← P_i + ê
	best.inFlight++               // n_i ← n_i + 1

	s.inFlight[call.ID] = call
	s.totalCharged += ê

	s.log = append(s.log, LogEvent{Type: "DISPATCH", AgentID: best.id, CallID: call.ID, Estimate: ê})
	return call
}
