// Package sim implements the deterministic simulation driver used by the evaluation
// harness. It maintains a virtual-time event queue and drives the RVT scheduler and
// alternative baselines under configurable workloads.
package sim

import (
	"container/heap"
	"math/rand"

	"github.com/atharvaarbat/agent-kernel/metrics"
	"github.com/atharvaarbat/agent-kernel/mock"
	"github.com/atharvaarbat/agent-kernel/scheduler"
)

// ─── event queue ─────────────────────────────────────────────────────────────

type eventKind int

const (
	evComplete eventKind = iota
	evSubmit
)

type event struct {
	time    float64 // simulation seconds
	kind    eventKind
	agentID scheduler.AgentID
	callID  int64
	actual  float64 // actual cost (token-denominated)
	latency float64 // wall-clock seconds from dispatch to completion
	call    *scheduler.Call
	index   int // heap position
}

type eventHeap []*event

func (h eventHeap) Len() int            { return len(h) }
func (h eventHeap) Less(i, j int) bool  { return h[i].time < h[j].time }
func (h eventHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i]; h[i].index = i; h[j].index = j }
func (h *eventHeap) Push(x interface{}) { *h = append(*h, x.(*event)) }
func (h *eventHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// ─── Sched interface (RVT or baseline) ───────────────────────────────────────

// Sched is the common interface satisfied by both the RVT scheduler and the baselines.
type Sched interface {
	Submit(agentID scheduler.AgentID, call *scheduler.Call) error
	Complete(agentID scheduler.AgentID, callID int64, actualCost float64) error
	Stats() map[scheduler.AgentID]*scheduler.AgentStats
}

// ─── Workload ────────────────────────────────────────────────────────────────

// WorkloadKind selects the six §7.2 workloads.
type WorkloadKind int

const (
	WorkloadUniform     WorkloadKind = iota // §7.2: uniform demand, uniform weights
	WorkloadSkewed                          // 10:1 demand skew, equal weights
	WorkloadAdversarial                     // agent 0 always submits max-cost calls
	WorkloadHeavyTail                       // heavier-tailed output lengths (σ=2.5)
	WorkloadMixedPrio                       // mixed weights (3:1), high-weight agents demand 3×
	WorkloadTwoResource                     // two-resource bottleneck (even-ID agents 3× latency)
)

// AgentConfig describes one agent in the experiment.
type AgentConfig struct {
	ID            scheduler.AgentID
	Weight        float64
	CallsPerEpoch int // informational; not used directly by sim
}

// Config configures one simulation run.
type Config struct {
	Workload          WorkloadKind
	Agents            []AgentConfig
	TotalCalls        int     // stop after this many completions
	EpochLen          float64 // epoch length in seconds (informational)
	MockConfig        mock.Config
	Seed              int64
	MaxD              int
	GlobalConcurrency int // shared provider concurrency cap (0 = unlimited)
	ReconcileEnabled  bool
}

// ─── Simulation ──────────────────────────────────────────────────────────────

// Simulation drives one experiment run to completion.
type Simulation struct {
	cfg           Config
	sched         Sched
	provider      *mock.Provider
	heavyProvider *mock.Provider // WorkloadHeavyTail: σ=2.5
	rng           *rand.Rand
	events        eventHeap
	now           float64
	collector     *metrics.Collector
	completions   int
	callsLeft     map[scheduler.AgentID]int
}

// NewWithDispatcher creates a Simulation and returns both the sim and its dispatcher
// so the caller can wire them to a freshly created scheduler.
func NewWithDispatcher(cfg Config) (*Simulation, *SimDispatcher) {
	rng := rand.New(rand.NewSource(cfg.Seed))
	prov := mock.New(cfg.MockConfig)

	heavyCfg := cfg.MockConfig
	heavyCfg.LogNormalSigma = 2.5
	heavyProv := mock.New(heavyCfg)

	cmax := cfg.MockConfig.CMax
	d := float64(cfg.MaxD)

	sim := &Simulation{
		cfg:           cfg,
		provider:      prov,
		heavyProvider: heavyProv,
		rng:           rng,
		collector:     metrics.NewCollector(len(cfg.Agents), cmax, cmax, d),
		callsLeft:     make(map[scheduler.AgentID]int),
	}
	heap.Init(&sim.events)
	return sim, &SimDispatcher{sim: sim}
}

// SetSched sets the scheduler after construction (used with NewWithDispatcher).
func (s *Simulation) SetSched(sched Sched) { s.sched = sched }

// Now returns the current simulation time.
func (s *Simulation) Now() float64 { return s.now }

// ─── workload helpers ────────────────────────────────────────────────────────

// callBudget returns the total number of calls this agent will submit during the run.
// For workloads that require continuous backlog (cost-skew workloads), every agent
// gets a full TotalCalls budget so no agent exhausts while others are still active —
// the VTime ordering alone determines the actual call-count distribution.
func (s *Simulation) callBudget(ag AgentConfig) int {
	base := s.cfg.TotalCalls / len(s.cfg.Agents)
	switch s.cfg.Workload {
	case WorkloadSkewed, WorkloadAdversarial:
		// Unlimited budget: simulation stops by total-completions counter, not budget.
		return s.cfg.TotalCalls
	case WorkloadMixedPrio:
		if ag.Weight > 1.0 {
			return int(float64(base) * ag.Weight)
		}
	}
	return base
}

// generateCall returns a workload-appropriate call for the given agent.
// WorkloadSkewed / WorkloadAdversarial: agent[0] submits max-cost calls (cost=CMax);
// other agents submit cheap calls (MaxTokens=50 → cost ≈ 160 tokens), giving ~30:1
// cost ratio so VTime-based fairness is clearly visible against FCFS/RR.
func (s *Simulation) generateCall(agentID scheduler.AgentID) *scheduler.Call {
	call := s.provider.GenerateCall(s.rng, agentID)
	isFirst := agentID == s.cfg.Agents[0].ID
	switch s.cfg.Workload {
	case WorkloadSkewed, WorkloadAdversarial:
		if isFirst {
			call.MaxTokens = int64(s.cfg.MockConfig.CMax / s.cfg.MockConfig.Beta)
			call.InTokens = 50
		} else {
			call.InTokens = 10
			call.MaxTokens = 50 // capped output → cost ≈ 10+3×50=160 tokens
		}
	}
	return call
}

// latencyForAgent samples the service time for a dispatched call.
// WorkloadTwoResource gives even-ID agents 3× latency (resource-bottleneck model).
func (s *Simulation) latencyForAgent(agentID scheduler.AgentID) float64 {
	lat := s.provider.SampleLatency(s.rng)
	if s.cfg.Workload == WorkloadTwoResource && int(agentID)%2 == 0 {
		lat *= 3.0
	}
	return lat
}

// actualCostForAgent samples the actual token cost for a dispatched call.
// WorkloadSkewed/Adversarial: agent[0] costs CMax; others use the (cheap) call params.
// WorkloadHeavyTail: all agents use the heavy-tail provider.
func (s *Simulation) actualCostForAgent(agentID scheduler.AgentID, call *scheduler.Call) float64 {
	switch s.cfg.Workload {
	case WorkloadSkewed, WorkloadAdversarial:
		if agentID == s.cfg.Agents[0].ID {
			return s.cfg.MockConfig.CMax // always 5000
		}
		return s.provider.SampleActualCost(s.rng, call) // cheap (≈160, capped at MaxTokens=50)
	case WorkloadHeavyTail:
		return s.heavyProvider.SampleActualCost(s.rng, call)
	}
	return s.provider.SampleActualCost(s.rng, call)
}

// ─── Run ─────────────────────────────────────────────────────────────────────

// Run executes the simulation until cfg.TotalCalls completions, then returns a Summary.
func (s *Simulation) Run() *metrics.Summary {
	for _, ag := range s.cfg.Agents {
		s.callsLeft[ag.ID] = s.callBudget(ag)
		s.submitNext(ag.ID)
	}

	for s.completions < s.cfg.TotalCalls && len(s.events) > 0 {
		ev := heap.Pop(&s.events).(*event)
		s.now = ev.time

		switch ev.kind {
		case evComplete:
			// Pre-queue the next call before freeing the slot so tryDispatchAll
			// sees the completing agent's next call when selecting the winner.
			// Without this, n=2 always alternates regardless of VTime ordering.
			s.submitNext(ev.agentID)
			_ = s.sched.Complete(ev.agentID, ev.callID, ev.actual)
			s.completions++
			s.collector.RecordCompletion(s.now, ev.agentID, ev.latency, s.sched.Stats())

		case evSubmit:
			_ = s.sched.Submit(ev.agentID, ev.call)
		}
	}

	return s.collector.Summary()
}

// submitNext queues the agent's next call immediately if quota remains.
func (s *Simulation) submitNext(agentID scheduler.AgentID) {
	if s.callsLeft[agentID] > 0 {
		s.callsLeft[agentID]--
		call := s.generateCall(agentID)
		_ = s.sched.Submit(agentID, call)
	}
}

// ─── SimDispatcher ────────────────────────────────────────────────────────────

// SimDispatcher satisfies scheduler.Dispatcher and feeds completions back into the sim.
type SimDispatcher struct {
	sim *Simulation
}

func (d *SimDispatcher) Dispatch(agentID scheduler.AgentID, call *scheduler.Call) error {
	lat := d.sim.latencyForAgent(agentID)
	actual := d.sim.actualCostForAgent(agentID, call)
	heap.Push(&d.sim.events, &event{
		time:    d.sim.now + lat,
		kind:    evComplete,
		agentID: agentID,
		callID:  call.ID,
		actual:  actual,
		latency: lat,
		call:    call,
	})
	return nil
}
