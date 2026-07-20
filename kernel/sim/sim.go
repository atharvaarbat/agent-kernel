// Package sim implements the deterministic simulation driver used by the evaluation
// harness. It maintains a virtual-time event queue and drives the RVT scheduler and
// alternative baselines under configurable workloads.
package sim

import (
	"container/heap"
	"math/rand"

	"github.com/atharva-arbat/agent-kernel/metrics"
	"github.com/atharva-arbat/agent-kernel/mock"
	"github.com/atharva-arbat/agent-kernel/scheduler"
)

// ─── event queue ─────────────────────────────────────────────────────────────

type eventKind int

const (
	evComplete eventKind = iota
	evSubmit
)

type event struct {
	time       float64 // simulation seconds
	kind       eventKind
	agentID    scheduler.AgentID
	callID     int64
	actualCost float64
	latency    float64 // time from dispatch to completion (seconds)
	call       *scheduler.Call
	index      int // heap position
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

// ─── dispatcher (bridges scheduler → sim event queue) ────────────────────────

// simDispatcher is used by the RVT scheduler; when Dispatch is called it schedules
// a future Complete event in the simulation event queue.
type simDispatcher struct {
	sim *Simulation
}

func (d *simDispatcher) Dispatch(agentID scheduler.AgentID, call *scheduler.Call) error {
	lat := d.sim.provider.SampleLatency(d.sim.rng)
	actual := d.sim.provider.SampleActualCost(d.sim.rng, call)
	heap.Push(&d.sim.events, &event{
		time:       d.sim.now + lat,
		kind:       evComplete,
		agentID:    agentID,
		callID:     call.ID,
		actualCost: actual,
		latency:    lat,
		call:       call,
	})
	return nil
}

// ─── Workload ────────────────────────────────────────────────────────────────

// WorkloadKind selects the six §7.2 workloads.
type WorkloadKind int

const (
	WorkloadUniform      WorkloadKind = iota // §7.2: uniform demand
	WorkloadSkewed                           // 10:1 demand skew
	WorkloadAdversarial                      // adversarial bursts (one agent max-cost)
	WorkloadHeavyTail                        // heavy-tailed output lengths
	WorkloadMixedPrio                        // mixed priority with starvation-inducing low-prio
	WorkloadTwoResource                      // two-resource bottleneck
)

// AgentConfig describes one agent in the experiment.
type AgentConfig struct {
	ID       scheduler.AgentID
	Weight   float64
	CallsPerEpoch int // calls submitted per epoch (for demand-skew workloads)
}

// Config configures one simulation run.
type Config struct {
	Workload     WorkloadKind
	Agents       []AgentConfig
	TotalCalls   int           // stop after this many completions
	EpochLen     float64       // epoch length in seconds (for rate-limit windows)
	MockConfig   mock.Config
	Seed         int64
	MaxD         int
	ReconcileEnabled bool
}

// ─── Simulation ──────────────────────────────────────────────────────────────

// Simulation drives one experiment run to completion.
type Simulation struct {
	cfg        Config
	sched      Sched
	provider   *mock.Provider
	rng        *rand.Rand
	events     eventHeap
	now        float64
	collector  *metrics.Collector
	completions int
	callsLeft  map[scheduler.AgentID]int
}

// New creates a Simulation. The returned *Simulation must have its scheduler set via
// the returned *simDispatcher before calling Run.
func New(cfg Config, rvtSched *scheduler.Scheduler) *Simulation {
	rng := rand.New(rand.NewSource(cfg.Seed))
	prov := mock.New(cfg.MockConfig)
	disp := &simDispatcher{}
	// Wire the scheduler's dispatcher to this simulation.
	// (rvtSched was created by the caller; we replace its dispatcher here.)
	// Since New() in scheduler package takes the dispatcher at construction,
	// we use the RVT scheduler as-is and provide a wrapper.
	sim := &Simulation{
		cfg:       cfg,
		sched:     rvtSched,
		provider:  prov,
		rng:       rng,
		collector: metrics.NewCollector(len(cfg.Agents)),
		callsLeft: make(map[scheduler.AgentID]int),
	}
	disp.sim = sim
	heap.Init(&sim.events)
	return sim
}

// NewWithDispatcher creates a Simulation and returns both the sim and its dispatcher
// so the caller can wire them to a freshly created scheduler.
func NewWithDispatcher(cfg Config) (*Simulation, *SimDispatcher) {
	rng := rand.New(rand.NewSource(cfg.Seed))
	prov := mock.New(cfg.MockConfig)
	sim := &Simulation{
		cfg:       cfg,
		provider:  prov,
		rng:       rng,
		collector: metrics.NewCollector(len(cfg.Agents)),
		callsLeft: make(map[scheduler.AgentID]int),
	}
	heap.Init(&sim.events)
	disp := &SimDispatcher{sim: sim}
	return sim, disp
}

// SetSched sets the scheduler after construction (used with NewWithDispatcher).
func (s *Simulation) SetSched(sched Sched) { s.sched = sched }

// Run executes the simulation until cfg.TotalCalls completions.
func (s *Simulation) Run() *metrics.Summary {
	// Seed initial calls for all agents.
	for _, ag := range s.cfg.Agents {
		s.callsLeft[ag.ID] = s.cfg.TotalCalls / len(s.cfg.Agents)
		s.submitNext(ag.ID)
	}

	for s.completions < s.cfg.TotalCalls && len(s.events) > 0 {
		ev := heap.Pop(&s.events).(*event)
		s.now = ev.time

		switch ev.kind {
		case evComplete:
			_ = s.sched.Complete(ev.agentID, ev.callID, ev.actualCost)
			s.completions++
			s.collector.RecordCompletion(s.now, ev.agentID, ev.latency, s.sched.Stats())
			s.submitNext(ev.agentID)

		case evSubmit:
			_ = s.sched.Submit(ev.agentID, ev.call)
		}
	}

	return s.collector.Summary(s.cfg.MockConfig.CMax, s.cfg.MockConfig.CMax,
		float64(s.cfg.MaxD))
}

// submitNext schedules the agent's next call if it still has calls remaining.
func (s *Simulation) submitNext(agentID scheduler.AgentID) {
	if s.callsLeft[agentID] > 0 {
		s.callsLeft[agentID]--
		call := s.provider.GenerateCall(s.rng, agentID)
		// Submit immediately (zero think-time = continuously backlogged).
		_ = s.sched.Submit(agentID, call)
	}
}

// Now returns the current simulation time.
func (s *Simulation) Now() float64 { return s.now }

// SimDispatcher satisfies scheduler.Dispatcher and feeds completions back into the sim.
type SimDispatcher struct {
	sim *Simulation
}

func (d *SimDispatcher) Dispatch(agentID scheduler.AgentID, call *scheduler.Call) error {
	lat := d.sim.provider.SampleLatency(d.sim.rng)
	actual := d.sim.provider.SampleActualCost(d.sim.rng, call)
	heap.Push(&d.sim.events, &event{
		time:       d.sim.now + lat,
		kind:       evComplete,
		agentID:    agentID,
		callID:     call.ID,
		actualCost: actual,
		latency:    lat,
		call:       call,
	})
	return nil
}
