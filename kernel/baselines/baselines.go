// Package baselines implements the four comparison schedulers used in §7.2:
// FCFS, unweighted round-robin, Promise.all (no scheduling), and token-bucket-per-agent.
// Each exposes the same Submit/Complete/Stats interface as the RVT scheduler so the
// experiment runner can swap them without changes.
package baselines

import (
	"sync"

	"github.com/atharva-arbat/agent-kernel/scheduler"
)

// ─── FCFS (first-come, first-served) ─────────────────────────────────────────

// FCFS dispatches calls in strict arrival order, ignoring weights.
type FCFS struct {
	mu       sync.Mutex
	agents   map[scheduler.AgentID]*agentRec
	queue    []*queueEntry // global FIFO
	inFlight map[int64]*queueEntry
	maxD     int
	inflight int
	disp     scheduler.Dispatcher
	nextID   int64
}

type agentRec struct {
	id      scheduler.AgentID
	service float64
}

type queueEntry struct {
	agentID scheduler.AgentID
	call    *scheduler.Call
}

func NewFCFS(maxD int, disp scheduler.Dispatcher) *FCFS {
	return &FCFS{
		agents:   make(map[scheduler.AgentID]*agentRec),
		inFlight: make(map[int64]*queueEntry),
		maxD:     maxD,
		disp:     disp,
	}
}

func (f *FCFS) AddAgent(id scheduler.AgentID) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.agents[id] = &agentRec{id: id}
}

func (f *FCFS) Submit(agentID scheduler.AgentID, call *scheduler.Call) error {
	f.mu.Lock()
	if call.ID == 0 {
		f.nextID++
		call.ID = f.nextID
	}
	call.AgentID = agentID
	f.queue = append(f.queue, &queueEntry{agentID, call})
	dispatches := f.drain()
	f.mu.Unlock()
	for _, e := range dispatches {
		_ = f.disp.Dispatch(e.agentID, e.call)
	}
	return nil
}

func (f *FCFS) Complete(agentID scheduler.AgentID, callID int64, actualCost float64) error {
	f.mu.Lock()
	if ag := f.agents[agentID]; ag != nil {
		ag.service += actualCost
	}
	delete(f.inFlight, callID)
	f.inflight--
	dispatches := f.drain()
	f.mu.Unlock()
	for _, e := range dispatches {
		_ = f.disp.Dispatch(e.agentID, e.call)
	}
	return nil
}

func (f *FCFS) drain() []*queueEntry {
	var out []*queueEntry
	for f.inflight < f.maxD && len(f.queue) > 0 {
		e := f.queue[0]
		f.queue = f.queue[1:]
		f.inFlight[e.call.ID] = e
		f.inflight++
		out = append(out, e)
	}
	return out
}

func (f *FCFS) Stats() map[scheduler.AgentID]*scheduler.AgentStats {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make(map[scheduler.AgentID]*scheduler.AgentStats)
	for id, ag := range f.agents {
		out[id] = &scheduler.AgentStats{ID: ag.id, Service: ag.service, Weight: 1}
	}
	return out
}

// ─── Round-robin (unweighted) ────────────────────────────────────────────────

// RoundRobin cycles through agents in registration order, dispatching one call
// per agent per cycle regardless of weight.
type RoundRobin struct {
	mu       sync.Mutex
	order    []scheduler.AgentID
	queues   map[scheduler.AgentID][]*scheduler.Call
	services map[scheduler.AgentID]float64
	inFlight map[int64]scheduler.AgentID
	inflight int
	maxD     int
	pos      int
	disp     scheduler.Dispatcher
	nextID   int64
}

func NewRoundRobin(maxD int, disp scheduler.Dispatcher) *RoundRobin {
	return &RoundRobin{
		queues:   make(map[scheduler.AgentID][]*scheduler.Call),
		services: make(map[scheduler.AgentID]float64),
		inFlight: make(map[int64]scheduler.AgentID),
		maxD:     maxD,
		disp:     disp,
	}
}

func (r *RoundRobin) AddAgent(id scheduler.AgentID) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.order = append(r.order, id)
	r.queues[id] = nil
}

func (r *RoundRobin) Submit(agentID scheduler.AgentID, call *scheduler.Call) error {
	r.mu.Lock()
	if call.ID == 0 {
		r.nextID++
		call.ID = r.nextID
	}
	call.AgentID = agentID
	r.queues[agentID] = append(r.queues[agentID], call)
	dispatches := r.drain()
	r.mu.Unlock()
	for _, c := range dispatches {
		_ = r.disp.Dispatch(c.AgentID, c)
	}
	return nil
}

func (r *RoundRobin) Complete(agentID scheduler.AgentID, callID int64, actualCost float64) error {
	r.mu.Lock()
	r.services[agentID] += actualCost
	delete(r.inFlight, callID)
	r.inflight--
	dispatches := r.drain()
	r.mu.Unlock()
	for _, c := range dispatches {
		_ = r.disp.Dispatch(c.AgentID, c)
	}
	return nil
}

func (r *RoundRobin) drain() []*scheduler.Call {
	var out []*scheduler.Call
	if len(r.order) == 0 {
		return nil
	}
	for r.inflight < r.maxD {
		found := false
		for range r.order {
			id := r.order[r.pos%len(r.order)]
			r.pos++
			if q := r.queues[id]; len(q) > 0 {
				call := q[0]
				r.queues[id] = q[1:]
				r.inFlight[call.ID] = id
				r.inflight++
				out = append(out, call)
				found = true
				break
			}
		}
		if !found {
			break
		}
	}
	return out
}

func (r *RoundRobin) Stats() map[scheduler.AgentID]*scheduler.AgentStats {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[scheduler.AgentID]*scheduler.AgentStats)
	for _, id := range r.order {
		out[id] = &scheduler.AgentStats{ID: id, Service: r.services[id], Weight: 1,
			QueueLen: len(r.queues[id])}
	}
	return out
}

// ─── PromiseAll (no scheduling) ───────────────────────────────────────────────

// PromiseAll dispatches every call immediately with no scheduling policy.
// This models the framework-level "fire-and-forget" concurrency pattern.
type PromiseAll struct {
	mu       sync.Mutex
	services map[scheduler.AgentID]float64
	disp     scheduler.Dispatcher
	nextID   int64
}

func NewPromiseAll(disp scheduler.Dispatcher) *PromiseAll {
	return &PromiseAll{services: make(map[scheduler.AgentID]float64), disp: disp}
}

func (p *PromiseAll) AddAgent(id scheduler.AgentID) {
	p.mu.Lock()
	p.services[id] = 0
	p.mu.Unlock()
}

func (p *PromiseAll) Submit(agentID scheduler.AgentID, call *scheduler.Call) error {
	p.mu.Lock()
	if call.ID == 0 {
		p.nextID++
		call.ID = p.nextID
	}
	call.AgentID = agentID
	p.mu.Unlock()
	return p.disp.Dispatch(agentID, call)
}

func (p *PromiseAll) Complete(agentID scheduler.AgentID, _ int64, actualCost float64) error {
	p.mu.Lock()
	p.services[agentID] += actualCost
	p.mu.Unlock()
	return nil
}

func (p *PromiseAll) Stats() map[scheduler.AgentID]*scheduler.AgentStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make(map[scheduler.AgentID]*scheduler.AgentStats)
	for id, svc := range p.services {
		out[id] = &scheduler.AgentStats{ID: id, Service: svc, Weight: 1}
	}
	return out
}

// ─── TokenBucket (per-agent rate limiter) ────────────────────────────────────

// TokenBucket implements per-agent token-bucket rate limiting (§5.3 folk remedy).
// Each agent has a bucket of capacity C that refills at rate r tokens/s.
// A call is dispatched only when the bucket holds ≥ estimate tokens.
// This is not work-conserving: idle agents don't share their unused rate.
type TokenBucket struct {
	mu       sync.Mutex
	agents   map[scheduler.AgentID]*bucketState
	disp     scheduler.Dispatcher
	nextID   int64
}

type bucketState struct {
	capacity float64
	tokens   float64
	rate     float64 // tokens per second
	queue    []*scheduler.Call
	service  float64
	inflight int
}

func NewTokenBucket(disp scheduler.Dispatcher) *TokenBucket {
	return &TokenBucket{
		agents: make(map[scheduler.AgentID]*bucketState),
		disp:   disp,
	}
}

func (tb *TokenBucket) AddAgent(id scheduler.AgentID, capacity, rate float64) {
	tb.mu.Lock()
	tb.agents[id] = &bucketState{capacity: capacity, tokens: capacity, rate: rate}
	tb.mu.Unlock()
}

func (tb *TokenBucket) Submit(agentID scheduler.AgentID, call *scheduler.Call) error {
	tb.mu.Lock()
	if call.ID == 0 {
		tb.nextID++
		call.ID = tb.nextID
	}
	call.AgentID = agentID
	ag := tb.agents[agentID]
	ag.queue = append(ag.queue, call)
	dispatches := tb.drainAgent(agentID)
	tb.mu.Unlock()
	for _, c := range dispatches {
		_ = tb.disp.Dispatch(agentID, c)
	}
	return nil
}

func (tb *TokenBucket) Complete(agentID scheduler.AgentID, _ int64, actualCost float64) error {
	tb.mu.Lock()
	ag := tb.agents[agentID]
	ag.service += actualCost
	ag.inflight--
	dispatches := tb.drainAgent(agentID)
	tb.mu.Unlock()
	for _, c := range dispatches {
		_ = tb.disp.Dispatch(agentID, c)
	}
	return nil
}

func (tb *TokenBucket) drainAgent(id scheduler.AgentID) []*scheduler.Call {
	ag := tb.agents[id]
	var out []*scheduler.Call
	for len(ag.queue) > 0 {
		call := ag.queue[0]
		est := call.EstimateCost
		if est == 0 {
			est = 100 // default estimate if not set
		}
		if ag.tokens < est {
			break
		}
		ag.queue = ag.queue[1:]
		ag.tokens -= est
		ag.inflight++
		out = append(out, call)
	}
	return out
}

func (tb *TokenBucket) Stats() map[scheduler.AgentID]*scheduler.AgentStats {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	out := make(map[scheduler.AgentID]*scheduler.AgentStats)
	for id, ag := range tb.agents {
		out[id] = &scheduler.AgentStats{ID: id, Service: ag.service, Weight: 1,
			InFlight: ag.inflight, QueueLen: len(ag.queue)}
	}
	return out
}
