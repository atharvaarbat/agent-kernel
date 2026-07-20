// Package estimator implements the per-agent cost estimator used by the RVT scheduler.
package estimator

import (
	"github.com/atharva-arbat/agent-kernel/scheduler"
)

// EWMA implements an exponentially weighted moving average estimator.
// The per-agent EWMA is bootstrapped from a configurable prior, then updated on each
// completion. The estimate is clamped at MaxTokensCost to enforce Ê_max ≤ CMax + EMax.
//
// Setting ReconcileEnabled=false in scheduler.Params turns the scheduler into the
// "estimate-only" ablation that instantiates Proposition 2.
type EWMA struct {
	alpha        float64            // smoothing factor ∈ (0,1]; default 0.1
	prior        float64            // bootstrap value for agents with no history
	maxTokenCost float64            // upper clamp: max_tokens * cost_per_output_token
	state        map[scheduler.AgentID]float64
}

// New creates an EWMA estimator.
//   - alpha: smoothing factor (0 < alpha ≤ 1). Smaller = slower adaptation.
//   - prior: starting estimate for agents with no observation history.
//   - maxTokenCost: upper clamp (enforces Ê_max bound in A4).
func New(alpha, prior, maxTokenCost float64) *EWMA {
	return &EWMA{
		alpha:        alpha,
		prior:        prior,
		maxTokenCost: maxTokenCost,
		state:        make(map[scheduler.AgentID]float64),
	}
}

// Estimate returns the current EWMA estimate for agent id, clamped at maxTokenCost.
// If MaxTokens is set in the call, the clamp is min(ewma, call.MaxTokens * costPerToken);
// here we use the global maxTokenCost clamp (agent declares max_tokens at submit time).
func (e *EWMA) Estimate(id scheduler.AgentID, call *scheduler.Call) float64 {
	est, ok := e.state[id]
	if !ok {
		est = e.prior
	}
	// Apply per-call max_tokens clamp if caller set it.
	clamp := e.maxTokenCost
	if call.MaxTokens > 0 {
		callClamp := float64(call.MaxTokens) // simplified: tokens == cost in unit model
		if callClamp < clamp {
			clamp = callClamp
		}
	}
	if est > clamp {
		est = clamp
	}
	return est
}

// Update applies the EWMA update: ê_{t+1} = alpha*actual + (1-alpha)*ê_t.
func (e *EWMA) Update(id scheduler.AgentID, actualCost float64) {
	prev, ok := e.state[id]
	if !ok {
		prev = e.prior
	}
	e.state[id] = e.alpha*actualCost + (1-e.alpha)*prev
}

// Constant returns an Estimator that always returns the same value (useful for
// reproducing analytic examples and unit tests).
func Constant(v float64) scheduler.Estimator { return &constEstimator{v} }

type constEstimator struct{ v float64 }

func (c *constEstimator) Estimate(_ scheduler.AgentID, _ *scheduler.Call) float64 { return c.v }
func (c *constEstimator) Update(_ scheduler.AgentID, _ float64)                   {}
