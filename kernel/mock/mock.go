// Package mock provides a deterministic mock LLM backend for the evaluation harness.
// Output lengths follow a log-normal distribution (heavy-tailed, matching published
// LLM serving trace statistics). Cost = α·in_tokens + β·out_tokens.
package mock

import (
	"math"
	"math/rand"

	"github.com/atharvaarbat/agent-kernel/scheduler"
)

// Config parameterizes the mock provider distribution.
type Config struct {
	// Output-length distribution: ln(out_tokens) ~ N(LogNormalMu, LogNormalSigma^2).
	// Defaults: Mu=6.0, Sigma=1.5 → median ≈ 403 tokens, heavy upper tail.
	LogNormalMu    float64
	LogNormalSigma float64

	// Cost coefficients. Cost = Alpha*in_tokens + Beta*out_tokens.
	Alpha float64 // per-input-token cost (e.g. 0.001 for GPT-4-class model)
	Beta  float64 // per-output-token cost (typically 3–4× Alpha)

	// Latency: drawn from Exp(1/MeanLatency) seconds (not used by scheduler, only
	// reported in simulation time so metrics can compute wall-clock throughput).
	MeanLatency float64 // seconds

	// CMax and CMin are applied as hard caps on sampled actual cost.
	CMax float64
	CMin float64
}

// Default returns a Config calibrated to approximate typical mid-size LLM call
// distributions from published serving traces.
func Default() Config {
	return Config{
		LogNormalMu:    6.0,
		LogNormalSigma: 1.5,
		Alpha:          1.0,
		Beta:           3.0,
		MeanLatency:    2.0,
		CMax:           5000.0,
		CMin:           10.0,
	}
}

// Provider is a deterministic mock LLM that generates call outcomes from a seeded RNG.
type Provider struct {
	cfg Config
}

// New creates a Provider with the given config.
func New(cfg Config) *Provider { return &Provider{cfg: cfg} }

// SampleActualCost draws an actual cost for call c using rng.
// The output-token count is log-normal; in_tokens come from the call directly.
func (p *Provider) SampleActualCost(rng *rand.Rand, call *scheduler.Call) float64 {
	outTokens := p.sampleOutputTokens(rng, call)
	cost := p.cfg.Alpha*float64(call.InTokens) + p.cfg.Beta*float64(outTokens)
	if cost < p.cfg.CMin {
		cost = p.cfg.CMin
	}
	if cost > p.cfg.CMax {
		cost = p.cfg.CMax
	}
	return cost
}

// SampleLatency draws a service time in seconds for a call (exponential distribution).
func (p *Provider) SampleLatency(rng *rand.Rand) float64 {
	return rng.ExpFloat64() * p.cfg.MeanLatency
}

// GenerateCall creates a synthetic call for agent i with randomised in-token count
// and a max_tokens upper bound consistent with the configured CMax.
func (p *Provider) GenerateCall(rng *rand.Rand, agentID scheduler.AgentID) *scheduler.Call {
	inTok := int64(50 + rng.Intn(950)) // 50–1000 input tokens
	maxTok := int64(math.Ceil(p.cfg.CMax / p.cfg.Beta))
	return &scheduler.Call{AgentID: agentID, InTokens: inTok, MaxTokens: maxTok}
}

func (p *Provider) sampleOutputTokens(rng *rand.Rand, call *scheduler.Call) int64 {
	// Box-Muller to get N(0,1) from uniform RNG.
	u1 := rng.Float64()
	if u1 == 0 {
		u1 = 1e-10
	}
	u2 := rng.Float64()
	z := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	outF := math.Exp(p.cfg.LogNormalMu + p.cfg.LogNormalSigma*z)
	out := int64(math.Round(outF))
	if out < 1 {
		out = 1
	}
	// Clamp to max_tokens declared by the call (admission safety).
	if call.MaxTokens > 0 && out > call.MaxTokens {
		out = call.MaxTokens
	}
	return out
}
