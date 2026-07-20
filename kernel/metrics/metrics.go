// Package metrics computes the evaluation metrics defined in §7.2 of the paper.
package metrics

import (
	"math"
	"sort"

	"github.com/atharva-arbat/agent-kernel/scheduler"
)

// ─── Jain's fairness index ────────────────────────────────────────────────────

// JainIndex computes Jain's fairness index over the slice of normalized allocations x.
// J = (Σx_i)² / (n · Σx_i²) ∈ (0,1]; 1 = perfectly fair.
func JainIndex(x []float64) float64 {
	if len(x) == 0 {
		return 1
	}
	var sumX, sumX2 float64
	for _, v := range x {
		sumX += v
		sumX2 += v * v
	}
	n := float64(len(x))
	if sumX2 == 0 {
		return 1
	}
	return (sumX * sumX) / (n * sumX2)
}

// ─── Gap vs. Theorem 1 bound ─────────────────────────────────────────────────

// GapResult holds the measured gap and the theoretical bound for one snapshot.
type GapResult struct {
	MaxGap     float64 // max pairwise |A_i/w_i − A_j/w_j| observed
	Bound      float64 // Theorem 1 bound d*(C_max/w_i + Ê_max/w_j) for the worst pair
	Normalized float64 // MaxGap / Bound ∈ [0,1] (1 = bound exactly tight)
}

// ComputeGap computes the worst-case pairwise service gap among agents and compares
// it to the Theorem 1 bound parameterized by (cmax, emax, d).
func ComputeGap(stats map[scheduler.AgentID]*scheduler.AgentStats, cmax, emax, d float64) GapResult {
	type entry struct {
		w, a float64
	}
	var agents []entry
	for _, ag := range stats {
		if ag.InBacklog || ag.Service > 0 {
			agents = append(agents, entry{ag.Weight, ag.Service})
		}
	}
	if len(agents) < 2 {
		return GapResult{}
	}

	var maxGap, worstBound float64
	for i, ai := range agents {
		for _, aj := range agents[i+1:] {
			gapIJ := ai.a/ai.w - aj.a/aj.w
			gapJI := aj.a/aj.w - ai.a/ai.w
			boundIJ := d * (cmax/ai.w + emax/aj.w)
			boundJI := d * (cmax/aj.w + emax/ai.w)
			if gapIJ > maxGap {
				maxGap = gapIJ
				worstBound = boundIJ
			}
			if gapJI > maxGap {
				maxGap = gapJI
				worstBound = boundJI
			}
		}
	}
	norm := 0.0
	if worstBound > 0 {
		norm = maxGap / worstBound
	}
	return GapResult{MaxGap: maxGap, Bound: worstBound, Normalized: norm}
}

// ─── Latency percentiles ──────────────────────────────────────────────────────

// Percentile returns the p-th percentile (0–100) of latencies in seconds.
func Percentile(latencies []float64, p float64) float64 {
	if len(latencies) == 0 {
		return 0
	}
	s := make([]float64, len(latencies))
	copy(s, latencies)
	sort.Float64s(s)
	idx := (p / 100.0) * float64(len(s)-1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	if lo == hi {
		return s[lo]
	}
	frac := idx - float64(lo)
	return s[lo]*(1-frac) + s[hi]*frac
}

// ─── Collector (records per-event metrics during simulation) ──────────────────

// Collector accumulates per-event data and produces a Summary at the end.
type Collector struct {
	n          int
	latencies  []float64
	services   map[scheduler.AgentID][]float64 // per-agent service snapshots
	gapHistory []GapResult
	starvation int // completions where some backlogged agent waited >N epochs
}

// NewCollector creates a Collector for n agents.
func NewCollector(n int) *Collector {
	return &Collector{
		n:        n,
		services: make(map[scheduler.AgentID][]float64),
	}
}

// RecordCompletion is called after each Complete event.
func (c *Collector) RecordCompletion(simTime float64, agentID scheduler.AgentID,
	latency float64, stats map[scheduler.AgentID]*scheduler.AgentStats) {
	c.latencies = append(c.latencies, latency)
	for id, ag := range stats {
		c.services[id] = append(c.services[id], ag.Service/ag.Weight)
	}
}

// Summary computes aggregate statistics over the collected data.
func (c *Collector) Summary(cmax, emax, d float64) *Summary {
	// Build allocation vector for Jain index (last snapshot per agent).
	allocs := make([]float64, 0, c.n)
	for _, vs := range c.services {
		if len(vs) > 0 {
			allocs = append(allocs, vs[len(vs)-1])
		}
	}

	// Compute latest gap (simplified: uses last recorded services).
	latestStats := make(map[scheduler.AgentID]*scheduler.AgentStats)
	for id, vs := range c.services {
		if len(vs) > 0 {
			latestStats[id] = &scheduler.AgentStats{
				ID:      id,
				Weight:  1.0, // weight info not recorded here; use 1.0 as placeholder
				Service: vs[len(vs)-1],
			}
		}
	}

	return &Summary{
		JainIndex:  JainIndex(allocs),
		P50Latency: Percentile(c.latencies, 50),
		P99Latency: Percentile(c.latencies, 99),
		Starvation: c.starvation,
		GapHistory: c.gapHistory,
	}
}

// Summary is the final output of one simulation run.
type Summary struct {
	JainIndex  float64
	P50Latency float64 // seconds
	P99Latency float64 // seconds
	Starvation int     // count of epochs with at least one starved agent
	GapHistory []GapResult
	// Thm1Satisfied: true if MaxGap/Bound ≤ 1 for all snapshots.
	Thm1Satisfied bool
}
