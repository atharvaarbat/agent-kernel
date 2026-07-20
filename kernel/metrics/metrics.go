// Package metrics computes the evaluation metrics defined in §7.2 of the paper.
package metrics

import (
	"math"
	"sort"

	"github.com/atharvaarbat/agent-kernel/scheduler"
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
// it to the Theorem 1 bound parameterised by (cmax, emax, d).
func ComputeGap(stats map[scheduler.AgentID]*scheduler.AgentStats, cmax, emax, d float64) GapResult {
	type entry struct{ w, a float64 }
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
			if ai.w <= 0 || aj.w <= 0 {
				continue
			}
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

// Percentile returns the p-th percentile (0–100) of latencies.
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
	return s[lo]*(1-(idx-float64(lo))) + s[hi]*(idx-float64(lo))
}

// ─── Collector ────────────────────────────────────────────────────────────────

// Collector accumulates per-event data and produces a Summary at end of run.
type Collector struct {
	n         int
	cmax      float64
	emax      float64
	d         float64
	latencies []float64

	// Snapshot of last known stats per agent (retains actual weights).
	lastStats map[scheduler.AgentID]*scheduler.AgentStats

	// Running max of gap/bound across all snapshots (for H1b).
	maxGapRatio float64

	// Starvation: count of completions where ≥1 backlogged agent waited >threshold.
	starvation      int
	completionCount int
	lastServed      map[scheduler.AgentID]int // completion index when agent was last dispatched
}

// NewCollector creates a Collector. cmax/emax/d are the Theorem 1 parameters used
// to compute the gap bound incrementally at each completion.
func NewCollector(n int, cmax, emax, d float64) *Collector {
	return &Collector{
		n:          n,
		cmax:       cmax,
		emax:       emax,
		d:          d,
		lastStats:  make(map[scheduler.AgentID]*scheduler.AgentStats),
		lastServed: make(map[scheduler.AgentID]int),
	}
}

const starveThresholdMul = 3 // starved if not served in n*3 completions

// RecordCompletion is called after each Complete event with the current stats snapshot.
func (c *Collector) RecordCompletion(
	_ float64,
	_ scheduler.AgentID,
	latency float64,
	stats map[scheduler.AgentID]*scheduler.AgentStats,
) {
	c.latencies = append(c.latencies, latency)
	c.completionCount++

	// Track last-dispatched index for starvation detection.
	for id, ag := range stats {
		c.lastStats[id] = ag
		if ag.InFlight > 0 {
			c.lastServed[id] = c.completionCount
		}
	}

	// Starvation: any backlogged agent not served in n*starveThresholdMul completions.
	threshold := c.n * starveThresholdMul
	if threshold < 10 {
		threshold = 10
	}
	for id, ag := range stats {
		if ag.QueueLen > 0 && ag.InFlight == 0 {
			last, seen := c.lastServed[id]
			if !seen || c.completionCount-last > threshold {
				c.starvation++
				c.lastServed[id] = c.completionCount // reset to avoid double-counting
			}
		}
	}

	// Gap ratio update.
	if len(stats) >= 2 {
		gr := ComputeGap(stats, c.cmax, c.emax, c.d)
		if gr.Normalized > c.maxGapRatio {
			c.maxGapRatio = gr.Normalized
		}
	}
}

// Summary computes aggregate statistics over the collected data.
func (c *Collector) Summary() *Summary {
	// Jain index: use A_i/w_i for each agent (normalized allocation).
	allocs := make([]float64, 0, len(c.lastStats))
	for _, ag := range c.lastStats {
		if ag.Weight > 0 && ag.Service > 0 {
			allocs = append(allocs, ag.Service/ag.Weight)
		}
	}

	// Final gap ratio snapshot.
	finalGR := ComputeGap(c.lastStats, c.cmax, c.emax, c.d)
	maxRatio := c.maxGapRatio
	if finalGR.Normalized > maxRatio {
		maxRatio = finalGR.Normalized
	}

	return &Summary{
		JainIndex:     JainIndex(allocs),
		P50Latency:    Percentile(c.latencies, 50),
		P99Latency:    Percentile(c.latencies, 99),
		Starvation:    c.starvation,
		MaxGapRatio:   maxRatio,
		Thm1Satisfied: maxRatio <= 1.0+1e-9,
	}
}

// Summary is the final output of one simulation run.
type Summary struct {
	JainIndex     float64
	P50Latency    float64 // seconds
	P99Latency    float64 // seconds
	Starvation    int     // count of starvation events (§H1c)
	MaxGapRatio   float64 // max(gap/bound) across all snapshots (§H1b)
	Thm1Satisfied bool    // true iff MaxGapRatio ≤ 1 (Theorem 1 not violated)
}
