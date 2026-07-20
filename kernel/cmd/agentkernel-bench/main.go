// Command agentkernel-bench measures H0a: the wall-clock and token-spend overhead
// of RVT mediation vs. a minimal direct-accounting baseline.
//
// Usage: agentkernel-bench [-n 10] [-calls 50000] [-mean-latency-ms 2000] [-out results/bench.csv]
//
// Methodology:
//
//	Direct path: minimal per-agent service accounting (array update per call).
//	Mediated path: full Submit+Complete cycle through the RVT scheduler.
//	Overhead per call = mediated_ns_per_call - direct_ns_per_call.
//	Wall overhead % = overhead_per_call / mean_latency_ns * 100.
//	Token overhead = (Σestimate - Σactual) / Σactual * 100.
//	H0a gate: wall overhead < 5%.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/atharvaarbat/agent-kernel/estimator"
	"github.com/atharvaarbat/agent-kernel/scheduler"
)

var (
	flagN             = flag.Int("n", 10, "number of agents")
	flagCalls         = flag.Int("calls", 50000, "total calls per path")
	flagMeanLatencyMs = flag.Float64("mean-latency-ms", 2000, "mean LLM inference latency (ms)")
	flagOut           = flag.String("out", "results/bench.csv", "output CSV path")
)

// syncDispatcher counts dispatches without doing real work.
type syncDispatcher struct{ count int64 }

func (d *syncDispatcher) Dispatch(_ scheduler.AgentID, _ *scheduler.Call) error {
	atomic.AddInt64(&d.count, 1)
	return nil
}

func main() {
	flag.Parse()

	n := *flagN
	calls := *flagCalls
	meanLatencyNs := *flagMeanLatencyMs * 1e6 // ms → ns

	// ── Direct path: minimal per-agent accounting (baseline) ─────────────────
	service := make([]float64, n)
	directStart := time.Now()
	for i := 0; i < calls; i++ {
		idx := i % n
		service[idx] += 400.0 // representative average cost
	}
	directNs := time.Since(directStart).Nanoseconds()
	_ = service // prevent optimisation

	// ── Mediated path: full RVT Submit + Complete cycle ───────────────────────
	mediatedDisp := &syncDispatcher{}
	params := scheduler.Params{
		MaxD:             float64(calls), // no per-agent in-flight limit for bench
		GlobalMaxD:       0,              // no global limit; all dispatched immediately
		CMax:             5000,
		EMax:             5000,
		CMin:             10,
		ReconcileEnabled: true,
	}
	est := estimator.New(0.1, 2500, 5000)
	sched := scheduler.New(params, est, mediatedDisp)
	for i := 0; i < n; i++ {
		sched.AddAgent(scheduler.AgentID(i+1), 1.0)
	}

	const actualCost = 400.0
	var totalEstimated, totalActual float64
	mediatedStart := time.Now()
	for i := 0; i < calls; i++ {
		agentID := scheduler.AgentID(i%n + 1)
		call := &scheduler.Call{InTokens: 100, MaxTokens: 1667}
		_ = sched.Submit(agentID, call)
		totalEstimated += call.EstimateCost
		_ = sched.Complete(agentID, call.ID, actualCost)
		totalActual += actualCost
	}
	mediatedNs := time.Since(mediatedStart).Nanoseconds()

	// ── Overhead metrics ─────────────────────────────────────────────────────
	directPerCall := float64(directNs) / float64(calls)
	mediatedPerCall := float64(mediatedNs) / float64(calls)
	overheadPerCallNs := mediatedPerCall - directPerCall
	wallOverheadPct := 100.0 * overheadPerCallNs / meanLatencyNs

	tokenOverheadPct := 0.0
	if totalActual > 0 {
		tokenOverheadPct = 100.0 * (totalEstimated - totalActual) / totalActual
	}

	h0aPass := wallOverheadPct < 5.0

	fmt.Printf("H0a Overhead Benchmark  n=%d  calls=%d  mean_latency=%.0fms\n",
		n, calls, *flagMeanLatencyMs)
	fmt.Printf("  Direct path:   %.1fns/call  total=%dms\n", directPerCall, directNs/1e6)
	fmt.Printf("  Mediated path: %.1fns/call  total=%dms\n", mediatedPerCall, mediatedNs/1e6)
	fmt.Printf("  Sched overhead: %.1fns/call  (%.4f%% of %.0fms inference)\n",
		overheadPerCallNs, wallOverheadPct, *flagMeanLatencyMs)
	fmt.Printf("  Token overhead: %+.4f%%  (target: <1%%)\n", tokenOverheadPct)
	fmt.Printf("  H0a gate: %v  (wall overhead < 5%% of inference time)\n", h0aPass)

	if err := writeCSV(*flagOut, n, calls, directPerCall, mediatedPerCall,
		overheadPerCallNs, wallOverheadPct, tokenOverheadPct, h0aPass); err != nil {
		log.Fatalf("write CSV: %v", err)
	}
	fmt.Printf("  → %s\n", *flagOut)
}

func writeCSV(path string, n, calls int, directNs, mediatedNs, overheadNs, wallPct, tokenPct float64, pass bool) error {
	if err := os.MkdirAll("results", 0755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	_ = w.Write([]string{"metric", "value"})
	rows := [][]string{
		{"n_agents", strconv.Itoa(n)},
		{"total_calls", strconv.Itoa(calls)},
		{"direct_ns_per_call", strconv.FormatFloat(directNs, 'f', 2, 64)},
		{"mediated_ns_per_call", strconv.FormatFloat(mediatedNs, 'f', 2, 64)},
		{"overhead_ns_per_call", strconv.FormatFloat(overheadNs, 'f', 2, 64)},
		{"wall_overhead_pct", strconv.FormatFloat(wallPct, 'f', 6, 64)},
		{"token_overhead_pct", strconv.FormatFloat(tokenPct, 'f', 6, 64)},
		{"h0a_pass", strconv.FormatBool(pass)},
	}
	for _, row := range rows {
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return nil
}
