// Command agentkernel-eval runs one experiment configuration and writes results to CSV.
// Usage: agentkernel-eval -workload uniform -n 10 -calls 1000 -seed 42 -sched rvt -out results/run.csv
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/atharvaarbat/agent-kernel/baselines"
	"github.com/atharvaarbat/agent-kernel/estimator"
	"github.com/atharvaarbat/agent-kernel/metrics"
	"github.com/atharvaarbat/agent-kernel/mock"
	"github.com/atharvaarbat/agent-kernel/scheduler"
	"github.com/atharvaarbat/agent-kernel/sim"
)

var (
	flagWorkload   = flag.String("workload", "uniform", "uniform|skewed|adversarial|heavytail|mixed|tworesource")
	flagN          = flag.Int("n", 10, "number of agents")
	flagCalls      = flag.Int("calls", 1000, "total completions to simulate")
	flagSeed       = flag.Int64("seed", 42, "RNG seed")
	flagOut        = flag.String("out", "results/run.csv", "output CSV path")
	flagSched      = flag.String("sched", "rvt", "rvt|fcfs|rr|promiseall")
	flagNoRecon    = flag.Bool("no-reconcile", false, "disable reconciliation (Prop 2 ablation)")
	flagD          = flag.Int("d", 1, "max in-flight calls per agent")
	flagGlobalConc = flag.Int("global-concurrency", 1, "shared provider concurrency cap (1=single-server, 0=unlimited)")
	flagCMax       = flag.Float64("cmax", 5000, "C_max (max actual cost)")
	flagEMax       = flag.Float64("emax", 5000, "Ê_max (max estimate cost)")
	flagCMin       = flag.Float64("cmin", 10, "c_min (min actual cost)")
)

var workloadKinds = map[string]sim.WorkloadKind{
	"uniform":     sim.WorkloadUniform,
	"skewed":      sim.WorkloadSkewed,
	"adversarial": sim.WorkloadAdversarial,
	"heavytail":   sim.WorkloadHeavyTail,
	"mixed":       sim.WorkloadMixedPrio,
	"tworesource": sim.WorkloadTwoResource,
}

func main() {
	flag.Parse()
	start := time.Now()

	n := *flagN
	wkind, ok := workloadKinds[*flagWorkload]
	if !ok {
		log.Fatalf("unknown workload: %q (valid: uniform skewed adversarial heavytail mixed tworesource)", *flagWorkload)
	}

	agents := buildAgents(n, wkind)

	mockCfg := mock.Default()
	mockCfg.CMax = *flagCMax
	mockCfg.CMin = *flagCMin

	cfg := sim.Config{
		Workload:          wkind,
		Agents:            agents,
		TotalCalls:        *flagCalls,
		EpochLen:          60,
		MockConfig:        mockCfg,
		Seed:              *flagSeed,
		MaxD:              *flagD,
		GlobalConcurrency: *flagGlobalConc,
		ReconcileEnabled:  !*flagNoRecon,
	}

	simInst, disp := sim.NewWithDispatcher(cfg)
	sched := buildSched(*flagSched, agents, disp, cfg)
	simInst.SetSched(sched)

	summary := simInst.Run()

	elapsed := time.Since(start)
	fmt.Printf("sched=%-10s workload=%-12s n=%d calls=%d seed=%d  elapsed=%s\n",
		*flagSched, *flagWorkload, n, *flagCalls, *flagSeed, elapsed)
	fmt.Printf("  Jain=%.4f  P50=%.3fs  P99=%.3fs  starvation=%d  maxGapRatio=%.4f  thm1=%v\n",
		summary.JainIndex, summary.P50Latency, summary.P99Latency,
		summary.Starvation, summary.MaxGapRatio, summary.Thm1Satisfied)

	if err := writeCSV(*flagOut, *flagSched, *flagWorkload, summary, cfg); err != nil {
		log.Fatalf("write CSV: %v", err)
	}
	fmt.Printf("  → %s\n", *flagOut)
}

// buildAgents constructs per-agent configs for the given workload.
func buildAgents(n int, wkind sim.WorkloadKind) []sim.AgentConfig {
	agents := make([]sim.AgentConfig, n)
	for i := range agents {
		w := 1.0
		if wkind == sim.WorkloadMixedPrio && i < n/2 {
			w = 3.0 // first half are high-priority (weight 3)
		}
		agents[i] = sim.AgentConfig{
			ID:     scheduler.AgentID(i + 1),
			Weight: w,
		}
	}
	return agents
}

// buildSched constructs the requested scheduler and registers all agents.
// cfg.GlobalConcurrency is the shared provider cap (how many calls can be in-flight
// globally across all agents). 0 = unlimited.
func buildSched(name string, agents []sim.AgentConfig, disp *sim.SimDispatcher, cfg sim.Config) sim.Sched {
	gc := cfg.GlobalConcurrency
	if gc <= 0 {
		gc = len(agents) * cfg.MaxD // unlimited → n×d total slots
	}

	switch name {
	case "rvt":
		p := scheduler.Params{
			MaxD:             float64(cfg.MaxD),
			GlobalMaxD:       gc,
			CMax:             *flagCMax,
			EMax:             *flagEMax,
			CMin:             *flagCMin,
			ReconcileEnabled: cfg.ReconcileEnabled,
		}
		est := estimator.New(0.1, *flagCMax/2, *flagEMax)
		rvt := scheduler.New(p, est, disp)
		for _, ag := range agents {
			rvt.AddAgent(ag.ID, ag.Weight)
		}
		return rvt

	case "fcfs":
		f := baselines.NewFCFS(gc, disp)
		for _, ag := range agents {
			f.AddAgent(ag.ID)
		}
		return f

	case "rr":
		rr := baselines.NewRoundRobin(gc, disp)
		for _, ag := range agents {
			rr.AddAgent(ag.ID)
		}
		return rr

	case "promiseall":
		pa := baselines.NewPromiseAll(gc, disp)
		for _, ag := range agents {
			pa.AddAgent(ag.ID)
		}
		return pa

	default:
		log.Fatalf("unknown scheduler: %q (valid: rvt fcfs rr promiseall)", name)
		return nil
	}
}

func writeCSV(path, schedName, workloadName string, s *metrics.Summary, cfg sim.Config) error {
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
		{"sched", schedName},
		{"workload", workloadName},
		{"n_agents", strconv.Itoa(len(cfg.Agents))},
		{"total_calls", strconv.Itoa(cfg.TotalCalls)},
		{"seed", strconv.FormatInt(cfg.Seed, 10)},
		{"reconcile", strconv.FormatBool(cfg.ReconcileEnabled)},
		{"jain_index", strconv.FormatFloat(s.JainIndex, 'f', 6, 64)},
		{"p50_latency_s", strconv.FormatFloat(s.P50Latency, 'f', 6, 64)},
		{"p99_latency_s", strconv.FormatFloat(s.P99Latency, 'f', 6, 64)},
		{"starvation_events", strconv.Itoa(s.Starvation)},
		{"max_gap_ratio", strconv.FormatFloat(s.MaxGapRatio, 'f', 6, 64)},
		{"thm1_satisfied", strconv.FormatBool(s.Thm1Satisfied)},
	}
	for _, row := range rows {
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return nil
}
