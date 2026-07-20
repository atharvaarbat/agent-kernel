// Command agentkernel-eval runs one experiment configuration and writes results to CSV.
// Usage: agentkernel-eval -workload uniform -n 10 -calls 1000 -seed 42 -out results/run.csv
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/atharva-arbat/agent-kernel/estimator"
	"github.com/atharva-arbat/agent-kernel/metrics"
	"github.com/atharva-arbat/agent-kernel/mock"
	"github.com/atharva-arbat/agent-kernel/scheduler"
	"github.com/atharva-arbat/agent-kernel/sim"
)

var (
	flagWorkload  = flag.String("workload", "uniform", "uniform|skewed|adversarial|heavytail|mixed|tworesource")
	flagN         = flag.Int("n", 10, "number of agents")
	flagCalls     = flag.Int("calls", 1000, "total completions to simulate")
	flagSeed      = flag.Int64("seed", 42, "RNG seed")
	flagOut       = flag.String("out", "results/run.csv", "output CSV path")
	flagSched     = flag.String("sched", "rvt", "rvt|fcfs|rr|promiseall")
	flagNoRecon   = flag.Bool("no-reconcile", false, "disable reconciliation (Prop 2 ablation)")
	flagD         = flag.Int("d", 1, "max in-flight calls per agent")
	flagCMax      = flag.Float64("cmax", 5000, "C_max (max actual cost)")
	flagEMax      = flag.Float64("emax", 5000, "Ê_max (max estimate cost)")
	flagCMin      = flag.Float64("cmin", 10, "c_min (min actual cost)")
)

func main() {
	flag.Parse()
	start := time.Now()

	n := *flagN
	rng := rand.New(rand.NewSource(*flagSeed))

	// Build agent configs.
	agents := make([]sim.AgentConfig, n)
	for i := range agents {
		w := 1.0
		calls := *flagCalls / n
		switch *flagWorkload {
		case "skewed":
			if i == 0 {
				calls = (*flagCalls / n) * 10 // 10× demand for agent 0
			}
		}
		agents[i] = sim.AgentConfig{
			ID:            scheduler.AgentID(i + 1),
			Weight:        w,
			CallsPerEpoch: calls,
		}
	}

	mockCfg := mock.Default()
	mockCfg.CMax = *flagCMax
	mockCfg.CMin = *flagCMin

	cfg := sim.Config{
		Agents:           agents,
		TotalCalls:       *flagCalls,
		EpochLen:         60,
		MockConfig:       mockCfg,
		Seed:             *flagSeed,
		MaxD:             *flagD,
		ReconcileEnabled: !*flagNoRecon,
	}

	switch *flagWorkload {
	case "uniform":
		cfg.Workload = sim.WorkloadUniform
	case "skewed":
		cfg.Workload = sim.WorkloadSkewed
	case "adversarial":
		cfg.Workload = sim.WorkloadAdversarial
	case "heavytail":
		cfg.Workload = sim.WorkloadHeavyTail
	case "mixed":
		cfg.Workload = sim.WorkloadMixedPrio
	case "tworesource":
		cfg.Workload = sim.WorkloadTwoResource
	default:
		log.Fatalf("unknown workload: %s", *flagWorkload)
	}

	simInst, disp := sim.NewWithDispatcher(cfg)

	var sched sim.Sched
	switch *flagSched {
	case "rvt":
		p := scheduler.Params{
			MaxD:             float64(*flagD),
			CMax:             *flagCMax,
			EMax:             *flagEMax,
			CMin:             *flagCMin,
			ReconcileEnabled: !*flagNoRecon,
		}
		est := estimator.New(0.1, *flagCMax/2, *flagEMax)
		rvt := scheduler.New(p, est, disp)
		for _, ag := range agents {
			rvt.AddAgent(ag.ID, ag.Weight)
		}
		sched = rvt
	default:
		log.Fatalf("sched %q not yet wired to this runner (use rvt)", *flagSched)
	}

	_ = rng // used in workload customisation above

	simInst.SetSched(sched)
	summary := simInst.Run()

	elapsed := time.Since(start)
	fmt.Printf("Run complete in %s\n", elapsed)
	fmt.Printf("Jain index: %.4f\n", summary.JainIndex)
	fmt.Printf("P50 latency: %.3fs  P99: %.3fs\n", summary.P50Latency, summary.P99Latency)
	fmt.Printf("Starvation events: %d\n", summary.Starvation)

	if err := writeCSV(*flagOut, summary, cfg); err != nil {
		log.Fatalf("write CSV: %v", err)
	}
	fmt.Printf("Results written to %s\n", *flagOut)
}

func writeCSV(path string, s *metrics.Summary, cfg sim.Config) error {
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
		{"workload", fmt.Sprintf("%d", cfg.Workload)},
		{"n_agents", strconv.Itoa(len(cfg.Agents))},
		{"total_calls", strconv.Itoa(cfg.TotalCalls)},
		{"seed", strconv.FormatInt(cfg.Seed, 10)},
		{"jain_index", strconv.FormatFloat(s.JainIndex, 'f', 6, 64)},
		{"p50_latency_s", strconv.FormatFloat(s.P50Latency, 'f', 6, 64)},
		{"p99_latency_s", strconv.FormatFloat(s.P99Latency, 'f', 6, 64)},
		{"starvation_events", strconv.Itoa(s.Starvation)},
	}
	for _, row := range rows {
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return nil
}
