#!/usr/bin/env bash
# experiments/run.sh — full evaluation matrix for the paper (§7).
# Produces one CSV per combination in results/; run from repo root.
# All runs are deterministic; re-running produces byte-identical results/ CSVs.
# Usage: bash experiments/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS=results
mkdir -p "$RESULTS"

# Build binaries.
echo "=== Building binaries ==="
(cd kernel && go build -o ../bin/agentkernel-eval.exe  ./cmd/agentkernel-eval)
(cd kernel && go build -o ../bin/agentkernel-bench.exe ./cmd/agentkernel-bench)
EVAL=bin/agentkernel-eval.exe
BENCH=bin/agentkernel-bench.exe
echo "Build OK"

# ─── H0a: overhead benchmark ──────────────────────────────────────────────────
echo ""
echo "=== H0a: scheduling overhead benchmark ==="
"$BENCH" -n 10  -calls 100000 -out "$RESULTS/bench_n10.csv"
"$BENCH" -n 50  -calls 100000 -out "$RESULTS/bench_n50.csv"

# ─── H1a / H1b: fairness + Theorem-1 gate ─────────────────────────────────────
# Workload: skewed (30:1 cost ratio agent0 vs others), global-concurrency=1.
# Schedulers: rvt, fcfs, rr.
echo ""
echo "=== H1a/H1b: fairness (skewed workload, global-concurrency=1) ==="
for SCHED in rvt fcfs rr; do
  for N in 2 10 50; do
    CALLS=5000
    OUT="$RESULTS/${SCHED}_skewed_n${N}_c${CALLS}.csv"
    echo "  sched=$SCHED n=$N calls=$CALLS"
    "$EVAL" -workload skewed -n "$N" -calls "$CALLS" -seed 42 \
            -sched "$SCHED" -global-concurrency 1 -out "$OUT"
  done
done

# ─── Robustness: multiple seeds (RVT, skewed, n=10) ──────────────────────────
echo ""
echo "=== Seed robustness (skewed, n=10, RVT) ==="
for SEED in 42 123 999 2024 7777; do
  OUT="$RESULTS/rvt_skewed_n10_seed${SEED}.csv"
  echo "  seed=$SEED"
  "$EVAL" -workload skewed -n 10 -calls 5000 -seed "$SEED" \
          -sched rvt -global-concurrency 1 -out "$OUT"
done

# ─── Additional workloads (RVT only, n=10) ────────────────────────────────────
echo ""
echo "=== Additional workloads (RVT, n=10) ==="
for WL in uniform adversarial heavytail mixed tworesource; do
  CALLS=2000
  case "$WL" in adversarial|heavytail|skewed) CALLS=5000 ;; esac
  OUT="$RESULTS/rvt_${WL}_n10_c${CALLS}.csv"
  echo "  workload=$WL calls=$CALLS"
  "$EVAL" -workload "$WL" -n 10 -calls "$CALLS" -seed 42 \
          -sched rvt -global-concurrency 1 -out "$OUT"
done

# ─── Proposition 2 ablation: no reconciliation ────────────────────────────────
echo ""
echo "=== Prop 2 ablation (no reconciliation, skewed, n=10) ==="
"$EVAL" -workload skewed -n 10 -calls 5000 -seed 42 \
        -sched rvt -global-concurrency 1 -no-reconcile \
        -out "$RESULTS/rvt_skewed_n10_norecon.csv"

echo ""
echo "=== All runs complete. Results in $RESULTS/ ==="
echo "Run: python analysis/analyze.py --results-dir results --out-dir analysis"
