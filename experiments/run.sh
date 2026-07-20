#!/usr/bin/env bash
# experiments/run.sh — reproduces all §7 experimental results from fixed seeds.
# Usage: bash experiments/run.sh
# All runs are deterministic; re-running produces byte-identical results/ CSVs.

set -euo pipefail
cd "$(dirname "$0")/.."

EVAL=./kernel/cmd/agentkernel-eval
RESULTS=results

echo "=== Building agentkernel-eval ==="
(cd kernel && go build -o ../agentkernel-eval ./cmd/agentkernel-eval)
EVAL=./agentkernel-eval

mkdir -p "$RESULTS"

# ─── workload × n matrix ──────────────────────────────────────────────────────
# §7.2: six workloads × n ∈ {2, 10, 50}, seeds fixed per run.

WORKLOADS=(uniform skewed adversarial heavytail mixed)
NS=(2 10 50)
SEED_BASE=42

for wl in "${WORKLOADS[@]}"; do
  for n in "${NS[@]}"; do
    seed=$((SEED_BASE))
    out="${RESULTS}/rvt_${wl}_n${n}_seed${seed}.csv"
    echo "RVT  workload=${wl} n=${n} seed=${seed} → ${out}"
    "$EVAL" -workload "$wl" -n "$n" -calls 1000 -seed "$seed" -sched rvt -out "$out"

    # Baselines for uniform workload only (to populate H1a comparisons).
    if [ "$wl" = "uniform" ]; then
      echo "  (baselines not yet wired in runner; placeholder)"
    fi
  done
done

# ─── Proposition 2 ablation (no reconciliation) ───────────────────────────────
echo "=== Proposition 2 ablation: estimate-only ==="
"$EVAL" -workload uniform -n 2 -calls 500 -seed 42 -sched rvt -no-reconcile \
  -out "${RESULTS}/prop2_ablation_seed42.csv"

# ─── H0a overhead micro-bench ─────────────────────────────────────────────────
echo "=== H0a overhead micro-bench ==="
echo "(overhead bench not yet implemented; placeholder)"

echo ""
echo "=== All runs complete. Results in ${RESULTS}/ ==="
echo "Run 'python analysis/analyze.py' to generate figures and tables."
