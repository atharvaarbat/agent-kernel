![Banner](/banner.jpg)
# AgentKernel

**The Agent Is a Process: A Token-Fair Runtime for Multi-Agent LLM Systems**

AgentKernel is a client-side runtime that enforces **proportional-share fair scheduling** across concurrent LLM agents. It solves a problem no provider can solve alone: fair allocation of token throughput, request rate, and dollar budgets across agents that may use different models, providers, and non-model tools simultaneously.

Theoretical foundation: [*The Agent Is a Process*](paper/paper.tex) introduces **Reconciled Virtual Time (RVT)**, a proportional-share discipline for non-preemptible, unknown-cost quanta with a proven service-gap bound (Theorem 1), starvation freedom (Corollary 1), and two negative results delimiting the design space.

## Quick Start

```bash
# From the kernel directory
cd kernel

# Run the evaluation harness (skewed workload, 10 agents, RVT scheduler)
go run ./cmd/agentkernel-eval -workload skewed -n 10 -calls 5000 -seed 42 -sched rvt

# Run the overhead benchmark (H0a gate)
go run ./cmd/agentkernel-bench -n 10 -calls 100000

# Run property-based tests (10,000 random cases each)
go test ./scheduler/ -v -run TestLemma1Property -count=1
go test ./scheduler/ -v -run TestTheorem1Property -count=1
```

## Architecture

```
agent ──> infer/invoke ──> AgentKernel ──> Provider A
agent ──> send/recv  ──>  (scheduler)  ──> Provider B
agent ──> clock/rng  ──>  (ledger)     ──> Tool 1
                          (wal)        ──> Tool 2
```

The kernel enforces **complete mediation (I1)**: every model call, tool invocation, inter-agent message, clock read, and random draw passes through the kernel. This is what makes fair scheduling implementable — no provider, framework middleware, or non-mediating layer can see the full resource contention picture.

### Packages

| Package | Lines | Role |
|---|---|---|
| `scheduler/` | ~356 | Core RVT algorithm: virtual time, equalization, reconciliation, dispatch loop |
| `baselines/` | ~344 | Comparison schedulers: FCFS, Round-Robin, PromiseAll, TokenBucket |
| `sim/` | ~227 | Discrete-event simulation driver with 6 workload kinds |
| `metrics/` | ~189 | Jain fairness index, latency percentiles, gap ratio, starvation counts |
| `mock/` | ~87 | Deterministic mock LLM provider (log-normal output lengths) |
| `ledger/` | ~98 | Per-agent and per-session budget accounting (I5 conservation) |
| `wal/` | ~74 | Write-ahead log (I3: log-before-effect) |
| `estimator/` | ~64 | Per-agent EWMA cost estimator with clamped upper bound |
| `cmd/agentkernel-eval/` | ~173 | Experiment runner CLI |
| `cmd/agentkernel-bench/` | ~132 | Overhead benchmark CLI |

## The RVT Algorithm

RVT extends virtual-time fair queueing (SFQ, VTC) to calls that are **non-preemptible** and have **unknown cost at dispatch**:

1. **Estimate**: charge an estimated cost $\hat{e}(c)$ at dispatch, advancing virtual time $F_i \gets F_i + \hat{e}(c)/w_i$
2. **Dispatch**: always select the admissible backlogged agent with the smallest $F_i$
3. **Reconcile**: on completion, adjust $F_i \gets F_i + (a(c) - \hat{e}(c))/w_i$

**Fairness bound** (Theorem 1): For two continuously backlogged agents $i, j$ with equal weights and $d$ in-flight calls each:

$$\left|\frac{\Delta A_i}{w} - \frac{\Delta A_j}{w}\right| \le \frac{d\,(C_{\max} + \hat{E}_{\max})}{w}$$

The bound has two terms — the **non-preemptibility gap** ($C_{\max}$): the call the scheduler cannot recall; and the **phantom-charge gap** ($\hat{E}_{\max}$): the estimate not yet reconciled. Reconciliation, not prediction accuracy, is what buys fairness.

**Key results**:
- Without reconciliation, the service gap is **unbounded in time** (Proposition 2)
- Starvation freedom follows as a **corollary** of the bound (Corollary 1), not a separate mechanism
- No non-mediating layer can enforce the scheduler's premises (Proposition 3)

## Agent-Process Abstraction

An LLM agent is a **process**, not a chat session:

| PCB Field | Agent Analogue |
|---|---|
| Address space | Context region (working set isolation) |
| Registers/PC | Conversation state + pending calls |
| Scheduler state | Virtual time $F_i$, weight $w_i$ |
| Resource limits | Budget vector $(B_{\mathrm{tok}}, B_\$, B_{\mathrm{rate}})$ |
| File descriptors | Tool handles / capabilities |
| Signals | `cancel` (preemption by truncation) |
| IPC | `send`/`recv` + kernel message buffer |

Syscall surface: `infer`, `cancel`, `invoke`, `send`, `recv`, `clock`, `random`, `spawn`, `exit`.

## Evaluation

**H0a — Scheduling overhead**: 1847 ns/call at $n=10$ (0.0001% of 2s inference latency). Clears the 5% wall-clock gate.

**H1a — Jain fairness under ${\approx}31\!:\!1$ cost skew**: RVT achieves **Jain $\ge 0.999$** across $n \in \{2, 10, 50\}$. FCFS and Round-Robin fall to 0.121--0.531.

**H1b — Theorem 1 bound**: Maximum observed gap is **0.67$\times$** the theoretical bound across all tested scales. Removing reconciliation (Proposition 2 ablation) tightens it to exactly 1.0$\times$.

**Workload coverage**: Uniform, skewed (31:1), adversarial, heavy-tail ($\sigma_{\ln}=2.5$), mixed priority (3:1 weights), and two-resource bottleneck — all achieve Jain $\ge 0.998$ under RVT.

## CLI Reference

### `agentkernel-eval`

Run one experiment configuration and write results to CSV.

```
go run ./cmd/agentkernel-eval [flags]

Flags:
  -workload string        Workload kind: uniform, skewed, adversarial, heavytail,
                          mixed, tworesource (default "uniform")
  -n int                  Number of agents (default 10)
  -calls int              Total completions to simulate (default 1000)
  -seed int               PRNG seed (default 42)
  -sched string           Scheduler: rvt, fcfs, rr, promiseall (default "rvt")
  -out string             Output CSV path (default "results/run.csv")
  -no-reconcile           Disable reconciliation (Proposition 2 ablation)
  -d int                  Max in-flight calls per agent (default 1)
  -global-concurrency int Global provider slot limit (default 1)
  -cmax float             Max actual cost per call (default 5000)
  -emax float             Max estimate per call (default 5000)
  -cmin float             Min cost per call (default 10)
```

### `agentkernel-bench`

Measure scheduling overhead (H0a gate).

```
go run ./cmd/agentkernel-bench [flags]

Flags:
  -n int                  Number of agents (default 10)
  -calls int              Total calls per benchmark (default 50000)
  -mean-latency-ms float  Mean LLM inference latency in ms for overhead % calculation (default 2000)
  -out string             Output CSV path (default "results/bench.csv")
```

## Testing

```bash
# All unit and property-based tests
go test ./... -v

# Property-based tests run 10,000 random cases each
go test ./scheduler/ -v -run TestLemma1Property -count=1
go test ./scheduler/ -v -run TestTheorem1Property -count=1
go test ./scheduler/ -v -run TestProposition3Tightness -count=1
go test ./scheduler/ -v -run TestProposition2NoReconciliation -count=1

# Figure 2 walkthrough unit test
go test ./scheduler/ -v -run TestFigure2 -count=1
```

## Paper

The full paper is at [`paper/paper.tex`](paper/paper.tex). It includes:
- Problem statement and system model (§2)
- The agent-process abstraction (§3) with the necessity-of-mediation proof (§3.3)
- Reconciled Virtual Time (§4) with the fairness bound (§4.4) and negative results (§4.5)
- Multi-resource DRF extension (§5)
- Implementation description (§6)
- Evaluation (§7) with H0a/H1a/H1b results
- Related work and threats to validity (§8, §9)

## License

MIT
