# Execution Plan — *The Agent Is a Process* → publish-ready

**Paper:** `paper_v1.pdf` (Atharva Arbat, July 2026)
**Goal:** Make the paper trustable and complete: (1) *prove* the theorems as a standalone,
machine-checkable artifact, (2) *build* the minimum scheduler + harness needed to produce
real evaluation numbers, and (3) *replace every TBD* with measured values that are
internally consistent with the theory.

**Explicitly out of scope:** the full agent OS. No WASM sandbox, no capability
enforcement, no crash-recovery/WAL replay as a product feature, no distributed scheduling,
no security analysis. We build only what is load-bearing for the *proofs* and the *numbers*.

**Locked decisions (2026-07-21):**
- Proof rigor: **Lean 4 mechanized** (Lemma 1, Theorem 1, Prop 3) **+ property-based tests**.
- Implementation: **Go** for the scheduler core + mock harness; **Python** for analysis/plots.
- Data: **mock-only**. Live provider runs deferred (§7.3 fidelity claim will be softened).

---

## 1. Claim inventory (traceability matrix)

Every provable/measurable claim in the paper, mapped to a deliverable. This table *is* the
definition of "done" — nothing ships until each row is Green.

### 1a. Theory (Track A — Lean 4 + property tests)

| Paper artifact | Location | How we make it trustable |
|---|---|---|
| **Lemma 1** — accounting identity `w_i(F_i-F_i0) = ΔA_i + ΔP_i` | §4.3, L477 | Lean: prove as a **state invariant** maintained by every event (Submit/Dispatch/Complete). |
| **Theorem 1** — gap bound `A_i/w_i − A_j/w_j ≤ d(C_max/w_i + Ê_max/w_j)` | §4.4, L577 | Lean: theorem over reachable states of the transition system, given the dispatch-selection rule + A1–A5. Property test: fuzz workloads, assert bound holds. |
| **Corollary 1** — `≤ 2C_max + E_max` (d=1, unit weights) | §4.5, L649 | Lean: algebraic specialization (`Ê_max ≤ C_max + E_max`). |
| **Corollary 2** — starvation freedom / bounded work | §4.5, L661 | Derive in Lean from Thm 1; property test: no starvation over 1000-epoch runs. |
| **Proposition 2** — no reconciliation ⇒ gap unbounded in time | §4.5, L684 | Property test / simulation: the estimate-only ablation reproduces the `mC(C/ε−1)` growth. (Prose proof audited, not mechanized.) |
| **Proposition 3** — tightness of `C_max` term (gap reaches `C_max − c_min`) | §4.5, L701 | Lean + unit test: the concrete 2-agent instance hits the bound exactly. |
| **Proposition 1** — necessity of complete mediation | §3.3, L327 | **Not mechanized** (structural/architectural argument). Audit prose for rigor; back with the §7.4 escape-suite demonstration that each bypass is realizable. |
| **Appendix A** — continuous-time derivation of Lemma 1 | L1201 | Audit only; note it as an alternative view of the mechanized Lemma 1. |

### 1b. Numbers / TBDs (Tracks B–D)

| TBD / hypothesis | Location | Deliverable that fills it |
|---|---|---|
| "[TBD: headline fairness, service-gap, overhead numbers]" | Abstract, L44 | Final numbers from H1a/H1b/H0a runs. |
| "implemented in Go [TBD: confirm language/LOC]" | §6, L843 | Real Go LOC count (`scc`/`cloc`) for the kernel core. |
| Entire §7 is "[TBD … hypotheses, not results]" | §7, L879 | Result tables + figures replacing every bracketed placeholder. |
| **H0a** mediation overhead <5% wall, <1% token | §7.1, L886 | Micro-bench: mediated vs. direct-dispatch path in the same harness. |
| **H1a** Jain index >0.9 @10 agents/10:1 skew; FCFS/Promise.all <0.6 | §7.1, L893 | Scheduler experiment, uniform + skew workloads. |
| **H1b** max observed gap respects Theorem 1 | §7.1, L895 | **Theory↔experiment consistency check** (the flagship trust result). |
| **H1c** no agent waits > N epochs; zero starvation | §7.1, L898 | Mixed-priority workload with aging; starvation counter. |
| **H1d** DRF per-dimension fairness | §7.1, L900 | Two-resource bottleneck workload (secondary). |

---

## 2. Repository layout

```
agent-kernel/
  paper/                 # LaTeX source (see §7 open item) + edited sections
  proofs/                # Lean 4 project (Track A)
    RVT/
      Model.lean         # agent state, events, transition system
      Accounting.lean    # Lemma 1 (invariant)
      ServiceGap.lean    # Theorem 1 + Corollaries 1–2
      Tightness.lean     # Proposition 3
    lakefile.lean lean-toolchain
  kernel/                # Go module — minimal AgentKernel core (Track B)
    scheduler/           # RVT = Algorithm 1 (virtual time, dispatch, reconcile)
    ledger/              # budget vector, reservation, I5 conservation check
    wal/                 # append-before-effect log (for overhead accounting only)
    provider/            # uniform resource-vector adapter (Appendix B) + mock backend
    mock/                # deterministic mock-LLM: latency + heavy-tailed output length
    baselines/           # FCFS, round-robin, Promise.all, token-bucket-per-agent
    estimator/           # per-agent EWMA + max_tokens clamp; estimate-only ablation
    metrics/             # Jain index, gap-vs-bound, p50/p99 latency, starvation, util
    cmd/agentkernel-eval # experiment runner (deterministic, seeded)
    *_test.go            # property-based tests (gopter/rapid): Thm 1, Prop 2, Prop 3
  experiments/           # workload configs (6 workloads × n∈{2,10,50}), seeds, run.sh
  results/               # raw run outputs (CSV/JSON) committed for reproducibility
  analysis/              # Python: pandas + matplotlib → figures/ and tables/
    figures/  tables/
  PLAN.md
```

---

## 3. Workstreams

### Track A — Mechanized proofs (Lean 4)
Model RVT as an abstract transition system, not the Go code, so the proof is about the
*discipline* and survives implementation changes.

- **A1. State + events.** State = per-agent `(F_i, P_i, A_i, n_i, w_i)` + backlog set `B`.
  Events: `Submit`, `Dispatch(i,c,ê)`, `Complete(i,c,a)` exactly as Algorithm 1 (L513–575).
  Encode A1–A5 as hypotheses (total order = list of events; `d` bound; `a≤C_max`, `ê≤Ê_max`;
  non-discriminatory admission).
- **A2. Lemma 1** (`Accounting.lean`): prove `w_i·(F_i − F_i0) = (A_i − A_i0) + (P_i − P_i0)`
  is preserved by each event (the paper's per-event bookkeeping, L482–485). This is the
  keystone; everything else composes from it.
- **A3. Theorem 1** (`ServiceGap.lean`): reproduce the paper's Case-1 argument — least-VT
  selection gives `F_i(τ⁻) ≤ F_j(τ⁻)`, plug Lemma 1, bound `P_j ≤ d·Ê_max`, bound i's
  post-τ completions by `d·C_max`. Case 2 (i never dispatched) is trivial. Then Corollary 1
  (algebra) and Corollary 2 (re-apply Thm 1 in the j-over-i direction).
- **A4. Proposition 3** (`Tightness.lean`): the concrete 2-agent instance; prove the gap
  equals `C_max − c_min`, establishing the term cannot be dropped.
- **Scope note / stretch:** full end-to-end mechanization of the scheduler dynamics (the
  reachability of arbitrary event sequences) is the ambitious part. Baseline commitment =
  Lemma 1 invariant + Theorem 1 as a lemma over states satisfying the invariant + Prop 3
  instance. Reaching "any reachable state satisfies the invariant ⇒ bound" end-to-end is the
  stretch target; if it slips, the invariant + algebraic theorem + property tests still
  constitute a strong, honest claim.

### Track B — Scheduler core + mock harness (Go)
The *minimum* kernel that makes the numbers real, matching §6's component list (L846–875).

- **B1. RVT scheduler** = Algorithm 1 verbatim: virtual-time priority structure keyed on
  `F_i`, deterministic tie-break by agent id (I4), `Submit/Dispatch/Complete`, reconciliation
  line (L558). Single logical dispatch loop (A1) = the serialization + logging + determinism point.
- **B2. Estimator**: per-agent EWMA bootstrapped from per-model prior, `max_tokens` clamp
  (§6, L858). **Ablation flag** to remove reconciliation → instantiates Prop 2.
- **B3. Budget ledger**: tokens/dollars/rate-slots per agent+session; reservation charge +
  completion refund; continuous I5 conservation assertion (charged == metered).
- **B4. Mock-LLM provider**: deterministic, seeded. Latency + **heavy-tailed output-length**
  distributions (log-normal / Pareto) parameterized to match published LLM-serving trace
  statistics. Cost = `α·in + β·out` (VTC form, L200). No spend.
- **B5. WAL**: append-before-effect, present so its cost is included in the H0a overhead
  number (not exercised for replay).
- **B6. Baselines**: FCFS, unweighted round-robin, Promise.all (no scheduler),
  token-bucket-per-agent (§5.3 folk remedy). Plus reservation on/off.

### Track C — Experiments + metrics (Go, driven by configs)
- **C1. Six workloads** (§7.2, L906–913): uniform; 10:1 demand skew; adversarial bursts;
  heavy-tailed output; mixed priority w/ starvation-inducing low-prio; two-resource bottleneck.
  Each at `n ∈ {2, 10, 50}`, fixed seeds.
- **C2. Metrics** (§7.2, L915): Jain's index over sliding windows; **max pairwise service
  gap vs. the Theorem 1 bound computed from that run's own C_max/Ê_max/d/w**; p50/p99 dispatch
  latency per priority class; starvation counts over 1000-epoch runs; utilization (work-conservation).
- **C3. Property tests** (Go, `rapid`/`gopter`): (i) Thm 1 bound never violated across
  thousands of random admissible workloads; (ii) estimate-only mode makes the gap grow with
  m (Prop 2); (iii) Prop 3 instance hits `C_max − c_min`; (iv) ledger I5 holds every step.
- **C4. Overhead micro-bench** (H0a): mediated dispatch path (scheduler + ledger + WAL append)
  vs. a direct/unmediated dispatch of the same mock calls; report wall-clock % and per-syscall
  dispatch throughput under 50-agent concurrency.

### Track D — Analysis, figures, tables (Python)
- **D1.** Ingest `results/*.csv`, compute summary stats, regenerate Figure 2's numeric example
  from an actual run.
- **D2.** Produce the result tables/figures for §7: fairness bars (H1a), gap-vs-bound
  distribution + worst case (H1b), latency percentiles per class (H1c), DRF dominant-share
  (H1d), overhead table (H0a), Prop 2 growth curve (ablation).
- **D3.** Emit a single `numbers.tex`/JSON of macros so the paper pulls values from one source
  of truth (prevents abstract/section drift).

### Track E — Paper integration + proof audit
- **E1. Proof audit** (before mechanizing, to catch anything Lean would expose): verify the
  spots most likely to hide a gap —
  1. A3 "quiescent start" + continuously-backlogged ⇒ no equalization jump for i,j inside
     `[t0,t]` (SUBMIT L519 branch never fires for them).
  2. Theorem 1 step bounding i's post-τ completions by `d·C_max` (calls in flight *at* τ ≤ d).
  3. Corollary 1's `Ê_max ≤ C_max + E_max`.
  4. Prop 2's counting (`A1 ≈ mC²/ε`) and that it needs only an *optimistic*, not adversarial,
     estimator.
  5. Prop 3's deterministic tie-break assumption.
  (Initial read: proofs are sound in outline; audit + Lean will confirm.)
- **E2.** Replace every TBD/bracket with values from `numbers.tex`; set §6 LOC; write §7 results
  prose from the measured distributions.
- **E3. Honesty passes** (see §6 Risks): soften "calibrated from real API traces" → "from
  published trace statistics / synthetic distributions"; reframe H0a's LangGraph comparison to
  the in-harness unmediated baseline; mark live-provider fidelity (§7.3) as future work.

---

## 4. Phases & milestones

| Phase | Deliverable | Depends on | Gate |
|---|---|---|---|
| **P0** Scaffolding | Go module + Lean project + `results/` + CI that builds both | — | `go test ./...` and `lake build` green (empty) |
| **P1** Scheduler core | RVT (Alg 1), estimator, ledger, mock provider, WAL | P0 | Figure 2 cycle reproduced exactly by a unit test |
| **P2** Proof audit | E1 findings written up | read paper | No unfixable gap found (else escalate) |
| **P3** Property tests | C3 passing; Prop 2 & Prop 3 demonstrated | P1 | Thm 1 never violated over ≥10k fuzz cases |
| **P4** Lean proofs | Lemma 1 invariant + Theorem 1 + Cor 1–2 + Prop 3 | P2 | `lake build` proves them (no `sorry`) |
| **P5** Baselines + workloads | 4 baselines, 6 workloads × n∈{2,10,50} | P1 | Deterministic reruns byte-identical |
| **P6** Experiments run | `results/` populated; overhead micro-bench | P5 | All runs seeded + reproducible |
| **P7** Analysis | Figures, tables, `numbers.tex` | P6 | H1a/H1b/H1c/H0a targets evaluated (pass or honestly reported) |
| **P8** Paper integration | TBDs replaced; proofs referenced; honesty passes | P4,P7 | No "TBD"/bracket remains; numbers consistent |

Tracks A (P2→P4) and B/C/E5 (P1→P7) run in parallel after P0/P1.

---

## 5. Risks & scoping adjustments (honesty ledger)

These are places where the artifact will be *more modest than the current prose*; the paper
text must be adjusted so no claim outruns the evidence.

1. **"Calibrated from real API traces" (abstract, §7.2).** Mock-only ⇒ we calibrate from
   *published* latency/output-length statistics + synthetic heavy-tailed distributions.
   Rewrite the claim; keep live calibration as clearly-marked future work.
2. **H0a vs. LangGraph/AutoGen.** Without building against those frameworks we cannot get a
   true head-to-head. Reframe H0a as **mediated vs. unmediated dispatch in the same harness**
   — an honest measure of the mediation mechanism's tax; note the framework comparison as future.
3. **Proposition 1 not mechanized.** It's an architectural necessity argument, not an
   inequality. We back it with the §7.4 escape suite (each bypass realizable) + prose audit,
   and say so plainly.
4. **Lean end-to-end reachability** is the stretch boundary (see A "scope note"). If it slips,
   the paper claims "invariant + bound mechanized; dynamics validated by property tests" — still strong.
5. **Single node only** (already stated as scope, §9 L1044) — no change needed.

---

## 6. Definition of done (acceptance criteria)

- **Trust gate (the whole point):** for *every* experimental run, the measured max pairwise
  service gap ≤ the Theorem 1 bound computed from that run's own `C_max, Ê_max, d, w`. If any
  run violates it, that is a bug in the code or the theorem — investigate before publishing.
- Lean `lake build` proves Lemma 1, Theorem 1, Corollaries 1–2, Prop 3 with **zero `sorry`**.
- Property tests (≥10k randomized workloads) find no Thm 1 violation; reproduce Prop 2 growth
  and Prop 3 tightness.
- Every TBD/bracketed placeholder in the paper is replaced with a value traceable to
  `results/` via `numbers.tex`; abstract, §6, §7 agree by construction.
- One command (`experiments/run.sh` + `analysis/`) reproduces all figures/tables from seeds.
- Honesty ledger (§5) applied to the prose.

---

## 7. Open items to resolve before/at P8

- **Paper LaTeX source.** We have only the PDF. Completing the paper needs the `.tex`. Either
  (a) locate the source, or (b) reconstruct the sections we must edit (abstract, §6, §7, and
  proof cross-references). *Please point me at the source if it exists.*
- **N (aging bound) for H1c** — pick the configurable epoch bound to report.
- **Estimator prior** — the per-model bootstrap values for the mock (choose defaults, document).

---

## 8. Suggested first move

Start P0+P1 in parallel with P2: scaffold the Go module and Lean project, implement RVT +
the Figure-2 unit test, and write up the proof audit. That produces something runnable and a
verified proof skeleton within the first iteration, and everything else hangs off it.
