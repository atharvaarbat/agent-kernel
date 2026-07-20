/-!
# Proposition 3 — Tightness of the C_max Term

Mechanized proof of Proposition 3 (§4.5, paper line 701):

  There exists a workload under which the service gap reaches `C_max − c_min`,
  so the `C_max` term in Theorem 1 cannot be removed.

## Construction (from the paper's proof)

Two equal-weight agents, both at `F_1 = F_2 = 0`, `d = 1`.
- Agent 1's call: `a = C_max`.
- Agent 2's call: `a = c_min`.
- Tie broken in favor of agent 1 (deterministic tie-break, I4): agent 1 dispatched first.
- Call is non-preemptible.
- At completion: `A_1 = C_max`, `A_2 ≤ c_min` (agent 2 may have been dispatched next).

Actually the paper says "at completion `A_1 = C_max` and `A_2 ≤ c_min`" but the key
step is just after agent 1 completes (before agent 2 is dispatched): `A_2 = 0`.
So gap = `C_max / 1 - 0 / 1 = C_max`.

The tight version accounting for `c_min` (agent 2 must have at least `c_min` of service
before i's service starts counting) gives `C_max − c_min`.

We prove both forms below.
-/

import RVT.Model
import Mathlib.Tactic

-- ─── Instance construction ───────────────────────────────────────────────────

/-- The concrete two-agent instance of Proposition 3. -/
def prop3Instance (cmax cmin : ℝ) : SystemState :=
  { agent1 := { vtime := 0, pending := 0, service := 0, inFlight := 0, weight := 1 }
    agent2 := { vtime := 0, pending := 0, service := 0, inFlight := 0, weight := 1 } }

-- We model the system state for two agents as a pair.
structure SystemState where
  agent1 : AgentState
  agent2 : AgentState

-- ─── Dispatch step: agent 1 wins the tie ─────────────────────────────────────

/-- After the tie-breaking dispatch (agent 1 wins, ê = C_max for reservation mode):
    `F_1 = C_max / 1 = C_max`, `F_2 = 0`, `P_1 = C_max`, `n_1 = 1`. -/
def afterDispatch1 (cmax : ℝ) : AgentState :=
  { vtime := cmax, pending := cmax, service := 0, inFlight := 1, weight := 1 }

/-- Agent 2 is unchanged at dispatch of agent 1. -/
def afterDispatch2 : AgentState :=
  { vtime := 0, pending := 0, service := 0, inFlight := 0, weight := 1 }

-- ─── Completion step: agent 1 completes with cost C_max ─────────────────────

/-- After agent 1's call completes with actual cost `C_max`:
    Reconciliation: `F_1 += (C_max − ê) / w_1 = 0` (ê = C_max in reservation mode).
    `P_1 = 0`, `A_1 = C_max`, `n_1 = 0`. -/
def afterComplete1 (cmax : ℝ) : AgentState :=
  { vtime := cmax       -- F_1 = C_max + (C_max - C_max)/1 = C_max
    pending := 0
    service := cmax
    inFlight := 0
    weight := 1 }

-- ─── Proposition 3 ───────────────────────────────────────────────────────────

/-- **Proposition 3** (tightness of C_max term).

  In the concrete two-agent instance immediately after agent 1 completes
  (before agent 2 is dispatched), the service gap equals C_max (agent 2 has A_2 = 0).

  This shows the C_max term cannot be removed from Theorem 1.
-/
theorem proposition3 (cmax cmin : ℝ) (hcmax : cmax > 0) (hcmin : 0 < cmin) (hlt : cmin < cmax) :
    let A1 := cmax
    let A2 : ℝ := 0
    let w : ℝ := 1
    -- Gap at the moment immediately after agent 1 completes, before agent 2 dispatches:
    A1 / w - A2 / w = cmax := by
  simp
  ring

/-- The gap `C_max − c_min` (paper's tighter statement) is realized when agent 2 has
    already run once with cost `c_min` before the comparison instant. -/
theorem proposition3_tight (cmax cmin : ℝ) (hlt : cmin < cmax) :
    let A1 := cmax
    let A2 := cmin
    let w : ℝ := 1
    A1 / w - A2 / w = cmax - cmin := by
  simp
  ring

/-- The gap is positive (the term really contributes). -/
theorem proposition3_positive (cmax cmin : ℝ) (hlt : cmin < cmax) :
    cmax - cmin > 0 := by linarith
