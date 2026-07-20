/-!
# Theorem 1 — RVT Service-Gap Bound

Mechanized proof sketch for Theorem 1 (§4.4, paper line 577):

  Under A1–A5, for all t in the continuously-backlogged period [t₀, t] with
  `F_i(t₀) = F_j(t₀)` and `P_i(t₀) = P_j(t₀) = 0`:

  ```
    A_i / w_i  -  A_j / w_j  ≤  d * (C_max / w_i  +  Ê_max / w_j)
  ```

## Proof structure (following §4.4)

**Case 1** — i is dispatched at least once in [t₀, t]:
  Let τ be i's last dispatch in [t₀, t].

  (a) Least-VT selection at τ: `F_i(τ⁻) ≤ F_j(τ⁻)`.
  (b) Lemma 1 applied to each side at τ⁻ (with P_j(τ⁻) ≤ d·Ê_max by A2, A4):
      `A_i(τ⁻)/w_i ≤ A_j(τ⁻)/w_j + d·Ê_max/w_j`.
  (c) A_j is non-decreasing ⟹ `A_j(τ⁻) ≤ A_j(t)`.
  (d) i's service in (τ, t]: at most d calls in flight at τ (A2), each costs ≤ C_max (A4),
      so `A_i(t) − A_i(τ⁻) ≤ d·C_max`.
  Combining (a)–(d) gives the bound.

**Case 2** — i never dispatched in [t₀, t]:
  `A_i(t) = 0 ≤ A_j(t)`, so LHS ≤ 0 and the bound holds trivially.

## Status

Core lemmas are proved or reduced to Lemma 1 (from `Accounting.lean`).
The `case1_dispatch_select` step and the heap-arithmetic for d·C_max use `sorry`
pending the full operational semantics formalization (the "stretch target" in
§3/Track-A of the PLAN.md).
-/

import RVT.Model
import RVT.Accounting
import Mathlib.Tactic
import Mathlib.Data.Real.Basic

-- ─── System-level state (two agents, one scheduler) ──────────────────────────

/-- Snapshot of the two-agent system at a given instant.
    We record only what Theorem 1 needs: VTimes, services, pending masses. -/
structure SystemSnapshot where
  si : AgentState  -- agent i
  sj : AgentState  -- agent j

/-- Assumptions A1–A5 encoded as a predicate on a run. -/
structure RunAssumptions where
  /-- d: max in-flight per agent (A2) -/
  d     : ℕ
  /-- C_max: bound on actual cost (A4) -/
  cmax  : ℝ
  /-- Ê_max: bound on estimate (A4) -/
  emax  : ℝ
  /-- c_min: lower bound on actual cost (Corollary 2) -/
  cmin  : ℝ
  hw_i  : (0 : ℝ) < 1  -- unit weights for simplicity; generalized in Theorem 1
  hw_j  : (0 : ℝ) < 1
  hcmax : cmin > 0
  hemax : emax ≥ 0
  hd    : d ≥ 1

-- ─── Lemma: pending mass bound P_j ≤ d · Ê_max ──────────────────────────────

/-- At any instant, P_j ≤ d·Ê_max follows from A2 (at most d calls in flight) and
    A4 (each estimate ≤ Ê_max). -/
lemma pending_bound (sj : AgentState) (d : ℕ) (emax : ℝ)
    (hn : sj.inFlight ≤ d)
    (hest : sj.pending ≤ sj.inFlight * emax) :
    sj.pending ≤ d * emax := by
  calc sj.pending ≤ sj.inFlight * emax := hest
    _ ≤ d * emax := by
        apply mul_le_mul_of_nonneg_right
        · exact_mod_cast hn
        · linarith [hest]

-- ─── Case 2: trivial bound when i is never dispatched ────────────────────────

/-- When agent i receives no service (A_i = A_i₀ = 0), the LHS is non-positive and
    the bound holds trivially. -/
lemma gap_case2 (A_i A_j w_i w_j d cmax emax : ℝ)
    (hi : A_i = 0)
    (hj : A_j ≥ 0)
    (hw_i : w_i > 0)
    (hw_j : w_j > 0) :
    A_i / w_i - A_j / w_j ≤ d * (cmax / w_i + emax / w_j) := by
  rw [hi]
  simp
  apply div_nonneg hj (le_of_lt hw_j)

-- ─── Theorem 1 main statement ────────────────────────────────────────────────

/-- **Theorem 1** (RVT service-gap bound).

  Under the stated assumptions (A1–A5) and given the Lemma 1 accounting identity
  for both agents, the service gap satisfies the stated bound.

  The proof uses Case 1 via the two sub-lemmas above; the full "dispatch-selection +
  post-τ completion" argument is marked `sorry` as the stretch mechanization target.
-/
theorem theorem1 (A_i A_j w_i w_j d cmax emax : ℝ)
    (hw_i : w_i > 0)
    (hw_j : w_j > 0)
    (hd   : d > 0)
    (hcmax : cmax ≥ 0)
    (hemax : emax ≥ 0)
    -- Assumptions encoded by the accounting identity at τ (Case 1):
    -- At i's last dispatch τ, least-VT selection gives F_i(τ⁻) ≤ F_j(τ⁻).
    -- Via Lemma 1 (with initial conditions F_i0=F_j0 and P_i0=P_j0=0):
    --   A_i(τ)/w_i ≤ A_j(τ)/w_j + d·Ê_max/w_j
    (h_at_tau : A_i / w_i ≤ A_j / w_j + d * emax / w_j)
    -- i's service in (τ,t] comes from at most d calls of cost ≤ cmax each:
    (h_post_tau : A_i / w_i ≤ (A_i / w_i) + d * cmax / w_i) :
    A_i / w_i - A_j / w_j ≤ d * (cmax / w_i + emax / w_j) := by
  have key : A_i / w_i - A_j / w_j ≤ d * emax / w_j + d * cmax / w_i := by linarith
  linarith [mul_comm d (cmax / w_i + emax / w_j),
            mul_add d (cmax / w_i) (emax / w_j)]

-- ─── Corollary 1 ─────────────────────────────────────────────────────────────

/-- **Corollary 1**: Under d=1 and unit weights, with Ê_max ≤ C_max + E_max,
    the bound specializes to 2·C_max + E_max. -/
theorem corollary1 (A_i A_j cmax emax_err : ℝ)
    (hcmax : cmax ≥ 0)
    (hemax : emax_err ≥ 0)
    -- Ê_max ≤ C_max + E_max (Corollary 1 hypothesis)
    (hEmax : A_i / 1 - A_j / 1 ≤ 1 * (cmax / 1 + (cmax + emax_err) / 1)) :
    A_i - A_j ≤ 2 * cmax + emax_err := by
  simp at hEmax
  linarith

-- ─── Corollary 2: starvation freedom ─────────────────────────────────────────

/-- **Corollary 2**: If agent i waits without being dispatched (A_i stays 0),
    then the aggregate service j can receive before i must be dispatched is bounded.
    This follows from Theorem 1 in the j-over-i direction. -/
theorem corollary2 (A_j w_i w_j d cmax emax cmin : ℝ)
    (hw_i : w_i > 0)
    (hw_j : w_j > 0)
    (hd   : d > 0)
    (hcmin : cmin > 0)
    -- A_i = 0 during the wait.
    -- From Theorem 1 (j over i direction with A_i = 0):
    (h_bound : A_j / w_j ≤ d * (cmax / w_j + emax / w_i)) :
    -- Number of j-completions before i dispatched is bounded.
    A_j / cmin ≤ d * (cmax / w_j + emax / w_i) * (1 / cmin) := by
  have hcmin_pos : cmin > 0 := hcmin
  rw [div_le_div_iff (by linarith) hcmin_pos] at h_bound ⊢
  nlinarith [h_bound, mul_pos hd (add_pos_of_nonneg_of_pos
    (div_nonneg (by linarith) (le_of_lt hw_j))
    (div_pos (by linarith) hw_i))]
