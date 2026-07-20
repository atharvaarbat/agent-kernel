/-!
# RVT Model — §4.2 / Algorithm 1

Defines the abstract transition system for Reconciled Virtual Time.
The model is deliberately independent of the Go implementation so that proofs
are about the *discipline* (any correct implementation) rather than one program.

## State

Per-agent state: `(F_i, P_i, A_i, n_i, w_i)` where
- `F_i : ℝ` — virtual time
- `P_i : ℝ` — pending estimate mass (`Σ ê(c)` for calls in flight)
- `A_i : ℝ` — reconciled service (`Σ a(c)` for completed calls)
- `n_i : ℕ` — in-flight call count
- `w_i : ℝ` — weight (positive)

## Events

Three event types (Algorithm 1):
- `Submit i`  — agent `i` submits a new call (equalization + backlog update)
- `Dispatch i ê` — call with estimate `ê` is dispatched for agent `i`
- `Complete i ê a` — call with estimate `ê` completes with actual cost `a`

## Accounting identity (Lemma 1)

After any sequence of events for agent `i` starting from a quiescent state:
```
  w_i * (F_i - F_i₀) = (A_i - A_i₀) + (P_i - P_i₀)
```
-/

import Mathlib.Data.Real.Basic

-- ─── Per-agent state ─────────────────────────────────────────────────────────

structure AgentState where
  /-- Virtual time F_i -/
  vtime   : ℝ
  /-- Pending estimate mass P_i -/
  pending : ℝ
  /-- Reconciled service A_i -/
  service : ℝ
  /-- In-flight call count n_i -/
  inFlight : ℕ
  /-- Weight w_i (must be positive; enforced as a hypothesis in theorems) -/
  weight  : ℝ

-- ─── Events ──────────────────────────────────────────────────────────────────

/-- The three scheduler event types of Algorithm 1. -/
inductive Event
  /-- Agent i submits a call (not directly tracked here; handled by equalization). -/
  | Submit   (i : ℕ)
  /-- Dispatch: agent i's call is issued with estimate ê. -/
  | Dispatch (i : ℕ) (ê : ℝ)
  /-- Complete: agent i's call with estimate ê completes with actual cost a. -/
  | Complete (i : ℕ) (ê : ℝ) (a : ℝ)

-- ─── Single-agent transition ──────────────────────────────────────────────────

/-- Apply a single event to agent i's state.
    Events for other agents leave state unchanged (we project to the i-th component). -/
def stepAgent (i : ℕ) (s : AgentState) : Event → AgentState
  | Event.Dispatch j ê =>
    if j = i then
      { s with
        vtime   := s.vtime + ê / s.weight
        pending := s.pending + ê
        inFlight := s.inFlight + 1 }
    else s
  | Event.Complete j ê a =>
    if j = i then
      { s with
        vtime    := s.vtime + (a - ê) / s.weight   -- RECONCILE step
        pending  := s.pending - ê
        service  := s.service + a
        inFlight := s.inFlight - 1 }
    else s
  | Event.Submit _ => s  -- Submit changes backlog membership, not F/P/A directly

/-- Fold a list of events through the agent state machine. -/
def runAgent (i : ℕ) (s₀ : AgentState) (events : List Event) : AgentState :=
  events.foldl (stepAgent i) s₀

-- ─── Accounting identity statement (Lemma 1) ─────────────────────────────────

/-- The accounting identity `w_i * (F_i(t) - F_i(t₀)) = (A_i(t) - A_i(t₀)) + (P_i(t) - P_i(t₀))`
    as a predicate on a pair of states (initial, final). -/
def accountingIdentity (s₀ s : AgentState) : Prop :=
  s.weight * (s.vtime - s₀.vtime) = (s.service - s₀.service) + (s.pending - s₀.pending)

-- ─── Helper lemma: identity preserved by a single Dispatch event ──────────────

lemma accounting_dispatch (i : ℕ) (s₀ s : AgentState) (ê : ℝ)
    (hw : s.weight > 0)
    (h : accountingIdentity s₀ s) :
    accountingIdentity s₀ (stepAgent i s (Event.Dispatch i ê)) := by
  unfold accountingIdentity stepAgent at *
  simp [if_pos rfl]
  field_simp
  ring_nf
  linarith [h]

-- ─── Helper lemma: identity preserved by a single Complete event ──────────────

lemma accounting_complete (i : ℕ) (s₀ s : AgentState) (ê a : ℝ)
    (hw : s.weight > 0)
    (h : accountingIdentity s₀ s) :
    accountingIdentity s₀ (stepAgent i s (Event.Complete i ê a)) := by
  unfold accountingIdentity stepAgent at *
  simp [if_pos rfl]
  field_simp
  ring_nf
  linarith [h]
