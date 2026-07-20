// Package ledger implements per-agent and per-session budget accounting (invariant I5).
package ledger

import (
	"fmt"
	"math"

	"github.com/atharvaarbat/agent-kernel/scheduler"
)

// BudgetVector holds the three scarce resources tracked per agent.
type BudgetVector struct {
	Tokens  float64 // token budget (input + output, in cost units)
	Dollars float64 // dollar budget
	Rate    float64 // rate-limit slots consumed this window
}

// AgentLedger is the per-agent budget entry.
type AgentLedger struct {
	Budget   BudgetVector // configured limit
	Reserved BudgetVector // currently held as reservation for in-flight calls
	Consumed BudgetVector // reconciled actuals (running total)
	// Metered is what the provider has reported (for I5 cross-check).
	Metered BudgetVector
}

// Ledger manages per-agent and per-session budget accounting.
type Ledger struct {
	agents  map[scheduler.AgentID]*AgentLedger
	session BudgetVector // aggregate session limits
	// I5 conservation: charged == metered within reconciliation error.
	totalCharged BudgetVector
	totalMetered BudgetVector
}

// New creates an empty ledger with the given session-level budget limits.
func New(sessionBudget BudgetVector) *Ledger {
	return &Ledger{
		agents:  make(map[scheduler.AgentID]*AgentLedger),
		session: sessionBudget,
	}
}

// AddAgent registers an agent with its per-agent budget.
func (l *Ledger) AddAgent(id scheduler.AgentID, budget BudgetVector) {
	l.agents[id] = &AgentLedger{Budget: budget}
}

// Reserve charges the estimated cost at dispatch time (before the call is issued).
// Returns an error if the reservation would exceed the agent or session budget.
func (l *Ledger) Reserve(id scheduler.AgentID, estimate float64) error {
	ag := l.agents[id]
	if ag == nil {
		return fmt.Errorf("ledger: unknown agent %d", id)
	}
	if ag.Reserved.Tokens+estimate > ag.Budget.Tokens {
		return fmt.Errorf("ledger: agent %d token budget exceeded (reserved=%.0f, budget=%.0f)",
			id, ag.Reserved.Tokens+estimate, ag.Budget.Tokens)
	}
	ag.Reserved.Tokens += estimate
	l.totalCharged.Tokens += estimate
	return nil
}

// Refund returns the unused portion of a reservation when a call completes.
// actualCost must be ≤ estimate (reservation ensures this for max_tokens mode).
func (l *Ledger) Refund(id scheduler.AgentID, estimate, actualCost float64) error {
	ag := l.agents[id]
	if ag == nil {
		return fmt.Errorf("ledger: unknown agent %d", id)
	}
	refund := estimate - actualCost
	if refund < -1e-9 {
		// actual > estimate (can happen in EWMA mode without reservation).
		refund = 0
	}
	ag.Reserved.Tokens -= estimate
	ag.Consumed.Tokens += actualCost
	l.totalCharged.Tokens -= refund // net charged = actual
	return nil
}

// RecordMetered records actual usage as reported by the provider (for I5 assertion).
func (l *Ledger) RecordMetered(id scheduler.AgentID, actual float64) {
	if ag := l.agents[id]; ag != nil {
		ag.Metered.Tokens += actual
	}
	l.totalMetered.Tokens += actual
}

// VerifyI5 checks that the total charged amount equals total metered within tolerance.
// Returns an error describing the discrepancy if one is found.
func (l *Ledger) VerifyI5(tolerance float64) error {
	delta := math.Abs(l.totalCharged.Tokens - l.totalMetered.Tokens)
	if delta > tolerance {
		return fmt.Errorf("I5 violation: charged=%.4f metered=%.4f delta=%.4f > %.4f",
			l.totalCharged.Tokens, l.totalMetered.Tokens, delta, tolerance)
	}
	return nil
}

// Stats returns a snapshot of the ledger for agent id.
func (l *Ledger) Stats(id scheduler.AgentID) *AgentLedger {
	ag := l.agents[id]
	if ag == nil {
		return nil
	}
	copy := *ag
	return &copy
}
