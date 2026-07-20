// Package wal implements the write-ahead log (invariant I3: log-before-effect).
// The current implementation is in-memory; persistence and replay are out of scope for
// the evaluation but the interface is wired through the scheduler so overhead (H0a)
// is measured honestly.
package wal

import (
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"
)

// EventType distinguishes the three scheduler event kinds.
type EventType string

const (
	EventSubmit   EventType = "SUBMIT"
	EventDispatch EventType = "DISPATCH"
	EventComplete EventType = "COMPLETE"
)

// Entry is one WAL record.
type Entry struct {
	Seq      int64     `json:"seq"`
	WallTime time.Time `json:"wall_time"`
	Type     EventType `json:"type"`
	AgentID  int64     `json:"agent_id"`
	CallID   int64     `json:"call_id"`
	Estimate float64   `json:"estimate,omitempty"`
	Actual   float64   `json:"actual,omitempty"`
}

// WAL is a durable, append-only log. Append must complete before the associated
// effect is applied (I3).
type WAL struct {
	mu      sync.Mutex
	entries []Entry
	seq     int64
	writer  io.Writer // nil → in-memory only
}

// New creates an in-memory WAL. Pass a non-nil writer to also stream NDJSON to disk.
func New(w io.Writer) *WAL { return &WAL{writer: w} }

// Append records an entry and (if writer is set) flushes it to the backing writer.
// Must be called before the corresponding scheduler effect.
func (w *WAL) Append(e Entry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.seq++
	e.Seq = w.seq
	e.WallTime = time.Now()
	w.entries = append(w.entries, e)

	if w.writer != nil {
		b, err := json.Marshal(e)
		if err != nil {
			return fmt.Errorf("wal marshal: %w", err)
		}
		b = append(b, '\n')
		if _, err := w.writer.Write(b); err != nil {
			return fmt.Errorf("wal write: %w", err)
		}
	}
	return nil
}

// Len returns the number of entries appended.
func (w *WAL) Len() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.entries)
}

// Entries returns a copy of all entries (for testing / replay).
func (w *WAL) Entries() []Entry {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]Entry, len(w.entries))
	copy(out, w.entries)
	return out
}
