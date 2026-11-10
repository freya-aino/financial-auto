package connector

import (
	"context"
	"sync"
	"time"
)

// MeteredLimiter wraps a RateLimiter and measures how saturated it is: the
// share of wall time callers spent blocked in Wait since the last Reset.
// The ingestor reports it as `limiter_saturation` in `ingest.health.<source>`
// (0 = never waited, 1 = someone was always waiting).
type MeteredLimiter struct {
	Inner RateLimiter

	mu       sync.Mutex
	since    time.Time
	blocked  time.Duration
	requests int64
	total    int64
}

// NewMeteredLimiter wraps inner (nil inner = unlimited).
func NewMeteredLimiter(inner RateLimiter) *MeteredLimiter {
	return &MeteredLimiter{Inner: inner, since: time.Now()}
}

// Wait implements RateLimiter.
func (m *MeteredLimiter) Wait(ctx context.Context) error {
	if m.Inner == nil {
		m.mu.Lock()
		m.requests++
		m.total++
		m.mu.Unlock()
		return nil
	}
	start := time.Now()
	err := m.Inner.Wait(ctx)
	d := time.Since(start)
	m.mu.Lock()
	m.blocked += d
	m.requests++
	m.total++
	m.mu.Unlock()
	return err
}

// LimiterStats is one measurement window.
type LimiterStats struct {
	// Saturation is blocked time / window length, clamped to [0, 1].
	Saturation float64
	// Requests admitted in the window; Total since creation.
	Requests, Total int64
	Window          time.Duration
}

// Snapshot returns the current window's stats and, when reset is true, starts
// a new window.
func (m *MeteredLimiter) Snapshot(reset bool) LimiterStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	window := now.Sub(m.since)
	st := LimiterStats{Requests: m.requests, Total: m.total, Window: window}
	if window > 0 {
		st.Saturation = min(1, float64(m.blocked)/float64(window))
	}
	if reset {
		m.since, m.blocked, m.requests = now, 0, 0
	}
	return st
}

// BackfillCursorStore persists bounded backfill ranges (ingest.backfill_cursor
// rows keyed by (source, instrument|"", timescale, range_start); spec 10 §6.4).
// It is separate from CursorStore, whose rows are the open-ended live
// watermarks (range_start = epoch).
type BackfillCursorStore interface {
	// GetBackfill returns the stored cursor and status for a range; ok=false when none.
	GetBackfill(ctx context.Context, sourceID, instrumentID, timescale string, rangeStart time.Time) (cursor time.Time, status string, ok bool, err error)
	// SetBackfill upserts a range's cursor and status (pending|running|done|failed).
	SetBackfill(ctx context.Context, sourceID, instrumentID, timescale string, rangeStart, rangeEnd, cursor time.Time, status string) error
}

// MemBackfillCursors is an in-memory BackfillCursorStore.
type MemBackfillCursors struct {
	mu sync.Mutex
	m  map[string]memBackfill
}

type memBackfill struct {
	cursor time.Time
	status string
}

func NewMemBackfillCursors() *MemBackfillCursors {
	return &MemBackfillCursors{m: map[string]memBackfill{}}
}

func backfillKey(sourceID, instrumentID, timescale string, rangeStart time.Time) string {
	return sourceID + "\x00" + instrumentID + "\x00" + timescale + "\x00" + rangeStart.UTC().Format(time.RFC3339Nano)
}

func (c *MemBackfillCursors) GetBackfill(_ context.Context, sourceID, instrumentID, timescale string, rangeStart time.Time) (time.Time, string, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.m[backfillKey(sourceID, instrumentID, timescale, rangeStart)]
	return v.cursor, v.status, ok, nil
}

func (c *MemBackfillCursors) SetBackfill(_ context.Context, sourceID, instrumentID, timescale string, rangeStart, _ time.Time, cursor time.Time, status string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m[backfillKey(sourceID, instrumentID, timescale, rangeStart)] = memBackfill{cursor: cursor, status: status}
	return nil
}
