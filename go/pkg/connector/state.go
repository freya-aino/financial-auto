package connector

import (
	"context"
	"sync"
	"time"
)

// CursorStore persists per-source progress (Postgres ingest.backfill_cursor,
// spec 10 §6.4). scope distinguishes series within a source ("items",
// "AAPL/1m", ...); instrumentID may be empty for non-instrument series.
type CursorStore interface {
	GetCursor(ctx context.Context, sourceID, instrumentID, scope string) (time.Time, bool, error)
	SetCursor(ctx context.Context, sourceID, instrumentID, scope string, cursor time.Time) error
}

// MemCursors is an in-memory CursorStore for tests and no-Postgres runs.
type MemCursors struct {
	mu sync.Mutex
	m  map[string]time.Time
}

func NewMemCursors() *MemCursors { return &MemCursors{m: map[string]time.Time{}} }

func (c *MemCursors) GetCursor(_ context.Context, sourceID, instrumentID, scope string) (time.Time, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	t, ok := c.m[sourceID+"\x00"+instrumentID+"\x00"+scope]
	return t, ok, nil
}

func (c *MemCursors) SetCursor(_ context.Context, sourceID, instrumentID, scope string, cursor time.Time) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m[sourceID+"\x00"+instrumentID+"\x00"+scope] = cursor
	return nil
}

// Seen is a bounded set of string ids for in-process de-duplication (news
// guids, HN item ids). Oldest entries are evicted once cap is exceeded.
type Seen struct {
	mu    sync.Mutex
	cap   int
	order []string
	set   map[string]struct{}
}

func NewSeen(capacity int) *Seen {
	if capacity <= 0 {
		capacity = 10_000
	}
	return &Seen{cap: capacity, set: map[string]struct{}{}}
}

// Add records id and reports whether it was new.
func (s *Seen) Add(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.set[id]; ok {
		return false
	}
	s.set[id] = struct{}{}
	s.order = append(s.order, id)
	if len(s.order) > s.cap {
		drop := s.order[:len(s.order)-s.cap]
		for _, d := range drop {
			delete(s.set, d)
		}
		s.order = append([]string(nil), s.order[len(drop):]...)
	}
	return true
}

func (s *Seen) Has(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.set[id]
	return ok
}

func (s *Seen) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.set)
}
