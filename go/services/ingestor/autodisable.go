package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DefaultAutoDisableAfter is `catalog.auto_disable_after_failures_h` (spec 11 §7).
const DefaultAutoDisableAfter = 24 * time.Hour

// AutoDisabler sets a catalog source to `disabled` after it has been `down`
// (not by operator stop) without interruption for After (spec 11 §6.1
// "disabled … after persistent failure", §8 "status → disabled, note
// recorded"). The reason is persisted in catalog.source.status_note and
// announced on catalog.changed.<id>; the halt the outage raised stays until an
// operator re-enables the source (ctl.ingest.enable). The failure clock lives
// in memory: an ingestor restart starts it again.
type AutoDisabler struct {
	After time.Duration
	// InCatalog reports whether the id is a catalog source (the synthetic heartbeat is not).
	InCatalog func(id string) bool
	// Disable stops the connector, persists `disabled` with the note and publishes the change.
	Disable func(ctx context.Context, id, note string) error
	Log     func(msg string, args ...any)
	Now     func() time.Time

	mu      sync.Mutex
	since   map[string]time.Time
	reasons map[string]string
}

func (a *AutoDisabler) now() time.Time {
	if a.Now != nil {
		return a.Now()
	}
	return time.Now().UTC()
}

// Observe records one health frame (Tracker.OnFrame).
func (a *AutoDisabler) Observe(f Frame, deliberate bool) {
	if a == nil || a.After <= 0 {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.since == nil {
		a.since, a.reasons = map[string]time.Time{}, map[string]string{}
	}
	if f.Status != "down" || deliberate {
		delete(a.since, f.Source)
		delete(a.reasons, f.Source)
		return
	}
	if _, ok := a.since[f.Source]; !ok {
		a.since[f.Source] = a.now()
	}
	a.reasons[f.Source] = f.Reason
}

// Check disables every source down for at least After; returns the ids disabled.
func (a *AutoDisabler) Check(ctx context.Context) []string {
	if a == nil || a.After <= 0 {
		return nil
	}
	now := a.now()
	type due struct{ id, note string }
	var todo []due
	a.mu.Lock()
	for id, t := range a.since {
		if now.Sub(t) >= a.After && (a.InCatalog == nil || a.InCatalog(id)) {
			todo = append(todo, due{id, fmt.Sprintf("auto-disabled at %s: down since %s (≥ %s): %s",
				now.Format(time.RFC3339), t.Format(time.RFC3339), a.After, a.reasons[id])})
		}
	}
	a.mu.Unlock()
	var done []string
	for _, d := range todo {
		if err := a.Disable(ctx, d.id, d.note); err != nil {
			if a.Log != nil {
				a.Log("auto-disable failed", "source", d.id, "err", err)
			}
			continue
		}
		a.mu.Lock()
		delete(a.since, d.id)
		delete(a.reasons, d.id)
		a.mu.Unlock()
		if a.Log != nil {
			a.Log("source auto-disabled", "source", d.id, "note", d.note)
		}
		done = append(done, d.id)
	}
	return done
}

// Run checks every interval until ctx is done.
func (a *AutoDisabler) Run(ctx context.Context, interval time.Duration) {
	if a == nil || a.After <= 0 {
		return
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			a.Check(ctx)
		}
	}
}
