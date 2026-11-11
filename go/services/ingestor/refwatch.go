package main

import (
	"context"
	"strings"
	"time"

	"github.com/nats-io/nats.go"

	"github.com/fey/financial-auto/go/pkg/bus"
	"github.com/fey/financial-auto/go/pkg/catalog"
)

// RefWatcher reacts to change notices (spec 11 §4, spec 12 §6, D-067):
//
//   - `ref.changed.<entity>` (quarantine merge, symbol change, identifier or
//     contract-spec override, provisioning): the resolver cache is dropped, so
//     the next event of every (source, symbol) pair re-reads ref.listing /
//     ref.identifier instead of serving a stale id for up to the 1 h TTL.
//   - `catalog.changed.<id>` (a source enabled/disabled): the resolver cache is
//     dropped too, and the in-memory catalog statuses are re-read from
//     catalog.source, so a change written by another instance or by hand is
//     what `ctl.ingest.start|enable|backfill` gate on. Running connectors are
//     not started or stopped from here: that stays an explicit ctl verb.
//
// The resolver caches only (source, symbol) → instrument id; the asset class
// and currencies are not held in the ingestor, so nothing else needs flushing.
type RefWatcher struct {
	Invalidate func()                                           // resolver cache flush; nil = none
	Statuses   func(context.Context) (map[string]string, error) // catalog.source statuses; nil = none
	Catalog    *catalog.Catalog
	Counters   *bus.Counters
	Log        func(msg string, args ...any)
	Timeout    time.Duration // for the status reload, default 5 s
	// Overrides re-applies stored rate-limit/config overrides (Reloader.ApplyStoredOverrides); nil = none.
	Overrides func(context.Context) ([]string, error)
}

// Handle processes one notice by subject.
func (w *RefWatcher) Handle(subject string) {
	isCatalog := strings.HasPrefix(subject, "catalog.changed.")
	if !isCatalog && !strings.HasPrefix(subject, "ref.changed.") {
		return
	}
	if w.Invalidate != nil {
		w.Invalidate()
		w.count("refdata.invalidations")
	}
	if !isCatalog {
		return
	}
	w.count("catalog.changes")
	if w.Statuses == nil || w.Catalog == nil {
		return
	}
	timeout := w.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	st, err := w.Statuses(ctx)
	if err != nil {
		w.count("catalog.reload_errors")
		w.log("catalog reload", "subject", subject, "err", err)
		return
	}
	if changed := w.Catalog.ApplyStatuses(st); len(changed) > 0 {
		w.log("catalog statuses reloaded from postgres", "subject", subject, "sources", changed)
	}
	if w.Overrides != nil {
		if applied, err := w.Overrides(ctx); err != nil {
			w.count("catalog.reload_errors")
			w.log("catalog overrides reload", "subject", subject, "err", err)
		} else if len(applied) > 0 {
			w.log("catalog overrides applied from postgres", "subject", subject, "sources", applied)
		}
	}
}

// Subscribe attaches the watcher to `ref.changed.>` and `catalog.changed.>`.
func (w *RefWatcher) Subscribe(nc *nats.Conn) error {
	for _, subj := range []string{bus.RefChanged(">"), bus.CatalogChanged(">")} {
		// Handled off the NATS dispatcher: an override may restart a connector.
		if _, err := nc.Subscribe(subj, func(m *nats.Msg) { go w.Handle(m.Subject) }); err != nil {
			return err
		}
	}
	return nil
}

func (w *RefWatcher) count(key string) {
	if w.Counters != nil {
		w.Counters.Add(key, 1)
	}
}

func (w *RefWatcher) log(msg string, args ...any) {
	if w.Log != nil {
		w.Log(msg, args...)
	}
}
