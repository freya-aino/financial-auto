package main

import (
	"context"
	"errors"
	"testing"

	"github.com/fey/financial-auto/go/pkg/bus"
	"github.com/fey/financial-auto/go/pkg/catalog"
)

func TestRefWatcherFlushesOnRefAndCatalogChanges(t *testing.T) {
	cat, err := catalog.Parse([]byte(`
version: 1
sources:
  - {id: kraken, name: Kraken, status: live, auth: none}
  - {id: yahoo, name: Yahoo, status: live, auth: none}
`))
	if err != nil {
		t.Fatal(err)
	}
	flushes, reloads := 0, 0
	db := map[string]string{"kraken": "disabled", "yahoo": "live"}
	var dbErr error
	w := &RefWatcher{
		Invalidate: func() { flushes++ },
		Statuses: func(context.Context) (map[string]string, error) {
			reloads++
			return db, dbErr
		},
		Catalog:  cat,
		Counters: bus.NewCounters(),
	}

	w.Handle(bus.RefChanged("instrument"))
	if flushes != 1 || reloads != 0 {
		t.Fatalf("ref.changed: flushes=%d reloads=%d", flushes, reloads)
	}

	w.Handle(bus.CatalogChanged("kraken"))
	if flushes != 2 || reloads != 1 {
		t.Fatalf("catalog.changed: flushes=%d reloads=%d", flushes, reloads)
	}
	if s, _ := cat.ByID("kraken"); s.Status != "disabled" {
		t.Fatalf("kraken status not reloaded: %q", s.Status)
	}

	dbErr = errors.New("down")
	w.Handle(bus.CatalogChanged("yahoo")) // a failed reload still flushed the cache
	if flushes != 3 {
		t.Fatalf("flushes=%d", flushes)
	}
	snap := w.Counters.Snapshot()
	if snap["refdata.invalidations"] != 3 || snap["catalog.changes"] != 2 || snap["catalog.reload_errors"] != 1 {
		t.Fatalf("counters: %v", snap)
	}

	w.Handle("ingest.health.kraken") // unrelated subjects are ignored
	if flushes != 3 {
		t.Fatalf("unrelated subject flushed the cache")
	}
}
