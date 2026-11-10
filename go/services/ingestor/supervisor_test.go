package main

import (
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/bus"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"

	_ "github.com/fey/financial-auto/go/connectors/heartbeat"
	_ "github.com/fey/financial-auto/go/connectors/oanda"
	_ "github.com/fey/financial-auto/go/connectors/yahoo"
)

func testSupervisor(t *testing.T) *Supervisor {
	t.Helper()
	cat, err := catalog.Parse([]byte(`
version: 1
sources:
  - {id: yahoo, name: Yahoo, status: stub, auth: none}
  - {id: oanda, name: OANDA, status: planned, auth: none}
  - {id: nocode, name: No connector, status: live, auth: none}
`))
	if err != nil {
		t.Fatal(err)
	}
	counters := bus.NewCounters()
	run := func(ctx context.Context, _ *slog.Logger, _ connector.Connector) error { <-ctx.Done(); return nil }
	deps := func(catalog.Source) connector.Deps { return connector.Deps{Clock: connector.SystemClock{}} }
	return NewSupervisor(cat, slog.Default(), counters, deps, run)
}

func TestSupervisorStartStop(t *testing.T) {
	s := testSupervisor(t)
	ctx := context.Background()

	if err := s.Start(ctx, "heartbeat"); err != nil { // not in catalog, registered → implicit live
		t.Fatal(err)
	}
	if err := s.Start(ctx, "heartbeat"); !errors.Is(err, ErrConflict) {
		t.Fatalf("second start: want conflict, got %v", err)
	}
	if got := s.Running(); len(got) != 1 || got[0] != "heartbeat" {
		t.Fatalf("running %v", got)
	}
	if v := s.counters.Snapshot()["connectors.running"]; v != 1 {
		t.Fatalf("connectors.running = %v", v)
	}
	if err := s.Start(ctx, "oanda"); !errors.Is(err, ErrConflict) {
		t.Fatalf("planned: want conflict, got %v", err)
	}
	if err := s.Start(ctx, "nocode"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("no connector: want not_found, got %v", err)
	}
	if err := s.Start(ctx, "nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown: want not_found, got %v", err)
	}
	if err := s.Stop("yahoo"); !errors.Is(err, ErrConflict) {
		t.Fatalf("stop idle: want conflict, got %v", err)
	}
	if err := s.Stop("nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("stop unknown: want not_found, got %v", err)
	}
	if err := s.Stop("heartbeat"); err != nil {
		t.Fatal(err)
	}
	if len(s.Running()) != 0 || s.counters.Snapshot()["connectors.running"] != 0 {
		t.Fatalf("still running: %v", s.Running())
	}
	done := make(chan struct{})
	go func() { s.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Wait did not return")
	}
}
