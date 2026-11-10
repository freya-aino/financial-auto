package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sort"
	"sync"
	"time"

	"github.com/fey/financial-auto/go/pkg/bus"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// Control errors. Their prefixes are the contract with control-api
// (`not_found:` → 404, `conflict:` → 409, anything else → 422).
var (
	ErrNotFound = errors.New("not_found")
	ErrConflict = errors.New("conflict")
	ErrInvalid  = errors.New("invalid")
)

// Statuses from the catalog that may never be started at runtime.
var blockedStatuses = []string{"disabled", "planned"}

// Runner drives one initialized connector until ctx is done.
type Runner func(ctx context.Context, log *slog.Logger, c connector.Connector)

// RunLogger records connector lifecycles (Postgres ingest.connector_run).
type RunLogger interface {
	Start(ctx context.Context, sourceID string) (runID string, err error)
	End(ctx context.Context, runID, status string, events, errs int64, lastErr string) error
}

// Supervisor starts and stops connectors by source id (spec 10 §4.3) and
// keeps the `connectors.*` heartbeat counters current.
type Supervisor struct {
	cat      *catalog.Catalog
	log      *slog.Logger
	counters *bus.Counters
	run      Runner
	deps     func(src catalog.Source) connector.Deps

	// Secrets supplies SourceConfig.Secrets per source id (optional).
	Secrets func(id string) map[string]string
	// RunLog, when set, gets a row per start/stop of catalog sources.
	RunLog RunLogger

	mu      sync.Mutex
	running map[string]*instance
	wg      sync.WaitGroup
}

type instance struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// NewSupervisor wires the catalog, a runner and a per-source deps factory.
func NewSupervisor(cat *catalog.Catalog, log *slog.Logger, counters *bus.Counters, deps func(catalog.Source) connector.Deps, run Runner) *Supervisor {
	return &Supervisor{cat: cat, log: log, counters: counters, run: run, deps: deps, running: map[string]*instance{}}
}

// Resolve finds the catalog entry for a connector id and checks it may run.
// Registered connectors absent from the catalog (the synthetic heartbeat)
// get an implicit live entry.
func (s *Supervisor) Resolve(id string) (catalog.Source, error) {
	if id == "" {
		return catalog.Source{}, fmt.Errorf("%w: empty source id", ErrInvalid)
	}
	registered := slices.Contains(connector.Registered(), id)
	src, inCatalog := s.cat.ByID(id)
	switch {
	case !inCatalog && !registered:
		return catalog.Source{}, fmt.Errorf("%w: source %q is neither in the catalog nor a registered connector", ErrNotFound, id)
	case !registered:
		return catalog.Source{}, fmt.Errorf("%w: source %q has no connector", ErrNotFound, id)
	case !inCatalog:
		src = catalog.Source{ID: id, Name: id, Status: "live", Auth: "none"}
	}
	if slices.Contains(blockedStatuses, src.Status) {
		return catalog.Source{}, fmt.Errorf("%w: source %q has status %s", ErrConflict, id, src.Status)
	}
	return src, nil
}

// Start creates, initializes and runs a connector. Idempotency is the
// caller's: a second Start of a running id is a conflict.
func (s *Supervisor) Start(parent context.Context, id string) error {
	src, err := s.Resolve(id)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.running[id]; ok {
		return fmt.Errorf("%w: source %q is already running", ErrConflict, id)
	}
	c, err := connector.New(id)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrNotFound, err)
	}
	ctx, cancel := context.WithCancel(parent)
	cfg := connector.SourceConfig{Source: src}
	if s.Secrets != nil {
		cfg.Secrets = s.Secrets(id)
	}
	if err := c.Init(ctx, cfg, s.deps(src)); err != nil {
		cancel()
		return fmt.Errorf("init %s: %w", id, err)
	}
	inst := &instance{cancel: cancel, done: make(chan struct{})}
	s.running[id] = inst
	s.updateCountersLocked()
	_, inCatalog := s.cat.ByID(id)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer close(inst.done)
		var runID string
		var before map[string]float64
		if s.RunLog != nil && inCatalog {
			before = s.snapshot()
			if rid, err := s.RunLog.Start(parent, id); err != nil {
				s.log.Warn("run log start", "source", id, "err", err)
			} else {
				runID = rid
			}
		}
		s.run(ctx, s.log.With("source", id), c)
		if runID != "" {
			after := s.snapshot()
			status := "stopped"
			if ctx.Err() == nil {
				status = "failed" // ended on its own
			}
			ectx, ecancel := context.WithTimeout(context.Background(), 5*time.Second)
			if err := s.RunLog.End(ectx, runID, status, int64(after["events."+id]-before["events."+id]), int64(after["errors."+id]-before["errors."+id]), ""); err != nil {
				s.log.Warn("run log end", "source", id, "err", err)
			}
			ecancel()
		}
		s.mu.Lock()
		if cur, ok := s.running[id]; ok && cur == inst {
			delete(s.running, id) // connector ended on its own (error or no capability)
			s.updateCountersLocked()
		}
		s.mu.Unlock()
	}()
	s.log.Info("connector started", "source", id)
	return nil
}

// Stop cancels a running connector and waits for it to exit.
func (s *Supervisor) Stop(id string) error {
	if _, err := s.Resolve(id); err != nil && errors.Is(err, ErrNotFound) {
		return err
	}
	s.mu.Lock()
	inst, ok := s.running[id]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("%w: source %q is not running", ErrConflict, id)
	}
	delete(s.running, id)
	s.updateCountersLocked()
	s.mu.Unlock()
	inst.cancel()
	<-inst.done
	s.log.Info("connector stopped", "source", id)
	return nil
}

// Running lists running ids in sorted order.
func (s *Supervisor) Running() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := make([]string, 0, len(s.running))
	for id := range s.running {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// Wait blocks until every connector goroutine has exited.
func (s *Supervisor) Wait() { s.wg.Wait() }

func (s *Supervisor) snapshot() map[string]float64 {
	if s.counters == nil {
		return map[string]float64{}
	}
	return s.counters.Snapshot()
}

func (s *Supervisor) updateCountersLocked() {
	if s.counters == nil {
		return
	}
	s.counters.Set("connectors.running", float64(len(s.running)))
	for _, id := range connector.Registered() {
		v := 0.0
		if _, ok := s.running[id]; ok {
			v = 1
		}
		s.counters.Set("connectors.up."+id, v)
	}
}
