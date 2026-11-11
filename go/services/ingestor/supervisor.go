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

// Runner drives one initialized connector until ctx is done. A non-nil error
// means the connector ended on its own (health: `down`, halt raised).
type Runner func(ctx context.Context, log *slog.Logger, c connector.Connector) error

// StatusStore persists catalog statuses (refdata.Store); nil keeps them in memory.
type StatusStore interface {
	SetStatus(ctx context.Context, id, status string) error
}

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
	// Health, when set, is told about starts/stops (ingest.health.* frames).
	Health *Tracker
	// Statuses, when set, persists enable/disable (catalog.source.status).
	Statuses StatusStore
	// BackfillDeps, when set, builds deps on the low-priority backfill lane
	// (backfill jobs, gap-fill; spec 10 §6.4); nil uses the live deps.
	BackfillDeps func(src catalog.Source) connector.Deps

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
	src, inCatalog := s.cat.ByID(id)
	registered := connector.HasConnector(id, src.Adapter)
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
	c, err := connector.NewFor(id, src.Adapter)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrNotFound, err)
	}
	ctx, cancel := context.WithCancel(parent)
	cfg := connector.SourceConfig{Source: src}
	if s.Secrets != nil {
		cfg.Secrets = s.Secrets(id)
	}
	deps := s.deps(src)
	if err := c.Init(ctx, cfg, deps); err != nil {
		cancel()
		return fmt.Errorf("init %s: %w", id, err)
	}
	inst := &instance{cancel: cancel, done: make(chan struct{})}
	s.running[id] = inst
	s.updateCountersLocked()
	_, inCatalog := s.cat.ByID(id)
	if s.Health != nil {
		ml, _ := deps.Limiter.(*connector.MeteredLimiter)
		s.Health.Register(c, ml)
	}
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
		runErr := s.run(ctx, s.log.With("source", id), c)
		operator := ctx.Err() != nil
		if s.Health != nil {
			s.Health.Unregister(id, operator, runErr)
		}
		if runID != "" {
			after := s.snapshot()
			status, lastErr := "stopped", ""
			if !operator {
				status = "failed" // ended on its own
				if runErr != nil {
					lastErr = runErr.Error()
				}
			}
			ectx, ecancel := context.WithTimeout(context.Background(), 5*time.Second)
			if err := s.RunLog.End(ectx, runID, status, int64(after["events."+id]-before["events."+id]), int64(after["errors."+id]-before["errors."+id]), lastErr); err != nil {
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

// Instance creates and initialises a fresh connector for a catalog source
// outside the supervisor's lifecycle (backfill jobs, gap-fill, replay); with
// backfillLane its limiter is the source's low-priority backfill lane.
func (s *Supervisor) Instance(ctx context.Context, id string, backfillLane bool) (connector.Connector, catalog.Source, connector.Deps, error) {
	src, err := s.Resolve(id)
	if err != nil {
		return nil, src, connector.Deps{}, err
	}
	c, err := connector.NewFor(id, src.Adapter)
	if err != nil {
		return nil, src, connector.Deps{}, fmt.Errorf("%w: %v", ErrNotFound, err)
	}
	cfg := connector.SourceConfig{Source: src}
	if s.Secrets != nil {
		cfg.Secrets = s.Secrets(id)
	}
	deps := s.deps(src)
	if backfillLane && s.BackfillDeps != nil {
		deps = s.BackfillDeps(src)
	}
	if err := c.Init(ctx, cfg, deps); err != nil {
		return nil, src, deps, fmt.Errorf("init %s: %w", id, err)
	}
	return c, src, deps, nil
}

// Restart stops a running connector and starts it again with the current
// catalog entry (ctl.ingest.reload / configure). Not running → no-op, false.
func (s *Supervisor) Restart(ctx context.Context, id string) (bool, error) {
	if !s.IsRunning(id) {
		return false, nil
	}
	if err := s.Stop(id); err != nil {
		return false, err
	}
	return true, s.Start(ctx, id)
}

// IsRunning reports whether id has a live connector.
func (s *Supervisor) IsRunning(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.running[id]
	return ok
}

// SetStatus changes a catalog source's status in memory and, with Statuses
// set, in Postgres (which then wins over the YAML seed on every later start).
func (s *Supervisor) SetStatus(ctx context.Context, id, status string) error {
	if _, ok := s.cat.ByID(id); !ok {
		return fmt.Errorf("%w: source %q is not in the catalog", ErrNotFound, id)
	}
	if !slices.Contains(catalog.Statuses, status) {
		return fmt.Errorf("%w: status %q", ErrInvalid, status)
	}
	if s.Statuses != nil {
		if err := s.Statuses.SetStatus(ctx, id, status); err != nil {
			return fmt.Errorf("persist status: %w", err)
		}
	}
	s.cat.SetStatus(id, status)
	return nil
}

// Disable sets status `disabled` and stops the connector if it runs.
// Returns whether a connector was stopped.
func (s *Supervisor) Disable(ctx context.Context, id string) (bool, error) {
	if err := s.SetStatus(ctx, id, "disabled"); err != nil {
		return false, err
	}
	if !s.IsRunning(id) {
		return false, nil
	}
	err := s.Stop(id)
	return err == nil, err
}

// Enable sets status `live` (the source is operable again; a connector must
// be registered) and starts the connector unless it already runs.
// Returns whether a connector was started.
func (s *Supervisor) Enable(ctx context.Context, id string) (bool, error) {
	src, inCatalog := s.cat.ByID(id)
	if !connector.HasConnector(id, src.Adapter) {
		if !inCatalog {
			return false, fmt.Errorf("%w: source %q is not in the catalog", ErrNotFound, id)
		}
		return false, fmt.Errorf("%w: source %q has no connector; enable it once one exists", ErrConflict, id)
	}
	if err := s.SetStatus(ctx, id, "live"); err != nil {
		return false, err
	}
	if s.IsRunning(id) {
		return false, nil
	}
	if err := s.Start(ctx, id); err != nil {
		return false, err
	}
	return true, nil
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
