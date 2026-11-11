package main

import (
	"context"
	"errors"
	"log/slog"
	"math/rand/v2"
	"time"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
	"github.com/fey/financial-auto/go/pkg/tsdb"
	"github.com/fey/financial-auto/go/pkg/warehouse"
)

// Pipeline is the per-message path shared by live connectors and backfill
// jobs (spec 10 §6.1): archive raw → normalize → publish → TSDB → health.
type Pipeline struct {
	Pub      *bus.Publisher
	Writer   *tsdb.Writer // nil disables
	Archive  connector.RawArchive
	Counters *bus.Counters
	Health   *Tracker // nil disables health/gap/unresolved frames
	// SeedGaps returns inst|tf → last bar close for a source (persisted live
	// watermarks), so the first bar after a restart is checked against the cursor.
	SeedGaps func(ctx context.Context, source string) map[string]time.Time
	// NewsSeen de-duplicates news across sources and restarts of a connector by
	// the normalized-URL key (spec 10 §6.5); nil disables.
	NewsSeen *connector.Seen
	// Warehouse writes macro observations, filings and XBRL facts to wh.macro_observation, wh.filing
	// and wh.fundamental (spec 20 §6.1 (b)); nil without -postgres.
	Warehouse *warehouse.Writer
}

// Process handles one raw message. It returns the number of envelopes
// published and the newest event time among them (zero when none).
func (p *Pipeline) Process(ctx context.Context, log *slog.Logger, c connector.Connector, norm connector.Normalizer, gaps *GapDetector, raw connector.RawMessage) (int, time.Time) {
	id := c.ID()
	venue := c.Capabilities().Venue
	ref, err := p.Archive.Put(ctx, raw)
	if err != nil {
		p.fail(id, log, "archive", err)
	}
	_, nopArchive := p.Archive.(connector.NopArchive)
	envs, unresolved, err := norm.Normalize(raw)
	if err != nil {
		p.fail(id, log, "normalize", err)
		return 0, time.Time{}
	}
	p.Counters.Add("unresolved."+id, float64(len(unresolved)))
	for _, u := range unresolved {
		log.Warn("unresolved symbol", "symbol", u.VenueSymbol, "reason", u.Reason)
		if p.Health != nil {
			p.Health.PublishUnresolved(UnresolvedEvent{Source: id, VenueSymbol: u.VenueSymbol, Reason: u.Reason, At: time.Now().UTC()})
		}
	}
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"
	n := 0
	var newest time.Time
	for _, e := range envs {
		Finalize(e, ref, !nopArchive, backfill)
		if n := e.GetNews(); n != nil && p.NewsSeen != nil && !backfill {
			if !p.NewsSeen.Add(connector.NewsDedupKey(n.GetUrl(), n.GetTitle(), n.GetOutlet(), e.GetEventTime().AsTime())) {
				p.Counters.Add("dedup."+id, 1)
				continue
			}
		}
		subj, err := bus.SubjectFor(e, venue)
		if err != nil {
			p.Counters.Add("errors."+id, 1)
			continue
		}
		if err := p.Pub.PublishWithID(subj, e.EventId, e); err != nil {
			p.fail(id, log, "publish", err)
			continue
		}
		n++
		p.Counters.Add("events."+id, 1)
		if p.Writer != nil {
			p.Writer.AddEnvelope(e, venue)
		}
		if et := e.GetEventTime(); et != nil {
			t := et.AsTime()
			if t.After(newest) {
				newest = t
			}
			if bar, ok := e.Payload.(*fav1.Envelope_Bar); ok && gaps != nil && !backfill {
				gaps.Observe(e.InstrumentId, bar.Bar.GetTimeframe(), t)
			}
		}
	}
	if _, err := p.Warehouse.Write(ctx, envs); err != nil { // nil-safe; idempotent per vintage
		p.fail(id, log, "warehouse", err)
	}
	if p.Health != nil {
		p.Health.Events(id, n)
	}
	return n, newest
}

// Finalize applies the pipeline's envelope rules (shared by live, backfill and
// replay so replay reproduces the live bytes): raw ref or raw_missing flag
// (spec 10 §9, only when an archive is configured), backfill flag, schema
// version and the deterministic event id (connector.StableEventID).
func Finalize(e *fav1.Envelope, ref *fav1.RawRef, archiveConfigured, backfill bool) {
	if e.RawRef == nil && ref != nil {
		e.RawRef = ref
	}
	e.RawMissing = e.RawRef == nil && archiveConfigured
	e.Backfill = e.Backfill || backfill
	if e.SchemaVersion == 0 {
		e.SchemaVersion = 1
	}
	e.EventId = connector.StableEventID(e)
}

func (p *Pipeline) fail(id string, log *slog.Logger, stage string, err error) {
	p.Counters.Add("errors."+id, 1)
	if p.Health != nil {
		p.Health.Error(id, err)
	}
	log.Warn(stage, "err", err)
}

func (p *Pipeline) gaps(ctx context.Context, source string) *GapDetector {
	if p.Health == nil {
		return nil
	}
	var seed map[string]time.Time
	if p.SeedGaps != nil {
		seed = p.SeedGaps(ctx, source)
	}
	return NewGapDetector(source, seed, p.Health.PublishGap)
}

// Run drives one live connector: stream or poll, then Process per message.
// It returns the error the connector ended with (nil on ctx cancel).
func (p *Pipeline) Run(ctx context.Context, log *slog.Logger, c connector.Connector) error {
	id := c.ID()
	norm, ok := c.(connector.Normalizer)
	if !ok {
		log.Error("connector does not implement Normalizer")
		return errors.New("connector does not implement Normalizer")
	}
	raws := make(chan connector.RawMessage, 1024)
	errc := make(chan error, 1)
	go func() {
		defer close(raws)
		var err error
		switch {
		case c.Capabilities().Stream:
			err = c.(connector.Streamer).Stream(ctx, nil, raws)
		case c.Capabilities().Poll:
			err = p.pollLoop(ctx, id, c.(connector.Poller), raws)
		default:
			err = errors.New("connector has neither stream nor poll capability")
			log.Warn(err.Error())
		}
		if err != nil && !errors.Is(err, context.Canceled) && ctx.Err() == nil {
			p.fail(id, log, "connector stopped", err)
		}
		if ctx.Err() != nil {
			err = nil
		}
		errc <- err
	}()
	gaps := p.gaps(ctx, id)
	for raw := range raws {
		if p.Health != nil {
			p.Health.Activity(id)
		}
		p.Process(ctx, log, c, norm, gaps, raw)
	}
	return <-errc
}

// Breaker tunables (spec 10 §7 `breaker.failures_to_open`, `retry.max_backoff`).
const (
	BreakerFailuresToOpen = 5
	RetryMaxBackoff       = 60 * time.Second
)

// retryWait is the wait after the n-th consecutive failure: exponential from
// 5 s capped at the poll interval while the breaker is closed, then
// RetryMaxBackoff (or the interval when longer) once it is open, each with
// up to 20 % jitter so sources sharing a host do not retry in lockstep.
func retryWait(n int, interval time.Duration, jitter func() float64) time.Duration {
	var w time.Duration
	if n >= BreakerFailuresToOpen {
		w = max(RetryMaxBackoff, interval)
	} else {
		w = min(interval, 5*time.Second<<max(0, n-1))
	}
	if jitter != nil {
		w += time.Duration(float64(w) * 0.2 * jitter())
	}
	return w
}

// pollLoop polls on the connector's schedule. Failures are recorded and
// retried with exponential backoff and jitter; after BreakerFailuresToOpen
// consecutive failures the breaker opens and retries slow to
// RetryMaxBackoff (spec 10 §6.3) instead of ending the connector; only ctx
// stops it.
func (p *Pipeline) pollLoop(ctx context.Context, id string, poller connector.Poller, out chan<- connector.RawMessage) error {
	interval := poller.Schedule().Interval
	if interval <= 0 {
		interval = time.Minute
	}
	since := time.Now().UTC()
	failures := 0
	for {
		msgs, err := poller.Poll(ctx, connector.PollRequest{Since: since})
		if ctx.Err() != nil {
			return ctx.Err()
		}
		wait := interval
		if err != nil {
			p.Counters.Add("errors."+id, 1)
			if p.Health != nil {
				p.Health.Error(id, err)
			}
			failures++
			if failures == BreakerFailuresToOpen {
				p.Counters.Add("breaker_open."+id, 1)
			}
			wait = retryWait(failures, interval, rand.Float64)
		} else {
			failures = 0
			since = time.Now().UTC()
			if p.Health != nil {
				p.Health.Activity(id)
			}
			for _, m := range msgs {
				select {
				case out <- m:
				case <-ctx.Done():
					return ctx.Err()
				}
			}
		}
		t := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
	}
}
