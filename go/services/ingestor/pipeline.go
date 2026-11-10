package main

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
	"github.com/fey/financial-auto/go/pkg/tsdb"
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
		if e.RawRef == nil && ref != nil {
			e.RawRef = ref
		}
		subj, err := bus.SubjectFor(e, venue)
		if err != nil {
			p.Counters.Add("errors."+id, 1)
			continue
		}
		if err := p.Pub.Publish(subj, e); err != nil {
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
	if p.Health != nil {
		p.Health.Events(id, n)
	}
	return n, newest
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

// pollLoop polls on the connector's schedule. Failures are recorded and
// retried with exponential backoff (capped at the poll interval, spec 10 §6.3)
// instead of ending the connector; only ctx stops it.
func (p *Pipeline) pollLoop(ctx context.Context, id string, poller connector.Poller, out chan<- connector.RawMessage) error {
	interval := poller.Schedule().Interval
	if interval <= 0 {
		interval = time.Minute
	}
	since := time.Now().UTC()
	backoff := time.Duration(0)
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
			if backoff == 0 {
				backoff = min(interval, 5*time.Second)
			} else {
				backoff = min(interval, backoff*2)
			}
			wait = backoff
		} else {
			backoff = 0
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
