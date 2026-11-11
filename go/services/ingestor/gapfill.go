package main

import (
	"context"
	"errors"
	"log/slog"
	"maps"
	"sync"
	"time"

	"github.com/fey/financial-auto/go/pkg/bus"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// Gap-fill tunables (spec 10 §7 `gapfill.max_span`, `gapfill.queue`).
const (
	// GapFillMaxSpan: longer gaps are left to an explicit backfill job (they
	// are still reported on ingest.gap.<source>).
	GapFillMaxSpan = 24 * time.Hour
	// GapFillQueue bounds pending fills; extra requests are dropped and counted.
	GapFillQueue = 256
	// StreamSilence: a stream connector that delivers again after this much
	// silence gets the silent window gap-filled for its configured symbols.
	StreamSilence = StreamPeriod
)

// GapJob is one range to re-fetch.
type GapJob struct {
	Source, Symbol, InstrumentID, Timeframe string
	From, To                                time.Time
	Reason                                  string // "bar_gap" | "stream_silence"
}

func (j GapJob) key() string {
	return j.Source + "|" + j.Symbol + "|" + j.Timeframe + "|" + j.From.UTC().Format(time.RFC3339)
}

// GapFiller re-fetches gaps through the connector's Backfill on the
// low-priority backfill lane and runs the payloads through the pipeline with
// MetaGapFill set, so the envelopes carry `gap_filled` (and `backfill`)
// (spec 10 §6.3, §8). Envelopes the live connector did publish keep their
// stable event id, so JetStream drops the overlap. Sources whose connector
// cannot backfill only report the gap.
type GapFiller struct {
	Pipe *Pipeline
	// Instance returns a fresh connector initialised on the backfill lane.
	Instance func(ctx context.Context, source string) (connector.Connector, error)
	Counters *bus.Counters
	Log      *slog.Logger
	MaxSpan  time.Duration

	once     sync.Once
	jobs     chan GapJob
	mu       sync.Mutex
	inflight map[string]bool
}

func (g *GapFiller) init() {
	g.once.Do(func() {
		g.jobs = make(chan GapJob, GapFillQueue)
		g.inflight = map[string]bool{}
		if g.MaxSpan <= 0 {
			g.MaxSpan = GapFillMaxSpan
		}
	})
}

// Enqueue schedules a fill; false when skipped (too long, duplicate, queue full).
func (g *GapFiller) Enqueue(j GapJob) bool {
	if g == nil {
		return false
	}
	g.init()
	if j.Symbol == "" || !j.From.Before(j.To) {
		return false
	}
	if j.To.Sub(j.From) > g.MaxSpan {
		g.count("gapfill.too_long."+j.Source, 1)
		return false
	}
	g.mu.Lock()
	if g.inflight[j.key()] {
		g.mu.Unlock()
		return false
	}
	g.inflight[j.key()] = true
	g.mu.Unlock()
	select {
	case g.jobs <- j:
		g.count("gapfill.requested."+j.Source, 1)
		return true
	default:
		g.mu.Lock()
		delete(g.inflight, j.key())
		g.mu.Unlock()
		g.count("gapfill.dropped."+j.Source, 1)
		return false
	}
}

// Run fills queued gaps one at a time until ctx is done.
func (g *GapFiller) Run(ctx context.Context) {
	g.init()
	for {
		select {
		case <-ctx.Done():
			return
		case j := <-g.jobs:
			n, err := g.Fill(ctx, j)
			g.mu.Lock()
			delete(g.inflight, j.key())
			g.mu.Unlock()
			if err != nil {
				g.count("gapfill.failed."+j.Source, 1)
				if g.Log != nil {
					g.Log.Warn("gap fill failed", "source", j.Source, "symbol", j.Symbol, "from", j.From, "to", j.To, "err", err)
				}
				continue
			}
			g.count("gapfill.done."+j.Source, 1)
			g.count("gapfill.events."+j.Source, float64(n))
			if g.Log != nil {
				g.Log.Info("gap filled", "source", j.Source, "symbol", j.Symbol, "reason", j.Reason, "from", j.From, "to", j.To, "events", n)
			}
		}
	}
}

// ErrNoBackfill: the source's connector cannot re-fetch history.
var ErrNoBackfill = errors.New("connector cannot backfill")

// Fill runs one job synchronously and returns the envelopes published.
func (g *GapFiller) Fill(ctx context.Context, j GapJob) (int, error) {
	c, err := g.Instance(ctx, j.Source)
	if err != nil {
		return 0, err
	}
	bf, ok := c.(connector.Backfiller)
	norm, ok2 := c.(connector.Normalizer)
	if !ok || !ok2 || !c.Capabilities().Backfill {
		return 0, ErrNoBackfill
	}
	out := make(chan connector.RawMessage, 16)
	errc := make(chan error, 1)
	go func() {
		defer close(out)
		errc <- bf.Backfill(ctx, connector.BackfillRequest{VenueSymbol: j.Symbol, InstrumentID: j.InstrumentID, Timeframe: j.Timeframe, From: j.From, To: j.To}, out)
	}()
	log := g.Log
	if log == nil {
		log = slog.Default()
	}
	n := 0
	for raw := range out {
		meta := maps.Clone(raw.TransportMeta)
		if meta == nil {
			meta = map[string]string{}
		}
		meta[connector.MetaGapFill] = "1"
		meta[connector.MetaBackfill] = "1"
		raw.TransportMeta = meta
		k, _ := g.Pipe.Process(ctx, log.With("source", j.Source, "gapfill", true), c, norm, nil, raw)
		n += k
	}
	return n, <-errc
}

func (g *GapFiller) count(k string, v float64) {
	if g.Counters != nil {
		g.Counters.Add(k, v)
	}
}
