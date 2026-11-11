package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/proto"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// Work-queue identity (spec 21 §5, D-046): stream JOBS_INGEST on jobs.ingest.>,
// durable pull consumer "ingestor", jobs on jobs.ingest.backfill.
const (
	BackfillQueueKind = "ingest"
	BackfillConsumer  = "ingestor"
	BackfillJobKind   = "ingest.backfill"
	// progressAckEvery keeps the 60 s ack window open while a job runs.
	progressAckEvery = 20 * time.Second
)

// BackfillSpec is the JSON `Job.spec` of an ingest.backfill job.
type BackfillSpec struct {
	Source string `json:"source"`
	// Instruments are venue symbols; default: the catalog entry's config.symbols.
	// Text sources (no venue) run one query with an empty symbol.
	Instruments []string `json:"instruments,omitempty"`
	Timescale   string   `json:"timescale"`
	// Start/End are RFC 3339; End defaults to now, Start to End − BackfillDepth(timescale).
	Start string `json:"start,omitempty"`
	End   string `json:"end,omitempty"`
}

// BackfillResult is the JSON `JobResult.result`.
type BackfillResult struct {
	Source      string                 `json:"source"`
	Timescale   string                 `json:"timescale"`
	Start       time.Time              `json:"start"`
	End         time.Time              `json:"end"`
	Events      int                    `json:"events"`
	Instruments []BackfillInstrumentRs `json:"instruments"`
}

// BackfillInstrumentRs is one instrument's outcome.
type BackfillInstrumentRs struct {
	Symbol       string     `json:"symbol"`
	InstrumentID string     `json:"instrument_id,omitempty"`
	Events       int        `json:"events"`
	Cursor       *time.Time `json:"cursor,omitempty"`
	Status       string     `json:"status"` // done | skipped | failed
	Error        string     `json:"error,omitempty"`
}

// parseBackfillSpec validates and completes a spec against the catalog entry.
func parseBackfillSpec(raw []byte, now time.Time) (BackfillSpec, time.Time, time.Time, error) {
	var spec BackfillSpec
	if err := json.Unmarshal(raw, &spec); err != nil {
		return spec, time.Time{}, time.Time{}, fmt.Errorf("%w: spec: %v", ErrInvalid, err)
	}
	spec.Source = strings.TrimSpace(spec.Source)
	spec.Timescale = strings.TrimSpace(spec.Timescale)
	if spec.Source == "" {
		return spec, time.Time{}, time.Time{}, fmt.Errorf("%w: source is required", ErrInvalid)
	}
	if spec.Timescale == "" {
		spec.Timescale = "1m"
	}
	if _, ok := Period(spec.Timescale); !ok {
		return spec, time.Time{}, time.Time{}, fmt.Errorf("%w: timescale %q (use e.g. 1m, 1h, 1d)", ErrInvalid, spec.Timescale)
	}
	end := now
	if spec.End != "" {
		t, err := time.Parse(time.RFC3339, spec.End)
		if err != nil {
			return spec, time.Time{}, time.Time{}, fmt.Errorf("%w: end: %v", ErrInvalid, err)
		}
		end = t.UTC()
	}
	start := end.Add(-catalog.BackfillDepth(spec.Timescale))
	if spec.Start != "" {
		t, err := time.Parse(time.RFC3339, spec.Start)
		if err != nil {
			return spec, time.Time{}, time.Time{}, fmt.Errorf("%w: start: %v", ErrInvalid, err)
		}
		start = t.UTC()
	}
	if !start.Before(end) {
		return spec, time.Time{}, time.Time{}, fmt.Errorf("%w: start %s is not before end %s", ErrInvalid, start.Format(time.RFC3339), end.Format(time.RFC3339))
	}
	var syms []string
	for _, s := range spec.Instruments {
		if s = strings.TrimSpace(s); s != "" {
			syms = append(syms, s)
		}
	}
	spec.Instruments = syms
	spec.Start, spec.End = start.Format(time.RFC3339), end.Format(time.RFC3339)
	return spec, start, end, nil
}

// Backfiller runs the ingest.backfill work queue.
type Backfiller struct {
	JS       jetstream.JetStream
	Consumer jetstream.Consumer
	Sup      *Supervisor
	Pipe     *Pipeline
	Cursors  connector.BackfillCursorStore
	RunLog   RunLogger
	Log      *slog.Logger
	Counters *bus.Counters
	WorkerID string
}

// NewBackfiller ensures the JOBS_INGEST work queue and its consumer.
func NewBackfiller(ctx context.Context, nc *nats.Conn, sup *Supervisor, pipe *Pipeline, cursors connector.BackfillCursorStore, runLog RunLogger, log *slog.Logger, counters *bus.Counters, workerID string) (*Backfiller, error) {
	js, err := jetstream.New(nc)
	if err != nil {
		return nil, err
	}
	_, cons, err := bus.EnsureWorkQueue(ctx, js, BackfillQueueKind, BackfillConsumer)
	if err != nil {
		return nil, fmt.Errorf("work queue %s: %w", bus.WorkQueueName(BackfillQueueKind), err)
	}
	if cursors == nil {
		cursors = connector.NewMemBackfillCursors()
	}
	return &Backfiller{JS: js, Consumer: cons, Sup: sup, Pipe: pipe, Cursors: cursors, RunLog: runLog, Log: log, Counters: counters, WorkerID: workerID}, nil
}

// Enqueue validates a spec against the catalog, publishes the Job on
// jobs.ingest.backfill (Nats-Msg-Id = job id) and returns the job id.
func (b *Backfiller) Enqueue(ctx context.Context, raw []byte) (string, error) {
	spec, _, _, err := parseBackfillSpec(raw, time.Now().UTC())
	if err != nil {
		return "", err
	}
	if _, err := b.resolveBackfiller(spec.Source); err != nil {
		return "", err
	}
	specJSON, _ := json.Marshal(spec)
	id := ulid.Make().String()
	job := &fav1.Job{JobId: id, Kind: BackfillJobKind, Spec: specJSON, Priority: 3}
	data, err := proto.Marshal(job)
	if err != nil {
		return "", err
	}
	if _, err := b.JS.Publish(ctx, bus.Jobs(BackfillJobKind), data, jetstream.WithMsgID(id)); err != nil {
		return "", fmt.Errorf("enqueue: %w", err)
	}
	b.Counters.Add("backfill.enqueued", 1)
	return id, nil
}

// resolveBackfiller checks the source may run and its connector can backfill.
func (b *Backfiller) resolveBackfiller(source string) (catalog.Source, error) {
	src, err := b.Sup.Resolve(source)
	if err != nil {
		return src, err
	}
	c, err := connector.NewFor(source, src.Adapter)
	if err != nil {
		return src, fmt.Errorf("%w: %v", ErrNotFound, err)
	}
	if _, ok := c.(connector.Backfiller); !ok || !c.Capabilities().Backfill {
		return src, fmt.Errorf("%w: source %q cannot backfill", ErrConflict, source)
	}
	return src, nil
}

// Run pulls jobs until ctx is done, one job per lane: Run is started once per
// backfill lane (-backfill-lanes, default 2). Every job's requests go through
// its source's low-priority backfill lane (connector.LaneLimiter), which
// yields to the live connector and is capped at a share of the catalog rate,
// so parallel jobs never starve live polling (spec 10 §6.4).
func (b *Backfiller) Run(ctx context.Context) {
	for ctx.Err() == nil {
		batch, err := b.Consumer.Fetch(1, jetstream.FetchMaxWait(5*time.Second))
		if err != nil {
			if ctx.Err() == nil {
				b.Log.Warn("backfill fetch", "err", err)
				time.Sleep(time.Second)
			}
			continue
		}
		for msg := range batch.Messages() {
			b.handle(ctx, msg)
		}
		if err := batch.Error(); err != nil && ctx.Err() == nil && !errors.Is(err, nats.ErrTimeout) {
			b.Log.Warn("backfill batch", "err", err)
		}
	}
}

func (b *Backfiller) handle(ctx context.Context, msg jetstream.Msg) {
	var job fav1.Job
	if err := proto.Unmarshal(msg.Data(), &job); err != nil {
		b.Log.Warn("undecodable job; dropping", "err", err)
		_ = msg.Ack()
		return
	}
	log := b.Log.With("job", job.JobId)
	// Keep the ack window open while the job runs.
	done := make(chan struct{})
	go func() {
		t := time.NewTicker(progressAckEvery)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				_ = msg.InProgress()
			}
		}
	}()
	res, err := b.execute(ctx, log, &job)
	close(done)
	if ctx.Err() != nil && err != nil {
		// Shutting down mid-job: leave it for redelivery; cursors make the rerun resume.
		_ = msg.Nak()
		return
	}
	result := &fav1.JobResult{JobId: job.JobId, Status: fav1.JobResult_STATUS_OK, WorkerId: b.WorkerID}
	if err != nil {
		result.Status, result.Error = fav1.JobResult_STATUS_FAILED, err.Error()
		b.Counters.Add("backfill.failed", 1)
	} else {
		b.Counters.Add("backfill.done", 1)
	}
	if res != nil {
		result.Result, _ = json.Marshal(res)
	}
	if err := b.Pipe.Pub.Publish(bus.JobResult(BackfillJobKind), result); err != nil {
		log.Error("result publish failed; not acking", "err", err)
		return
	}
	if err := msg.Ack(); err != nil {
		log.Warn("ack failed", "err", err)
	}
}

func (b *Backfiller) progress(jobID string, fraction float64, message string) {
	_ = b.Pipe.Pub.Publish(bus.JobProgress(jobID), &fav1.JobProgress{JobId: jobID, Fraction: float32(fraction), Message: message})
}

// execute runs one job; a non-nil result is returned together with an error
// when some instruments failed.
func (b *Backfiller) execute(ctx context.Context, log *slog.Logger, job *fav1.Job) (*BackfillResult, error) {
	spec, start, end, err := parseBackfillSpec(job.Spec, time.Now().UTC())
	if err != nil {
		return nil, err
	}
	if _, err := b.resolveBackfiller(spec.Source); err != nil {
		return nil, err
	}
	// Low-priority lane (spec 10 §6.4): the job's requests yield to the live connector's.
	c, src, deps, err := b.Sup.Instance(ctx, spec.Source, true)
	if err != nil {
		return nil, err
	}
	norm, ok := c.(connector.Normalizer)
	if !ok {
		return nil, fmt.Errorf("%w: %s has no normalizer", ErrConflict, spec.Source)
	}
	bf := c.(connector.Backfiller)
	symbols := spec.Instruments
	if len(symbols) == 0 {
		symbols = src.Params().Symbols
	}
	if len(symbols) == 0 {
		if c.Capabilities().Venue != "" {
			return nil, fmt.Errorf("%w: %s needs instruments (none in the job or in config.symbols)", ErrInvalid, spec.Source)
		}
		symbols = []string{""} // text source: one query over the range
	}
	log.Info("backfill start", "source", spec.Source, "timescale", spec.Timescale, "start", start, "end", end, "instruments", symbols)
	var runID string
	if b.RunLog != nil {
		if rid, err := b.RunLog.Start(ctx, spec.Source); err == nil {
			runID = rid
		}
	}
	res := &BackfillResult{Source: spec.Source, Timescale: spec.Timescale, Start: start, End: end}
	var errs []error
	for i, sym := range symbols {
		b.progress(job.JobId, float64(i)/float64(len(symbols)), fmt.Sprintf("%s %s", spec.Source, sym))
		r := b.instrument(ctx, log, c, norm, bf, deps, spec, sym, start, end)
		res.Events += r.Events
		res.Instruments = append(res.Instruments, r)
		if r.Error != "" {
			errs = append(errs, fmt.Errorf("%s: %s", sym, r.Error))
		}
		if ctx.Err() != nil {
			break
		}
	}
	b.progress(job.JobId, 1, "done")
	err = errors.Join(errs...)
	if runID != "" {
		status, lastErr := "done", ""
		if err != nil {
			status, lastErr = "failed", err.Error()
		}
		ectx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_ = b.RunLog.End(ectx, runID, status, int64(res.Events), int64(len(errs)), lastErr)
		cancel()
	}
	log.Info("backfill end", "events", res.Events, "errors", len(errs))
	return res, err
}

func (b *Backfiller) instrument(ctx context.Context, log *slog.Logger, c connector.Connector, norm connector.Normalizer, bf connector.Backfiller, deps connector.Deps, spec BackfillSpec, sym string, start, end time.Time) BackfillInstrumentRs {
	r := BackfillInstrumentRs{Symbol: sym, Status: "done"}
	// Only market-data sources (with a venue) resolve symbols to instruments; for macro series,
	// news queries or CIKs resolving would provision bogus instruments (spec 12 §6). Their
	// cursor is kept per series instead: timescale "<tf>|<series>".
	cursorScope := spec.Timescale
	if sym != "" && deps.Resolver != nil && c.Capabilities().Venue != "" {
		r.InstrumentID, _ = deps.Resolver.Resolve(spec.Source, sym)
	} else if sym != "" {
		cursorScope = spec.Timescale + "|" + sym
	}
	cur, status, ok, err := b.Cursors.GetBackfill(ctx, spec.Source, r.InstrumentID, cursorScope, start)
	if err != nil {
		r.Status, r.Error = "failed", "cursor: "+err.Error()
		return r
	}
	if ok && status == "done" {
		r.Status = "skipped"
		if !cur.IsZero() {
			r.Cursor = &cur
		}
		return r
	}
	req := connector.BackfillRequest{VenueSymbol: sym, InstrumentID: r.InstrumentID, Timeframe: spec.Timescale, From: start, To: end}
	if ok && !cur.IsZero() && cur.After(start) {
		req.Cursor = cur.Format(time.RFC3339)
		req.From = cur
	}
	_ = b.Cursors.SetBackfill(ctx, spec.Source, r.InstrumentID, cursorScope, start, end, cur, "running")

	out := make(chan connector.RawMessage, 16)
	errc := make(chan error, 1)
	go func() {
		defer close(out)
		errc <- bf.Backfill(ctx, req, out)
	}()
	newest := cur
	for raw := range out {
		if b.Pipe.Health != nil {
			b.Pipe.Health.Activity(spec.Source)
		}
		n, t := b.Pipe.Process(ctx, log, c, norm, nil, raw)
		r.Events += n
		if t.After(newest) {
			newest = t
			_ = b.Cursors.SetBackfill(ctx, spec.Source, r.InstrumentID, cursorScope, start, end, newest, "running")
		}
	}
	if !newest.IsZero() {
		r.Cursor = &newest
	}
	if err := <-errc; err != nil && !errors.Is(err, context.Canceled) {
		r.Status, r.Error = "failed", err.Error()
		_ = b.Cursors.SetBackfill(ctx, spec.Source, r.InstrumentID, cursorScope, start, end, newest, "failed")
		return r
	}
	if ctx.Err() != nil {
		r.Status, r.Error = "failed", "interrupted"
		return r
	}
	_ = b.Cursors.SetBackfill(ctx, spec.Source, r.InstrumentID, cursorScope, start, end, newest, "done")
	return r
}
