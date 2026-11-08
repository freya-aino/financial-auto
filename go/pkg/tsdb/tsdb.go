// Package tsdb writes canonical market events to VictoriaMetrics through its
// JSON line import API (spec 20 §5.3: metric + bounded labels, ms timestamps,
// VM timestamp = event_time).
package tsdb

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// Metric names per spec 20 §5.3.
const (
	MetricTradePrice = "fa_trade_price"
	MetricTradeSize  = "fa_trade_size"
)

// Point is one sample of one series.
type Point struct {
	Metric string
	Labels map[string]string
	Value  float64
	TsMs   int64
}

// Stats are the writer's cumulative counters.
type Stats struct {
	Written uint64 // samples accepted by VM
	Errors  uint64 // failed flushes (each may hold many samples)
	Dropped uint64 // samples dropped because the buffer was full
}

// Writer batches points and POSTs them to <url>/api/v1/import.
type Writer struct {
	url       string
	client    *http.Client
	batch     int
	every     time.Duration
	maxBuffer int

	mu    sync.Mutex
	buf   []Point
	stats Stats
	wake  chan struct{}
}

// Options tune the writer; zero values take the defaults.
type Options struct {
	BatchSize     int           // flush when this many points are buffered (default 500)
	FlushInterval time.Duration // flush at least this often (default 1 s)
	MaxBuffer     int           // drop points beyond this (default 50 000)
	Client        *http.Client  // default: 10 s timeout
}

// New creates a writer for a VictoriaMetrics base URL such as http://127.0.0.1:8428.
func New(url string, o Options) *Writer {
	if o.BatchSize <= 0 {
		o.BatchSize = 500
	}
	if o.FlushInterval <= 0 {
		o.FlushInterval = time.Second
	}
	if o.MaxBuffer <= 0 {
		o.MaxBuffer = 50_000
	}
	if o.Client == nil {
		o.Client = &http.Client{Timeout: 10 * time.Second}
	}
	return &Writer{
		url: strings.TrimRight(url, "/"), client: o.Client, batch: o.BatchSize, every: o.FlushInterval,
		maxBuffer: o.MaxBuffer, wake: make(chan struct{}, 1),
	}
}

// Add buffers points; a full batch wakes the flusher.
func (w *Writer) Add(pts ...Point) {
	w.mu.Lock()
	for _, p := range pts {
		if len(w.buf) >= w.maxBuffer {
			w.stats.Dropped++
			continue
		}
		w.buf = append(w.buf, p)
	}
	full := len(w.buf) >= w.batch
	w.mu.Unlock()
	if full {
		select {
		case w.wake <- struct{}{}:
		default:
		}
	}
}

// AddEnvelope converts a canonical envelope to points (trades only for now)
// and buffers them. venue is the connector's subject venue token.
func (w *Writer) AddEnvelope(e *fav1.Envelope, venue string) {
	w.Add(TradePoints(e, venue)...)
}

// TradePoints maps a Trade envelope to fa_trade_price / fa_trade_size samples.
// Non-trade envelopes yield nothing.
func TradePoints(e *fav1.Envelope, venue string) []Point {
	t := e.GetTrade()
	if t == nil || e.GetInstrumentId() == "" {
		return nil
	}
	ts := e.GetEventTime()
	if ts == nil {
		ts = e.GetIngestTime()
	}
	ms := ts.AsTime().UnixMilli()
	labels := map[string]string{"inst": bus.SubjectToken(e.InstrumentId), "venue": venue, "src": e.SourceId}
	return []Point{
		{Metric: MetricTradePrice, Labels: labels, Value: t.Price, TsMs: ms},
		{Metric: MetricTradeSize, Labels: labels, Value: t.Size, TsMs: ms},
	}
}

// Stats returns a copy of the counters.
func (w *Writer) Stats() Stats {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.stats
}

// Run flushes on a timer, when a batch fills, and once more on ctx done.
func (w *Writer) Run(ctx context.Context, onErr func(error)) {
	t := time.NewTicker(w.every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			w.flush(context.Background(), onErr)
			return
		case <-t.C:
		case <-w.wake:
		}
		w.flush(ctx, onErr)
	}
}

// Flush sends everything buffered now (used by tests and shutdown).
func (w *Writer) Flush(ctx context.Context) error {
	var ferr error
	w.flush(ctx, func(err error) { ferr = err })
	return ferr
}

func (w *Writer) flush(ctx context.Context, onErr func(error)) {
	w.mu.Lock()
	pts := w.buf
	w.buf = nil
	w.mu.Unlock()
	if len(pts) == 0 {
		return
	}
	err := w.send(ctx, pts)
	w.mu.Lock()
	if err == nil {
		w.stats.Written += uint64(len(pts))
	} else {
		w.stats.Errors++
	}
	w.mu.Unlock()
	if err != nil && onErr != nil {
		onErr(err)
	}
}

// line is VictoriaMetrics' JSON import format (one series per line).
type line struct {
	Metric     map[string]string `json:"metric"`
	Values     []float64         `json:"values"`
	Timestamps []int64           `json:"timestamps"`
}

// Encode groups points by series into import lines.
func Encode(pts []Point) ([]byte, error) {
	order := []string{}
	lines := map[string]*line{}
	for _, p := range pts {
		key := seriesKey(p)
		l, ok := lines[key]
		if !ok {
			m := make(map[string]string, len(p.Labels)+1)
			m["__name__"] = p.Metric
			for k, v := range p.Labels {
				m[k] = v
			}
			l = &line{Metric: m}
			lines[key] = l
			order = append(order, key)
		}
		l.Values = append(l.Values, p.Value)
		l.Timestamps = append(l.Timestamps, p.TsMs)
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	for _, k := range order {
		if err := enc.Encode(lines[k]); err != nil {
			return nil, err
		}
	}
	return b.Bytes(), nil
}

func seriesKey(p Point) string {
	var sb strings.Builder
	sb.WriteString(p.Metric)
	// Labels are few and fixed (inst, venue, src); iterate in a stable order.
	for _, k := range []string{"inst", "venue", "src", "tf", "lvl"} {
		if v, ok := p.Labels[k]; ok {
			sb.WriteString("|" + k + "=" + v)
		}
	}
	return sb.String()
}

func (w *Writer) send(ctx context.Context, pts []Point) error {
	body, err := Encode(pts)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, w.url+"/api/v1/import", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/stream+json")
	res, err := w.client.Do(req)
	if err != nil {
		return fmt.Errorf("tsdb import: %w", err)
	}
	defer res.Body.Close()
	if res.StatusCode/100 != 2 {
		msg, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return fmt.Errorf("tsdb import: %s: %s", res.Status, strings.TrimSpace(string(msg)))
	}
	return nil
}
