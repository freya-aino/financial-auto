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
	"strconv"
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
	MetricTradeCount = "fa_trade_count" // trades folded into the sample (same-millisecond VWAP, D-038)
	MetricQuoteBidPx = "fa_quote_bid_px"
	MetricQuoteBidSz = "fa_quote_bid_sz"
	MetricQuoteAskPx = "fa_quote_ask_px"
	MetricQuoteAskSz = "fa_quote_ask_sz"
	MetricBarOpen    = "fa_bar_open"
	MetricBarHigh    = "fa_bar_high"
	MetricBarLow     = "fa_bar_low"
	MetricBarClose   = "fa_bar_close"
	MetricBarVolume  = "fa_bar_volume"
	MetricBarVwap    = "fa_bar_vwap"
	MetricBookBidPx  = "fa_book_bid_px"
	MetricBookBidSz  = "fa_book_bid_sz"
	MetricBookAskPx  = "fa_book_ask_px"
	MetricBookAskSz  = "fa_book_ask_sz"
)

// MaxBookLevels bounds the `lvl` label (spec 20 §5.3: N ≤ 20).
const MaxBookLevels = 20

// Point is one sample of one series.
type Point struct {
	Metric string
	Labels map[string]string
	Value  float64
	TsMs   int64
}

// Stats are the writer's cumulative counters.
type Stats struct {
	Written    uint64 // samples accepted by VM
	Errors     uint64 // failed flushes (each may hold many samples)
	Dropped    uint64 // samples dropped because the buffer was full
	Aggregated uint64 // trades folded into an earlier trade of the same series and millisecond (VWAP)
	Late       uint64 // trades older than their series' newest emitted millisecond, written unaggregated
}

// tradeBucket accumulates the trades of one series within one millisecond:
// the written price is the volume-weighted average, the size the sum (D-038).
type tradeBucket struct {
	labels map[string]string
	ms     int64
	pv, sz float64 // Σ price·size, Σ size
	last   float64 // last price, used when Σ size == 0
	n      int     // trades folded in
	opened time.Time
}

func (b *tradeBucket) points() []Point {
	px := b.last
	if b.sz > 0 {
		px = b.pv / b.sz
	}
	return []Point{
		{Metric: MetricTradePrice, Labels: b.labels, Value: px, TsMs: b.ms},
		{Metric: MetricTradeSize, Labels: b.labels, Value: b.sz, TsMs: b.ms},
		{Metric: MetricTradeCount, Labels: b.labels, Value: float64(b.n), TsMs: b.ms},
	}
}

// Writer batches points and POSTs them to <url>/api/v1/import.
type Writer struct {
	url       string
	client    *http.Client
	batch     int
	every     time.Duration
	maxBuffer int

	holdFor time.Duration
	now     func() time.Time

	mu      sync.Mutex
	buf     []Point
	trades  map[string]*tradeBucket // open (newest-millisecond) bucket per trade series
	emitted map[string]int64        // newest millisecond already emitted per trade series
	stats   Stats
	wake    chan struct{}
}

// Options tune the writer; zero values take the defaults.
type Options struct {
	BatchSize     int           // flush when this many points are buffered (default 500)
	FlushInterval time.Duration // flush at least this often (default 1 s)
	MaxBuffer     int           // drop points beyond this (default 50 000)
	Client        *http.Client  // default: 10 s timeout
	// TradeHold is how long the newest millisecond of a trade series stays open for
	// further same-millisecond trades before it is written (default FlushInterval).
	// A newer millisecond on the same series closes it at once.
	TradeHold time.Duration
	Now       func() time.Time // clock for TradeHold (tests); default time.Now
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
	if o.TradeHold <= 0 {
		o.TradeHold = o.FlushInterval
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	return &Writer{
		url: strings.TrimRight(url, "/"), client: o.Client, batch: o.BatchSize, every: o.FlushInterval,
		maxBuffer: o.MaxBuffer, holdFor: o.TradeHold, now: o.Now, wake: make(chan struct{}, 1),
		trades: map[string]*tradeBucket{}, emitted: map[string]int64{},
	}
}

// Add buffers points; a full batch wakes the flusher.
func (w *Writer) Add(pts ...Point) {
	w.mu.Lock()
	w.bufferLocked(pts)
	full := len(w.buf) >= w.batch
	w.mu.Unlock()
	if full {
		w.kick()
	}
}

func (w *Writer) bufferLocked(pts []Point) {
	for _, p := range pts {
		if len(w.buf) >= w.maxBuffer {
			w.stats.Dropped++
			continue
		}
		w.buf = append(w.buf, p)
	}
}

func (w *Writer) kick() {
	select {
	case w.wake <- struct{}{}:
	default:
	}
}

// AddEnvelope converts a canonical market envelope (trade, quote, bar, book)
// to points and buffers them. venue is the connector's subject venue token.
// Trades go through the same-millisecond VWAP aggregation (AddTrade).
func (w *Writer) AddEnvelope(e *fav1.Envelope, venue string) {
	if t := e.GetTrade(); t != nil {
		if labels, ms, ok := base(e, venue); ok {
			w.AddTrade(labels, ms, t.Price, t.Size)
		}
		return
	}
	w.Add(Points(e, venue)...)
}

// AddTrade aggregates trades per series (inst, venue, src) and millisecond
// (D-038): VictoriaMetrics keeps one sample per series and timestamp, so
// several fills in one millisecond are written as a single sample whose
// fa_trade_price is the VWAP Σp·s/Σs (the last price when Σs = 0) and whose
// fa_trade_size is Σs. The newest millisecond of a series stays open until a
// newer one arrives or TradeHold passes; a trade older than the newest
// emitted millisecond of its series cannot be merged any more and is written
// as-is (Stats.Late).
func (w *Writer) AddTrade(labels map[string]string, ms int64, price, size float64) {
	key := seriesKey(Point{Metric: MetricTradePrice, Labels: labels})
	w.mu.Lock()
	b := w.trades[key]
	switch {
	case b != nil && ms == b.ms:
		b.pv += price * size
		b.sz += size
		b.last = price
		b.n++
		w.stats.Aggregated++
	case b != nil && ms < b.ms, b == nil && w.isEmittedLocked(key, ms):
		w.stats.Late++
		w.bufferLocked((&tradeBucket{labels: labels, ms: ms, pv: price * size, sz: size, last: price, n: 1}).points())
	default:
		if b != nil { // a newer millisecond closes the open one
			w.closeLocked(key, b)
		}
		w.trades[key] = &tradeBucket{labels: labels, ms: ms, pv: price * size, sz: size, last: price, n: 1, opened: w.now()}
	}
	full := len(w.buf) >= w.batch
	w.mu.Unlock()
	if full {
		w.kick()
	}
}

func (w *Writer) isEmittedLocked(key string, ms int64) bool {
	last, ok := w.emitted[key]
	return ok && ms <= last
}

func (w *Writer) closeLocked(key string, b *tradeBucket) {
	w.bufferLocked(b.points())
	w.emitted[key] = b.ms
	delete(w.trades, key)
}

// closeTradesLocked moves open trade buckets older than TradeHold (all of
// them with force) into the write buffer.
func (w *Writer) closeTradesLocked(force bool) {
	now := w.now()
	for key, b := range w.trades {
		if force || now.Sub(b.opened) >= w.holdFor {
			w.closeLocked(key, b)
		}
	}
}

// Points maps any market-data envelope to its samples; other payloads yield nothing.
func Points(e *fav1.Envelope, venue string) []Point {
	switch e.GetPayload().(type) {
	case *fav1.Envelope_Trade:
		return TradePoints(e, venue)
	case *fav1.Envelope_Quote:
		return QuotePoints(e, venue)
	case *fav1.Envelope_Bar:
		return BarPoints(e, venue)
	case *fav1.Envelope_Book:
		return BookPoints(e, venue)
	}
	return nil
}

// base returns the common labels and ms timestamp, or ok=false when the
// envelope has no instrument.
func base(e *fav1.Envelope, venue string) (map[string]string, int64, bool) {
	if e.GetInstrumentId() == "" {
		return nil, 0, false
	}
	ts := e.GetEventTime()
	if ts == nil {
		ts = e.GetIngestTime()
	}
	labels := map[string]string{"inst": bus.SubjectToken(e.InstrumentId), "venue": venue, "src": e.SourceId}
	return labels, ts.AsTime().UnixMilli(), true
}

// TradePoints maps one Trade envelope to fa_trade_price / fa_trade_size /
// fa_trade_count (= 1) samples, without aggregation (the Writer aggregates
// same-millisecond trades, AddTrade). Non-trade envelopes yield nothing.
func TradePoints(e *fav1.Envelope, venue string) []Point {
	t := e.GetTrade()
	labels, ms, ok := base(e, venue)
	if t == nil || !ok {
		return nil
	}
	return (&tradeBucket{labels: labels, ms: ms, pv: t.Price * t.Size, sz: t.Size, last: t.Price, n: 1}).points()
}

// QuotePoints maps a Quote envelope to fa_quote_{bid,ask}_{px,sz}.
func QuotePoints(e *fav1.Envelope, venue string) []Point {
	q := e.GetQuote()
	labels, ms, ok := base(e, venue)
	if q == nil || !ok {
		return nil
	}
	return []Point{
		{Metric: MetricQuoteBidPx, Labels: labels, Value: q.BidPrice, TsMs: ms},
		{Metric: MetricQuoteBidSz, Labels: labels, Value: q.BidSize, TsMs: ms},
		{Metric: MetricQuoteAskPx, Labels: labels, Value: q.AskPrice, TsMs: ms},
		{Metric: MetricQuoteAskSz, Labels: labels, Value: q.AskSize, TsMs: ms},
	}
}

// BarPoints maps a Bar envelope to fa_bar_{open,high,low,close,volume[,vwap]}
// with the extra `tf` label. VWAP is written only when the source set it.
func BarPoints(e *fav1.Envelope, venue string) []Point {
	b := e.GetBar()
	labels, ms, ok := base(e, venue)
	if b == nil || !ok {
		return nil
	}
	labels["tf"] = b.GetTimeframe()
	pts := []Point{
		{Metric: MetricBarOpen, Labels: labels, Value: b.Open, TsMs: ms},
		{Metric: MetricBarHigh, Labels: labels, Value: b.High, TsMs: ms},
		{Metric: MetricBarLow, Labels: labels, Value: b.Low, TsMs: ms},
		{Metric: MetricBarClose, Labels: labels, Value: b.Close, TsMs: ms},
		{Metric: MetricBarVolume, Labels: labels, Value: b.Volume, TsMs: ms},
	}
	if b.Vwap != nil {
		pts = append(pts, Point{Metric: MetricBarVwap, Labels: labels, Value: b.GetVwap(), TsMs: ms})
	}
	return pts
}

// BookPoints maps a BookSnapshot to fa_book_{bid,ask}_{px,sz} per level with
// the `lvl` label (0 = top of book), truncated to MaxBookLevels.
func BookPoints(e *fav1.Envelope, venue string) []Point {
	bk := e.GetBook()
	labels, ms, ok := base(e, venue)
	if bk == nil || !ok {
		return nil
	}
	var pts []Point
	side := func(levels []*fav1.BookLevel, px, sz string) {
		for i, l := range levels {
			if i >= MaxBookLevels {
				break
			}
			ll := make(map[string]string, len(labels)+1)
			for k, v := range labels {
				ll[k] = v
			}
			ll["lvl"] = strconv.Itoa(i)
			pts = append(pts,
				Point{Metric: px, Labels: ll, Value: l.Price, TsMs: ms},
				Point{Metric: sz, Labels: ll, Value: l.Size, TsMs: ms})
		}
	}
	side(bk.Bids, MetricBookBidPx, MetricBookBidSz)
	side(bk.Asks, MetricBookAskPx, MetricBookAskSz)
	return pts
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
			w.mu.Lock()
			w.closeTradesLocked(true)
			w.mu.Unlock()
			w.flush(context.Background(), onErr)
			return
		case <-t.C:
		case <-w.wake:
		}
		w.flush(ctx, onErr)
	}
}

// Flush sends everything buffered now, open trade buckets included (used by tests and shutdown).
func (w *Writer) Flush(ctx context.Context) error {
	w.mu.Lock()
	w.closeTradesLocked(true)
	w.mu.Unlock()
	var ferr error
	w.flush(ctx, func(err error) { ferr = err })
	return ferr
}

func (w *Writer) flush(ctx context.Context, onErr func(error)) {
	w.mu.Lock()
	w.closeTradesLocked(false)
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
