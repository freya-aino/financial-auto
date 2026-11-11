package tsdb

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

func TestTradePointsAndEncode(t *testing.T) {
	at := time.Date(2026, 1, 2, 3, 4, 5, 600_000_000, time.UTC)
	e := &fav1.Envelope{
		SourceId: "heartbeat", InstrumentId: "01900000-0000-7000-8000-000000000001",
		EventTime: timestamppb.New(at),
		Payload:   &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: 101.5, Size: 3}},
	}
	pts := TradePoints(e, "sim")
	if len(pts) != 3 || pts[0].Metric != MetricTradePrice || pts[1].Metric != MetricTradeSize || pts[2].Metric != MetricTradeCount || pts[2].Value != 1 {
		t.Fatalf("points: %+v", pts)
	}
	if pts[0].TsMs != at.UnixMilli() || pts[0].Labels["inst"] != "01900000000070008000000000000001" || pts[0].Labels["venue"] != "sim" {
		t.Fatalf("point: %+v", pts[0])
	}
	if TradePoints(&fav1.Envelope{Payload: &fav1.Envelope_News{News: &fav1.NewsItem{}}}, "x") != nil {
		t.Fatal("news must not produce points")
	}
	// Two samples of the same series collapse into one line.
	pts = append(pts, Point{Metric: MetricTradePrice, Labels: pts[0].Labels, Value: 102, TsMs: pts[0].TsMs + 1})
	b, err := Encode(pts)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 3 {
		t.Fatalf("want 3 lines, got %d: %s", len(lines), b)
	}
	var l line
	if err := json.Unmarshal([]byte(lines[0]), &l); err != nil {
		t.Fatal(err)
	}
	if l.Metric["__name__"] != MetricTradePrice || len(l.Values) != 2 || l.Values[1] != 102 {
		t.Fatalf("line: %+v", l)
	}
}

func TestWriterFlushesAndCounts(t *testing.T) {
	var got atomic.Int32
	var fail atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" { // series-budget poll (budget.go)
			return
		}
		if r.URL.Path != "/api/v1/import" {
			t.Errorf("path %s", r.URL.Path)
		}
		if fail.Load() {
			http.Error(w, "boom", 500)
			return
		}
		b, _ := io.ReadAll(r.Body)
		got.Add(int32(strings.Count(string(b), "\n")))
		w.WriteHeader(204)
	}))
	defer srv.Close()

	w := New(srv.URL, Options{BatchSize: 2, FlushInterval: time.Hour})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); w.Run(ctx, nil) }()

	w.Add(Point{Metric: "a", Value: 1}, Point{Metric: "b", Value: 2})
	deadline := time.Now().Add(2 * time.Second)
	for w.Stats().Written < 2 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if s := w.Stats(); s.Written != 2 || got.Load() != 2 {
		t.Fatalf("stats %+v, lines %d", s, got.Load())
	}
	fail.Store(true)
	w.Add(Point{Metric: "c", Value: 3})
	if err := w.Flush(ctx); err == nil {
		t.Fatal("want error from 500")
	}
	if s := w.Stats(); s.Errors != 1 || s.Written != 2 {
		t.Fatalf("stats after error %+v", s)
	}
	cancel()
	<-done
}

func TestQuoteBarBookPoints(t *testing.T) {
	at := time.Date(2026, 1, 2, 3, 4, 0, 0, time.UTC)
	inst := "01900000-0000-7000-8000-000000000002"
	vwap := 10.5
	cases := []struct {
		name    string
		env     *fav1.Envelope
		metrics []string
		labels  map[string]string
	}{
		{"quote", &fav1.Envelope{SourceId: "lemon", InstrumentId: inst, EventTime: timestamppb.New(at),
			Payload: &fav1.Envelope_Quote{Quote: &fav1.Quote{BidPrice: 9, BidSize: 1, AskPrice: 11, AskSize: 2}}},
			[]string{MetricQuoteBidPx, MetricQuoteBidSz, MetricQuoteAskPx, MetricQuoteAskSz}, map[string]string{"venue": "lemon"}},
		{"bar", &fav1.Envelope{SourceId: "yahoo", InstrumentId: inst, EventTime: timestamppb.New(at),
			Payload: &fav1.Envelope_Bar{Bar: &fav1.Bar{Timeframe: "1m", Open: 1, High: 2, Low: 0.5, Close: 1.5, Volume: 100, Vwap: &vwap}}},
			[]string{MetricBarOpen, MetricBarHigh, MetricBarLow, MetricBarClose, MetricBarVolume, MetricBarVwap}, map[string]string{"tf": "1m"}},
		{"book", &fav1.Envelope{SourceId: "x", InstrumentId: inst, EventTime: timestamppb.New(at),
			Payload: &fav1.Envelope_Book{Book: &fav1.BookSnapshot{
				Bids: []*fav1.BookLevel{{Price: 9, Size: 1}, {Price: 8, Size: 2}},
				Asks: []*fav1.BookLevel{{Price: 11, Size: 3}}}}},
			[]string{MetricBookBidPx, MetricBookBidSz, MetricBookBidPx, MetricBookBidSz, MetricBookAskPx, MetricBookAskSz}, map[string]string{"lvl": "0"}},
	}
	for _, c := range cases {
		pts := Points(c.env, "venue-token")
		if len(pts) != len(c.metrics) {
			t.Fatalf("%s: got %d points, want %d: %+v", c.name, len(pts), len(c.metrics), pts)
		}
		for i, p := range pts {
			if p.Metric != c.metrics[i] {
				t.Errorf("%s[%d]: metric %s, want %s", c.name, i, p.Metric, c.metrics[i])
			}
			if p.TsMs != at.UnixMilli() || p.Labels["inst"] != "01900000000070008000000000000002" {
				t.Errorf("%s[%d]: %+v", c.name, i, p)
			}
		}
		for k, v := range c.labels {
			if k == "venue" {
				v = "venue-token"
			}
			if pts[0].Labels[k] != v {
				t.Errorf("%s: label %s = %q, want %q", c.name, k, pts[0].Labels[k], v)
			}
		}
	}
	// bar without vwap writes five metrics; book level 1 carries lvl=1
	bar := cases[1].env
	bar.GetBar().Vwap = nil
	if n := len(BarPoints(bar, "v")); n != 5 {
		t.Fatalf("bar without vwap: %d points", n)
	}
	if bk := BookPoints(cases[2].env, "v"); bk[2].Labels["lvl"] != "1" || bk[4].Labels["lvl"] != "0" {
		t.Fatalf("book levels: %+v", bk)
	}
	// Different lvl values are distinct series when encoded.
	b, err := Encode(BookPoints(cases[2].env, "v"))
	if err != nil {
		t.Fatal(err)
	}
	if n := len(strings.Split(strings.TrimSpace(string(b)), "\n")); n != 6 {
		t.Fatalf("book encode: %d lines, want 6", n)
	}
}

// Same-millisecond trades on one series become a single VWAP sample (D-038);
// different series, newer milliseconds and late trades behave as documented.
func TestTradeVWAPAggregation(t *testing.T) {
	type sample struct {
		v  float64
		ts int64
	}
	var mu sync.Mutex
	got := map[string][]sample{} // "<metric>|<src>" → samples
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" { // series-budget poll (budget.go)
			return
		}
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		defer mu.Unlock()
		for _, ln := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			var l line
			if err := json.Unmarshal([]byte(ln), &l); err != nil {
				t.Errorf("bad line %q: %v", ln, err)
				continue
			}
			k := l.Metric["__name__"] + "|" + l.Metric["src"]
			for i := range l.Values {
				got[k] = append(got[k], sample{l.Values[i], l.Timestamps[i]})
			}
		}
		w.WriteHeader(204)
	}))
	defer srv.Close()

	clock := time.Unix(1_700_000_000, 0)
	w := New(srv.URL, Options{FlushInterval: time.Hour, TradeHold: time.Second, Now: func() time.Time { return clock }})
	inst := "01900000-0000-7000-8000-000000000003"
	trade := func(src string, ms int64, px, sz float64) {
		w.AddEnvelope(&fav1.Envelope{SourceId: src, InstrumentId: inst, EventTime: timestamppb.New(time.UnixMilli(ms)),
			Payload: &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: px, Size: sz}}}, "v")
	}
	trade("a", 1000, 10, 1)
	trade("a", 1000, 13, 2) // same ms → VWAP (10·1+13·2)/3 = 12, size 3
	trade("a", 1000, 99, 0) // zero size: no weight, but counted
	trade("b", 1000, 50, 1) // other series, same ms: independent
	trade("a", 1001, 20, 1) // newer ms closes 1000 at once
	trade("a", 1000, 7, 1)  // late: older than the newest emitted ms → written unaggregated

	// Nothing held is flushed before TradeHold passes; the closed 1000 bucket is.
	w.flush(context.Background(), nil)
	mu.Lock()
	if s := got["fa_trade_price|a"]; len(s) != 2 || s[0] != (sample{12, 1000}) || s[1] != (sample{7, 1000}) {
		t.Fatalf("price a after first flush: %+v", s)
	}
	if s := got["fa_trade_size|a"]; len(s) != 2 || s[0] != (sample{3, 1000}) {
		t.Fatalf("size a: %+v", s)
	}
	if len(got["fa_trade_price|b"]) != 0 {
		t.Fatalf("b flushed before hold: %+v", got["fa_trade_price|b"])
	}
	mu.Unlock()

	clock = clock.Add(time.Second)
	w.flush(context.Background(), nil)
	mu.Lock()
	if s := got["fa_trade_price|a"]; len(s) != 3 || s[2] != (sample{20, 1001}) {
		t.Fatalf("price a after hold: %+v", s)
	}
	if s := got["fa_trade_price|b"]; len(s) != 1 || s[0] != (sample{50, 1000}) {
		t.Fatalf("price b: %+v", s)
	}
	mu.Unlock()
	if s := got["fa_trade_count|a"]; len(s) != 3 || s[0] != (sample{3, 1000}) || s[1] != (sample{1, 1000}) {
		t.Fatalf("count a: %+v", s)
	}
	if st := w.Stats(); st.Aggregated != 2 || st.Late != 1 || st.Written != 12 {
		t.Fatalf("stats %+v", st)
	}

	// All-zero-size millisecond keeps the last price; Flush closes open buckets.
	trade("c", 2000, 5, 0)
	trade("c", 2000, 6, 0)
	if err := w.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if s := got["fa_trade_price|c"]; len(s) != 1 || s[0] != (sample{6, 2000}) {
		t.Fatalf("price c: %+v", s)
	}
}
