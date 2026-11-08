package tsdb

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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
	if len(pts) != 2 || pts[0].Metric != MetricTradePrice || pts[1].Metric != MetricTradeSize {
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
	if len(lines) != 2 {
		t.Fatalf("want 2 lines, got %d: %s", len(lines), b)
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
