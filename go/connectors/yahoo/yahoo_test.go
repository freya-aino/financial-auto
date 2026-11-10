package yahoo

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

type fixedClock struct{ t time.Time }

func (f fixedClock) Now() time.Time { return f.t }

func newTest(t *testing.T, now time.Time) (*Connector, *httptest.Server, *atomic.Int32) {
	t.Helper()
	fixture, err := os.ReadFile("testdata/chart_aapl_1m.json")
	if err != nil {
		t.Fatal(err)
	}
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if !strings.HasPrefix(r.URL.Path, "/v8/finance/chart/AAPL") || r.URL.Query().Get("interval") != "1m" {
			t.Errorf("unexpected request %s", r.URL)
		}
		w.Write(fixture)
	}))
	t.Cleanup(srv.Close)
	res := connector.NewStaticResolver()
	res.Add(ID, "AAPL", "01900000-0000-7000-8000-00000000aaaa")
	c := &Connector{BaseURL: srv.URL}
	src := catalog.Source{ID: ID, Config: &catalog.Params{Symbols: []string{"AAPL"}}}
	deps := connector.Deps{Resolver: res, Clock: fixedClock{now}, Cursors: connector.NewMemCursors()}
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, deps); err != nil {
		t.Fatal(err)
	}
	return c, srv, &hits
}

func TestPollNormalizeDedupAndCursor(t *testing.T) {
	// Fixture bars open at 1794234600..1794234840 (5 × 1m). "Now" is just
	// after the 4th bar closed, so the 5th is still forming.
	now := time.Unix(1794234850, 0).UTC()
	c, _, _ := newTest(t, now)
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 1 {
		t.Fatalf("poll: %v %d", err, len(raws))
	}
	if raws[0].TransportMeta[connector.MetaSymbol] != "AAPL" || raws[0].ContentType != "application/json" {
		t.Fatalf("raw meta: %+v", raws[0].TransportMeta)
	}
	envs, unres, err := c.Normalize(raws[0])
	if err != nil || len(unres) != 0 {
		t.Fatalf("normalize: %v %v", err, unres)
	}
	if len(envs) != 8 { // 4 completed bars × (bar + synthetic trade)
		t.Fatalf("want 8 envelopes, got %d", len(envs))
	}
	bar, trade := envs[0], envs[1]
	if b := bar.GetBar(); b == nil || b.Timeframe != "1m" || b.Open != 344.93499755859375 || b.Volume != 1599598 {
		t.Fatalf("bar: %v", bar)
	}
	if tr := trade.GetTrade(); tr == nil || tr.Price != bar.GetBar().Close || tr.Size != 1599598 || tr.VenueTradeId != "bar:1794234600" {
		t.Fatalf("trade: %v", trade)
	}
	if bar.EventTime.AsTime() != time.Unix(1794234660, 0).UTC() || bar.InstrumentId != "01900000-0000-7000-8000-00000000aaaa" || bar.SourceId != ID {
		t.Fatalf("envelope: %v", bar)
	}
	if bar.PublishTime.AsTime() != now || bar.IngestTime.AsTime() != now {
		t.Fatal("publish/ingest must be the receive time")
	}
	// Same payload again: nothing new.
	envs, _, _ = c.Normalize(raws[0])
	if len(envs) != 0 {
		t.Fatalf("dedup failed: %d", len(envs))
	}
	// Later poll once the 5th bar closed emits exactly it.
	raws[0].ReceivedAt = now.Add(time.Minute)
	envs, _, _ = c.Normalize(raws[0])
	if len(envs) != 2 || envs[0].GetBar().Close != 347.9075012207031 {
		t.Fatalf("5th bar: %v", envs)
	}
	cur, ok, _ := c.deps.Cursors.GetCursor(context.Background(), ID, "01900000-0000-7000-8000-00000000aaaa", "1m")
	if !ok || cur != time.Unix(1794234900, 0).UTC() {
		t.Fatalf("cursor %v %v", cur, ok)
	}
	// A fresh connector resumes from the persisted cursor.
	c2 := &Connector{BaseURL: c.BaseURL}
	if err := c2.Init(context.Background(), c.cfg, c.deps); err != nil {
		t.Fatal(err)
	}
	raws[0].ReceivedAt = now.Add(time.Hour)
	if envs, _, _ := c2.Normalize(raws[0]); len(envs) != 0 {
		t.Fatalf("resume from cursor failed: %d", len(envs))
	}
}

func TestUnresolvedSymbolAndErrors(t *testing.T) {
	c, _, _ := newTest(t, time.Unix(1794234850, 0).UTC())
	raws, _ := c.Poll(context.Background(), connector.PollRequest{})
	raws[0].TransportMeta[connector.MetaSymbol] = "ZZZZ"
	envs, unres, err := c.Normalize(raws[0])
	if err != nil || len(envs) != 0 || len(unres) != 1 || unres[0].VenueSymbol != "ZZZZ" {
		t.Fatalf("%v %v %v", envs, unres, err)
	}
	raw := raws[0]
	raw.Bytes = []byte(`{"chart":{"result":null,"error":{"code":"Not Found","description":"No data found"}}}`)
	if _, _, err := c.Normalize(raw); err == nil || !strings.Contains(err.Error(), "No data found") {
		t.Fatalf("want yahoo error, got %v", err)
	}
	if err := (&Connector{}).Init(context.Background(), connector.SourceConfig{Source: catalog.Source{Config: &catalog.Params{Timeframe: "1m"}}}, connector.Deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestBackfillChunksAndEmitsAll(t *testing.T) {
	now := time.Unix(1794234850, 0).UTC()
	c, _, hits := newTest(t, now)
	out := make(chan connector.RawMessage, 16)
	// 15 days of 1m bars → 3 requests of ≤ 7 days.
	req := connector.BackfillRequest{VenueSymbol: "AAPL", Timeframe: "1m", From: now.Add(-15 * 24 * time.Hour), To: now}
	if err := c.Backfill(context.Background(), req, out); err != nil {
		t.Fatal(err)
	}
	close(out)
	n := 0
	for raw := range out {
		n++
		if raw.TransportMeta[connector.MetaBackfill] != "1" {
			t.Fatal("backfill meta missing")
		}
		envs, _, err := c.Normalize(raw)
		if err != nil || len(envs) != 10 { // all 5 bars, no live de-dup
			t.Fatalf("backfill normalize: %v %d", err, len(envs))
		}
	}
	if n != 3 || hits.Load() != 3 {
		t.Fatalf("chunks=%d hits=%d", n, hits.Load())
	}
	if err := c.Backfill(context.Background(), connector.BackfillRequest{Timeframe: "3m"}, out); err == nil {
		t.Fatal("bad timeframe must error")
	}
}
