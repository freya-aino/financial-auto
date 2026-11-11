package secedgar

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

type fixedClock struct{ t time.Time }

func (f fixedClock) Now() time.Time { return f.t }

// Fixture: Apple submissions trimmed to 5 recent filings accepted
// 2026-11-05T23:30:13Z .. 2026-11-09T22:00:55Z, plus one older page (2015).
var now = time.Date(2026, 11, 11, 2, 0, 0, 0, time.UTC)

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func newTest(t *testing.T, cur connector.CursorStore, extra map[string]string, clock time.Time) (*Connector, *atomic.Int32, *atomic.Int32) {
	t.Helper()
	sub, page := fixture(t, "submissions_aapl.json"), fixture(t, "submissions_aapl_001.json")
	var subs, pages atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if ua := r.Header.Get("User-Agent"); ua != "financial-auto ingestor" && ua != extra["user_agent"] {
			w.WriteHeader(http.StatusForbidden)
			w.Write([]byte("<html>Undeclared Automated Tool</html>"))
			return
		}
		switch r.URL.Path {
		case "/submissions/CIK0000320193.json":
			subs.Add(1)
			w.Write(sub)
		case "/submissions/CIK0000320193-submissions-001.json":
			pages.Add(1)
			w.Write(page)
		default:
			t.Errorf("unexpected %s", r.URL)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)
	if extra == nil {
		extra = map[string]string{}
	}
	if _, ok := extra["ciks"]; !ok {
		extra["ciks"] = "320193"
	}
	c := &Connector{BaseURL: srv.URL}
	src := catalog.Source{ID: ID, Config: &catalog.Params{Extra: extra}}
	deps := connector.Deps{Clock: fixedClock{clock}, Cursors: cur}
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, deps); err != nil {
		t.Fatal(err)
	}
	return c, &subs, &pages
}

func TestPollNormalizeDedupCursor(t *testing.T) {
	cur := connector.NewMemCursors()
	c, subs, _ := newTest(t, cur, nil, now)
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 1 || subs.Load() != 1 {
		t.Fatalf("poll: %v %d", err, len(raws))
	}
	envs, unres, err := c.Normalize(raws[0])
	if err != nil || len(unres) != 0 || len(envs) != 5 { // all 5 within the 7-day first-run lookback
		t.Fatalf("normalize: %v %v %d", err, unres, len(envs))
	}
	e := envs[1]
	f := e.GetFiling()
	if f.FormType != "8-K" || f.IssuerCik != "0000320193" || f.Accession != "0001140361-26-042798" || f.IssuerLei != "" {
		t.Fatalf("filing: %v", f)
	}
	if len(f.DocumentRefs) != 1 || f.DocumentRefs[0] != "https://www.sec.gov/Archives/edgar/data/320193/000114036126042798/ef20085077_8k.htm" {
		t.Fatalf("refs: %v", f.DocumentRefs)
	}
	acc := time.Date(2026, 11, 9, 21, 30, 40, 0, time.UTC)
	if e.EventTime.AsTime() != acc || e.PublishTime.AsTime() != acc || f.FilingTime.AsTime() != acc || e.IngestTime.AsTime() != now || e.InstrumentId != "" {
		t.Fatalf("times: %v", e)
	}
	if again, _, _ := c.Normalize(raws[0]); len(again) != 0 {
		t.Fatalf("dedup: %d", len(again))
	}
	newest := time.Date(2026, 11, 9, 22, 0, 55, 0, time.UTC)
	if got, ok, _ := cur.GetCursor(context.Background(), ID, "", "cik:0000320193"); !ok || got != newest {
		t.Fatalf("cursor %v %v", got, ok)
	}
	// Restart with the cursor: nothing is re-emitted.
	c2, _, _ := newTest(t, cur, nil, now)
	if envs, _, _ := c2.Normalize(raws[0]); len(envs) != 0 {
		t.Fatalf("after restart: %d", len(envs))
	}
}

func TestFirstRunLookback(t *testing.T) {
	// 2 days lookback from 2026-11-11T02:00Z keeps filings accepted after 2026-11-09T02:00Z.
	c, _, _ := newTest(t, nil, map[string]string{"lookback": "48h"}, now)
	raws, _ := c.Poll(context.Background(), connector.PollRequest{})
	envs, _, _ := c.Normalize(raws[0])
	if len(envs) != 2 {
		t.Fatalf("lookback: %d", len(envs))
	}
}

func TestUserAgent(t *testing.T) {
	c, _, _ := newTest(t, nil, map[string]string{"user_agent": "acme research bot"}, now)
	if c.UserAgent != "acme research bot" {
		t.Fatal(c.UserAgent)
	}
	if _, err := c.Poll(context.Background(), connector.PollRequest{}); err != nil {
		t.Fatal(err)
	}
	// A UA the server refuses surfaces as an error (403 is not retried).
	bad, _, _ := newTest(t, nil, nil, now)
	bad.UserAgent = ""
	bad.Fetch.Prepare = func(r *http.Request) { r.Header.Set("User-Agent", "something else") }
	if _, err := bad.Poll(context.Background(), connector.PollRequest{}); !connector.IsHTTPStatus(err, http.StatusForbidden) {
		t.Fatalf("want 403, got %v", err)
	}
}

func TestDeterministicPayload(t *testing.T) {
	raw := connector.RawMessage{SourceID: ID, ReceivedAt: now, Bytes: fixture(t, "submissions_aapl.json"),
		TransportMeta: map[string]string{connector.MetaEndpoint: "submissions", metaCIK: "0000320193"}}
	a, _, _ := newTest(t, nil, nil, now)
	b, _, _ := newTest(t, nil, nil, now)
	ea, _, _ := a.Normalize(raw)
	eb, _, _ := b.Normalize(raw)
	if len(ea) != 5 || len(eb) != 5 {
		t.Fatalf("%d %d", len(ea), len(eb))
	}
	for i := range ea {
		ea[i].EventId, eb[i].EventId = "", ""
		if !proto.Equal(ea[i], eb[i]) {
			t.Fatalf("envelope %d differs", i)
		}
	}
}

func TestBackfillRange(t *testing.T) {
	c, subs, pages := newTest(t, nil, nil, now)
	run := func(from, to time.Time, sym string) int {
		out := make(chan connector.RawMessage, 8)
		if err := c.Backfill(context.Background(), connector.BackfillRequest{VenueSymbol: sym, From: from, To: to}, out); err != nil {
			t.Fatal(err)
		}
		close(out)
		n := 0
		for raw := range out {
			if raw.TransportMeta[connector.MetaBackfill] != "1" {
				t.Fatal("backfill meta")
			}
			envs, _, err := c.Normalize(raw)
			if err != nil {
				t.Fatal(err)
			}
			n += len(envs)
		}
		return n
	}
	// Recent range: [11-06, 11-09T22:00:00) → 3 filings, older page not fetched.
	if n := run(time.Date(2026, 11, 6, 0, 0, 0, 0, time.UTC), time.Date(2026, 11, 9, 22, 0, 0, 0, time.UTC), ""); n != 3 || pages.Load() != 0 {
		t.Fatalf("recent: %d pages=%d", n, pages.Load())
	}
	// 2015 range: the older page is fetched; its 3 filings fall on 2015-10-02.
	if n := run(time.Date(2015, 10, 1, 0, 0, 0, 0, time.UTC), time.Date(2015, 10, 3, 0, 0, 0, 0, time.UTC), "CIK0000320193"); n != 3 || pages.Load() != 1 {
		t.Fatalf("page: %d pages=%d", n, pages.Load())
	}
	if subs.Load() != 2 {
		t.Fatalf("subs %d", subs.Load())
	}
	// Backfill never touches the live cursor/seen set.
	raws, _ := c.Poll(context.Background(), connector.PollRequest{})
	if envs, _, _ := c.Normalize(raws[0]); len(envs) != 5 {
		t.Fatalf("live after backfill: %d", len(envs))
	}
}

func TestEstimatedTimesAndParse(t *testing.T) {
	evt, pub, ok := times("", "2026-11-09")
	if !ok || evt != time.Date(2026, 11, 9, 5, 0, 0, 0, time.UTC) || pub != time.Date(2026, 11, 10, 3, 0, 0, 0, time.UTC) {
		t.Fatalf("estimate %v %v", evt, pub)
	}
	for _, s := range []string{"320193", "0000320193", "CIK0000320193"} {
		if n, err := ParseCIK(s); err != nil || n != 320193 {
			t.Fatalf("%s: %v", s, err)
		}
	}
	if _, err := ParseCIK("AAPL"); err == nil {
		t.Fatal("ticker is not a CIK")
	}
	if _, _, err := (&Connector{}).Normalize(connector.RawMessage{Bytes: []byte("<html>")}); err == nil {
		t.Fatal("html must fail")
	}
}
