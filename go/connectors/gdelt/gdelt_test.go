package gdelt

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

type fixedClock struct{ t time.Time }

func (f fixedClock) Now() time.Time { return f.t }

var now = time.Date(2026, 11, 11, 2, 0, 0, 0, time.UTC)

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func newTest(t *testing.T, cur connector.CursorStore, h http.HandlerFunc) *Connector {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	c := &Connector{BaseURL: srv.URL}
	c.Fetch = connector.NewFetcher(nil)
	c.Fetch.Retries = 1 // keep 429 tests fast
	c.Fetch.Sleep = func(context.Context, time.Duration) error { return nil }
	src := catalog.Source{ID: ID, Config: &catalog.Params{Queries: []string{"stock market"}}}
	deps := connector.Deps{Clock: fixedClock{now}, Cursors: cur}
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, deps); err != nil {
		t.Fatal(err)
	}
	return c
}

func serve(t *testing.T, body []byte, status int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		if q.Get("query") != "stock market" || q.Get("mode") != "artlist" || q.Get("format") != "json" {
			t.Errorf("query %s", r.URL.RawQuery)
		}
		w.WriteHeader(status)
		w.Write(body)
	}
}

func TestPollNormalizeDedupCursor(t *testing.T) {
	cur := connector.NewMemCursors()
	c := newTest(t, cur, serve(t, fixture(t, "artlist_stock_market.json"), 200))
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 1 {
		t.Fatalf("poll: %v %d", err, len(raws))
	}
	if raws[0].TransportMeta["query"] != "stock market" || !strings.Contains(raws[0].TransportMeta[connector.MetaURL], "timespan=1h") {
		t.Fatalf("meta: %+v", raws[0].TransportMeta)
	}
	envs, _, err := c.Normalize(raws[0])
	if err != nil || len(envs) != 3 {
		t.Fatalf("normalize: %v %d", err, len(envs))
	}
	e := envs[0]
	n := e.GetNews()
	if n.Title != "Taiwan, China tap market rescue funds. Why wont Korea?" || n.Outlet != "koreatimes.co.kr" || n.Language != "en" {
		t.Fatalf("news: %v", n)
	}
	if len(n.EntitiesRaw) != 2 || n.EntitiesRaw[0] != "query:stock market" || n.EntitiesRaw[1] != "country:South Korea" {
		t.Fatalf("entities: %v", n.EntitiesRaw)
	}
	if n.DedupKey != DedupKey("https://www.koreatimes.co.kr/economy/20261111/taiwan-china-tap-market-rescue-funds-why-wont-korea") || !strings.HasPrefix(n.DedupKey, "url:") || len(n.DedupKey) != 44 {
		t.Fatalf("dedup %q", n.DedupKey)
	}
	want := time.Date(2026, 11, 11, 1, 30, 0, 0, time.UTC)
	if e.EventTime.AsTime() != want || e.PublishTime.AsTime() != want || e.IngestTime.AsTime() != now || e.SourceId != ID || e.InstrumentId != "" {
		t.Fatalf("envelope times: %v", e)
	}
	// Re-normalizing the same answer emits nothing; cursor = newest seendate.
	if again, _, _ := c.Normalize(raws[0]); len(again) != 0 {
		t.Fatalf("dedup: %d", len(again))
	}
	if got, ok, _ := cur.GetCursor(context.Background(), ID, "", cursorScope); !ok || got != want {
		t.Fatalf("cursor %v %v", got, ok)
	}
	// Restart: everything at or before the persisted cursor is skipped.
	c2 := newTest(t, cur, serve(t, fixture(t, "artlist_stock_market.json"), 200))
	if envs, _, _ := c2.Normalize(raws[0]); len(envs) != 0 {
		t.Fatalf("after restart: %d", len(envs))
	}
}

func TestDeterministicPayload(t *testing.T) {
	raw := connector.RawMessage{SourceID: ID, ReceivedAt: now, Bytes: fixture(t, "artlist_stock_market.json"),
		TransportMeta: map[string]string{"query": "stock market", connector.MetaBackfill: "1"}}
	a, _, _ := newTest(t, nil, serve(t, nil, 200)).Normalize(raw)
	b, _, _ := newTest(t, nil, serve(t, nil, 200)).Normalize(raw)
	if len(a) != 3 || len(a) != len(b) {
		t.Fatalf("%d %d", len(a), len(b))
	}
	for i := range a {
		a[i].EventId, b[i].EventId = "", ""
		if !proto.Equal(a[i], b[i]) {
			t.Fatalf("envelope %d differs", i)
		}
	}
}

func TestErrorBodies(t *testing.T) {
	// 200 + plain text (bad query) and 429 + plain text (rate limit) are errors.
	for _, tc := range []struct {
		file   string
		status int
	}{{"bad_query.txt", 200}, {"ratelimit.txt", 429}} {
		c := newTest(t, nil, serve(t, fixture(t, tc.file), tc.status))
		if raws, err := c.Poll(context.Background(), connector.PollRequest{}); err == nil || len(raws) != 0 {
			t.Fatalf("%s: want error, got %v %d", tc.file, err, len(raws))
		}
		if _, _, err := c.Normalize(connector.RawMessage{Bytes: fixture(t, tc.file)}); err == nil {
			t.Fatalf("%s: normalize must fail", tc.file)
		}
	}
	// {} is an empty, valid answer.
	c := newTest(t, nil, serve(t, fixture(t, "empty.json"), 200))
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 1 {
		t.Fatalf("empty poll: %v", err)
	}
	if envs, _, err := c.Normalize(raws[0]); err != nil || len(envs) != 0 {
		t.Fatalf("empty normalize: %v %d", err, len(envs))
	}
}

func TestBackfillWindows(t *testing.T) {
	var hits atomic.Int32
	var ranges []string
	body := fixture(t, "artlist_stock_market.json")
	c := newTest(t, nil, func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		q := r.URL.Query()
		if q.Get("timespan") != "" || q.Get("maxrecords") != "250" {
			t.Errorf("backfill query %s", r.URL.RawQuery)
		}
		ranges = append(ranges, q.Get("startdatetime")+"-"+q.Get("enddatetime"))
		w.Write(body)
	})
	from := time.Date(2026, 11, 9, 0, 0, 0, 0, time.UTC)
	to := time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)
	out := make(chan connector.RawMessage, 10)
	if err := c.Backfill(context.Background(), connector.BackfillRequest{From: from, To: to}, out); err != nil {
		t.Fatal(err)
	}
	close(out)
	if hits.Load() != 2 || ranges[0] != "20261109000000-20261110000000" || ranges[1] != "20261110000000-20261110120000" {
		t.Fatalf("ranges %v", ranges)
	}
	total := 0
	for raw := range out {
		if raw.TransportMeta[connector.MetaBackfill] != "1" {
			t.Fatal("backfill meta")
		}
		envs, _, err := c.Normalize(raw)
		if err != nil {
			t.Fatal(err)
		}
		total += len(envs)
	}
	if total != 6 { // backfill does not de-dup across windows
		t.Fatalf("total %d", total)
	}
	// Depth is clamped to 90 days.
	ranges = nil
	hits.Store(0)
	c.Window = 30 * 24 * time.Hour
	out2 := make(chan connector.RawMessage, 100)
	if err := c.Backfill(context.Background(), connector.BackfillRequest{From: to.AddDate(-2, 0, 0), To: to}, out2); err != nil {
		t.Fatal(err)
	}
	if hits.Load() != 3 || !strings.HasPrefix(ranges[0], to.Add(-MaxBackfillDepth).Format(rangeLayout)) {
		t.Fatalf("clamp: %d %v", hits.Load(), ranges)
	}
	// Errors end the backfill.
	bad := newTest(t, nil, serve(t, fixture(t, "bad_query.txt"), 200))
	if err := bad.Backfill(context.Background(), connector.BackfillRequest{From: from, To: to}, make(chan connector.RawMessage, 4)); err == nil {
		t.Fatal("want backfill error")
	}
}

func TestNormalizeURLAndLanguage(t *testing.T) {
	a := NormalizeURL("HTTPS://WWW.Example.com/a/b/?utm_source=x&id=1#frag")
	if a != "https://example.com/a/b?id=1" {
		t.Fatalf("%q", a)
	}
	if DedupKey("https://example.com/a/b?id=1&utm_medium=y") != DedupKey("https://www.example.com/a/b/?id=1") {
		t.Fatal("equivalent URLs must share a key")
	}
	if Language("English") != "en" || Language("Klingon") != "klingon" {
		t.Fatal("language map")
	}
}
