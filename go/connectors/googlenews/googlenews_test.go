package googlenews

import (
	"context"
	"github.com/fey/financial-auto/go/connectors/rss"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

type fixedClock struct{ t time.Time }

func (f fixedClock) Now() time.Time { return f.t }

func newTest(t *testing.T, cur connector.CursorStore) (*Connector, *int) {
	t.Helper()
	fixture, _ := os.ReadFile("testdata/search_apple_stock.xml")
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		q := r.URL.Query()
		if q.Get("q") != "Apple stock" || q.Get("hl") != "en-US" || q.Get("gl") != "US" || q.Get("ceid") != "US:en" {
			t.Errorf("query %s", r.URL.RawQuery)
		}
		w.Write(fixture)
	}))
	t.Cleanup(srv.Close)
	c := &Connector{BaseURL: srv.URL}
	src := catalog.Source{ID: ID, Config: &catalog.Params{Queries: []string{"Apple stock"}}}
	deps := connector.Deps{Clock: fixedClock{time.Unix(1_800_000_000, 0).UTC()}, Cursors: cur}
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, deps); err != nil {
		t.Fatal(err)
	}
	return c, &hits
}

func TestPollNormalizeDedupCursor(t *testing.T) {
	cur := connector.NewMemCursors()
	c, hits := newTest(t, cur)
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 1 || *hits != 1 {
		t.Fatalf("poll: %v %d %d", err, len(raws), *hits)
	}
	if raws[0].ContentType != "application/rss+xml" || raws[0].TransportMeta["query"] != "Apple stock" {
		t.Fatalf("raw: %+v", raws[0].TransportMeta)
	}
	envs, unres, err := c.Normalize(raws[0])
	if err != nil || len(unres) != 0 || len(envs) != 3 {
		t.Fatalf("normalize: %v %v %d", err, unres, len(envs))
	}
	n := envs[0].GetNews()
	if n.Title != "Where Will Apple Stock Be in 5 Years?" || n.Outlet != "Yahoo Finance" || n.Language != "en" {
		t.Fatalf("news: %v", n)
	}
	if n.DedupKey == "" || n.DedupKey != "CBMilAFBVV95cUxObEJWcVExS0JUOTB4RW8zdzRSNDh5dThnbDM2X2JfTlpBTENiRmJQdjhfczUwUEVVbWp4elAtQkJzTmo4Rnd3UmNxT0FmZjF3QUkwRkNfZldWY3h2cm5fMjBqMVpua3JJbkhNem0yU21DNnd4cGRmSzE0SnVZRURqZEM2VDgxN1otcEh6am5DVnk2djRy" {
		t.Fatalf("dedup key %q", n.DedupKey)
	}
	if n.Url == "" || n.EntitiesRaw[0] != "query:Apple stock" {
		t.Fatalf("news: %v", n)
	}
	want := time.Date(2026, 11, 9, 17, 10, 0, 0, time.UTC)
	if envs[0].EventTime.AsTime() != want || envs[0].PublishTime.AsTime() != want || envs[0].IngestTime.AsTime() != time.Unix(1_800_000_000, 0).UTC() {
		t.Fatalf("times: %v", envs[0])
	}
	if envs, _, _ := c.Normalize(raws[0]); len(envs) != 0 {
		t.Fatal("same feed again must emit nothing")
	}
	at, ok, _ := cur.GetCursor(context.Background(), ID, "", cursorScope)
	if !ok || at != want {
		t.Fatalf("cursor %v %v (newest pubDate)", at, ok)
	}
	// Restart with the cursor: all three items are at or before it.
	c2, _ := newTest(t, cur)
	if envs, _, _ := c2.Normalize(raws[0]); len(envs) != 0 {
		t.Fatalf("restart emitted %d", len(envs))
	}
}

func TestHelpers(t *testing.T) {
	if title, outlet := splitOutlet("A - B - Reuters", "Reuters"); title != "A - B" || outlet != "Reuters" {
		t.Fatal(title, outlet)
	}
	if title, outlet := splitOutlet("A - B", ""); title != "A" || outlet != "B" {
		t.Fatal(title, outlet)
	}
	if _, err := rss.ParseDate("garbage"); err == nil {
		t.Fatal("expected error")
	}
	c := &Connector{BaseURL: "https://x", HL: "de", GL: "DE", CEID: "DE:de"}
	if got := c.URL("a b"); got != "https://x?ceid=DE%3Ade&gl=DE&hl=de&q=a+b" {
		t.Fatal(got)
	}
	if _, _, err := c.Normalize(connector.RawMessage{Bytes: []byte("<rss><channel><item><title>x</title></item></channel>")}); err == nil {
		t.Fatal("malformed xml must error")
	}
}
