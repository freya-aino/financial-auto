package hackernews

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/connector"
)

type fixedClock struct{ t time.Time }

func (f fixedClock) Now() time.Time { return f.t }

func serve(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	top, _ := os.ReadFile("testdata/topstories.json")
	item, _ := os.ReadFile("testdata/item_8863.json")
	alg, _ := os.ReadFile("testdata/algolia_search_by_date.json")
	var items atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "stories.json"):
			w.Write(top)
		case strings.HasPrefix(r.URL.Path, "/item/"):
			items.Add(1)
			if strings.Contains(r.URL.Path, "50431058") {
				w.Write([]byte("null"))
				return
			}
			// All fixture ids answer with the same recorded story (id 8863) except
			// the time, so de-dup is exercised on the real id.
			w.Write(item)
		case strings.HasPrefix(r.URL.Path, "/search_by_date"):
			if r.URL.Query().Get("page") != "0" {
				w.Write([]byte(`{"hits":[],"nbPages":0}`))
				return
			}
			w.Write(alg)
		default:
			t.Errorf("unexpected %s", r.URL)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, &items
}

func newTest(t *testing.T, cur connector.CursorStore) (*Connector, *atomic.Int32) {
	srv, items := serve(t)
	c := &Connector{BaseURL: srv.URL, AlgoliaURL: srv.URL, Lists: []string{"topstories"}}
	deps := connector.Deps{Clock: fixedClock{time.Unix(1_800_000_000, 0).UTC()}, Cursors: cur}
	if err := c.Init(context.Background(), connector.SourceConfig{}, deps); err != nil {
		t.Fatal(err)
	}
	return c, items
}

func TestPollNormalizeDedup(t *testing.T) {
	cur := connector.NewMemCursors()
	c, items := newTest(t, cur)
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(raws) != 4 || items.Load() != 5 { // 5 ids in the fixture list, one is null
		t.Fatalf("raws=%d items=%d", len(raws), items.Load())
	}
	var total int
	for _, raw := range raws {
		envs, unres, err := c.Normalize(raw)
		if err != nil || len(unres) != 0 {
			t.Fatal(err)
		}
		total += len(envs)
	}
	if total != 1 { // every item payload carries id 8863 → one NewsItem
		t.Fatalf("want 1 news item after de-dup, got %d", total)
	}
	envs, _, _ := c.Normalize(raws[0])
	if len(envs) != 0 {
		t.Fatal("re-normalizing a seen item must emit nothing")
	}
	// Second poll: the list ids are now all seen → no item fetches.
	before := items.Load()
	raws, _ = c.Poll(context.Background(), connector.PollRequest{})
	if len(raws) != 0 || items.Load() != before {
		t.Fatalf("second poll fetched %d items", items.Load()-before)
	}
	at, ok, _ := cur.GetCursor(context.Background(), ID, "", cursorScope)
	if !ok || at != time.Unix(1175714200, 0).UTC() {
		t.Fatalf("cursor %v %v", at, ok)
	}
	// Fresh connector with the persisted cursor skips the (old) story.
	c2, _ := newTest(t, cur)
	raws, _ = c2.Poll(context.Background(), connector.PollRequest{})
	for _, raw := range raws {
		if envs, _, _ := c2.Normalize(raw); len(envs) != 0 {
			t.Fatal("story at cursor time must be skipped after restart")
		}
	}
}

func TestNormalizeItemFields(t *testing.T) {
	c, _ := newTest(t, nil)
	item, _ := os.ReadFile("testdata/item_8863.json")
	now := time.Unix(1_800_000_000, 0).UTC()
	envs, _, err := c.Normalize(connector.RawMessage{SourceID: ID, ReceivedAt: now, Bytes: item, TransportMeta: map[string]string{connector.MetaEndpoint: "item"}})
	if err != nil || len(envs) != 1 {
		t.Fatalf("%v %v", envs, err)
	}
	n := envs[0].GetNews()
	if n.Title != "My YC app: Dropbox - Throw away your USB drive" || n.Url != "http://www.getdropbox.com/u/2/screencast.html" {
		t.Fatalf("news: %v", n)
	}
	if n.DedupKey != "hn:8863" || n.Outlet != Outlet || n.Language != "en" || n.EntitiesRaw[0] != "author:dhouston" {
		t.Fatalf("news: %v", n)
	}
	e := envs[0]
	if e.EventTime.AsTime() != time.Unix(1175714200, 0).UTC() || e.PublishTime.AsTime() != e.EventTime.AsTime() || e.IngestTime.AsTime() != now || e.InstrumentId != "" {
		t.Fatalf("envelope: %v", e)
	}
	// Within one run an older story than the newest published one is still new.
	older := []byte(`{"id":4,"type":"story","time":1175714100,"title":"older"}`)
	if envs, _, _ := c.Normalize(connector.RawMessage{Bytes: older, TransportMeta: map[string]string{}}); len(envs) != 1 {
		t.Fatal("older story in the same run must be emitted")
	}
	// Comments and dead items are dropped; Ask HN text is cleaned.
	ask := []byte(`{"id":1,"type":"story","time":1900000000,"title":"Ask HN: x &amp; y","text":"a<p>b &quot;c&quot;"}`)
	envs, _, _ = c.Normalize(connector.RawMessage{Bytes: ask, TransportMeta: map[string]string{}})
	if len(envs) != 1 || envs[0].GetNews().Body != "a\nb \"c\"" || envs[0].GetNews().Title != "Ask HN: x & y" || envs[0].GetNews().Url != "https://news.ycombinator.com/item?id=1" {
		t.Fatalf("ask: %v", envs)
	}
	for _, bad := range []string{`{"id":2,"type":"comment","time":1900000001}`, `{"id":3,"type":"story","time":1900000001,"dead":true}`} {
		if envs, _, _ := c.Normalize(connector.RawMessage{Bytes: []byte(bad), TransportMeta: map[string]string{}}); len(envs) != 0 {
			t.Fatalf("%s must be dropped", bad)
		}
	}
}

func TestBackfillAlgolia(t *testing.T) {
	c, _ := newTest(t, nil)
	out := make(chan connector.RawMessage, 4)
	err := c.Backfill(context.Background(), connector.BackfillRequest{From: time.Unix(1762000000, 0), To: time.Unix(1762003600, 0)}, out)
	if err != nil {
		t.Fatal(err)
	}
	close(out)
	var envs []string
	for raw := range out {
		es, _, err := c.Normalize(raw)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range es {
			envs = append(envs, e.GetNews().DedupKey)
		}
	}
	if len(envs) != 2 || envs[0] != "hn:45781462" || envs[1] != "hn:45781458" {
		t.Fatalf("backfill items: %v", envs)
	}
}
