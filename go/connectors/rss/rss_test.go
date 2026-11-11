package rss

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

func TestParseRSSAndAtom(t *testing.T) {
	b, _ := os.ReadFile("testdata/feed.rss")
	f, err := Parse(b)
	if err != nil || f.Title != "Example Markets" || len(f.Items) != 3 {
		t.Fatalf("rss: %+v %v", f, err)
	}
	if f.Items[0].Key() != "a-1" || !f.Items[0].Published.Equal(time.Date(2026, 11, 10, 14, 0, 0, 0, time.UTC)) {
		t.Fatalf("item 0 %+v", f.Items[0])
	}
	if f.Items[1].Key() != "https://example.com/b" || f.Items[1].Published.Hour() != 15 {
		t.Fatalf("item 1 (dc:date) %+v", f.Items[1])
	}
	b, _ = os.ReadFile("testdata/feed.atom")
	a, err := Parse(b)
	if err != nil || a.Language != "de" || len(a.Items) != 2 || a.Items[0].Link != "https://example.org/1" || a.Items[0].Published.Hour() != 11 || a.Items[1].Published.Hour() != 13 {
		t.Fatalf("atom: %+v %v", a, err)
	}
	if _, err := Parse([]byte("<html/>")); err == nil {
		t.Fatal("html must be rejected")
	}
}

func TestAdapterPollNormalizeDedupCursor(t *testing.T) {
	rssB, _ := os.ReadFile("testdata/feed.rss")
	atomB, _ := os.ReadFile("testdata/feed.atom")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/atom" {
			w.Write(atomB)
			return
		}
		w.Write(rssB)
	}))
	defer srv.Close()
	c, err := connector.NewFor("outlet_feeds", Adapter)
	if err != nil || c.ID() != "outlet_feeds" {
		t.Fatalf("adapter: %v %v", c, err)
	}
	src := catalog.Source{ID: "outlet_feeds", Adapter: Adapter, Config: &catalog.Params{Extra: map[string]string{"feeds": srv.URL + "/rss, " + srv.URL + "/atom", "poll_interval": "1m"}}}
	cur := connector.NewMemCursors()
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{Cursors: cur}); err != nil {
		t.Fatal(err)
	}
	p := c.(connector.Poller)
	if p.Schedule().Interval != time.Minute {
		t.Fatal("interval")
	}
	raws, err := p.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 2 {
		t.Fatalf("poll %d %v", len(raws), err)
	}
	n := c.(connector.Normalizer)
	e1, _, _ := n.Normalize(raws[0])
	e2, _, _ := n.Normalize(raws[1])
	if len(e1) != 2 || len(e2) != 2 {
		t.Fatalf("envelopes %d %d", len(e1), len(e2))
	}
	if g := e1[0].GetNews(); g.Outlet != "Example Markets" || g.Language != "en" || g.Title != "Rates hold & markets rally" || e1[0].SourceId != "outlet_feeds" {
		t.Fatalf("news %+v", g)
	}
	if again, _, _ := n.Normalize(raws[0]); len(again) != 0 {
		t.Fatal("re-poll must be de-duplicated")
	}
	// restart: the cursor suppresses what was already published
	c2, _ := connector.NewFor("outlet_feeds", Adapter)
	if err := c2.Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{Cursors: cur}); err != nil {
		t.Fatal(err)
	}
	if again, _, _ := c2.(connector.Normalizer).Normalize(raws[1]); len(again) != 0 {
		t.Fatalf("after restart %d", len(again))
	}
	bad := catalog.Source{ID: "x", Adapter: Adapter}
	if err := c2.Init(context.Background(), connector.SourceConfig{Source: bad}, connector.Deps{}); err == nil {
		t.Fatal("no feeds must fail Init")
	}
}
