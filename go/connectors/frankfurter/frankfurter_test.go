package frankfurter

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

type fixedClock struct{ t time.Time }

func (f fixedClock) Now() time.Time { return f.t }

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func newTest(t *testing.T, now time.Time, cursors connector.CursorStore, handler http.HandlerFunc) *Connector {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	c := &Connector{BaseURL: srv.URL}
	src := catalog.Source{ID: ID, Config: &catalog.Params{Extra: map[string]string{"base": "EUR", "currencies": "USD, GBP"}}}
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{Clock: fixedClock{now}, Cursors: cursors}); err != nil {
		t.Fatal(err)
	}
	return c
}

func TestPollNormalizeDedupAndCursor(t *testing.T) {
	now := time.Date(2026, 11, 10, 17, 0, 0, 0, time.UTC)
	body := fixture(t, "range_eur_usd_gbp.json")
	var paths []string
	cursors := connector.NewMemCursors()
	c := newTest(t, now, cursors, func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path+"?"+r.URL.RawQuery)
		w.Write(body)
	})
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 1 {
		t.Fatalf("poll: %v %d", err, len(raws))
	}
	if paths[0] != "/2026-11-03..?base=EUR&symbols=USD%2CGBP" {
		t.Fatalf("request %q", paths[0])
	}
	envs, unres, err := c.Normalize(raws[0])
	if err != nil || len(unres) != 0 {
		t.Fatalf("normalize: %v %v", err, unres)
	}
	if len(envs) != 12 { // 6 business days × 2 currencies
		t.Fatalf("want 12 envelopes, got %d", len(envs))
	}
	first := envs[0]
	m := first.GetMacro()
	if m == nil || m.SeriesId != "EUR_GBP" || m.Period != "2026-11-03" || m.Value != 0.83358 || m.Revision {
		t.Fatalf("first: %v", first)
	}
	// 16:00 CET on 2026-11-03 = 15:00 UTC.
	if want := time.Date(2026, 11, 3, 15, 0, 0, 0, time.UTC); !m.Vintage.AsTime().Equal(want) || !first.EventTime.AsTime().Equal(want) {
		t.Fatalf("vintage %v", m.Vintage.AsTime())
	}
	if !first.PublishTime.AsTime().Equal(now) || !first.PublishTimeEstimated || first.InstrumentId != "" || first.SourceId != ID {
		t.Fatalf("envelope: %v", first)
	}
	if last := envs[11].GetMacro(); last.SeriesId != "EUR_USD" || last.Period != "2026-11-10" || last.Value != 1.1214 {
		t.Fatalf("last: %v", last)
	}
	// Same payload: nothing new.
	if again, _, _ := c.Normalize(raws[0]); len(again) != 0 {
		t.Fatalf("dedup failed: %d", len(again))
	}
	cur, ok, _ := cursors.GetCursor(context.Background(), ID, "", cursorScope)
	if !ok || !cur.Equal(time.Date(2026, 11, 10, 0, 0, 0, 0, time.UTC)) {
		t.Fatalf("cursor %v %v", cur, ok)
	}
	// Restart: a fresh connector with the persisted cursor skips everything
	// up to 2026-11-10 and asks from cursor − 7 days.
	c2 := newTest(t, now, cursors, func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		w.Write(fixture(t, "latest_eur.json"))
	})
	raws, err = c2.Poll(context.Background(), connector.PollRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if paths[len(paths)-1] != "/2026-11-03.." {
		t.Fatalf("restart request %q", paths[len(paths)-1])
	}
	if envs, _, _ := c2.Normalize(raws[0]); len(envs) != 0 {
		t.Fatalf("restart re-emitted %d", len(envs))
	}
}

func TestLatestShapeAndDeterminism(t *testing.T) {
	raw := connector.RawMessage{SourceID: ID, ReceivedAt: time.Date(2026, 11, 11, 3, 0, 0, 0, time.UTC), Bytes: fixture(t, "latest_eur.json"),
		TransportMeta: map[string]string{connector.MetaEndpoint: "range"}}
	norm := func() []*fav1Env {
		c := newTest(t, raw.ReceivedAt, nil, func(http.ResponseWriter, *http.Request) {})
		envs, _, err := c.Normalize(raw)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]*fav1Env, len(envs))
		for i, e := range envs {
			out[i] = &fav1Env{payload: e.GetMacro(), event: e.EventTime.AsTime(), publish: e.PublishTime.AsTime()}
		}
		return out
	}
	a, b := norm(), norm()
	if len(a) != 2 || len(b) != 2 {
		t.Fatalf("got %d/%d", len(a), len(b))
	}
	for i := range a {
		if !proto.Equal(a[i].payload, b[i].payload) || !a[i].event.Equal(b[i].event) || !a[i].publish.Equal(b[i].publish) {
			t.Fatalf("non-deterministic at %d: %v vs %v", i, a[i].payload, b[i].payload)
		}
	}
	if a[0].payload.SeriesId != "EUR_GBP" || a[1].payload.SeriesId != "EUR_USD" || a[1].payload.Period != "2026-11-10" {
		t.Fatalf("latest: %v %v", a[0].payload, a[1].payload)
	}
}

type fav1Env struct {
	payload        *fav1.MacroObservation
	event, publish time.Time
}

func TestBackfillWindowsAndRange(t *testing.T) {
	now := time.Date(2026, 11, 11, 0, 0, 0, 0, time.UTC)
	var reqs []string
	c := newTest(t, now, connector.NewMemCursors(), func(w http.ResponseWriter, r *http.Request) {
		reqs = append(reqs, r.URL.Path)
		w.Write(fixture(t, "range_eur_usd_gbp.json"))
	})
	out := make(chan connector.RawMessage, 8)
	from := time.Date(2026, 11, 4, 0, 0, 0, 0, time.UTC)
	to := time.Date(2026, 11, 10, 0, 0, 0, 0, time.UTC)
	if err := c.Backfill(context.Background(), connector.BackfillRequest{From: from, To: to}, out); err != nil {
		t.Fatal(err)
	}
	close(out)
	if len(reqs) != 1 || reqs[0] != "/2026-11-04..2026-11-09" {
		t.Fatalf("requests %v", reqs)
	}
	raw := <-out
	envs, _, err := c.Normalize(raw)
	if err != nil {
		t.Fatal(err)
	}
	// 11-04, 05, 06, 09 inside [from, to) × 2 currencies; 11-03 and 11-10 dropped.
	if len(envs) != 8 {
		t.Fatalf("want 8, got %d", len(envs))
	}
	for _, e := range envs {
		if !e.PublishTime.AsTime().Equal(e.GetMacro().Vintage.AsTime()) {
			t.Fatal("backfill publish_time must be the estimated vintage")
		}
	}
	// Backfill re-emits (no live de-dup) and does not move the live cursor.
	if again, _, _ := c.Normalize(raw); len(again) != 8 {
		t.Fatalf("backfill must not de-dup: %d", len(again))
	}
	if _, ok, _ := c.deps.Cursors.GetCursor(context.Background(), ID, "", cursorScope); ok {
		t.Fatal("backfill moved the live cursor")
	}

	// Multi-year range: one request per year, clamped to the first ECB date.
	reqs = nil
	out = make(chan connector.RawMessage, 64)
	if err := c.Backfill(context.Background(), connector.BackfillRequest{From: time.Date(1990, 1, 1, 0, 0, 0, 0, time.UTC), To: time.Date(2001, 6, 1, 0, 0, 0, 0, time.UTC)}, out); err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 3 || reqs[0] != "/1999-01-04..2000-01-03" || reqs[2] != "/2001-01-04..2001-05-31" {
		t.Fatalf("windows %v", reqs)
	}
}

func TestErrorAnswer(t *testing.T) {
	c := newTest(t, time.Now(), nil, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"message":"not found"}`))
	})
	if _, err := c.Poll(context.Background(), connector.PollRequest{}); err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("want 404 error, got %v", err)
	}
	if _, _, err := c.Normalize(connector.RawMessage{Bytes: []byte(`{"message":"not found"}`)}); err == nil {
		t.Fatal("want error for message-only body")
	}
	if _, _, err := c.Normalize(connector.RawMessage{Bytes: []byte(`<html>`)}); err == nil {
		t.Fatal("want decode error")
	}
}

func TestInitRejectsBadCurrency(t *testing.T) {
	c := &Connector{}
	src := catalog.Source{ID: ID, Config: &catalog.Params{Extra: map[string]string{"currencies": "USDX"}}}
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{}); err == nil {
		t.Fatal("want error")
	}
}
