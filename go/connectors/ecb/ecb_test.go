package ecb

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

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

// server answers per flow with a fixture and records request URLs.
func newTest(t *testing.T, now time.Time, cursors connector.CursorStore, queries []string, files map[string]string) (*Connector, *[]string) {
	t.Helper()
	var mu sync.Mutex
	var reqs []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		reqs = append(reqs, r.URL.String())
		mu.Unlock()
		flow := strings.Split(strings.TrimPrefix(r.URL.Path, "/"), "/")[0]
		name, ok := files[flow]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			w.Write([]byte(`{"title":"Not Found","status":404}`))
			return
		}
		if strings.HasSuffix(name, ".html") {
			w.WriteHeader(http.StatusBadRequest)
		}
		w.Write(fixture(t, name))
	}))
	t.Cleanup(srv.Close)
	c := &Connector{BaseURL: srv.URL}
	src := catalog.Source{ID: ID, Config: &catalog.Params{Queries: queries}}
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{Clock: fixedClock{now}, Cursors: cursors}); err != nil {
		t.Fatal(err)
	}
	return c, &reqs
}

var now = time.Date(2026, 11, 11, 2, 0, 0, 0, time.UTC)

func TestPollHICPVintagesAndCursor(t *testing.T) {
	cursors := connector.NewMemCursors()
	c, reqs := newTest(t, now, cursors, []string{"HICP/M.U2.N.000000.4D0.ANR"}, map[string]string{"HICP": "hicp_history.json"})
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 1 {
		t.Fatalf("poll: %v %d", err, len(raws))
	}
	if u := (*reqs)[0]; !strings.HasPrefix(u, "/HICP/M.U2.N.000000.4D0.ANR?") || !strings.Contains(u, "includeHistory=true") || !strings.Contains(u, "lastNObservations=3") {
		t.Fatalf("request %s", u)
	}
	envs, _, err := c.Normalize(raws[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(envs) != 4 {
		t.Fatalf("want 4, got %d", len(envs))
	}
	e := envs[3]
	m := e.GetMacro()
	if m.SeriesId != "HICP:M:U2:N:000000:4D0:ANR" || m.Period != "2026-10" || m.Value != 4 || m.Revision {
		t.Fatalf("obs: %v", m)
	}
	vintage := time.Date(2026, 11, 4, 10, 6, 25, 0, time.UTC) // 11:06:25+01:00
	if !m.Vintage.AsTime().Equal(vintage) || !e.PublishTime.AsTime().Equal(vintage) {
		t.Fatalf("vintage %v publish %v", m.Vintage.AsTime(), e.PublishTime.AsTime())
	}
	if !e.EventTime.AsTime().Equal(time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)) || !e.IngestTime.AsTime().Equal(now) {
		t.Fatalf("event %v", e.EventTime.AsTime())
	}
	if e.PublishTimeEstimated {
		t.Fatal("a real validFrom vintage is not estimated")
	}
	if again, _, _ := c.Normalize(raws[0]); len(again) != 0 {
		t.Fatalf("dedup: %d", len(again))
	}
	cur, ok, _ := cursors.GetCursor(context.Background(), ID, "", "q:HICP/M.U2.N.000000.4D0.ANR")
	if !ok || !cur.Equal(vintage) {
		t.Fatalf("cursor %v %v", cur, ok)
	}
	// Restart with the cursor: nothing re-emitted, a later revision would be.
	c2, _ := newTest(t, now, cursors, []string{"HICP/M.U2.N.000000.4D0.ANR"}, map[string]string{"HICP": "hicp_history.json"})
	if envs, _, _ := c2.Normalize(raws[0]); len(envs) != 0 {
		t.Fatalf("restart re-emitted %d", len(envs))
	}
}

func TestRevisionsDeleteAndNull(t *testing.T) {
	// Synthetic: 2026-09 first released 2026-10-01 as 3.7, revised to 3.8 on
	// 2026-10-16; a null value and a Delete dataset are ignored.
	obs, err := Parse("HICP", fixture(t, "hicp_revision_synthetic.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(obs) != 5 {
		t.Fatalf("want 5 observations, got %d: %+v", len(obs), obs)
	}
	var sep []Obs
	for _, o := range obs {
		if o.Period == "2026-09" {
			sep = append(sep, o)
		}
	}
	if len(sep) != 2 || sep[0].Value != 3.7 || sep[0].Revision || sep[1].Value != 3.8 || !sep[1].Revision || !sep[0].Vintage.Before(sep[1].Vintage) {
		t.Fatalf("revision handling: %+v", sep)
	}
}

func TestEstimatedVintageForHistorylessFlow(t *testing.T) {
	c, _ := newTest(t, now, nil, []string{"EXR/D.USD.EUR.SP00.A"}, map[string]string{"EXR": "exr_usd.json"})
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil {
		t.Fatal(err)
	}
	envs, _, err := c.Normalize(raws[0])
	if err != nil || len(envs) != 2 {
		t.Fatalf("normalize: %v %d", err, len(envs))
	}
	m := envs[1].GetMacro()
	// validFrom == prepared → estimate 2026-11-10 16:00 CET = 15:00 UTC; live publish = ingest.
	if m.SeriesId != "EXR:D:USD:EUR:SP00:A" || m.Period != "2026-11-10" || m.Value != 1.1214 ||
		!m.Vintage.AsTime().Equal(time.Date(2026, 11, 10, 15, 0, 0, 0, time.UTC)) || !envs[1].PublishTime.AsTime().Equal(now) {
		t.Fatalf("exr: %v publish %v", m, envs[1].PublishTime.AsTime())
	}
	if !envs[1].PublishTimeEstimated {
		t.Fatal("estimated vintage must set publish_time_estimated")
	}
	// Determinism: a second connector yields identical payloads.
	c2, _ := newTest(t, now, nil, []string{"EXR/D.USD.EUR.SP00.A"}, map[string]string{"EXR": "exr_usd.json"})
	envs2, _, _ := c2.Normalize(raws[0])
	for i := range envs {
		if !proto.Equal(envs[i].GetMacro(), envs2[i].GetMacro()) || !envs[i].PublishTime.AsTime().Equal(envs2[i].PublishTime.AsTime()) {
			t.Fatalf("non-deterministic at %d", i)
		}
	}
}

func TestMultipleQueriesPartialFailureAndWAF(t *testing.T) {
	c, _ := newTest(t, now, nil, []string{"FM/B.U2.EUR.4F.KR.MRR_FR.LEV", "XYZ/A.B", "BLOCK/D.X"},
		map[string]string{"FM": "fm_mro.json", "BLOCK": "waf_block.html"})
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 1 {
		t.Fatalf("partial failure must still return the good query: %v %d", err, len(raws))
	}
	envs, _, err := c.Normalize(raws[0])
	if err != nil || len(envs) != 2 {
		t.Fatalf("mro: %v %d", err, len(envs))
	}
	if m := envs[1].GetMacro(); m.SeriesId != "FM:B:U2:EUR:4F:KR:MRR_FR:LEV" || m.Period != "2026-09-16" || m.Value != 2.65 {
		t.Fatalf("mro: %v", m)
	}
	c2, _ := newTest(t, now, nil, []string{"BLOCK/D.X"}, map[string]string{"BLOCK": "waf_block.html"})
	if _, err := c2.Poll(context.Background(), connector.PollRequest{}); err == nil || !strings.Contains(err.Error(), "firewall") {
		t.Fatalf("want firewall error, got %v", err)
	}
	if _, _, err := c.Normalize(connector.RawMessage{Bytes: []byte("<html>"), TransportMeta: map[string]string{metaQuery: "FM/x"}}); err == nil {
		t.Fatal("want decode error")
	}
}

func TestBackfillWindows(t *testing.T) {
	c, reqs := newTest(t, now, connector.NewMemCursors(), []string{"HICP/M.U2.N.000000.4D0.ANR", "FM/B.U2.EUR.4F.KR.MRR_FR.LEV"},
		map[string]string{"HICP": "hicp_history.json", "FM": "fm_mro.json"})
	out := make(chan connector.RawMessage, 32)
	from := time.Date(2015, 1, 1, 0, 0, 0, 0, time.UTC)
	to := time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)
	if err := c.Backfill(context.Background(), connector.BackfillRequest{From: from, To: to}, out); err != nil {
		t.Fatal(err)
	}
	close(out)
	if len(*reqs) != 6 { // 2015-2019, 2020-2024, 2025-2026 × 2 queries
		t.Fatalf("requests %v", *reqs)
	}
	if u := (*reqs)[2]; !strings.Contains(u, "startPeriod=2025-01-01") || !strings.Contains(u, "endPeriod=2026-10-31") {
		t.Fatalf("last window %s", u)
	}
	total := 0
	for raw := range out {
		envs, _, err := c.Normalize(raw)
		if err != nil {
			t.Fatal(err)
		}
		total += len(envs)
	}
	if total != 3*4+3*2 {
		t.Fatalf("backfill events %d", total)
	}
	if _, ok, _ := c.deps.Cursors.GetCursor(context.Background(), ID, "", "q:HICP/M.U2.N.000000.4D0.ANR"); ok {
		t.Fatal("backfill moved the live cursor")
	}
	// A single query via VenueSymbol.
	*reqs = nil
	out = make(chan connector.RawMessage, 4)
	if err := c.Backfill(context.Background(), connector.BackfillRequest{VenueSymbol: "FM/B.U2.EUR.4F.KR.MRR_FR.LEV", From: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), To: to}, out); err != nil {
		t.Fatal(err)
	}
	if len(*reqs) != 1 || !strings.HasPrefix((*reqs)[0], "/FM/") {
		t.Fatalf("single query %v", *reqs)
	}
}

func TestPeriodBounds(t *testing.T) {
	d := func(y, m, dd int) time.Time { return time.Date(y, time.Month(m), dd, 0, 0, 0, 0, time.UTC) }
	for p, want := range map[string][2]time.Time{
		"2026":       {d(2026, 1, 1), d(2027, 1, 1)},
		"2026-S2":    {d(2026, 7, 1), d(2027, 1, 1)},
		"2026-Q3":    {d(2026, 7, 1), d(2026, 10, 1)},
		"2026-10":    {d(2026, 10, 1), d(2026, 11, 1)},
		"2026-W01":   {d(2025, 12, 29), d(2026, 1, 5)},
		"2026-11-10": {d(2026, 11, 10), d(2026, 11, 11)},
	} {
		s, e, err := PeriodBounds(p)
		if err != nil || !s.Equal(want[0]) || !e.Equal(want[1]) {
			t.Errorf("%s: %v %v %v", p, s, e, err)
		}
	}
	if _, _, err := PeriodBounds("2026-X"); err == nil {
		t.Error("want error")
	}
}

func TestInitRejectsBadQuery(t *testing.T) {
	c := &Connector{}
	src := catalog.Source{ID: ID, Config: &catalog.Params{Queries: []string{"not a query"}}}
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{}); err == nil {
		t.Fatal("want error")
	}
}
