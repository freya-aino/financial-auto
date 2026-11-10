package refdata

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/catalog"
)

func TestUUIDv5(t *testing.T) {
	// RFC 4122 appendix / python: uuid5(NAMESPACE_DNS, "python.org")
	if got := UUIDv5("6ba7b810-9dad-11d1-80b4-00c04fd430c8", "python.org"); got != "886313e1-3b8a-5372-9b90-0c9aee199e5d" {
		t.Fatalf("uuid5 = %s", got)
	}
	if ProvisionalID("yahoo", "AAPL") != ProvisionalID("yahoo", "AAPL") || ProvisionalID("yahoo", "AAPL") == ProvisionalID("lemon", "AAPL") {
		t.Fatal("provisional ids must be deterministic per (source, symbol)")
	}
}

// Live tests run only with FA_POSTGRES_URL (compose: postgres://fa:fa@127.0.0.1:5432/fa).
func liveStore(t *testing.T) *Store {
	t.Helper()
	url := os.Getenv("FA_POSTGRES_URL")
	if url == "" {
		t.Skip("FA_POSTGRES_URL not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	s, err := Open(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return s
}

func TestLiveResolverCursorsRunLog(t *testing.T) {
	s := liveStore(t)
	ctx := context.Background()
	cat, err := catalog.Load("../../../config/sources.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SyncCatalog(ctx, cat); err != nil {
		t.Fatal(err)
	}
	var n int
	if err := s.Pool.QueryRow(ctx, `select count(*) from catalog.source`).Scan(&n); err != nil || n < len(cat.Sources) {
		t.Fatalf("catalog rows %d (%v)", n, err)
	}

	sym := "TEST_" + time.Now().UTC().Format("20060102150405.000")
	var errs []error
	r := NewResolver(s, nil, func(err error) { errs = append(errs, err) })
	id, ok := r.Resolve("yahoo", sym)
	if !ok || id != ProvisionalID("yahoo", sym) || len(errs) != 0 {
		t.Fatalf("resolve: %s %v %v", id, ok, errs)
	}
	if r.Created("yahoo") != 1 {
		t.Fatal("expected one provisional creation")
	}
	// Second resolver (cold cache) finds the row through ref.listing, creates nothing.
	r2 := NewResolver(s, nil, nil)
	if id2, ok := r2.Resolve("yahoo", sym); !ok || id2 != id || r2.Created("yahoo") != 0 {
		t.Fatalf("lookup: %s %v created=%d", id2, ok, r2.Created("yahoo"))
	}
	var status, qstatus string
	if err := s.Pool.QueryRow(ctx, `select status from ref.instrument where id = $1`, id).Scan(&status); err != nil || status != "provisional" {
		t.Fatalf("instrument status %q %v", status, err)
	}
	if err := s.Pool.QueryRow(ctx, `select status from ingest.quarantine where source_id = 'yahoo' and venue_symbol = $1`, sym).Scan(&qstatus); err != nil || qstatus != "open" {
		t.Fatalf("quarantine %q %v", qstatus, err)
	}
	t.Cleanup(func() {
		s.Pool.Exec(ctx, `delete from ingest.quarantine where venue_symbol = $1`, sym)
		s.Pool.Exec(ctx, `delete from ref.identifier where instrument_id = $1`, id)
		s.Pool.Exec(ctx, `delete from ref.listing where instrument_id = $1`, id)
		s.Pool.Exec(ctx, `delete from ingest.backfill_cursor where instrument_id = $1`, id)
		s.Pool.Exec(ctx, `delete from ref.instrument where id = $1`, id)
	})

	c := NewCursors(s)
	if _, ok, err := c.GetCursor(ctx, "hackernews", "", "test-"+sym); ok || err != nil {
		t.Fatalf("empty cursor: %v %v", ok, err)
	}
	at := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	if err := c.SetCursor(ctx, "hackernews", "", "test-"+sym, at); err != nil {
		t.Fatal(err)
	}
	if err := c.SetCursor(ctx, "hackernews", "", "test-"+sym, at.Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	got, ok, err := c.GetCursor(ctx, "hackernews", "", "test-"+sym)
	if err != nil || !ok || !got.Equal(at.Add(time.Hour)) {
		t.Fatalf("cursor %v %v %v", got, ok, err)
	}
	if err := c.SetCursor(ctx, "yahoo", id, "1m", at); err != nil {
		t.Fatal(err)
	}
	if got, ok, _ := c.GetCursor(ctx, "yahoo", id, "1m"); !ok || !got.Equal(at) {
		t.Fatalf("instrument cursor %v %v", got, ok)
	}
	t.Cleanup(func() { s.Pool.Exec(ctx, `delete from ingest.backfill_cursor where timescale = $1`, "test-"+sym) })

	l := NewRunLog(s)
	runID, err := l.Start(ctx, "yahoo")
	if err != nil {
		t.Fatal(err)
	}
	if err := l.End(ctx, runID, "stopped", 10, 1, "boom"); err != nil {
		t.Fatal(err)
	}
	var ev int64
	var last *string
	if err := s.Pool.QueryRow(ctx, `select events, last_error from ingest.connector_run where id = $1 and ended_at is not null`, runID).Scan(&ev, &last); err != nil || ev != 10 || last == nil || *last != "boom" {
		t.Fatalf("run row: %d %v %v", ev, last, err)
	}
	t.Cleanup(func() { s.Pool.Exec(ctx, `delete from ingest.connector_run where id = $1`, runID) })
}
