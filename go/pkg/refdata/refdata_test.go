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

func TestLiveStatusesBackfillCursorsProvisionHook(t *testing.T) {
	s := liveStore(t)
	ctx := context.Background()
	cat, err := catalog.Load("../../../config/sources.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SyncCatalog(ctx, cat); err != nil {
		t.Fatal(err)
	}
	// Statuses round trip: DB status wins over the YAML seed.
	st, err := s.Statuses(ctx)
	if err != nil || st["yahoo"] == "" {
		t.Fatalf("statuses %v %v", st, err)
	}
	orig := st["frankfurter"]
	if err := s.SetStatus(ctx, "frankfurter", "disabled"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.SetStatus(ctx, "frankfurter", orig) })
	st, _ = s.Statuses(ctx)
	if st["frankfurter"] != "disabled" {
		t.Fatalf("status not persisted: %q", st["frankfurter"])
	}
	if err := s.SyncCatalog(ctx, cat); err != nil {
		t.Fatal(err)
	}
	st, _ = s.Statuses(ctx)
	if st["frankfurter"] != "disabled" {
		t.Fatalf("sync overwrote the DB status: %q", st["frankfurter"])
	}
	if err := s.SetStatus(ctx, "no_such_source_xyz", "live"); err == nil {
		t.Fatal("expected error for unknown id")
	}

	// Provision hook fires once per new pair and the quarantine row carries the instrument id.
	sym := "TESTHOOK_" + time.Now().UTC().Format("20060102150405.000")
	var hooked []string
	r := NewResolver(s, nil, nil)
	r.OnProvision = func(src, symbol, id string) { hooked = append(hooked, src+"/"+symbol+"/"+id) }
	id, ok := r.Resolve("yahoo", sym)
	if !ok || len(hooked) != 1 || hooked[0] != "yahoo/"+sym+"/"+id {
		t.Fatalf("hook %v (%s %v)", hooked, id, ok)
	}
	r.Resolve("yahoo", sym)
	if len(hooked) != 1 {
		t.Fatal("hook fired for a cached pair")
	}
	var qinst string
	if err := s.Pool.QueryRow(ctx, `select instrument_id::text from ingest.quarantine where source_id = 'yahoo' and venue_symbol = $1`, sym).Scan(&qinst); err != nil || qinst != id {
		t.Fatalf("quarantine instrument_id %q %v", qinst, err)
	}
	t.Cleanup(func() {
		s.Pool.Exec(ctx, `delete from ingest.quarantine where venue_symbol = $1`, sym)
		s.Pool.Exec(ctx, `delete from ingest.backfill_cursor where instrument_id = $1`, id)
		s.Pool.Exec(ctx, `delete from ref.identifier where instrument_id = $1`, id)
		s.Pool.Exec(ctx, `delete from ref.listing where instrument_id = $1`, id)
		s.Pool.Exec(ctx, `delete from ref.instrument where id = $1`, id)
	})

	// Bounded backfill ranges are independent of the live watermark row.
	bc := NewBackfillCursors(s)
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	end := start.Add(24 * time.Hour)
	if _, _, ok, err := bc.GetBackfill(ctx, "yahoo", id, "1m", start); ok || err != nil {
		t.Fatalf("empty range %v %v", ok, err)
	}
	if err := bc.SetBackfill(ctx, "yahoo", id, "1m", start, end, time.Time{}, "pending"); err != nil {
		t.Fatal(err)
	}
	if err := bc.SetBackfill(ctx, "yahoo", id, "1m", start, end, start.Add(time.Hour), "running"); err != nil {
		t.Fatal(err)
	}
	cur, status, ok, err := bc.GetBackfill(ctx, "yahoo", id, "1m", start)
	if err != nil || !ok || status != "running" || !cur.Equal(start.Add(time.Hour)) {
		t.Fatalf("range %v %q %v %v", cur, status, ok, err)
	}
	if err := bc.SetBackfill(ctx, "yahoo", id, "1m", start, end, time.Time{}, "done"); err != nil {
		t.Fatal(err)
	}
	cur, status, _, _ = bc.GetBackfill(ctx, "yahoo", id, "1m", start)
	if status != "done" || !cur.Equal(start.Add(time.Hour)) {
		t.Fatalf("done keeps cursor: %v %q", cur, status)
	}
	if _, ok, _ := NewCursors(s).GetCursor(ctx, "yahoo", id, "1m"); ok {
		t.Fatal("bounded range must not show up as the live watermark")
	}
	rows, err := s.ListCursors(ctx, "yahoo", 50)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range rows {
		if r.InstrumentID == id && !r.Live && r.Status == "done" {
			found = true
		}
	}
	if !found {
		t.Fatalf("ListCursors missing the range: %+v", rows)
	}
}

func TestClassifyYahooSymbols(t *testing.T) {
	h := DefaultHints["yahoo"]
	cases := map[string]Class{
		"AAPL":     {"equity", "equity", "", "USD"},
		"EURUSD=X": {"fx", "fx_pair", "EUR", "USD"},
		"BTC-USD":  {"crypto", "crypto_spot", "BTC", "USD"},
		"^GSPC":    {AssetClass: "index", Kind: "index"},
		"GC=F":     {AssetClass: "derivative", Kind: "future", Quote: "USD"},
		"SAP.DE":   {"equity", "equity", "", "EUR"},
	}
	for sym, want := range cases {
		if got := Classify("yahoo", sym, h); got != want {
			t.Errorf("%s: got %+v want %+v", sym, got, want)
		}
	}
	if got := Classify("oanda", "EUR_USD", DefaultHints["oanda"]); got != (Class{"fx", "fx_pair", "EUR", "USD"}) {
		t.Errorf("oanda: %+v", got)
	}
	if got := Classify("kraken", "XBT/EUR", Hint{AssetClass: "crypto", Kind: "crypto_spot"}); got != (Class{"crypto", "crypto_spot", "XBT", "EUR"}) {
		t.Errorf("kraken: %+v", got)
	}
}
