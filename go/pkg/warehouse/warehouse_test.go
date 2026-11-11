package warehouse

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

var vintage = time.Date(2026, 11, 4, 10, 6, 25, 0, time.UTC)

func macro(series, period string, v float64, rev bool) *fav1.Envelope {
	return &fav1.Envelope{SourceId: "ecb", Payload: &fav1.Envelope_Macro{Macro: &fav1.MacroObservation{
		SeriesId: series, Period: period, Value: v, Vintage: timestamppb.New(vintage), Revision: rev,
	}}}
}

func TestMacroRows(t *testing.T) {
	envs := []*fav1.Envelope{
		macro("HICP:M:U2:N:000000:4D0:ANR", "2026-10", 4, false),
		{SourceId: "yahoo", Payload: &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: 1}}},
		macro("X", "", 1, false),              // no period
		macro("X", "2026", math.NaN(), false), // not finite
		{SourceId: "ecb", Payload: &fav1.Envelope_Macro{Macro: &fav1.MacroObservation{SeriesId: "X", Period: "2026"}}}, // no vintage
		macro("X", "2026", 2.5, true),
	}
	rows, invalid := MacroRows(envs)
	if invalid != 3 || len(rows) != 2 {
		t.Fatalf("rows %d invalid %d", len(rows), invalid)
	}
	if r := rows[0]; r.SourceID != "ecb" || r.SeriesID != "HICP:M:U2:N:000000:4D0:ANR" || r.Period != "2026-10" || !r.Vintage.Equal(vintage) || r.Value != 4 || r.Revision {
		t.Fatalf("row %+v", r)
	}
	if !rows[1].Revision {
		t.Fatal("revision flag lost")
	}
}

// fakeDB records queued rows and reports duplicates for keys already seen.
type fakeDB struct {
	keys map[string]bool
	fail bool
}

type fakeResults struct {
	pgx.BatchResults // methods not overridden here are never called
	tags             []bool
	i                int
	fail             bool
}

func (f *fakeDB) SendBatch(_ context.Context, b *pgx.Batch) pgx.BatchResults {
	r := &fakeResults{fail: f.fail}
	for _, q := range b.QueuedQueries {
		k := q.Arguments[0].(string) + "|" + q.Arguments[1].(string) + "|" + q.Arguments[2].(string) + "|" + q.Arguments[3].(time.Time).String()
		r.tags = append(r.tags, !f.keys[k])
		f.keys[k] = true
	}
	return r
}

func (r *fakeResults) Exec() (pgconn.CommandTag, error) {
	if r.fail {
		return pgconn.CommandTag{}, errors.New("boom")
	}
	ok := r.tags[r.i]
	r.i++
	if ok {
		return pgconn.NewCommandTag("INSERT 0 1"), nil
	}
	return pgconn.NewCommandTag("INSERT 0 0"), nil
}
func (r *fakeResults) Query() (pgx.Rows, error) { return nil, errors.New("unused") }
func (r *fakeResults) QueryRow() pgx.Row        { return nil }
func (r *fakeResults) Close() error             { return nil }

func TestWriterIdempotentCounts(t *testing.T) {
	db := &fakeDB{keys: map[string]bool{}}
	w := NewWriter(db)
	envs := []*fav1.Envelope{macro("A", "2026-10", 1, false), macro("B", "2026-10", 2, false)}
	n, err := w.Write(context.Background(), envs)
	if err != nil || n != 2 {
		t.Fatalf("first write %d %v", n, err)
	}
	n, err = w.Write(context.Background(), envs)
	if err != nil || n != 0 {
		t.Fatalf("second write %d %v", n, err)
	}
	if st := w.Stats(); st.Inserted != 2 || st.Duplicates != 2 || st.Errors != 0 {
		t.Fatalf("stats %+v", st)
	}
	// No macro envelopes: no batch at all.
	if n, err := w.Write(context.Background(), []*fav1.Envelope{{Payload: &fav1.Envelope_News{News: &fav1.NewsItem{}}}}); n != 0 || err != nil {
		t.Fatal("non-macro write")
	}
	db.fail = true
	if _, err := w.Write(context.Background(), []*fav1.Envelope{macro("C", "2026", 1, false)}); err == nil {
		t.Fatal("want error")
	}
	if w.Stats().Errors != 1 {
		t.Fatal("error not counted")
	}
	var nilW *Writer
	if n, err := nilW.Write(context.Background(), envs); n != 0 || err != nil {
		t.Fatal("nil writer must be a no-op")
	}
}

// TestPostgres runs against a real database when FA_TEST_POSTGRES_URL is set
// (e.g. postgres://fa:fa@127.0.0.1:5432/fa); everything happens inside a
// rolled-back transaction.
func TestPostgres(t *testing.T) {
	url := os.Getenv("FA_TEST_POSTGRES_URL")
	if url == "" {
		t.Skip("FA_TEST_POSTGRES_URL not set")
	}
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close(ctx)
	tx, err := conn.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	w := NewWriter(tx)
	envs := []*fav1.Envelope{macro("TEST:warehouse", "2026-10", 4, false), macro("TEST:warehouse", "2026-09", 3.8, true)}
	if n, err := w.Write(ctx, envs); err != nil || n != 2 {
		t.Fatalf("insert %d %v", n, err)
	}
	if n, err := w.Write(ctx, envs); err != nil || n != 0 {
		t.Fatalf("idempotent re-insert %d %v", n, err)
	}
	var count int
	var rev bool
	if err := tx.QueryRow(ctx, `select count(*), bool_or(revision) from wh.macro_observation where source_id='ecb' and series_id='TEST:warehouse'`).Scan(&count, &rev); err != nil {
		t.Fatal(err)
	}
	if count != 2 || !rev {
		t.Fatalf("rows %d revision %v", count, rev)
	}
}

func TestFilingRows(t *testing.T) {
	at := time.Date(2026, 11, 10, 21, 5, 0, 0, time.UTC)
	envs := []*fav1.Envelope{
		{SourceId: "sec_edgar", EventId: "E1", PublishTime: timestamppb.New(at), Payload: &fav1.Envelope_Filing{Filing: &fav1.FilingEvent{IssuerCik: "0000320193", FormType: "8-K", Accession: "0000320193-26-000001", FilingTime: timestamppb.New(at)}}},
		{SourceId: "sec_edgar", Payload: &fav1.Envelope_Filing{Filing: &fav1.FilingEvent{FormType: "8-K"}}},
		{SourceId: "x", Payload: &fav1.Envelope_Trade{Trade: &fav1.Trade{}}},
	}
	rows, invalid := FilingRows(envs)
	if len(rows) != 1 || invalid != 1 || rows[0].Accession != "0000320193-26-000001" || rows[0].PublishTime == nil || rows[0].DocumentRefs == nil {
		t.Fatalf("rows %+v invalid %d", rows, invalid)
	}
}

func fact(field, unit, period string, v float64, filed time.Time) *fav1.Envelope {
	end := time.Date(2026, 9, 26, 0, 0, 0, 0, time.UTC)
	return &fav1.Envelope{SourceId: "sec_edgar", Payload: &fav1.Envelope_Fundamental{Fundamental: &fav1.FundamentalFact{
		IssuerCik: "0000320193", Field: field, Unit: unit, Period: period, PeriodEnd: timestamppb.New(end),
		Value: v, Vintage: timestamppb.New(filed), Form: "10-K", Accession: "0000320193-26-000026", FiscalYear: "2026", FiscalPeriod: "FY",
	}}}
}

func TestFundamentalRows(t *testing.T) {
	filed := time.Date(2026, 11, 4, 3, 0, 0, 0, time.UTC)
	envs := []*fav1.Envelope{
		fact("us-gaap:NetIncomeLoss", "USD", "2025-09-28..2026-09-26", 132175000000, filed),
		fact("us-gaap:Assets", "USD", "", 1, filed),                                                                                               // no period
		fact("us-gaap:Assets", "USD", "2026-09-26", math.Inf(1), filed),                                                                           // not finite
		{SourceId: "sec_edgar", Payload: &fav1.Envelope_Fundamental{Fundamental: &fav1.FundamentalFact{IssuerCik: "1", Field: "f", Period: "p"}}}, // no vintage
		macro("X", "2026", 1, false),
	}
	rows, invalid := FundamentalRows(envs)
	if len(rows) != 1 || invalid != 3 {
		t.Fatalf("rows %d invalid %d", len(rows), invalid)
	}
	r := rows[0]
	if r.IssuerCIK != "0000320193" || r.Field != "us-gaap:NetIncomeLoss" || r.Unit != "USD" || !r.Vintage.Equal(filed) ||
		r.PeriodEnd == nil || r.PeriodStart != nil || r.Form != "10-K" || r.FiscalPeriod != "FY" {
		t.Fatalf("row %+v", r)
	}
}

// TestPostgresFundamentals: like TestPostgres (FA_TEST_POSTGRES_URL, rolled back);
// needs migration 20261113000012.
func TestPostgresFundamentals(t *testing.T) {
	url := os.Getenv("FA_TEST_POSTGRES_URL")
	if url == "" {
		t.Skip("FA_TEST_POSTGRES_URL not set")
	}
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close(ctx)
	tx, err := conn.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	filed := time.Date(2099, 1, 2, 3, 0, 0, 0, time.UTC)
	w := NewWriter(tx)
	envs := []*fav1.Envelope{
		fact("test:Concept", "USD", "2098-01-01..2098-12-31", 1, filed),
		fact("test:Concept", "EUR", "2098-01-01..2098-12-31", 2, filed),                   // other unit: own row
		fact("test:Concept", "USD", "2098-01-01..2098-12-31", 3, filed.Add(24*time.Hour)), // revision: new vintage
	}
	if n, err := w.Write(ctx, envs); err != nil || n != 3 {
		t.Fatalf("insert %d %v", n, err)
	}
	if n, err := w.Write(ctx, envs); err != nil || n != 0 {
		t.Fatalf("idempotent re-insert %d %v", n, err)
	}
	// as-of read: latest vintage <= t
	var v float64
	if err := tx.QueryRow(ctx, `select value from wh.fundamental where issuer_cik='0000320193' and field='test:Concept' and unit='USD'
		and vintage <= $1 order by vintage desc limit 1`, filed.Add(time.Hour)).Scan(&v); err != nil || v != 1 {
		t.Fatalf("as-of %v %v", v, err)
	}
}
