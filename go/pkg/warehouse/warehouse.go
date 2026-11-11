// Package warehouse is the ingestor's Postgres write path for slow canonical
// records (spec 20 §6.1 (b)). It writes FilingEvent envelopes into wh.filing
// (idempotent on source_id + accession) and MacroObservation envelopes into
// wh.macro_observation, one row per (source_id, series_id, period, vintage),
// idempotently: a replayed or re-polled vintage is a no-op (`on conflict do
// nothing`), so live polls, backfill jobs and raw-archive replays may all
// write the same observation (spec 10 §6.2: revisions are new vintages, never
// overwrites).
package warehouse

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// MacroRow is one wh.macro_observation row.
type MacroRow struct {
	SourceID string
	SeriesID string
	Period   string
	Vintage  time.Time
	Value    float64
	Revision bool
}

// MacroRows extracts the rows of the macro envelopes in envs. Envelopes of
// other kinds are ignored; macro envelopes missing a key field or with a
// non-finite value are counted in invalid and skipped.
func MacroRows(envs []*fav1.Envelope) (rows []MacroRow, invalid int) {
	for _, e := range envs {
		m := e.GetMacro()
		if m == nil {
			continue
		}
		if e.GetSourceId() == "" || m.GetSeriesId() == "" || m.GetPeriod() == "" || m.GetVintage() == nil ||
			math.IsNaN(m.GetValue()) || math.IsInf(m.GetValue(), 0) {
			invalid++
			continue
		}
		rows = append(rows, MacroRow{
			SourceID: e.GetSourceId(), SeriesID: m.GetSeriesId(), Period: m.GetPeriod(),
			Vintage: m.GetVintage().AsTime().UTC(), Value: m.GetValue(), Revision: m.GetRevision(),
		})
	}
	return rows, invalid
}

// InsertMacroSQL is idempotent on the table's primary key.
const InsertMacroSQL = `insert into wh.macro_observation (source_id, series_id, period, vintage, value, revision)
values ($1, $2, $3, $4, $5, $6)
on conflict (source_id, series_id, period, vintage) do nothing`

// FilingRow is one wh.filing row (migration 20261111000045).
type FilingRow struct {
	SourceID, Accession, IssuerCIK, IssuerLEI, FormType, EventID string
	FilingTime                                                   time.Time
	PublishTime                                                  *time.Time
	DocumentRefs                                                 []string
}

// FilingRows extracts the rows of the filing envelopes in envs; filings
// without an accession, form type or filing time are counted in invalid.
func FilingRows(envs []*fav1.Envelope) (rows []FilingRow, invalid int) {
	for _, e := range envs {
		f := e.GetFiling()
		if f == nil {
			continue
		}
		if e.GetSourceId() == "" || f.GetAccession() == "" || f.GetFormType() == "" || f.GetFilingTime() == nil {
			invalid++
			continue
		}
		r := FilingRow{SourceID: e.GetSourceId(), Accession: f.GetAccession(), IssuerCIK: f.GetIssuerCik(), IssuerLEI: f.GetIssuerLei(),
			FormType: f.GetFormType(), EventID: e.GetEventId(), FilingTime: f.GetFilingTime().AsTime().UTC(), DocumentRefs: f.GetDocumentRefs()}
		if r.DocumentRefs == nil {
			r.DocumentRefs = []string{}
		}
		if pt := e.GetPublishTime(); pt != nil {
			t := pt.AsTime().UTC()
			r.PublishTime = &t
		}
		rows = append(rows, r)
	}
	return rows, invalid
}

// InsertFilingSQL is idempotent on (source_id, accession).
const InsertFilingSQL = `insert into wh.filing (source_id, accession, issuer_cik, issuer_lei, form_type, filing_time, publish_time, document_refs, event_id)
values ($1, $2, nullif($3, ''), nullif($4, ''), $5, $6, $7, $8, $9)
on conflict (source_id, accession) do nothing`

// DB is the subset of *pgxpool.Pool the writer uses.
type DB interface {
	SendBatch(ctx context.Context, b *pgx.Batch) pgx.BatchResults
}

// Stats are cumulative writer counters.
type Stats struct {
	Inserted   uint64 // new rows
	Duplicates uint64 // rows that already existed
	Invalid    uint64 // macro envelopes that could not be written
	Errors     uint64 // failed batches
}

// Writer inserts macro observations synchronously, one batch per call.
type Writer struct {
	DB      DB
	Timeout time.Duration // per batch (default 10 s)

	inserted, duplicates, invalid, errors atomic.Uint64
}

// NewWriter builds a Writer over a pool (or any DB).
func NewWriter(db DB) *Writer { return &Writer{DB: db} }

// Write inserts the macro observations among envs and returns how many rows
// were new. Non-macro envelopes are ignored, so callers may pass everything a
// connector produced.
func (w *Writer) Write(ctx context.Context, envs []*fav1.Envelope) (int, error) {
	if w == nil || w.DB == nil {
		return 0, nil
	}
	rows, invalid := MacroRows(envs)
	filings, finvalid := FilingRows(envs)
	w.invalid.Add(uint64(invalid + finvalid))
	if len(rows) == 0 && len(filings) == 0 {
		return 0, nil
	}
	timeout := w.Timeout
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	b := &pgx.Batch{}
	labels := make([]string, 0, len(rows)+len(filings))
	for _, r := range rows {
		b.Queue(InsertMacroSQL, r.SourceID, r.SeriesID, r.Period, r.Vintage, r.Value, r.Revision)
		labels = append(labels, "macro "+r.SourceID+"/"+r.SeriesID+"/"+r.Period)
	}
	for _, f := range filings {
		b.Queue(InsertFilingSQL, f.SourceID, f.Accession, f.IssuerCIK, f.IssuerLEI, f.FormType, f.FilingTime, f.PublishTime, f.DocumentRefs, f.EventID)
		labels = append(labels, "filing "+f.SourceID+"/"+f.Accession)
	}
	br := w.DB.SendBatch(ctx, b)
	n := 0
	var firstErr error
	for i := range labels {
		tag, err := br.Exec()
		if err != nil {
			if firstErr == nil {
				firstErr = fmt.Errorf("warehouse: %s: %w", labels[i], err)
			}
			continue
		}
		if tag.RowsAffected() > 0 {
			n++
		} else {
			w.duplicates.Add(1)
		}
	}
	if err := br.Close(); err != nil && firstErr == nil {
		firstErr = fmt.Errorf("warehouse: %w", err)
	}
	w.inserted.Add(uint64(n))
	if firstErr != nil {
		w.errors.Add(1)
	}
	return n, firstErr
}

// Stats returns the cumulative counters.
func (w *Writer) Stats() Stats {
	return Stats{Inserted: w.inserted.Load(), Duplicates: w.duplicates.Load(), Invalid: w.invalid.Load(), Errors: w.errors.Load()}
}
