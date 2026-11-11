// Package refdata is the Postgres side of ingestion bookkeeping (specs 10, 12):
// a symbol resolver over ref.listing / ref.identifier that provisions
// unknown symbols with deterministic UUIDv5 ids, the catalog.source sync the
// ingest.* foreign keys need, connector run logging (ingest.connector_run) and
// progress cursors (ingest.backfill_cursor).
package refdata

import (
	"context"
	"crypto/sha1"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// Namespace is the UUIDv5 namespace for provisional instruments:
// uuid5(Namespace, "<source_id>\x00<venue_symbol>").
const Namespace = "7d3a2b1c-5e4f-5a6b-8c9d-0e1f2a3b4c5d"

// Hint tells the resolver what to create for an unknown symbol of a source.
type Hint struct {
	Venue      string // ref.venue.code, defaults to the source id
	VenueKind  string // exchange | broker | dex | otc
	VenueName  string
	AssetClass string
	Kind       string // ref.instrument.kind
	Scheme     string // ref.identifier.scheme for the venue symbol (e.g. "isin")
}

// DefaultHints cover the first-wave connectors.
var DefaultHints = map[string]Hint{
	"yahoo":     {Venue: "yahoo", VenueKind: "otc", VenueName: "Yahoo Finance (aggregated)", AssetClass: "equity", Kind: "equity", Scheme: "yahoo_symbol"},
	"lemon":     {Venue: "lemon", VenueKind: "broker", VenueName: "lemon.markets", AssetClass: "equity", Kind: "equity", Scheme: "isin"},
	"oanda":     {Venue: "oanda", VenueKind: "broker", VenueName: "OANDA", AssetClass: "fx", Kind: "fx_pair", Scheme: "oanda_instrument"},
	"kraken":    {Venue: "kraken", VenueKind: "exchange", VenueName: "Kraken", AssetClass: "crypto", Kind: "crypto_spot", Scheme: "kraken_symbol"},
	"heartbeat": {Venue: "sim", VenueKind: "otc", VenueName: "Simulated", AssetClass: "synthetic", Kind: "index", Scheme: "sim_symbol"},
}

// The other keyless sources (frankfurter, ecb, gdelt, sec_edgar, hackernews,
// googlenews) carry no venue (connector.Capabilities.Venue == ""): their
// normalizers never call the resolver and backfill skips symbol resolution
// for them, so they provision no instruments and need no hint.

// Store wraps a pgx pool.
type Store struct {
	Pool *pgxpool.Pool
}

// Open connects to Postgres (url like postgres://fa:fa@127.0.0.1:5432/fa) and pings.
func Open(ctx context.Context, url string) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		return nil, fmt.Errorf("refdata: %w", err)
	}
	cfg.MaxConns = 4
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("refdata: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("refdata: ping: %w", err)
	}
	return &Store{Pool: pool}, nil
}

func (s *Store) Close() { s.Pool.Close() }

// SyncCatalog upserts config/sources.yaml into catalog.source. Status is kept
// when the row exists (Postgres is authoritative at runtime, spec 11 §3).
func (s *Store) SyncCatalog(ctx context.Context, cat *catalog.Catalog) error {
	b := &pgx.Batch{}
	for _, src := range cat.Sources {
		b.Queue(`insert into catalog.source (id, name, category, capabilities, asset_classes, transport, auth, status, tos_notes, notes)
			values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
			on conflict (id) do update set name = excluded.name, category = excluded.category, capabilities = excluded.capabilities,
			  asset_classes = excluded.asset_classes, transport = excluded.transport, auth = excluded.auth,
			  tos_notes = excluded.tos_notes, notes = excluded.notes, updated_at = now()`,
			src.ID, src.Name, orDefault(src.Category, "uncategorized"), nz(src.Capabilities), nz(src.AssetClasses), nz(src.Transport), src.Auth, src.Status, src.TosNotes, src.Notes)
	}
	res := s.Pool.SendBatch(ctx, b)
	defer res.Close()
	for range cat.Sources {
		if _, err := res.Exec(); err != nil {
			return fmt.Errorf("refdata: sync catalog: %w", err)
		}
	}
	return nil
}

// Statuses returns catalog.source.status per id. After SyncCatalog the
// ingestor applies them to its in-memory catalog: the YAML status is only the
// seed, the Postgres status wins (spec 11 §8, D-054).
func (s *Store) Statuses(ctx context.Context) (map[string]string, error) {
	rows, err := s.Pool.Query(ctx, `select id, status from catalog.source`)
	if err != nil {
		return nil, fmt.Errorf("refdata: statuses: %w", err)
	}
	defer rows.Close()
	out := map[string]string{}
	for rows.Next() {
		var id, st string
		if err := rows.Scan(&id, &st); err != nil {
			return nil, err
		}
		out[id] = st
	}
	return out, rows.Err()
}

// SetStatus persists a source status (ctl.ingest.enable|disable).
func (s *Store) SetStatus(ctx context.Context, id, status string) error {
	tag, err := s.Pool.Exec(ctx, `update catalog.source set status = $2, updated_at = now() where id = $1`, id, status)
	if err != nil {
		return fmt.Errorf("refdata: set status: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("refdata: set status: source %q not in catalog.source", id)
	}
	return nil
}

func nz(a []string) []string {
	if a == nil {
		return []string{}
	}
	return a
}

func orDefault(s, d string) string {
	if s == "" {
		return d
	}
	return s
}

// UUIDv5 computes RFC 4122 version-5 UUID of name under namespace (both as
// canonical strings). Used for provisional instruments so the same symbol
// always maps to the same id on every machine.
func UUIDv5(namespace, name string) string {
	ns := strings.ReplaceAll(namespace, "-", "")
	nsb := make([]byte, 16)
	for i := 0; i < 16; i++ {
		var v byte
		fmt.Sscanf(ns[2*i:2*i+2], "%02x", &v)
		nsb[i] = v
	}
	h := sha1.New()
	h.Write(nsb)
	h.Write([]byte(name))
	sum := h.Sum(nil)[:16]
	sum[6] = (sum[6] & 0x0f) | 0x50
	sum[8] = (sum[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", sum[0:4], sum[4:6], sum[6:8], sum[8:10], sum[10:16])
}

// ProvisionalID is the instrument id a source/symbol pair gets when unknown.
func ProvisionalID(sourceID, venueSymbol string) string {
	return UUIDv5(Namespace, sourceID+"\x00"+venueSymbol)
}

// Resolver implements connector.SymbolResolver on Postgres with a cache.
// Unknown (source, symbol) pairs are provisioned: a ref.venue row for the
// source's venue code if missing, a ref.instrument with status 'provisional'
// and the deterministic UUIDv5 id, a ref.listing and a ref.identifier row,
// plus an ingest.quarantine row (status 'open') so the UI's manual queue shows
// the auto-created instrument for review (spec 12 §6 proposal variant).
type Resolver struct {
	store   *Store
	hints   map[string]Hint
	timeout time.Duration
	onErr   func(error)

	// OnProvision, when set, is called after an unknown pair was provisioned
	// (the ingestor publishes `ingest.unresolved.<source>` from it, D-054).
	OnProvision func(sourceID, venueSymbol, instrumentID string)

	// CacheTTL bounds how long a mapping is served from memory (spec 12 §7
	// resolver.cache_ttl, default 1 h); Invalidate drops everything at once
	// (the ingestor calls it on `ref.changed.*`, e.g. after a quarantine merge
	// or a symbol change).
	CacheTTL time.Duration
	now      func() time.Time

	mu    sync.RWMutex
	cache map[string]cached
	// Quarantined counts provisional creations per source (for counters/tests).
	created map[string]int
}

// NewResolver builds a resolver; hints default to DefaultHints and onErr may be nil.
func NewResolver(store *Store, hints map[string]Hint, onErr func(error)) *Resolver {
	if hints == nil {
		hints = DefaultHints
	}
	return &Resolver{store: store, hints: hints, timeout: 5 * time.Second, onErr: onErr, cache: map[string]cached{}, created: map[string]int{},
		CacheTTL: time.Hour, now: time.Now}
}

type cached struct {
	id string
	at time.Time
}

// Invalidate drops every cached mapping (reference data changed).
func (r *Resolver) Invalidate() {
	r.mu.Lock()
	r.cache = map[string]cached{}
	r.mu.Unlock()
}

func (r *Resolver) cachedID(key string) (string, bool) {
	r.mu.RLock()
	c, ok := r.cache[key]
	r.mu.RUnlock()
	if !ok || (r.CacheTTL > 0 && r.now().Sub(c.at) > r.CacheTTL) {
		return "", false
	}
	return c.id, true
}

// Resolve maps (source, symbol) → instrument UUID. It never blocks the hot
// path for long: lookups are cached; database errors are reported through
// onErr and yield ok=false (the event goes to the connector's unresolved list).
func (r *Resolver) Resolve(sourceID, venueSymbol string) (string, bool) {
	key := sourceID + "\x00" + venueSymbol
	if id, ok := r.cachedID(key); ok {
		return id, true
	}
	ctx, cancel := context.WithTimeout(context.Background(), r.timeout)
	defer cancel()
	id, created, err := r.resolveDB(ctx, sourceID, venueSymbol)
	if err != nil {
		if r.onErr != nil {
			r.onErr(fmt.Errorf("refdata: resolve %s/%s: %w", sourceID, venueSymbol, err))
		}
		return "", false
	}
	r.mu.Lock()
	r.cache[key] = cached{id: id, at: r.now()}
	if created {
		r.created[sourceID]++
	}
	r.mu.Unlock()
	if created && r.OnProvision != nil {
		r.OnProvision(sourceID, venueSymbol, id)
	}
	return id, true
}

// Created returns how many provisional instruments were created per source.
func (r *Resolver) Created(sourceID string) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.created[sourceID]
}

// Lookup finds an existing mapping without provisioning.
func (r *Resolver) Lookup(ctx context.Context, sourceID, venueSymbol string) (string, bool, error) {
	var id string
	err := r.store.Pool.QueryRow(ctx, `
		select l.instrument_id::text from ref.listing l
		where l.venue_symbol = $2 and now() >= l.valid_from and now() < l.valid_to
		  and (l.source_id = $1 or l.venue_id = (select id from ref.venue where code = $3))
		order by (l.source_id = $1) desc limit 1`, sourceID, venueSymbol, r.hint(sourceID).Venue).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		h := r.hint(sourceID)
		err = r.store.Pool.QueryRow(ctx, `select instrument_id::text from ref.identifier
			where scheme = $1 and value = $2 and now() >= valid_from and now() < valid_to limit 1`, h.Scheme, venueSymbol).Scan(&id)
	}
	if errors.Is(err, pgx.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return id, true, nil
}

func (r *Resolver) hint(sourceID string) Hint {
	h, ok := r.hints[sourceID]
	if !ok {
		h = Hint{AssetClass: "unknown", Kind: "unknown"}
	}
	if h.Venue == "" {
		h.Venue = sourceID
	}
	if h.VenueKind == "" {
		h.VenueKind = "otc"
	}
	if h.VenueName == "" {
		h.VenueName = sourceID
	}
	if h.Scheme == "" {
		h.Scheme = sourceID + "_symbol"
	}
	return h
}

func (r *Resolver) resolveDB(ctx context.Context, sourceID, venueSymbol string) (string, bool, error) {
	if id, ok, err := r.Lookup(ctx, sourceID, venueSymbol); err != nil || ok {
		return id, false, err
	}
	h := r.hint(sourceID)
	id := ProvisionalID(sourceID, venueSymbol)
	venueID := UUIDv5(Namespace, "venue\x00"+h.Venue)
	tx, err := r.store.Pool.Begin(ctx)
	if err != nil {
		return "", false, err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `insert into ref.venue (id, code, name, timezone, kind, always_open) values ($1,$2,$3,'UTC',$4,$5)
		on conflict (code) do nothing`, venueID, h.Venue, h.VenueName, h.VenueKind, h.AssetClass == "crypto"); err != nil {
		return "", false, fmt.Errorf("venue: %w", err)
	}
	if err := tx.QueryRow(ctx, `select id from ref.venue where code = $1`, h.Venue).Scan(&venueID); err != nil {
		return "", false, err
	}
	cl := Classify(sourceID, venueSymbol, h)
	if _, err := tx.Exec(ctx, `insert into ref.instrument (id, asset_class, kind, name, status, base_ccy, quote_ccy) values ($1,$2,$3,$4,'provisional',$5,$6)
		on conflict (id) do nothing`, id, cl.AssetClass, cl.Kind, venueSymbol+" ("+sourceID+")", nullable(cl.Base), nullable(cl.Quote)); err != nil {
		return "", false, fmt.Errorf("instrument: %w", err)
	}
	if _, err := tx.Exec(ctx, `insert into ref.listing (instrument_id, venue_id, venue_symbol, source_id) values ($1,$2,$3,$4)
		on conflict do nothing`, id, venueID, venueSymbol, sourceID); err != nil {
		return "", false, fmt.Errorf("listing: %w", err)
	}
	if _, err := tx.Exec(ctx, `insert into ref.identifier (instrument_id, scheme, value) values ($1,$2,$3) on conflict do nothing`, id, h.Scheme, venueSymbol); err != nil {
		return "", false, fmt.Errorf("identifier: %w", err)
	}
	if _, err := tx.Exec(ctx, `insert into ingest.quarantine (source_id, venue_symbol, sample_raw_ref, status, instrument_id) values ($1,$2,$3,'open',$4)
		on conflict (source_id, venue_symbol) do update set last_seen = now(), count = ingest.quarantine.count + 1,
		  instrument_id = coalesce(ingest.quarantine.instrument_id, excluded.instrument_id)`,
		sourceID, venueSymbol, "provisional instrument "+id+" auto-created by ingestor", id); err != nil {
		return "", false, fmt.Errorf("quarantine: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return "", false, err
	}
	return id, true, nil
}

// Class is what Classify infers from a venue symbol.
type Class struct {
	AssetClass, Kind string
	Base, Quote      string // currencies, empty when unknown
}

var (
	yahooFX     = regexp.MustCompile(`^([A-Z]{3})([A-Z]{3})=X$`)
	yahooCrypto = regexp.MustCompile(`^([A-Z0-9]{2,10})-(USD|EUR|USDT|USDC|GBP|BTC|ETH)$`)
	yahooPlain  = regexp.MustCompile(`^[A-Z.]{1,6}$`)
	oandaPair   = regexp.MustCompile(`^([A-Z]{3})_([A-Z]{3})$`)
	slashPair   = regexp.MustCompile(`^([A-Z0-9]{2,10})/([A-Z0-9]{2,10})$`)
	// krakenLegacy is Kraken's X/Z-prefixed pair name: XXBTZEUR, XETHZUSD, XETHXXBT.
	krakenLegacy = regexp.MustCompile(`^[XZ]([A-Z0-9]{3})[XZ]([A-Z0-9]{3})$`)
	krakenBase   = regexp.MustCompile(`^[A-Z0-9]{2,10}$`)
)

// krakenQuotes are the quote assets of Kraken's altname pairs (XBTEUR, SOLUSDT),
// longest first so USDT wins over USD.
var krakenQuotes = []string{"USDT", "USDC", "EUR", "USD", "GBP", "JPY", "CAD", "CHF", "AUD", "XBT", "ETH", "DAI"}

// krakenAsset maps Kraken's own asset codes to the common ones.
var krakenAsset = map[string]string{"XBT": "BTC", "XDG": "DOGE"}

// fiat is the set of ISO currencies a Kraken pair can be quoted in; a pair of two of
// them is an fx pair, not crypto.
var fiat = map[string]bool{"EUR": true, "USD": true, "GBP": true, "JPY": true, "CAD": true, "CHF": true, "AUD": true}

// KrakenPair splits a Kraken pair name (altname `XBTEUR`, legacy `XXBTZEUR`,
// wsname `XBT/EUR`) into base and quote with Kraken's asset codes normalised
// (XBT → BTC, XDG → DOGE); ok is false for an unrecognised shape.
func KrakenPair(symbol string) (base, quote string, ok bool) {
	s := strings.ToUpper(strings.TrimSpace(symbol))
	if m := slashPair.FindStringSubmatch(s); m != nil {
		base, quote = m[1], m[2]
	} else if m := krakenLegacy.FindStringSubmatch(s); m != nil {
		base, quote = m[1], m[2]
	} else {
		for _, q := range krakenQuotes {
			if b, found := strings.CutSuffix(s, q); found && krakenBase.MatchString(b) {
				base, quote = b, q
				break
			}
		}
	}
	if base == "" {
		return "", "", false
	}
	if a, ok := krakenAsset[base]; ok {
		base = a
	}
	if a, ok := krakenAsset[quote]; ok {
		quote = a
	}
	return base, quote, true
}

// yahooSuffixCcy maps Yahoo exchange suffixes to the listing currency.
var yahooSuffixCcy = map[string]string{".DE": "EUR", ".F": "EUR", ".PA": "EUR", ".AS": "EUR", ".MI": "EUR", ".MC": "EUR", ".L": "GBP", ".SW": "CHF", ".TO": "CAD", ".T": "JPY", ".HK": "HKD"}

// Classify infers asset class, kind and currencies from a source's symbol
// syntax (spec 12 §6) so provisional instruments get the right risk caps
// (D-061) before an operator or a global identifier confirms them. Unknown
// shapes fall back to the source hint.
func Classify(sourceID, symbol string, h Hint) Class {
	c := Class{AssetClass: h.AssetClass, Kind: h.Kind}
	switch sourceID {
	case "yahoo":
		switch {
		case yahooFX.MatchString(symbol):
			m := yahooFX.FindStringSubmatch(symbol)
			return Class{"fx", "fx_pair", m[1], m[2]}
		case yahooCrypto.MatchString(symbol):
			m := yahooCrypto.FindStringSubmatch(symbol)
			return Class{"crypto", "crypto_spot", m[1], m[2]}
		case strings.HasPrefix(symbol, "^"):
			return Class{AssetClass: "index", Kind: "index"}
		case strings.HasSuffix(symbol, "=F"):
			return Class{AssetClass: "derivative", Kind: "future", Quote: "USD"}
		}
		for suf, ccy := range yahooSuffixCcy {
			if strings.HasSuffix(symbol, suf) {
				c.Quote = ccy
				return c
			}
		}
		if yahooPlain.MatchString(symbol) {
			c.Quote = "USD"
		}
	case "kraken":
		if b, q, ok := KrakenPair(symbol); ok {
			if fiat[b] && fiat[q] {
				return Class{"fx", "fx_pair", b, q}
			}
			return Class{"crypto", "crypto_spot", b, q}
		}
	case "oanda":
		if m := oandaPair.FindStringSubmatch(symbol); m != nil {
			return Class{"fx", "fx_pair", m[1], m[2]}
		}
	default:
		if m := slashPair.FindStringSubmatch(symbol); m != nil && (h.AssetClass == "crypto" || h.AssetClass == "fx") {
			kind := "crypto_spot"
			if h.AssetClass == "fx" {
				kind = "fx_pair"
			}
			return Class{h.AssetClass, kind, m[1], m[2]}
		}
	}
	return c
}

// Cursors implements connector.CursorStore on ingest.backfill_cursor. A live
// cursor row is keyed (source, instrument|null, timescale=scope,
// range_start=epoch); range_end and cursor both hold the progress time.
type Cursors struct{ store *Store }

func NewCursors(s *Store) *Cursors { return &Cursors{store: s} }

var epoch = time.Unix(0, 0).UTC()

func (c *Cursors) GetCursor(ctx context.Context, sourceID, instrumentID, scope string) (time.Time, bool, error) {
	var t *time.Time
	err := c.store.Pool.QueryRow(ctx, `select cursor from ingest.backfill_cursor
		where source_id = $1 and instrument_id is not distinct from $2::uuid and timescale = $3 and range_start = $4`,
		sourceID, nullable(instrumentID), scope, epoch).Scan(&t)
	if errors.Is(err, pgx.ErrNoRows) || (err == nil && t == nil) {
		return time.Time{}, false, nil
	}
	if err != nil {
		return time.Time{}, false, err
	}
	return t.UTC(), true, nil
}

func (c *Cursors) SetCursor(ctx context.Context, sourceID, instrumentID, scope string, cursor time.Time) error {
	id := UUIDv5(Namespace, "cursor\x00"+sourceID+"\x00"+instrumentID+"\x00"+scope)
	_, err := c.store.Pool.Exec(ctx, `insert into ingest.backfill_cursor (id, source_id, instrument_id, timescale, range_start, range_end, cursor, status)
		values ($1,$2,$3::uuid,$4,$5,$6,$6,'running')
		on conflict (source_id, instrument_id, timescale, range_start) do update
		  set range_end = greatest(ingest.backfill_cursor.range_end, excluded.range_end), cursor = excluded.cursor, updated_at = now()`,
		id, sourceID, nullable(instrumentID), scope, epoch, cursor.UTC())
	return err
}

func nullable(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// CursorRow is one ingest.backfill_cursor row (for ctl.ingest.status / the UI).
type CursorRow struct {
	SourceID     string     `json:"source_id"`
	InstrumentID string     `json:"instrument_id,omitempty"`
	Timescale    string     `json:"timescale"`
	RangeStart   time.Time  `json:"range_start"`
	RangeEnd     time.Time  `json:"range_end"`
	Cursor       *time.Time `json:"cursor,omitempty"`
	Status       string     `json:"status"`
	UpdatedAt    time.Time  `json:"updated_at"`
	// Live is true for the open-ended watermark rows (range_start = epoch).
	Live bool `json:"live"`
}

// ListCursors returns a source's cursor rows, newest first (all sources when sourceID is "").
func (s *Store) ListCursors(ctx context.Context, sourceID string, limit int) ([]CursorRow, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := s.Pool.Query(ctx, `select source_id, coalesce(instrument_id::text, ''), timescale, range_start, range_end, cursor, status, updated_at
		from ingest.backfill_cursor where ($1 = '' or source_id = $1) order by updated_at desc limit $2`, sourceID, limit)
	if err != nil {
		return nil, fmt.Errorf("refdata: cursors: %w", err)
	}
	defer rows.Close()
	var out []CursorRow
	for rows.Next() {
		var r CursorRow
		if err := rows.Scan(&r.SourceID, &r.InstrumentID, &r.Timescale, &r.RangeStart, &r.RangeEnd, &r.Cursor, &r.Status, &r.UpdatedAt); err != nil {
			return nil, err
		}
		r.Live = r.RangeStart.Equal(epoch)
		out = append(out, r)
	}
	return out, rows.Err()
}

// BackfillCursors implements connector.BackfillCursorStore: one row per
// bounded range (source, instrument|null, timescale, range_start), so a
// redelivered or restarted job resumes at `cursor` and a finished one is a no-op.
type BackfillCursors struct{ store *Store }

func NewBackfillCursors(s *Store) *BackfillCursors { return &BackfillCursors{store: s} }

func (c *BackfillCursors) GetBackfill(ctx context.Context, sourceID, instrumentID, timescale string, rangeStart time.Time) (time.Time, string, bool, error) {
	var cur *time.Time
	var status string
	err := c.store.Pool.QueryRow(ctx, `select cursor, status from ingest.backfill_cursor
		where source_id = $1 and instrument_id is not distinct from $2::uuid and timescale = $3 and range_start = $4`,
		sourceID, nullable(instrumentID), timescale, rangeStart.UTC()).Scan(&cur, &status)
	if errors.Is(err, pgx.ErrNoRows) {
		return time.Time{}, "", false, nil
	}
	if err != nil {
		return time.Time{}, "", false, err
	}
	if cur == nil {
		return time.Time{}, status, true, nil
	}
	return cur.UTC(), status, true, nil
}

func (c *BackfillCursors) SetBackfill(ctx context.Context, sourceID, instrumentID, timescale string, rangeStart, rangeEnd, cursor time.Time, status string) error {
	id := UUIDv5(Namespace, "backfill\x00"+sourceID+"\x00"+instrumentID+"\x00"+timescale+"\x00"+rangeStart.UTC().Format(time.RFC3339Nano))
	var cur *time.Time
	if !cursor.IsZero() {
		u := cursor.UTC()
		cur = &u
	}
	_, err := c.store.Pool.Exec(ctx, `insert into ingest.backfill_cursor (id, source_id, instrument_id, timescale, range_start, range_end, cursor, status)
		values ($1,$2,$3::uuid,$4,$5,$6,$7,$8)
		on conflict (source_id, instrument_id, timescale, range_start) do update
		  set range_end = excluded.range_end, cursor = coalesce(excluded.cursor, ingest.backfill_cursor.cursor), status = excluded.status, updated_at = now()`,
		id, sourceID, nullable(instrumentID), timescale, rangeStart.UTC(), rangeEnd.UTC(), cur, status)
	return err
}

// RunLog records connector lifecycles in ingest.connector_run.
type RunLog struct{ store *Store }

func NewRunLog(s *Store) *RunLog { return &RunLog{store: s} }

// Start inserts a 'running' row and returns its id.
func (l *RunLog) Start(ctx context.Context, sourceID string) (string, error) {
	id := UUIDv5(Namespace, fmt.Sprintf("run\x00%s\x00%d", sourceID, time.Now().UnixNano()))
	_, err := l.store.Pool.Exec(ctx, `insert into ingest.connector_run (id, source_id, status) values ($1,$2,'running')`, id, sourceID)
	return id, err
}

// End closes the row with final counters.
func (l *RunLog) End(ctx context.Context, runID, status string, events, errs int64, lastErr string) error {
	_, err := l.store.Pool.Exec(ctx, `update ingest.connector_run set ended_at = now(), status = $2, events = $3, errors = $4, last_error = nullif($5,'') where id = $1`,
		runID, status, events, errs, lastErr)
	return err
}

var _ connector.SymbolResolver = (*Resolver)(nil)
var _ connector.CursorStore = (*Cursors)(nil)
var _ connector.BackfillCursorStore = (*BackfillCursors)(nil)
