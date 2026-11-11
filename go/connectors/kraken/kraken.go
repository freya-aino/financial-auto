// Package kraken is the Kraken public market-data connector for catalog id "kraken" (specs 10, 11).
//
// It polls the keyless public REST API once a minute per pair in the catalog
// entry's `config.symbols` (Kraken pair names such as XBTEUR, ETHEUR, SOLEUR):
//
//	GET https://api.kraken.com/0/public/OHLC?pair=<p>&interval=<min>  → Bar   (md.bar.<tf>.kraken.<inst>)
//	GET https://api.kraken.com/0/public/Trades?pair=<p>&since=<ns>    → Trade (md.trade.kraken.<inst>)
//	GET https://api.kraken.com/0/public/Ticker?pair=<p>               → Quote (md.quote.kraken.<inst>)
//
// Kraken answers with its own result key (XBTEUR → XXBTZEUR, ETHEUR →
// XETHZEUR, SOLEUR → SOLEUR); the connector maps it back to the configured
// symbol through TransportMeta, so resolution always uses the catalog symbol.
// Bars are stamped at their close (event_time = open + tf); the last OHLC row
// is still forming and is never emitted live. Trades carry Kraken's trade_id
// as venue_trade_id and the aggressor side (b/s). Quotes take the ticker's
// best bid/ask with their lot volumes; the ticker has no timestamp, so
// event_time = ingest_time. publish_time = ingest_time for every event
// (Kraken gives no availability time, spec 10 §6.2).
//
// De-duplication (live): bars per symbol by open time, persisted as an
// ingest.backfill_cursor row (instrument, scope = tf, cursor = bar close);
// trades per symbol by trade_id, with the Trades `since` cursor kept in memory
// and the newest emitted trade time persisted (instrument, scope "trades").
// Without a trade cursor the first poll asks for the last poll interval only.
//
// Backfill re-reads OHLC with `since`; Kraken only ever returns the latest 720
// intervals (12 h of 1m bars, 30 d of 1h, ~2 y of 1d), so older ranges come
// back empty. Rows are filtered to open time in [From, To) and only completed
// bars are emitted.
//
// Verified live 2026-11-11: OHLC, Trades and Ticker all 200 without a key;
// unknown pairs answer 200 with {"error":["EQuery:Unknown asset pair"]}.
package kraken

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const (
	ID             = "kraken"
	Venue          = "kraken"
	DefaultBaseURL = "https://api.kraken.com"
	tradesScope    = "trades"
	// MaxOHLCRows is how many intervals Kraken returns at most.
	MaxOHLCRows = 720
)

// DefaultSymbols apply when the catalog entry has no config.symbols.
var DefaultSymbols = []string{"XBTEUR"}

// Intervals maps canonical timeframes to Kraken's interval (minutes).
var Intervals = map[string]int{"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector implements Connector, Poller, Backfiller and Normalizer.
type Connector struct {
	BaseURL string
	Fetch   *connector.Fetcher

	cfg     connector.SourceConfig
	deps    connector.Deps
	symbols []string
	tf      string

	mu        sync.Mutex
	lastBar   map[string]int64  // symbol → open time (unix) of the last emitted live bar
	lastTrade map[string]int64  // symbol → highest emitted trade_id
	since     map[string]string // symbol → Trades `since` cursor (ns) for the next poll
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Backfill: true, Venue: Venue,
		Events: []connector.EventKind{connector.KindBar, connector.KindTrade, connector.KindQuote}}
}

func (c *Connector) Init(ctx context.Context, cfg connector.SourceConfig, deps connector.Deps) error {
	if deps.Clock == nil {
		deps.Clock = connector.SystemClock{}
	}
	c.cfg, c.deps = cfg, deps
	if c.BaseURL == "" {
		c.BaseURL = DefaultBaseURL
	}
	if c.Fetch == nil {
		c.Fetch = connector.NewFetcher(deps.Limiter)
	}
	p := cfg.Source.Params()
	c.symbols = p.Symbols
	if len(c.symbols) == 0 {
		c.symbols = DefaultSymbols
	}
	c.tf = p.Timeframe
	if c.tf == "" {
		c.tf = "1m"
	}
	if _, ok := Intervals[c.tf]; !ok {
		return fmt.Errorf("kraken: unsupported timeframe %q", c.tf)
	}
	c.lastBar, c.lastTrade, c.since = map[string]int64{}, map[string]int64{}, map[string]string{}
	if deps.Cursors != nil {
		dur := tfDur(c.tf)
		for _, s := range c.symbols {
			inst := c.instrument(s)
			if inst == "" {
				continue
			}
			if t, ok, err := deps.Cursors.GetCursor(ctx, ID, inst, c.tf); err == nil && ok {
				c.lastBar[s] = t.Add(-dur).Unix() // cursor is bar close
			}
			if t, ok, err := deps.Cursors.GetCursor(ctx, ID, inst, tradesScope); err == nil && ok {
				c.since[s] = strconv.FormatInt(t.UnixNano(), 10)
			}
		}
	}
	return nil
}

func tfDur(tf string) time.Duration { return time.Duration(Intervals[tf]) * time.Minute }

func (c *Connector) instrument(symbol string) string {
	if c.deps.Resolver == nil {
		return ""
	}
	id, _ := c.deps.Resolver.Resolve(ID, symbol)
	return id
}

func (c *Connector) Schedule() connector.PollSchedule {
	return connector.PollSchedule{Interval: time.Minute}
}

// Poll fetches OHLC, Trades and Ticker for every configured pair.
func (c *Connector) Poll(ctx context.Context, _ connector.PollRequest) ([]connector.RawMessage, error) {
	var out []connector.RawMessage
	var firstErr error
	note := func(err error) {
		if firstErr == nil {
			firstErr = err
		}
	}
	for _, s := range c.symbols {
		pair := url.QueryEscape(s)
		reqs := []struct{ endpoint, u string }{
			{"ohlc", fmt.Sprintf("%s/0/public/OHLC?pair=%s&interval=%d", c.BaseURL, pair, Intervals[c.tf])},
			{"trades", fmt.Sprintf("%s/0/public/Trades?pair=%s&since=%s", c.BaseURL, pair, c.tradeSince(s))},
			{"ticker", fmt.Sprintf("%s/0/public/Ticker?pair=%s", c.BaseURL, pair)},
		}
		for _, r := range reqs {
			b, err := c.Fetch.Get(ctx, r.u)
			if err != nil {
				if ctx.Err() != nil {
					return out, ctx.Err()
				}
				note(fmt.Errorf("kraken %s %s: %w", r.endpoint, s, err))
				continue
			}
			if r.endpoint == "trades" {
				c.advanceSince(s, b)
			}
			out = append(out, c.raw(s, r.endpoint, r.u, b, false))
		}
	}
	if len(out) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// tradeSince is the Trades cursor for symbol: the last response's `last`, the
// persisted cursor, or (first poll) one poll interval back.
func (c *Connector) tradeSince(symbol string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if s, ok := c.since[symbol]; ok {
		return s
	}
	return strconv.FormatInt(c.deps.Clock.Now().Add(-c.Schedule().Interval).UnixNano(), 10)
}

func (c *Connector) advanceSince(symbol string, b []byte) {
	var r struct {
		Error  []string `json:"error"`
		Result struct {
			Last json.RawMessage `json:"last"`
		} `json:"result"`
	}
	if json.Unmarshal(b, &r) != nil || len(r.Error) > 0 || len(r.Result.Last) == 0 {
		return
	}
	last := strings.Trim(string(r.Result.Last), `"`)
	if last == "" || last == "0" {
		return
	}
	c.mu.Lock()
	c.since[symbol] = last
	c.mu.Unlock()
}

func (c *Connector) raw(symbol, endpoint, u string, b []byte, backfill bool) connector.RawMessage {
	meta := map[string]string{connector.MetaSymbol: symbol, connector.MetaEndpoint: endpoint, connector.MetaURL: u}
	if endpoint == "ohlc" {
		meta[connector.MetaTf] = c.tf
	}
	if backfill {
		meta[connector.MetaBackfill] = "1"
	}
	return connector.RawMessage{SourceID: ID, ReceivedAt: c.deps.Clock.Now(), TransportMeta: meta, Bytes: b, ContentType: "application/json"}
}

// Backfill reads OHLC since From; Kraken returns at most the latest 720
// intervals, so the reachable depth is 720 × tf before now.
func (c *Connector) Backfill(ctx context.Context, req connector.BackfillRequest, out chan<- connector.RawMessage) error {
	tf := req.Timeframe
	if tf == "" {
		tf = c.tf
	}
	iv, ok := Intervals[tf]
	if !ok {
		return fmt.Errorf("kraken: unsupported timeframe %q", tf)
	}
	sym := req.VenueSymbol
	if sym == "" {
		return fmt.Errorf("kraken: backfill needs a pair")
	}
	from, to := req.From, req.To
	if cur, err := time.Parse(time.RFC3339, req.Cursor); req.Cursor != "" && err == nil && cur.After(from) {
		from = cur
	}
	if to.IsZero() {
		to = c.deps.Clock.Now()
	}
	if !from.Before(to) {
		return nil
	}
	u := fmt.Sprintf("%s/0/public/OHLC?pair=%s&interval=%d&since=%d", c.BaseURL, url.QueryEscape(sym), iv, from.Unix()-1)
	b, err := c.Fetch.Get(ctx, u)
	if err != nil {
		return fmt.Errorf("kraken backfill %s: %w", sym, err)
	}
	m := c.raw(sym, "ohlc", u, b, true)
	m.TransportMeta[connector.MetaTf] = tf
	m.TransportMeta["from"] = strconv.FormatInt(from.Unix(), 10)
	m.TransportMeta["to"] = strconv.FormatInt(to.Unix(), 10)
	select {
	case out <- m:
	case <-ctx.Done():
		return ctx.Err()
	}
	return nil
}

// response is Kraken's envelope; result holds one pair key (plus "last").
type response struct {
	Error  []string                   `json:"error"`
	Result map[string]json.RawMessage `json:"result"`
}

// pairData returns the single pair entry of a result (any key but "last").
func (r response) pairData() (json.RawMessage, bool) {
	keys := make([]string, 0, len(r.Result))
	for k := range r.Result {
		if k != "last" {
			keys = append(keys, k)
		}
	}
	if len(keys) == 0 {
		return nil, false
	}
	sort.Strings(keys)
	return r.Result[keys[0]], true
}

func num(r json.RawMessage) (float64, error) {
	s := strings.Trim(string(r), `"`)
	return strconv.ParseFloat(s, 64)
}

// Normalize maps one OHLC, Trades or Ticker payload to envelopes.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var resp response
	if err := json.Unmarshal(raw.Bytes, &resp); err != nil {
		return nil, nil, fmt.Errorf("kraken: decode: %w", err)
	}
	if len(resp.Error) > 0 {
		return nil, nil, fmt.Errorf("kraken: %s", strings.Join(resp.Error, "; "))
	}
	data, ok := resp.pairData()
	if !ok {
		return nil, nil, nil
	}
	symbol := raw.TransportMeta[connector.MetaSymbol]
	inst, ok := c.resolve(symbol)
	if !ok {
		return nil, []connector.Unresolved{{SourceID: ID, VenueSymbol: symbol, Reason: "no instrument for kraken pair"}}, nil
	}
	switch ep := raw.TransportMeta[connector.MetaEndpoint]; ep {
	case "ohlc":
		return c.normOHLC(raw, symbol, inst, data)
	case "trades":
		return c.normTrades(raw, symbol, inst, data)
	case "ticker":
		return c.normTicker(raw, inst, data)
	default:
		return nil, nil, fmt.Errorf("kraken: unknown endpoint %q", ep)
	}
}

func (c *Connector) resolve(symbol string) (string, bool) {
	if c.deps.Resolver == nil || symbol == "" {
		return "", false
	}
	return c.deps.Resolver.Resolve(ID, symbol)
}

func (c *Connector) normOHLC(raw connector.RawMessage, symbol, inst string, data json.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var rows [][]json.RawMessage
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, nil, fmt.Errorf("kraken: decode ohlc: %w", err)
	}
	tf := raw.TransportMeta[connector.MetaTf]
	if tf == "" {
		tf = c.tf
	}
	if _, ok := Intervals[tf]; !ok {
		return nil, nil, fmt.Errorf("kraken: unsupported timeframe %q", tf)
	}
	dur := tfDur(tf)
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"
	var from, to int64 = 0, 1<<62 - 1
	if v, err := strconv.ParseInt(raw.TransportMeta["from"], 10, 64); err == nil {
		from = v
	}
	if v, err := strconv.ParseInt(raw.TransportMeta["to"], 10, 64); err == nil {
		to = v
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	last := c.lastBar[symbol]
	var envs []*fav1.Envelope
	var maxTs int64
	for i, row := range rows {
		if len(row) < 8 {
			continue
		}
		ts, err := strconv.ParseInt(strings.Trim(string(row[0]), `"`), 10, 64)
		if err != nil {
			continue
		}
		var v [6]float64
		bad := false
		for j := range v {
			if v[j], err = num(row[j+1]); err != nil {
				bad = true
			}
		}
		count, err := strconv.ParseUint(strings.Trim(string(row[7]), `"`), 10, 64)
		if bad || err != nil {
			continue
		}
		closeAt := time.Unix(ts, 0).UTC().Add(dur)
		if closeAt.After(raw.ReceivedAt) {
			continue // still forming
		}
		if backfill {
			if ts < from || ts >= to {
				continue
			}
		} else if ts <= last || i == len(rows)-1 {
			continue // already emitted, or Kraken's last (unconfirmed) row
		}
		vwap := v[4]
		bar := &fav1.Bar{Timeframe: tf, Open: v[0], High: v[1], Low: v[2], Close: v[3], Vwap: &vwap, Volume: v[5], TradeCount: count}
		e := envelope(inst, closeAt, raw.ReceivedAt)
		e.Payload = &fav1.Envelope_Bar{Bar: bar}
		envs = append(envs, e)
		if ts > maxTs {
			maxTs = ts
		}
	}
	if !backfill && maxTs > last {
		c.lastBar[symbol] = maxTs
		c.setCursor(inst, tf, time.Unix(maxTs, 0).UTC().Add(dur))
	}
	return envs, nil, nil
}

func (c *Connector) normTrades(raw connector.RawMessage, symbol, inst string, data json.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var rows [][]json.RawMessage
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, nil, fmt.Errorf("kraken: decode trades: %w", err)
	}
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"
	c.mu.Lock()
	defer c.mu.Unlock()
	lastID := c.lastTrade[symbol]
	var envs []*fav1.Envelope
	var maxID int64
	var newest time.Time
	for _, row := range rows {
		if len(row) < 4 {
			continue
		}
		price, err1 := num(row[0])
		size, err2 := num(row[1])
		at, err3 := decimalTime(string(row[2]))
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}
		var tid int64
		if len(row) >= 7 {
			tid, _ = strconv.ParseInt(strings.Trim(string(row[6]), `"`), 10, 64)
		}
		vid := strconv.FormatInt(tid, 10)
		if tid == 0 {
			vid = fmt.Sprintf("%s:%d", symbol, at.UnixNano())
		} else if !backfill && tid <= lastID {
			continue
		}
		side := fav1.Side_SIDE_UNSPECIFIED
		switch strings.Trim(string(row[3]), `"`) {
		case "b":
			side = fav1.Side_SIDE_BUY
		case "s":
			side = fav1.Side_SIDE_SELL
		}
		e := envelope(inst, at, raw.ReceivedAt)
		e.Payload = &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: price, Size: size, Aggressor: side, VenueTradeId: vid}}
		envs = append(envs, e)
		if tid > maxID {
			maxID = tid
		}
		if at.After(newest) {
			newest = at
		}
	}
	if !backfill && maxID > lastID {
		c.lastTrade[symbol] = maxID
		c.setCursor(inst, tradesScope, newest)
	}
	return envs, nil, nil
}

// decimalTime parses Kraken's decimal seconds ("1794362222.1962714") from
// the literal text, without float rounding, so replays are exact.
func decimalTime(s string) (time.Time, error) {
	s = strings.Trim(s, `"`)
	sec, frac, _ := strings.Cut(s, ".")
	secs, err := strconv.ParseInt(sec, 10, 64)
	if err != nil {
		return time.Time{}, err
	}
	if len(frac) > 9 {
		frac = frac[:9]
	}
	var ns int64
	if frac != "" {
		if ns, err = strconv.ParseInt(frac+strings.Repeat("0", 9-len(frac)), 10, 64); err != nil {
			return time.Time{}, err
		}
	}
	return time.Unix(secs, ns).UTC(), nil
}

func (c *Connector) normTicker(raw connector.RawMessage, inst string, data json.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var t struct {
		A []json.RawMessage `json:"a"`
		B []json.RawMessage `json:"b"`
	}
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, nil, fmt.Errorf("kraken: decode ticker: %w", err)
	}
	if len(t.A) < 3 || len(t.B) < 3 {
		return nil, nil, fmt.Errorf("kraken: ticker without bid/ask")
	}
	ap, e1 := num(t.A[0])
	as, e2 := num(t.A[2])
	bp, e3 := num(t.B[0])
	bs, e4 := num(t.B[2])
	if e1 != nil || e2 != nil || e3 != nil || e4 != nil {
		return nil, nil, fmt.Errorf("kraken: ticker numbers")
	}
	e := envelope(inst, raw.ReceivedAt, raw.ReceivedAt)
	e.Payload = &fav1.Envelope_Quote{Quote: &fav1.Quote{BidPrice: bp, BidSize: bs, AskPrice: ap, AskSize: as}}
	return []*fav1.Envelope{e}, nil, nil
}

func (c *Connector) setCursor(inst, scope string, t time.Time) {
	if c.deps.Cursors == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	_ = c.deps.Cursors.SetCursor(ctx, ID, inst, scope, t)
	cancel()
}

func envelope(inst string, at, ingest time.Time) *fav1.Envelope {
	return &fav1.Envelope{
		EventId:       ulid.Make().String(),
		SourceId:      ID,
		InstrumentId:  inst,
		EventTime:     timestamppb.New(at),
		PublishTime:   timestamppb.New(ingest),
		IngestTime:    timestamppb.New(ingest),
		SchemaVersion: 1,
	}
}

// Symbols lists the configured pairs (for logs and tests).
func (c *Connector) Symbols() []string { return append([]string(nil), c.symbols...) }
