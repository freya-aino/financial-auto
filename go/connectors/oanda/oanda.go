// Package oanda is the OANDA v20 connector for catalog id "oanda" (specs 10, 11).
//
// Implemented against the public v20 reference (https://developer.oanda.com/rest-live-v20/):
//
//	GET https://stream-fxpractice.oanda.com/v3/accounts/<id>/pricing/stream?instruments=EUR_USD,...  → Quote per PRICE line
//	GET https://api-fxpractice.oanda.com/v3/instruments/<i>/candles?granularity=M1&from=&to=&price=M → Bar (backfill)
//
// Auth: `Authorization: Bearer <api_key>` plus the account id, from secrets
// `api_key` / `account_id` (FA_OANDA_API_KEY, FA_OANDA_ACCOUNT_ID or
// [sources.oanda] in config/secrets.toml, D-020). config.extra.env selects
// "practice" (default) or "live" hosts. Instruments come from config.symbols.
// Quotes carry the best bid/ask price with the quoted liquidity as size.
//
// Status: implemented, unverified against the live API (no key).
package oanda

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const (
	ID    = "oanda"
	Venue = "oanda"
)

// Hosts per environment.
var Hosts = map[string]struct{ REST, Stream string }{
	"practice": {"https://api-fxpractice.oanda.com", "https://stream-fxpractice.oanda.com"},
	"live":     {"https://api-fxtrade.oanda.com", "https://stream-fxtrade.oanda.com"},
}

// Granularities maps canonical timeframes to v20 granularity codes.
var Granularities = map[string]struct {
	Code string
	Dur  time.Duration
}{
	"1m": {"M1", time.Minute}, "5m": {"M5", 5 * time.Minute}, "15m": {"M15", 15 * time.Minute},
	"1h": {"H1", time.Hour}, "1d": {"D", 24 * time.Hour},
}

// MaxCandles is v20's per-request cap.
const MaxCandles = 5000

var (
	ErrNoAPIKey  = errors.New("oanda: missing secret api_key (FA_OANDA_API_KEY)")
	ErrNoAccount = errors.New("oanda: missing secret account_id (FA_OANDA_ACCOUNT_ID)")
)

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector implements Connector, Streamer, Backfiller and Normalizer.
type Connector struct {
	RESTURL   string
	StreamURL string
	Fetch     *connector.Fetcher
	// StreamClient has no timeout (long-lived response); tests override it.
	StreamClient *http.Client
	// Reconnect delay after a dropped stream (default 5 s).
	Reconnect time.Duration

	cfg         connector.SourceConfig
	deps        connector.Deps
	key, acct   string
	instruments []string
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Stream: true, Backfill: true, Events: []connector.EventKind{connector.KindQuote, connector.KindBar}, Venue: Venue}
}

func (c *Connector) Init(_ context.Context, cfg connector.SourceConfig, deps connector.Deps) error {
	if deps.Clock == nil {
		deps.Clock = connector.SystemClock{}
	}
	c.cfg, c.deps = cfg, deps
	c.key, c.acct = cfg.Secret("api_key"), cfg.Secret("account_id")
	if c.key == "" {
		return ErrNoAPIKey
	}
	if c.acct == "" {
		return ErrNoAccount
	}
	p := cfg.Source.Params()
	env := p.Extra["env"]
	if env == "" {
		env = "practice"
	}
	h, ok := Hosts[env]
	if !ok {
		return fmt.Errorf("oanda: unknown env %q", env)
	}
	if c.RESTURL == "" {
		c.RESTURL = h.REST
	}
	if c.StreamURL == "" {
		c.StreamURL = h.Stream
	}
	if c.Fetch == nil {
		c.Fetch = connector.NewFetcher(deps.Limiter)
	}
	c.Fetch.Prepare = c.auth
	if c.StreamClient == nil {
		c.StreamClient = &http.Client{}
	}
	if c.Reconnect == 0 {
		c.Reconnect = 5 * time.Second
	}
	c.instruments = p.Symbols
	if len(c.instruments) == 0 {
		return errors.New("oanda: config.symbols (instruments) is empty")
	}
	return nil
}

func (c *Connector) auth(r *http.Request) {
	r.Header.Set("Authorization", "Bearer "+c.key)
	r.Header.Set("Accept-Datetime-Format", "RFC3339")
	if r.Header.Get("User-Agent") == "" {
		r.Header.Set("User-Agent", connector.UserAgent)
	}
}

// Stream reads the pricing stream, one RawMessage per PRICE line, reconnecting
// until ctx is done. subs, when given, override the configured instruments.
func (c *Connector) Stream(ctx context.Context, subs []connector.Subscription, out chan<- connector.RawMessage) error {
	insts := c.instruments
	if len(subs) > 0 {
		insts = insts[:0:0]
		for _, s := range subs {
			insts = append(insts, s.VenueSymbol)
		}
	}
	u := fmt.Sprintf("%s/v3/accounts/%s/pricing/stream?instruments=%s", c.StreamURL, url.PathEscape(c.acct), url.QueryEscape(strings.Join(insts, ",")))
	for {
		err := c.streamOnce(ctx, u, out)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if connector.IsHTTPStatus(err, http.StatusUnauthorized) || connector.IsHTTPStatus(err, http.StatusForbidden) {
			return err // credentials problem: do not hammer
		}
		t := time.NewTimer(c.Reconnect)
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
	}
}

func (c *Connector) streamOnce(ctx context.Context, u string, out chan<- connector.RawMessage) error {
	if c.deps.Limiter != nil {
		if err := c.deps.Limiter.Wait(ctx); err != nil {
			return err
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	c.auth(req)
	res, err := c.StreamClient.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode/100 != 2 {
		var sb bytes.Buffer
		sb.ReadFrom(res.Body)
		return &connector.HTTPError{Status: res.StatusCode, Body: sb.String()}
	}
	sc := bufio.NewScanner(res.Body)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 || bytes.Contains(line, []byte(`"HEARTBEAT"`)) {
			continue
		}
		m := connector.RawMessage{
			SourceID: ID, ReceivedAt: c.deps.Clock.Now(), Bytes: append([]byte(nil), line...), ContentType: "application/json",
			TransportMeta: map[string]string{connector.MetaEndpoint: "pricing", connector.MetaURL: u},
		}
		select {
		case out <- m:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return sc.Err()
}

// Backfill fetches mid candles in ≤ MaxCandles windows over [From, To).
func (c *Connector) Backfill(ctx context.Context, req connector.BackfillRequest, out chan<- connector.RawMessage) error {
	tf := req.Timeframe
	if tf == "" {
		tf = "1m"
	}
	g, ok := Granularities[tf]
	if !ok {
		return fmt.Errorf("oanda: unsupported timeframe %q", tf)
	}
	to := req.To
	if to.IsZero() {
		to = c.deps.Clock.Now()
	}
	window := g.Dur * MaxCandles
	for start := req.From; start.Before(to); start = start.Add(window) {
		end := start.Add(window)
		if end.After(to) {
			end = to
		}
		u := fmt.Sprintf("%s/v3/instruments/%s/candles?granularity=%s&price=M&from=%s&to=%s", c.RESTURL, url.PathEscape(req.VenueSymbol), g.Code,
			url.QueryEscape(start.UTC().Format(time.RFC3339)), url.QueryEscape(end.UTC().Format(time.RFC3339)))
		b, err := c.Fetch.Get(ctx, u)
		if err != nil {
			return fmt.Errorf("oanda backfill %s: %w", req.VenueSymbol, err)
		}
		m := connector.RawMessage{
			SourceID: ID, ReceivedAt: c.deps.Clock.Now(), Bytes: b, ContentType: "application/json",
			TransportMeta: map[string]string{connector.MetaEndpoint: "candles", connector.MetaURL: u, connector.MetaTf: tf, connector.MetaSymbol: req.VenueSymbol, connector.MetaBackfill: "1"},
		}
		select {
		case out <- m:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

type price struct {
	Type       string `json:"type"`
	Time       string `json:"time"`
	Instrument string `json:"instrument"`
	Bids       []struct {
		Price     string  `json:"price"`
		Liquidity float64 `json:"liquidity"`
	} `json:"bids"`
	Asks []struct {
		Price     string  `json:"price"`
		Liquidity float64 `json:"liquidity"`
	} `json:"asks"`
	ErrorMessage string `json:"errorMessage"`
}

type candles struct {
	Instrument  string `json:"instrument"`
	Granularity string `json:"granularity"`
	Candles     []struct {
		Complete bool    `json:"complete"`
		Volume   float64 `json:"volume"`
		Time     string  `json:"time"`
		Mid      struct {
			O, H, L, C string
		} `json:"mid"`
	} `json:"candles"`
	ErrorMessage string `json:"errorMessage"`
}

func f(s string) (float64, error) { return strconv.ParseFloat(s, 64) }

// Normalize maps a PRICE line to a Quote or a candles page to Bars (complete candles only).
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	switch raw.TransportMeta[connector.MetaEndpoint] {
	case "candles":
		return c.normalizeCandles(raw)
	default:
		return c.normalizePrice(raw)
	}
}

func (c *Connector) envelope(inst string, at, ingest time.Time) *fav1.Envelope {
	return &fav1.Envelope{
		EventId: ulid.Make().String(), SourceId: ID, InstrumentId: inst,
		EventTime: timestamppb.New(at), PublishTime: timestamppb.New(ingest), IngestTime: timestamppb.New(ingest), SchemaVersion: 1,
	}
}

func (c *Connector) normalizePrice(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var p price
	if err := json.Unmarshal(raw.Bytes, &p); err != nil {
		return nil, nil, fmt.Errorf("oanda: decode price: %w", err)
	}
	if p.ErrorMessage != "" {
		return nil, nil, fmt.Errorf("oanda: %s", p.ErrorMessage)
	}
	if p.Type != "PRICE" || len(p.Bids) == 0 || len(p.Asks) == 0 {
		return nil, nil, nil
	}
	inst, ok := c.deps.Resolver.Resolve(ID, p.Instrument)
	if !ok {
		return nil, []connector.Unresolved{{SourceID: ID, VenueSymbol: p.Instrument, Reason: "no instrument for oanda symbol"}}, nil
	}
	at, err := time.Parse(time.RFC3339Nano, p.Time)
	if err != nil {
		return nil, nil, fmt.Errorf("oanda: bad time %q", p.Time)
	}
	bid, err1 := f(p.Bids[0].Price)
	ask, err2 := f(p.Asks[0].Price)
	if err1 != nil || err2 != nil {
		return nil, nil, fmt.Errorf("oanda: bad price in %s", raw.Bytes)
	}
	e := c.envelope(inst, at.UTC(), raw.ReceivedAt)
	e.Payload = &fav1.Envelope_Quote{Quote: &fav1.Quote{BidPrice: bid, BidSize: p.Bids[0].Liquidity, AskPrice: ask, AskSize: p.Asks[0].Liquidity}}
	return []*fav1.Envelope{e}, nil, nil
}

func (c *Connector) normalizeCandles(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var cs candles
	if err := json.Unmarshal(raw.Bytes, &cs); err != nil {
		return nil, nil, fmt.Errorf("oanda: decode candles: %w", err)
	}
	if cs.ErrorMessage != "" {
		return nil, nil, fmt.Errorf("oanda: %s", cs.ErrorMessage)
	}
	symbol := cs.Instrument
	if symbol == "" {
		symbol = raw.TransportMeta[connector.MetaSymbol]
	}
	tf := raw.TransportMeta[connector.MetaTf]
	if tf == "" {
		for k, g := range Granularities {
			if g.Code == cs.Granularity {
				tf = k
			}
		}
	}
	g, ok := Granularities[tf]
	if !ok {
		return nil, nil, fmt.Errorf("oanda: unsupported timeframe %q", tf)
	}
	inst, ok := c.deps.Resolver.Resolve(ID, symbol)
	if !ok {
		return nil, []connector.Unresolved{{SourceID: ID, VenueSymbol: symbol, Reason: "no instrument for oanda symbol"}}, nil
	}
	var envs []*fav1.Envelope
	for _, k := range cs.Candles {
		if !k.Complete {
			continue
		}
		at, err := time.Parse(time.RFC3339Nano, k.Time)
		if err != nil {
			return envs, nil, fmt.Errorf("oanda: bad time %q", k.Time)
		}
		o, e1 := f(k.Mid.O)
		h, e2 := f(k.Mid.H)
		l, e3 := f(k.Mid.L)
		cl, e4 := f(k.Mid.C)
		if e1 != nil || e2 != nil || e3 != nil || e4 != nil {
			return envs, nil, fmt.Errorf("oanda: bad candle %+v", k.Mid)
		}
		e := c.envelope(inst, at.UTC().Add(g.Dur), raw.ReceivedAt)
		e.Payload = &fav1.Envelope_Bar{Bar: &fav1.Bar{Timeframe: tf, Open: o, High: h, Low: l, Close: cl, Volume: k.Volume, TradeCount: uint64(k.Volume)}}
		envs = append(envs, e)
	}
	return envs, nil, nil
}
