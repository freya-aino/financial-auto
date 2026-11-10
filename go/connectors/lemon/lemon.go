// Package lemon is the lemon.markets market-data connector for catalog id "lemon" (specs 10, 11).
//
// Implemented against the public API reference (https://docs.lemon.markets):
//
//	GET https://data.lemon.markets/v1/quotes/latest?isin=A,B&mic=XMUN   → Quote
//	GET https://data.lemon.markets/v1/trades/latest?isin=A,B&mic=XMUN   → Trade
//	GET https://data.lemon.markets/v1/ohlc/{m1,h1,d1}?isin=A&from=&to=  → Bar (backfill)
//
// Auth: `Authorization: Bearer <api_key>` from secret `api_key`
// (FA_LEMON_API_KEY or [sources.lemon] in config/secrets.toml, D-020).
// Venue symbols are ISINs; the MIC comes from config.extra.mic (default XMUN).
// Execution is broker-gateway's job, not this connector's.
//
// Status: implemented, unverified against the live API (no key).
package lemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
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
	ID             = "lemon"
	Venue          = "lemon"
	DefaultBaseURL = "https://data.lemon.markets/v1"
	DefaultMIC     = "XMUN"
)

// ErrNoAPIKey is returned by Init when no api_key secret is configured.
var ErrNoAPIKey = errors.New("lemon: missing secret api_key (FA_LEMON_API_KEY)")

// timeframes maps canonical timeframes to lemon's OHLC path segment.
var timeframes = map[string]struct {
	Path string
	Dur  time.Duration
}{"1m": {"m1", time.Minute}, "1h": {"h1", time.Hour}, "1d": {"d1", 24 * time.Hour}}

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector implements Connector, Poller, Backfiller and Normalizer.
type Connector struct {
	BaseURL string
	Fetch   *connector.Fetcher

	cfg   connector.SourceConfig
	deps  connector.Deps
	isins []string
	mic   string

	mu   sync.Mutex
	last map[string]time.Time // isin+kind → last emitted event time (live de-dup)
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Backfill: true, Events: []connector.EventKind{connector.KindQuote, connector.KindTrade, connector.KindBar}, Venue: Venue}
}

func (c *Connector) Init(_ context.Context, cfg connector.SourceConfig, deps connector.Deps) error {
	if deps.Clock == nil {
		deps.Clock = connector.SystemClock{}
	}
	c.cfg, c.deps = cfg, deps
	key := cfg.Secret("api_key")
	if key == "" {
		return ErrNoAPIKey
	}
	if c.BaseURL == "" {
		c.BaseURL = DefaultBaseURL
	}
	if c.Fetch == nil {
		c.Fetch = connector.NewFetcher(deps.Limiter)
	}
	c.Fetch.Prepare = func(r *http.Request) { r.Header.Set("Authorization", "Bearer "+key) }
	p := cfg.Source.Params()
	c.isins = p.Symbols
	c.mic = p.Extra["mic"]
	if c.mic == "" {
		c.mic = DefaultMIC
	}
	if len(c.isins) == 0 {
		return errors.New("lemon: config.symbols (ISINs) is empty")
	}
	c.last = map[string]time.Time{}
	return nil
}

func (c *Connector) Schedule() connector.PollSchedule {
	return connector.PollSchedule{Interval: time.Minute}
}

func (c *Connector) raw(endpoint, u string, b []byte, backfill bool) connector.RawMessage {
	meta := map[string]string{connector.MetaEndpoint: endpoint, connector.MetaURL: u}
	if backfill {
		meta[connector.MetaBackfill] = "1"
	}
	return connector.RawMessage{SourceID: ID, ReceivedAt: c.deps.Clock.Now(), TransportMeta: meta, Bytes: b, ContentType: "application/json"}
}

// Poll fetches the latest quote and trade for all ISINs (two requests per poll;
// lemon accepts up to 10 ISINs per call, so batch by 10).
func (c *Connector) Poll(ctx context.Context, _ connector.PollRequest) ([]connector.RawMessage, error) {
	var out []connector.RawMessage
	for i := 0; i < len(c.isins); i += 10 {
		batch := c.isins[i:min(i+10, len(c.isins))]
		for _, ep := range []string{"quotes", "trades"} {
			u := fmt.Sprintf("%s/%s/latest?isin=%s&mic=%s", c.BaseURL, ep, strings.Join(batch, ","), url.QueryEscape(c.mic))
			b, err := c.Fetch.Get(ctx, u)
			if err != nil {
				return out, fmt.Errorf("lemon %s: %w", ep, err)
			}
			out = append(out, c.raw(ep, u, b, false))
		}
	}
	return out, nil
}

// Backfill pages OHLC bars for one ISIN over [From, To).
func (c *Connector) Backfill(ctx context.Context, req connector.BackfillRequest, out chan<- connector.RawMessage) error {
	tf := req.Timeframe
	if tf == "" {
		tf = "1m"
	}
	t, ok := timeframes[tf]
	if !ok {
		return fmt.Errorf("lemon: unsupported timeframe %q", tf)
	}
	to := req.To
	if to.IsZero() {
		to = c.deps.Clock.Now()
	}
	next := fmt.Sprintf("%s/ohlc/%s?isin=%s&mic=%s&from=%s&to=%s&limit=100", c.BaseURL, t.Path, url.QueryEscape(req.VenueSymbol),
		url.QueryEscape(c.mic), req.From.UTC().Format("2006-01-02T15:04:05"), to.UTC().Format("2006-01-02T15:04:05"))
	for page := 0; next != "" && page < 10_000; page++ {
		b, err := c.Fetch.Get(ctx, next)
		if err != nil {
			return fmt.Errorf("lemon backfill: %w", err)
		}
		m := c.raw("ohlc", next, b, true)
		m.TransportMeta[connector.MetaTf] = tf
		select {
		case out <- m:
		case <-ctx.Done():
			return ctx.Err()
		}
		var pg struct {
			Next *string `json:"next"`
		}
		if err := json.Unmarshal(b, &pg); err != nil {
			return fmt.Errorf("lemon backfill: decode: %w", err)
		}
		next = ""
		if pg.Next != nil && *pg.Next != "" {
			next = c.rebase(*pg.Next)
		}
	}
	return nil
}

// rebase keeps the path and query of a pagination link but points it at
// BaseURL (so proxies/tests work and the API host cannot redirect us elsewhere).
func (c *Connector) rebase(link string) string {
	u, err := url.Parse(link)
	if err != nil {
		return ""
	}
	base, err := url.Parse(c.BaseURL)
	if err != nil {
		return ""
	}
	// lemon's next links are absolute under /v1, which BaseURL already carries.
	path := strings.TrimPrefix(u.Path, "/v1")
	return base.Scheme + "://" + base.Host + strings.TrimSuffix(base.Path, "/") + path + "?" + u.RawQuery
}

// row is the union of lemon's quote / trade / ohlc result rows.
type row struct {
	ISIN string  `json:"isin"`
	MIC  string  `json:"mic"`
	T    string  `json:"t"`
	B    float64 `json:"b"`
	BV   float64 `json:"b_v"`
	A    float64 `json:"a"`
	AV   float64 `json:"a_v"`
	P    float64 `json:"p"`
	V    float64 `json:"v"`
	O    float64 `json:"o"`
	H    float64 `json:"h"`
	L    float64 `json:"l"`
	C    float64 `json:"c"`
	PBV  float64 `json:"pbv"`
}

type page struct {
	Results []row  `json:"results"`
	Status  string `json:"status"`
	ErrMsg  string `json:"error_message"`
}

func parseTime(s string) (time.Time, error) {
	for _, layout := range []string{"2006-01-02T15:04:05.000-07:00", time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("lemon: bad time %q", s)
}

// Normalize maps a quotes/trades/ohlc page to envelopes.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var pg page
	if err := json.Unmarshal(raw.Bytes, &pg); err != nil {
		return nil, nil, fmt.Errorf("lemon: decode: %w", err)
	}
	if pg.Status == "error" {
		return nil, nil, fmt.Errorf("lemon: api error: %s", pg.ErrMsg)
	}
	ep := raw.TransportMeta[connector.MetaEndpoint]
	tf := raw.TransportMeta[connector.MetaTf]
	if tf == "" {
		tf = "1m"
	}
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"
	var envs []*fav1.Envelope
	var unres []connector.Unresolved
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, r := range pg.Results {
		inst, ok := c.deps.Resolver.Resolve(ID, r.ISIN)
		if !ok {
			unres = append(unres, connector.Unresolved{SourceID: ID, VenueSymbol: r.ISIN, Reason: "no instrument for ISIN"})
			continue
		}
		at, err := parseTime(r.T)
		if err != nil {
			return envs, unres, err
		}
		key := r.ISIN + "/" + ep
		if !backfill && !at.After(c.last[key]) {
			continue // latest unchanged since the previous poll
		}
		if !backfill {
			c.last[key] = at
		}
		e := &fav1.Envelope{
			EventId: ulid.Make().String(), SourceId: ID, InstrumentId: inst,
			EventTime: timestamppb.New(at), PublishTime: timestamppb.New(raw.ReceivedAt), IngestTime: timestamppb.New(raw.ReceivedAt),
			SchemaVersion: 1,
		}
		switch ep {
		case "quotes":
			e.Payload = &fav1.Envelope_Quote{Quote: &fav1.Quote{BidPrice: r.B, BidSize: r.BV, AskPrice: r.A, AskSize: r.AV}}
		case "trades":
			e.Payload = &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: r.P, Size: r.V, VenueTradeId: r.ISIN + "@" + strconv.FormatInt(at.UnixMilli(), 10)}}
		case "ohlc":
			d := timeframes[tf].Dur
			e.EventTime = timestamppb.New(at.Add(d)) // bar close
			bar := &fav1.Bar{Timeframe: tf, Open: r.O, High: r.H, Low: r.L, Close: r.C, Volume: r.V}
			if r.V > 0 && r.PBV > 0 {
				v := r.PBV / r.V
				bar.Vwap = &v
			}
			e.Payload = &fav1.Envelope_Bar{Bar: bar}
		default:
			return nil, nil, fmt.Errorf("lemon: unknown endpoint %q", ep)
		}
		envs = append(envs, e)
	}
	return envs, unres, nil
}
