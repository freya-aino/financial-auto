// Package yahoo is the Yahoo Finance connector for catalog id "yahoo" (specs 10, 11).
//
// It polls the unofficial chart endpoint
//
//	https://query2.finance.yahoo.com/v8/finance/chart/<symbol>?interval=1m&range=1d
//
// for every symbol in the catalog entry's `config.symbols`, and backfills with
// `period1`/`period2` windows. Each completed bar becomes a Bar envelope on
// `md.bar.<tf>.yahoo.<inst>` plus a synthetic Trade (price = close, size =
// volume, venue_trade_id = "bar:<unix>") on `md.trade.yahoo.<inst>` so the
// `md.trade.>` consumers (strategy features, evo-worker eval, D-044) see
// keyless data. Bars are stamped at their close (event_time = open + tf);
// publish_time = ingest_time (Yahoo gives no availability time, spec 10 §6.2).
//
// Verified live 2026-11-10: query1 answered 429 to a browser-like user agent,
// query2 answered 200 to the ingestor's own user agent. No crumb/cookie needed
// for /v8/finance/chart. Research use only (catalog tos_notes).
package yahoo

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// ID matches config/sources.yaml.
const (
	ID    = "yahoo"
	Venue = "yahoo"
	// BaseURL is overridable for tests.
	DefaultBaseURL = "https://query2.finance.yahoo.com"
)

// DefaultSymbols apply when the catalog entry has no config.symbols.
var DefaultSymbols = []string{"SPY"}

// Intervals maps canonical timeframes to Yahoo's interval parameter and
// the maximum window one request may span.
var Intervals = map[string]struct {
	Param  string
	Dur    time.Duration
	MaxReq time.Duration
}{
	"1m":  {"1m", time.Minute, 7 * 24 * time.Hour},
	"5m":  {"5m", 5 * time.Minute, 60 * 24 * time.Hour},
	"15m": {"15m", 15 * time.Minute, 60 * 24 * time.Hour},
	"1h":  {"60m", time.Hour, 730 * 24 * time.Hour},
	"1d":  {"1d", 24 * time.Hour, 50 * 365 * 24 * time.Hour},
}

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector implements Connector, Poller, Backfiller and Normalizer.
type Connector struct {
	BaseURL string
	Fetch   *connector.Fetcher

	cfg     connector.SourceConfig
	deps    connector.Deps
	symbols []string
	tf      string

	mu   sync.Mutex
	last map[string]int64 // symbol → unix of the last emitted live bar
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Backfill: true, Events: []connector.EventKind{connector.KindBar, connector.KindTrade}, Venue: Venue}
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
		return fmt.Errorf("yahoo: unsupported timeframe %q", c.tf)
	}
	c.last = map[string]int64{}
	if deps.Cursors != nil {
		for _, s := range c.symbols {
			if t, ok, err := deps.Cursors.GetCursor(ctx, ID, c.instrument(s), c.tf); err == nil && ok {
				c.last[s] = t.Unix() - int64(Intervals[c.tf].Dur/time.Second) // cursor is bar close
			}
		}
	}
	return nil
}

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

// Poll fetches the latest day of bars for every configured symbol.
func (c *Connector) Poll(ctx context.Context, _ connector.PollRequest) ([]connector.RawMessage, error) {
	var out []connector.RawMessage
	var firstErr error
	for _, s := range c.symbols {
		u := fmt.Sprintf("%s/v8/finance/chart/%s?interval=%s&range=1d&includePrePost=false", c.BaseURL, url.PathEscape(s), Intervals[c.tf].Param)
		b, err := c.Fetch.Get(ctx, u)
		if err != nil {
			if ctx.Err() != nil {
				return out, ctx.Err()
			}
			if firstErr == nil {
				firstErr = fmt.Errorf("yahoo %s: %w", s, err)
			}
			continue
		}
		out = append(out, c.raw(s, u, b, false))
	}
	if len(out) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

func (c *Connector) raw(symbol, u string, b []byte, backfill bool) connector.RawMessage {
	meta := map[string]string{connector.MetaSymbol: symbol, connector.MetaTf: c.tf, connector.MetaURL: u, connector.MetaEndpoint: "chart"}
	if backfill {
		meta[connector.MetaBackfill] = "1"
	}
	return connector.RawMessage{SourceID: ID, ReceivedAt: c.deps.Clock.Now(), TransportMeta: meta, Bytes: b, ContentType: "application/json"}
}

// Backfill walks [From, To) in windows Yahoo accepts for the timeframe.
func (c *Connector) Backfill(ctx context.Context, req connector.BackfillRequest, out chan<- connector.RawMessage) error {
	tf := req.Timeframe
	if tf == "" {
		tf = c.tf
	}
	iv, ok := Intervals[tf]
	if !ok {
		return fmt.Errorf("yahoo: unsupported timeframe %q", tf)
	}
	from, to := req.From, req.To
	if cur, err := time.Parse(time.RFC3339, req.Cursor); req.Cursor != "" && err == nil && cur.After(from) {
		from = cur
	}
	if to.IsZero() {
		to = c.deps.Clock.Now()
	}
	for start := from; start.Before(to); start = start.Add(iv.MaxReq) {
		end := start.Add(iv.MaxReq)
		if end.After(to) {
			end = to
		}
		u := fmt.Sprintf("%s/v8/finance/chart/%s?interval=%s&period1=%d&period2=%d&includePrePost=false",
			c.BaseURL, url.PathEscape(req.VenueSymbol), iv.Param, start.Unix(), end.Unix())
		b, err := c.Fetch.Get(ctx, u)
		if err != nil {
			return fmt.Errorf("yahoo backfill %s: %w", req.VenueSymbol, err)
		}
		m := c.raw(req.VenueSymbol, u, b, true)
		m.TransportMeta[connector.MetaTf] = tf
		select {
		case out <- m:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// chart is the subset of the v8 chart response we read.
type chart struct {
	Chart struct {
		Result []struct {
			Meta struct {
				Symbol   string `json:"symbol"`
				Currency string `json:"currency"`
			} `json:"meta"`
			Timestamp  []int64 `json:"timestamp"`
			Indicators struct {
				Quote []struct {
					Open   []*float64 `json:"open"`
					High   []*float64 `json:"high"`
					Low    []*float64 `json:"low"`
					Close  []*float64 `json:"close"`
					Volume []*float64 `json:"volume"`
				} `json:"quote"`
			} `json:"indicators"`
		} `json:"result"`
		Error *struct {
			Code        string `json:"code"`
			Description string `json:"description"`
		} `json:"error"`
	} `json:"chart"`
}

// Normalize emits one Bar and one synthetic Trade per completed bar. Live
// polls skip bars already emitted and the still-open bar; backfill emits all.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var ch chart
	if err := json.Unmarshal(raw.Bytes, &ch); err != nil {
		return nil, nil, fmt.Errorf("yahoo: decode: %w", err)
	}
	if ch.Chart.Error != nil {
		return nil, nil, fmt.Errorf("yahoo: %s: %s", ch.Chart.Error.Code, ch.Chart.Error.Description)
	}
	if len(ch.Chart.Result) == 0 || len(ch.Chart.Result[0].Indicators.Quote) == 0 {
		return nil, nil, nil
	}
	res := ch.Chart.Result[0]
	q := res.Indicators.Quote[0]
	symbol := raw.TransportMeta[connector.MetaSymbol]
	if symbol == "" {
		symbol = res.Meta.Symbol
	}
	tf := raw.TransportMeta[connector.MetaTf]
	if tf == "" {
		tf = c.tf
	}
	iv, ok := Intervals[tf]
	if !ok {
		return nil, nil, fmt.Errorf("yahoo: unsupported timeframe %q", tf)
	}
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"

	inst, ok := c.deps.Resolver.Resolve(ID, symbol)
	if !ok {
		return nil, []connector.Unresolved{{SourceID: ID, VenueSymbol: symbol, Reason: "no instrument for yahoo symbol"}}, nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	last := c.last[symbol]
	var envs []*fav1.Envelope
	var maxTs int64
	for i, ts := range res.Timestamp {
		if i >= len(q.Close) || q.Close[i] == nil || q.Open[i] == nil || q.High[i] == nil || q.Low[i] == nil {
			continue // Yahoo pads missing minutes with nulls
		}
		closeAt := time.Unix(ts, 0).UTC().Add(iv.Dur)
		if !backfill {
			if ts <= last || closeAt.After(raw.ReceivedAt) {
				continue // already emitted or still forming
			}
		}
		vol := 0.0
		if i < len(q.Volume) && q.Volume[i] != nil {
			vol = *q.Volume[i]
		}
		bar := &fav1.Bar{Timeframe: tf, Open: *q.Open[i], High: *q.High[i], Low: *q.Low[i], Close: *q.Close[i], Volume: vol}
		be := c.envelope(inst, closeAt, raw.ReceivedAt)
		be.Payload = &fav1.Envelope_Bar{Bar: bar}
		te := c.envelope(inst, closeAt, raw.ReceivedAt)
		te.Payload = &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: *q.Close[i], Size: vol, VenueTradeId: "bar:" + strconv.FormatInt(ts, 10)}}
		envs = append(envs, be, te)
		if ts > maxTs {
			maxTs = ts
		}
	}
	if !backfill && maxTs > last {
		c.last[symbol] = maxTs
		if c.deps.Cursors != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = c.deps.Cursors.SetCursor(ctx, ID, inst, tf, time.Unix(maxTs, 0).UTC().Add(iv.Dur))
			cancel()
		}
	}
	return envs, nil, nil
}

func (c *Connector) envelope(inst string, at, ingest time.Time) *fav1.Envelope {
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

// Symbols lists the configured symbols (for logs and tests).
func (c *Connector) Symbols() []string { return append([]string(nil), c.symbols...) }
