// Package frankfurter is the Frankfurter (ECB reference rate mirror) connector
// for catalog id "frankfurter" (specs 10, 11).
//
// It polls the keyless JSON API
//
//	https://api.frankfurter.dev/v1/<start>..?base=EUR&symbols=USD,GBP,...
//
// (the old api.frankfurter.app host answers 301 to it) and emits one
// MacroObservation per (currency, business day) on
// `macro.obs.frankfurter.<BASE>_<CCY>`; value = units of CCY per 1 BASE.
//
// Why MacroObservation and not Quote: the ECB reference rates are a single
// daily fixing ("for information purposes only"), not a tradable bid/ask from
// a venue. Emitting them as Quotes would provision fake EUR/USD instruments on
// a "frankfurter" venue and feed md.quote consumers a once-a-day price with no
// spread. As macro observations they land in wh.macro_observation with a
// vintage and stay point-in-time correct; the risk-engine keeps its own direct
// eurofxref feed for accounting FX (D-061).
//
// Time semantics (spec 10 §6.2): Frankfurter gives no publication time. The ECB
// publishes the rates around 16:00 CET/CEST on the reference day, so vintage =
// <date> 16:00 Europe/Berlin, an estimate, so every envelope carries
// publish_time_estimated = true. event_time = vintage. publish_time = ingest_time for live polls and the estimated vintage
// for backfill. The vintage is deterministic, so re-polling the same day
// never creates a second vintage in the warehouse. The ECB does not revise
// reference rates: revision is always false.
//
// De-duplication: in memory per (series, period) plus a persisted cursor
// (ingest.backfill_cursor, scope "rates") holding the newest reference date
// published; each poll asks for [cursor − 7 days, today] so a missed day is
// caught up and days at or before the cursor are skipped.
//
// Backfill: [From, To) in one-year windows of the same range endpoint (data
// starts 1999-01-04), every row emitted (MetaBackfill=1).
//
// Verified live 2026-11-11: /v1/latest and /v1/<a>..<b> answered 200 without
// any key; a 2-year range returns every business day (no down-sampling).
package frankfurter

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"
	_ "time/tzdata" // Europe/Berlin in distroless/alpine images

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const (
	ID             = "frankfurter"
	DefaultBaseURL = "https://api.frankfurter.dev/v1"
	cursorScope    = "rates"
	dateLayout     = "2006-01-02"
	// lookback is how far before the cursor each live poll re-asks.
	lookback = 7 * 24 * time.Hour
)

// DefaultCurrencies apply when config.extra.currencies is empty.
var DefaultCurrencies = []string{"USD", "GBP", "JPY", "CHF"}

// FirstDate is the first ECB reference date.
var FirstDate = time.Date(1999, 1, 4, 0, 0, 0, 0, time.UTC)

var berlin = func() *time.Location {
	l, err := time.LoadLocation("Europe/Berlin")
	if err != nil {
		return time.FixedZone("CET", 3600)
	}
	return l
}()

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector implements Connector, Poller, Backfiller and Normalizer.
type Connector struct {
	BaseURL string
	Fetch   *connector.Fetcher

	deps       connector.Deps
	base       string
	currencies []string
	seen       *connector.Seen

	mu            sync.Mutex
	floor, cursor time.Time // reference dates (UTC midnight)
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Backfill: true, Events: []connector.EventKind{connector.KindMacro}}
}

func (c *Connector) Init(ctx context.Context, cfg connector.SourceConfig, deps connector.Deps) error {
	if deps.Clock == nil {
		deps.Clock = connector.SystemClock{}
	}
	c.deps = deps
	if c.BaseURL == "" {
		c.BaseURL = DefaultBaseURL
	}
	if c.Fetch == nil {
		c.Fetch = connector.NewFetcher(deps.Limiter)
	}
	extra := cfg.Source.Params().Extra
	c.base = strings.ToUpper(strings.TrimSpace(extra["base"]))
	if c.base == "" {
		c.base = "EUR"
	}
	c.currencies = nil
	for _, s := range strings.Split(extra["currencies"], ",") {
		if s = strings.ToUpper(strings.TrimSpace(s)); s != "" && s != c.base {
			c.currencies = append(c.currencies, s)
		}
	}
	if len(c.currencies) == 0 {
		c.currencies = DefaultCurrencies
	}
	for _, s := range append([]string{c.base}, c.currencies...) {
		if len(s) != 3 {
			return fmt.Errorf("frankfurter: invalid currency %q", s)
		}
	}
	c.seen = connector.NewSeen(50_000)
	if deps.Cursors != nil {
		if t, ok, err := deps.Cursors.GetCursor(ctx, ID, "", cursorScope); err != nil {
			return fmt.Errorf("frankfurter: cursor: %w", err)
		} else if ok {
			c.floor, c.cursor = t, t
		}
	}
	return nil
}

// Schedule: the rates change once per business day; hourly polling bounds the
// live latency after the ~16:00 CET release to an hour at ~24 requests a day.
func (c *Connector) Schedule() connector.PollSchedule {
	return connector.PollSchedule{Interval: time.Hour}
}

// URL builds a range request [from, to]; a zero to means "up to the latest".
func (c *Connector) URL(from, to time.Time) string {
	v := url.Values{"base": {c.base}, "symbols": {strings.Join(c.currencies, ",")}}
	r := from.Format(dateLayout) + ".."
	if !to.IsZero() {
		r += to.Format(dateLayout)
	}
	return c.BaseURL + "/" + r + "?" + v.Encode()
}

// Poll asks for [cursor − 7 d, latest] (or the last 7 days without a cursor).
func (c *Connector) Poll(ctx context.Context, _ connector.PollRequest) ([]connector.RawMessage, error) {
	now := c.deps.Clock.Now().UTC()
	c.mu.Lock()
	from := c.cursor
	c.mu.Unlock()
	if from.IsZero() {
		from = now
	}
	from = from.Add(-lookback)
	u := c.URL(from, time.Time{})
	b, err := c.Fetch.Get(ctx, u)
	if err != nil {
		return nil, fmt.Errorf("frankfurter: %w", err)
	}
	return []connector.RawMessage{c.raw(u, b, false)}, nil
}

func (c *Connector) raw(u string, b []byte, backfill bool) connector.RawMessage {
	meta := map[string]string{connector.MetaEndpoint: "range", connector.MetaURL: u}
	if backfill {
		meta[connector.MetaBackfill] = "1"
	}
	return connector.RawMessage{SourceID: ID, ReceivedAt: c.deps.Clock.Now(), TransportMeta: meta, Bytes: b, ContentType: "application/json"}
}

// Backfill walks [From, To) in one-year windows.
func (c *Connector) Backfill(ctx context.Context, req connector.BackfillRequest, out chan<- connector.RawMessage) error {
	from, to := req.From.UTC(), req.To.UTC()
	if to.IsZero() {
		to = c.deps.Clock.Now().UTC()
	}
	if from.Before(FirstDate) {
		from = FirstDate
	}
	from = truncDay(from)
	for start := from; start.Before(to); start = start.AddDate(1, 0, 0) {
		end := start.AddDate(1, 0, -1) // inclusive end date
		if !end.Before(to) {
			end = truncDay(to.Add(-time.Nanosecond))
		}
		u := c.URL(start, end)
		b, err := c.Fetch.Get(ctx, u)
		if err != nil {
			if connector.IsHTTPStatus(err, 404) {
				continue // no business day in the window
			}
			return fmt.Errorf("frankfurter backfill: %w", err)
		}
		m := c.raw(u, b, true)
		m.TransportMeta["from"] = req.From.UTC().Format(time.RFC3339)
		m.TransportMeta["to"] = to.Format(time.RFC3339)
		select {
		case out <- m:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// response covers both /latest (date + flat rates) and range answers
// (start_date/end_date + rates keyed by date).
type response struct {
	Base    string          `json:"base"`
	Date    string          `json:"date"`
	Rates   json.RawMessage `json:"rates"`
	Message string          `json:"message"`
}

// Vintage is the estimated ECB publication time of a reference date's rates.
func Vintage(day time.Time) time.Time {
	return time.Date(day.Year(), day.Month(), day.Day(), 16, 0, 0, 0, berlin).UTC()
}

// SeriesID names the series of one currency pair (NATS-token safe).
func SeriesID(base, ccy string) string { return base + "_" + ccy }

type obs struct {
	day   time.Time
	ccy   string
	value float64
}

func parse(b []byte) (string, []obs, error) {
	var r response
	if err := json.Unmarshal(b, &r); err != nil {
		return "", nil, fmt.Errorf("frankfurter: decode: %w", err)
	}
	if r.Message != "" && len(r.Rates) == 0 {
		return "", nil, fmt.Errorf("frankfurter: %s", r.Message)
	}
	byDay := map[string]map[string]float64{}
	if r.Date != "" {
		var flat map[string]float64
		if err := json.Unmarshal(r.Rates, &flat); err != nil {
			return "", nil, fmt.Errorf("frankfurter: decode rates: %w", err)
		}
		byDay[r.Date] = flat
	} else if len(r.Rates) > 0 {
		if err := json.Unmarshal(r.Rates, &byDay); err != nil {
			return "", nil, fmt.Errorf("frankfurter: decode rates: %w", err)
		}
	}
	var out []obs
	for d, rates := range byDay {
		day, err := time.Parse(dateLayout, d)
		if err != nil {
			return "", nil, fmt.Errorf("frankfurter: date %q: %w", d, err)
		}
		for ccy, v := range rates {
			if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
				continue
			}
			out = append(out, obs{day: day, ccy: ccy, value: v})
		}
	}
	// Deterministic order: by day, then currency.
	sort.Slice(out, func(i, j int) bool {
		if !out[i].day.Equal(out[j].day) {
			return out[i].day.Before(out[j].day)
		}
		return out[i].ccy < out[j].ccy
	})
	return r.Base, out, nil
}

// Normalize maps a latest/range answer to MacroObservations. Live polls skip
// reference dates at or before the persisted cursor and (series, date) pairs
// already published; backfill emits every row inside the requested range.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	base, rows, err := parse(raw.Bytes)
	if err != nil {
		return nil, nil, err
	}
	if base == "" {
		base = c.base
	}
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"
	var lo, hi time.Time
	if backfill {
		lo, _ = time.Parse(time.RFC3339, raw.TransportMeta["from"])
		hi, _ = time.Parse(time.RFC3339, raw.TransportMeta["to"])
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	newest := c.cursor
	var envs []*fav1.Envelope
	for _, o := range rows {
		vintage := Vintage(o.day)
		if backfill {
			// Frankfurter snaps a weekend start back to the previous business day.
			if (!lo.IsZero() && o.day.Before(truncDay(lo))) || (!hi.IsZero() && !o.day.Before(hi)) {
				continue
			}
		} else {
			key := o.ccy + "|" + o.day.Format(dateLayout)
			if !c.floor.IsZero() && !o.day.After(c.floor) {
				c.seen.Add(key)
				continue
			}
			if !c.seen.Add(key) {
				continue
			}
			if o.day.After(newest) {
				newest = o.day
			}
		}
		publish := vintage
		if !backfill {
			publish = raw.ReceivedAt
		}
		envs = append(envs, &fav1.Envelope{
			EventId:       ulid.Make().String(),
			SourceId:      ID,
			EventTime:     timestamppb.New(vintage),
			PublishTime:   timestamppb.New(publish),
			IngestTime:    timestamppb.New(raw.ReceivedAt),
			SchemaVersion: 1,
			// No availability time from the source: publish_time is derived.
			PublishTimeEstimated: true,
			Payload: &fav1.Envelope_Macro{Macro: &fav1.MacroObservation{
				SeriesId: SeriesID(base, o.ccy),
				Period:   o.day.Format(dateLayout),
				Value:    o.value,
				Vintage:  timestamppb.New(vintage),
			}},
		})
	}
	if newest.After(c.cursor) {
		c.cursor = newest
		if c.deps.Cursors != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = c.deps.Cursors.SetCursor(ctx, ID, "", cursorScope, newest)
			cancel()
		}
	}
	return envs, nil, nil
}

func truncDay(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}
