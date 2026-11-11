// Package ecb is the ECB Data Portal (SDMX 2.1 REST) connector for catalog id
// "ecb" (specs 10, 11).
//
// For every SDMX query "<FLOW>/<KEY>" in the catalog entry's `config.queries`
// it polls
//
//	https://data-api.ecb.europa.eu/service/data/<FLOW>/<KEY>?format=jsondata
//	    &detail=dataonly&includeHistory=true&lastNObservations=3
//
// and emits one MacroObservation per (series, period, vintage) on
// `macro.obs.ecb.<series>`, where <series> is the flow and the series key
// joined with ':' (e.g. `HICP:M:U2:N:000000:4D0:ANR`) so it stays one NATS
// token and maps back to the SDMX key by replacing ':' with '.'.
//
// Vintages (spec 10 §6.2): with includeHistory=true the portal answers one
// dataset per published version, each with `validFrom` = the time that
// version went public. Each (series, period, validFrom) is its own event:
// vintage = publish_time = validFrom, and revision = true when the same
// payload holds an earlier vintage of that period (e.g. HICP flash → final).
// "Delete" datasets and null values are skipped. Some flows (EXR) keep no
// history: their single dataset's validFrom equals the response's `prepared`
// time. Such a vintage would change with every request, so it is replaced
// by a deterministic *estimate*: daily periods → <date> 16:00 Europe/Berlin
// (the ECB reference rate release), other periods → the end of the period;
// publish_time is then ingest_time for live polls and the estimate for
// backfill, and the envelope carries publish_time_estimated = true.
// event_time = end of the reference period.
//
// De-duplication: in memory per (series, period, vintage) plus one persisted
// cursor per query (ingest.backfill_cursor, scope "q:<FLOW>/<KEY>") holding
// the newest vintage published; vintages at or before it are skipped.
//
// Backfill: startPeriod/endPeriod windows of five years over [From, To) per
// query, with history, every version emitted (MetaBackfill=1).
//
// Verified live 2026-11-11: HICP, FM (MRO/DFR) and EXR answered 200 without a
// key. The portal's WAF answers an HTML "access blocked" page (status 400 or
// 504) to bursts and to some clients (it blocked curl's default user agent
// but not the ingestor's); the catalog limit keeps requests ≥ 10 s apart and
// HTML bodies are reported as errors.
package ecb

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	_ "time/tzdata" // Europe/Berlin in alpine images

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const (
	ID             = "ecb"
	DefaultBaseURL = "https://data-api.ecb.europa.eu/service/data"
	// LiveObservations is lastNObservations for live polls.
	LiveObservations = 3
	metaQuery        = "query"
	// syntheticSlack: a validFrom this close to `prepared` is the response time, not a vintage.
	syntheticSlack = 5 * time.Second
)

// DefaultQueries apply when the catalog entry has no config.queries: euro
// area HICP annual rate, MRO and deposit facility rates, USD reference rate.
var DefaultQueries = []string{
	"HICP/M.U2.N.000000.4D0.ANR",
	"FM/B.U2.EUR.4F.KR.MRR_FR.LEV",
	"FM/D.U2.EUR.4F.KR.DFR.LEV",
	"EXR/D.USD.EUR.SP00.A",
}

var queryRe = regexp.MustCompile(`^[A-Z0-9_]+/[A-Za-z0-9_.+*-]*$`)

var berlin = func() *time.Location {
	l, err := time.LoadLocation("Europe/Berlin")
	if err != nil {
		return time.FixedZone("CET", 3600)
	}
	return l
}()

// Adapter is the catalog `adapter:` name of the generic SDMX-JSON connector
// (spec 11 §6.3): the same code serves any SDMX 2.1 REST endpoint answering
// `format=jsondata` (ECB, OECD, IMF SDMX Central …). Its catalog entry names
// the endpoint in `config.extra.base_url` (…/service/data or …/rest/data)
// and the series as `config.queries` "<FLOW>/<KEY>"; there are no default
// queries. History-less daily series get the ECB-style estimated vintage
// (16:00 Europe/Berlin, flagged publish_time_estimated). No catalog entry uses
// the adapter yet (Eurostat answers JSON-stat, not SDMX-JSON, and needs its own
// parser).
const Adapter = "sdmx"

func init() {
	connector.Register(ID, func() connector.Connector { return &Connector{} })
	connector.RegisterAdapter(Adapter, func(id string) connector.Connector { return &Connector{SourceID: id} })
}

// Connector implements Connector, Poller, Backfiller and Normalizer.
type Connector struct {
	BaseURL string
	Fetch   *connector.Fetcher
	// SourceID is the catalog id when running as the generic `sdmx` adapter ("" = ecb).
	SourceID string

	deps    connector.Deps
	queries []string
	seen    *connector.Seen

	mu sync.Mutex
	// floor is the per-query cursor loaded at Init; cursor the newest vintage published.
	floor, cursor map[string]time.Time
}

func (c *Connector) ID() string { return c.id() }

func (c *Connector) id() string {
	if c.SourceID != "" {
		return c.SourceID
	}
	return ID
}

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Backfill: true, Events: []connector.EventKind{connector.KindMacro}}
}

func (c *Connector) Init(ctx context.Context, cfg connector.SourceConfig, deps connector.Deps) error {
	if deps.Clock == nil {
		deps.Clock = connector.SystemClock{}
	}
	c.deps = deps
	adapter := c.SourceID != "" && c.SourceID != ID
	if adapter && c.BaseURL == "" {
		c.BaseURL = strings.TrimRight(cfg.Source.Params().Extra["base_url"], "/")
		if c.BaseURL == "" {
			return fmt.Errorf("sdmx %s: config.extra.base_url is required", c.SourceID)
		}
	}
	if c.BaseURL == "" {
		c.BaseURL = DefaultBaseURL
	}
	if c.Fetch == nil {
		c.Fetch = connector.NewFetcher(deps.Limiter)
	}
	c.queries = nil
	for _, q := range cfg.Source.Params().Queries {
		if q = strings.TrimSpace(q); q != "" {
			c.queries = append(c.queries, q)
		}
	}
	if len(c.queries) == 0 {
		if adapter {
			return fmt.Errorf("sdmx %s: config.queries is empty", c.SourceID)
		}
		c.queries = DefaultQueries
	}
	for _, q := range c.queries {
		if !queryRe.MatchString(q) {
			return fmt.Errorf("ecb: query %q is not <FLOW>/<KEY>", q)
		}
	}
	c.seen = connector.NewSeen(50_000)
	c.floor, c.cursor = map[string]time.Time{}, map[string]time.Time{}
	if deps.Cursors != nil {
		for _, q := range c.queries {
			t, ok, err := deps.Cursors.GetCursor(ctx, c.id(), "", scope(q))
			if err != nil {
				return fmt.Errorf("ecb: cursor: %w", err)
			}
			if ok {
				c.floor[q], c.cursor[q] = t, t
			}
		}
	}
	return nil
}

func scope(query string) string { return "q:" + query }

// Schedule: releases are at most daily per series; hourly polling keeps the
// live latency ≤ 1 h at a handful of requests per hour.
func (c *Connector) Schedule() connector.PollSchedule {
	return connector.PollSchedule{Interval: time.Hour}
}

// URL builds a data request for one query with extra parameters.
func (c *Connector) URL(query string, extra url.Values) string {
	v := url.Values{"format": {"jsondata"}, "detail": {"dataonly"}, "includeHistory": {"true"}}
	for k, vs := range extra {
		v[k] = vs
	}
	return c.BaseURL + "/" + query + "?" + v.Encode()
}

// Poll fetches the last observations (with their history) of every query.
func (c *Connector) Poll(ctx context.Context, _ connector.PollRequest) ([]connector.RawMessage, error) {
	var out []connector.RawMessage
	var firstErr error
	for _, q := range c.queries {
		u := c.URL(q, url.Values{"lastNObservations": {strconv.Itoa(LiveObservations)}})
		b, err := c.get(ctx, u)
		if err != nil {
			if ctx.Err() != nil {
				return out, ctx.Err()
			}
			if firstErr == nil {
				firstErr = fmt.Errorf("ecb %s: %w", q, err)
			}
			continue
		}
		out = append(out, c.raw(q, u, b, false))
	}
	if len(out) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// get fetches u and rejects the WAF's HTML pages.
func (c *Connector) get(ctx context.Context, u string) ([]byte, error) {
	b, err := c.Fetch.Get(ctx, u)
	if err != nil {
		if bytes.Contains([]byte(err.Error()), []byte("<html")) {
			return nil, fmt.Errorf("blocked by the ECB portal firewall (HTML answer): %w", err)
		}
		return nil, err
	}
	if t := bytes.TrimSpace(b); len(t) > 0 && t[0] == '<' {
		return nil, fmt.Errorf("unexpected HTML answer (portal firewall?)")
	}
	return b, nil
}

func (c *Connector) raw(query, u string, b []byte, backfill bool) connector.RawMessage {
	meta := map[string]string{connector.MetaEndpoint: "sdmx-json", connector.MetaURL: u, metaQuery: query}
	if backfill {
		meta[connector.MetaBackfill] = "1"
	}
	return connector.RawMessage{SourceID: c.id(), ReceivedAt: c.deps.Clock.Now(), TransportMeta: meta, Bytes: b, ContentType: "application/json"}
}

// Backfill walks [From, To) in five-year startPeriod/endPeriod windows for
// every query (or only VenueSymbol when it names one of them).
func (c *Connector) Backfill(ctx context.Context, req connector.BackfillRequest, out chan<- connector.RawMessage) error {
	queries := c.queries
	if s := strings.TrimSpace(req.VenueSymbol); s != "" {
		if !queryRe.MatchString(s) {
			return fmt.Errorf("ecb: query %q is not <FLOW>/<KEY>", s)
		}
		queries = []string{s}
	}
	from, to := req.From.UTC(), req.To.UTC()
	if to.IsZero() {
		to = c.deps.Clock.Now().UTC()
	}
	if min := time.Date(1990, 1, 1, 0, 0, 0, 0, time.UTC); from.Before(min) {
		from = min // the portal's series start in the 1990s at the earliest relevant point
	}
	for _, q := range queries {
		for start := from; start.Before(to); start = start.AddDate(5, 0, 0) {
			end := start.AddDate(5, 0, 0).Add(-24 * time.Hour)
			if !end.Before(to) {
				end = to.Add(-time.Nanosecond)
			}
			u := c.URL(q, url.Values{"startPeriod": {start.Format("2006-01-02")}, "endPeriod": {end.Format("2006-01-02")}})
			b, err := c.get(ctx, u)
			if err != nil {
				if connector.IsHTTPStatus(err, 404) {
					continue // no observations in the window
				}
				return fmt.Errorf("ecb backfill %s: %w", q, err)
			}
			select {
			case out <- c.raw(q, u, b, true):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	return nil
}

// message is the subset of SDMX-JSON 1.0 read here.
type message struct {
	Header struct {
		Prepared string `json:"prepared"`
	} `json:"header"`
	DataSets []struct {
		Action    string `json:"action"`
		ValidFrom string `json:"validFrom"`
		Series    map[string]struct {
			Observations map[string][]*float64 `json:"observations"`
		} `json:"series"`
	} `json:"dataSets"`
	Structure struct {
		Dimensions struct {
			Series []struct {
				ID     string `json:"id"`
				Values []struct {
					ID string `json:"id"`
				} `json:"values"`
			} `json:"series"`
			Observation []struct {
				ID     string `json:"id"`
				Values []struct {
					ID string `json:"id"`
				} `json:"values"`
			} `json:"observation"`
		} `json:"dimensions"`
	} `json:"structure"`
}

// Obs is one parsed (series, period, vintage) value.
type Obs struct {
	Series    string
	Period    string
	Value     float64
	Vintage   time.Time
	Estimated bool
	Revision  bool
}

const sdmxTime = "2006-01-02T15:04:05.000Z07:00"

func parseTime(s string) (time.Time, error) {
	for _, l := range []string{sdmxTime, time.RFC3339Nano, time.RFC3339} {
		if t, err := time.Parse(l, s); err == nil {
			return t.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("ecb: time %q", s)
}

// Parse decodes an SDMX-JSON answer into observations sorted by series,
// period and vintage, with revision flags and estimated vintages resolved.
func Parse(flow string, b []byte) ([]Obs, error) {
	var m message
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("ecb: decode: %w", err)
	}
	if len(m.Structure.Dimensions.Observation) == 0 {
		return nil, nil // empty answer
	}
	periods := m.Structure.Dimensions.Observation[0].Values
	var prepared time.Time
	if m.Header.Prepared != "" {
		prepared, _ = parseTime(m.Header.Prepared)
	}
	var out []Obs
	for _, ds := range m.DataSets {
		if !strings.EqualFold(ds.Action, "Replace") && !strings.EqualFold(ds.Action, "Information") {
			continue // Delete: an observation withdrawn; nothing to record as a value
		}
		var vintage time.Time
		estimated := true
		if ds.ValidFrom != "" {
			t, err := parseTime(ds.ValidFrom)
			if err != nil {
				return nil, err
			}
			if prepared.IsZero() || prepared.Sub(t).Abs() > syntheticSlack {
				vintage, estimated = t, false
			}
		}
		for skey, s := range ds.Series {
			sid, err := seriesID(flow, skey, m)
			if err != nil {
				return nil, err
			}
			for oidx, vals := range s.Observations {
				i, err := strconv.Atoi(oidx)
				if err != nil || i < 0 || i >= len(periods) {
					return nil, fmt.Errorf("ecb: observation index %q", oidx)
				}
				if len(vals) == 0 || vals[0] == nil || math.IsNaN(*vals[0]) || math.IsInf(*vals[0], 0) {
					continue
				}
				p := periods[i].ID
				o := Obs{Series: sid, Period: p, Value: *vals[0], Vintage: vintage, Estimated: estimated}
				if estimated {
					v, err := EstimatedVintage(p)
					if err != nil {
						return nil, err
					}
					o.Vintage = v
				}
				out = append(out, o)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Series != b.Series {
			return a.Series < b.Series
		}
		if a.Period != b.Period {
			return a.Period < b.Period
		}
		return a.Vintage.Before(b.Vintage)
	})
	// Drop exact duplicates; flag later vintages of a period as revisions.
	dedup := out[:0]
	for i, o := range out {
		if i > 0 {
			prev := out[i-1]
			if prev.Series == o.Series && prev.Period == o.Period {
				if prev.Vintage.Equal(o.Vintage) {
					continue
				}
				o.Revision = true
			}
		}
		dedup = append(dedup, o)
	}
	return dedup, nil
}

func seriesID(flow, key string, m message) (string, error) {
	dims := m.Structure.Dimensions.Series
	parts := strings.Split(key, ":")
	if len(parts) != len(dims) {
		return "", fmt.Errorf("ecb: series key %q does not match %d dimensions", key, len(dims))
	}
	ids := []string{flow}
	for i, p := range parts {
		k, err := strconv.Atoi(p)
		if err != nil || k < 0 || k >= len(dims[i].Values) {
			return "", fmt.Errorf("ecb: series key %q", key)
		}
		ids = append(ids, dims[i].Values[k].ID)
	}
	return strings.Join(ids, ":"), nil
}

var (
	reYear = regexp.MustCompile(`^(\d{4})$`)
	reHalf = regexp.MustCompile(`^(\d{4})-S([12])$`)
	reQtr  = regexp.MustCompile(`^(\d{4})-Q([1-4])$`)
	reMon  = regexp.MustCompile(`^(\d{4})-(\d{2})$`)
	reWeek = regexp.MustCompile(`^(\d{4})-W(\d{2})$`)
	reDay  = regexp.MustCompile(`^(\d{4})-(\d{2})-(\d{2})$`)
)

// PeriodBounds returns [start, end) of an SDMX time period in UTC.
func PeriodBounds(p string) (time.Time, time.Time, error) {
	atoi := func(s string) int { n, _ := strconv.Atoi(s); return n }
	date := func(y, m, d int) time.Time { return time.Date(y, time.Month(m), d, 0, 0, 0, 0, time.UTC) }
	switch {
	case reDay.MatchString(p):
		s, err := time.Parse("2006-01-02", p)
		return s, s.AddDate(0, 0, 1), err
	case reMon.MatchString(p):
		g := reMon.FindStringSubmatch(p)
		s := date(atoi(g[1]), atoi(g[2]), 1)
		return s, s.AddDate(0, 1, 0), nil
	case reQtr.MatchString(p):
		g := reQtr.FindStringSubmatch(p)
		s := date(atoi(g[1]), 3*(atoi(g[2])-1)+1, 1)
		return s, s.AddDate(0, 3, 0), nil
	case reHalf.MatchString(p):
		g := reHalf.FindStringSubmatch(p)
		s := date(atoi(g[1]), 6*(atoi(g[2])-1)+1, 1)
		return s, s.AddDate(0, 6, 0), nil
	case reYear.MatchString(p):
		s := date(atoi(p), 1, 1)
		return s, s.AddDate(1, 0, 0), nil
	case reWeek.MatchString(p):
		g := reWeek.FindStringSubmatch(p)
		// ISO week 1 contains January 4th.
		jan4 := date(atoi(g[1]), 1, 4)
		monday := jan4.AddDate(0, 0, -((int(jan4.Weekday()) + 6) % 7))
		s := monday.AddDate(0, 0, 7*(atoi(g[2])-1))
		return s, s.AddDate(0, 0, 7), nil
	}
	return time.Time{}, time.Time{}, fmt.Errorf("ecb: unsupported period %q", p)
}

// EstimatedVintage is the deterministic release-time estimate for a period
// whose answer carries no version history.
func EstimatedVintage(period string) (time.Time, error) {
	start, end, err := PeriodBounds(period)
	if err != nil {
		return time.Time{}, err
	}
	if reDay.MatchString(period) {
		return time.Date(start.Year(), start.Month(), start.Day(), 16, 0, 0, 0, berlin).UTC(), nil
	}
	return end, nil
}

// Normalize maps one SDMX-JSON answer to MacroObservations.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	q := raw.TransportMeta[metaQuery]
	flow, _, _ := strings.Cut(q, "/")
	if flow == "" {
		return nil, nil, fmt.Errorf("ecb: raw message without query meta")
	}
	obs, err := Parse(flow, raw.Bytes)
	if err != nil {
		return nil, nil, err
	}
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"
	c.mu.Lock()
	defer c.mu.Unlock()
	floor, newest := c.floor[q], c.cursor[q]
	var envs []*fav1.Envelope
	for _, o := range obs {
		if !backfill {
			key := o.Series + "|" + o.Period + "|" + o.Vintage.Format(time.RFC3339Nano)
			if !floor.IsZero() && !o.Vintage.After(floor) {
				c.seen.Add(key)
				continue
			}
			if !c.seen.Add(key) {
				continue
			}
			if o.Vintage.After(newest) {
				newest = o.Vintage
			}
		}
		_, end, err := PeriodBounds(o.Period)
		if err != nil {
			return nil, nil, err
		}
		publish := o.Vintage
		if o.Estimated && !backfill {
			publish = raw.ReceivedAt
		}
		envs = append(envs, &fav1.Envelope{
			EventId:              ulid.Make().String(),
			SourceId:             c.id(),
			EventTime:            timestamppb.New(end),
			PublishTime:          timestamppb.New(publish),
			IngestTime:           timestamppb.New(raw.ReceivedAt),
			SchemaVersion:        1,
			PublishTimeEstimated: o.Estimated,
			Payload: &fav1.Envelope_Macro{Macro: &fav1.MacroObservation{
				SeriesId: o.Series,
				Period:   o.Period,
				Value:    o.Value,
				Vintage:  timestamppb.New(o.Vintage),
				Revision: o.Revision,
			}},
		})
	}
	if !backfill && newest.After(c.cursor[q]) {
		c.cursor[q] = newest
		if c.deps.Cursors != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = c.deps.Cursors.SetCursor(ctx, c.id(), "", scope(q), newest)
			cancel()
		}
	}
	return envs, nil, nil
}
