// Package gdelt is the GDELT DOC 2.0 connector for catalog id "gdelt" (specs 10, 11).
//
// It polls, per query in the catalog entry's `config.queries`,
//
//	https://api.gdeltproject.org/api/v2/doc/doc?query=<q>&mode=artlist&format=json&maxrecords=<n>&sort=datedesc&timespan=<span>
//
// and maps every article to a NewsItem on `news.item.gdelt`. GDELT gives no
// article publication time, only `seendate` (when its crawler first saw the
// article, UTC, 15-minute granularity); it is used for event_time and
// publish_time (the article was demonstrably public by then).
//
// dedup_key = "url:" + hex(sha1(normalized URL)) (spec 10 §6.5): scheme and
// host lower-cased, a leading "www." dropped, fragment and utm_* parameters
// removed, trailing "/" trimmed. Live de-duplication is by that key in memory
// plus a persisted cursor (ingest.backfill_cursor, scope "items") holding the
// newest seendate published; after a restart articles at or before it are
// skipped (same floor logic as google_news).
//
// Backfill walks [From, To) in windows (default 1 day, maxrecords 250 each,
// newest first) with startdatetime/enddatetime. The DOC API only searches a
// rolling window of roughly the last 3 months, so From is clamped to
// To − 90 days; a window with more than maxrecords articles is truncated by
// GDELT (no paging exists).
//
// Error handling: GDELT answers 429 with a plain-text body when called more
// than once per 5 s (the Fetcher retries those), and 200 with a plain-text
// body for invalid queries ("Your query was too short or too long."); any
// non-JSON body is an error. An empty result is `{}`.
//
// Verified live 2026-11-11: 200 with {"articles":[{url,url_mobile,title,
// seendate,socialimage,domain,language,sourcecountry}]}; 429 text when
// rate-limited; 200 text for a one-letter query.
package gdelt

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
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
	ID             = "gdelt"
	DefaultBaseURL = "https://api.gdeltproject.org/api/v2/doc/doc"
	cursorScope    = "items"
	// MaxRecords is GDELT's per-request cap for artlist.
	MaxRecords = 250
	// MaxBackfillDepth is how far back the DOC API searches.
	MaxBackfillDepth = 90 * 24 * time.Hour
	seenLayout       = "20060102T150405Z"
	rangeLayout      = "20060102150405"
)

// DefaultQueries apply when the catalog entry has no config.queries.
var DefaultQueries = []string{`"stock market" sourcelang:english`}

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector implements Connector, Poller, Backfiller and Normalizer.
type Connector struct {
	BaseURL string
	Fetch   *connector.Fetcher
	// Timespan is the live look-back per poll (config.extra.timespan, default "1h").
	Timespan string
	// MaxRecords per live poll (config.extra.maxrecords, default 250).
	MaxRecords int
	// Window is the backfill request span (config.extra.window, default 24h).
	Window time.Duration

	cfg     connector.SourceConfig
	deps    connector.Deps
	queries []string
	seen    *connector.Seen

	mu            sync.Mutex
	floor, cursor time.Time
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Backfill: true, Events: []connector.EventKind{connector.KindNews}}
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
	c.queries = p.Queries
	if len(c.queries) == 0 {
		c.queries = DefaultQueries
	}
	if c.Timespan == "" {
		c.Timespan = p.Extra["timespan"]
	}
	if c.Timespan == "" {
		c.Timespan = "1h"
	}
	if c.MaxRecords == 0 {
		if n, err := strconv.Atoi(p.Extra["maxrecords"]); err == nil && n > 0 {
			c.MaxRecords = n
		} else {
			c.MaxRecords = MaxRecords
		}
	}
	c.MaxRecords = min(c.MaxRecords, MaxRecords)
	if c.Window == 0 {
		if d, err := time.ParseDuration(p.Extra["window"]); err == nil && d > 0 {
			c.Window = d
		} else {
			c.Window = 24 * time.Hour
		}
	}
	c.seen = connector.NewSeen(50_000)
	if deps.Cursors != nil {
		if t, ok, err := deps.Cursors.GetCursor(ctx, ID, "", cursorScope); err != nil {
			return fmt.Errorf("gdelt: cursor: %w", err)
		} else if ok {
			c.floor, c.cursor = t, t
		}
	}
	return nil
}

// Schedule: GDELT refreshes every 15 minutes.
func (c *Connector) Schedule() connector.PollSchedule {
	return connector.PollSchedule{Interval: 15 * time.Minute}
}

// URL builds the live artlist URL for a query.
func (c *Connector) URL(query string) string {
	v := url.Values{"query": {query}, "mode": {"artlist"}, "format": {"json"}, "sort": {"datedesc"},
		"maxrecords": {strconv.Itoa(c.MaxRecords)}, "timespan": {c.Timespan}}
	return c.BaseURL + "?" + v.Encode()
}

// RangeURL builds a backfill artlist URL for [from, to).
func (c *Connector) RangeURL(query string, from, to time.Time) string {
	v := url.Values{"query": {query}, "mode": {"artlist"}, "format": {"json"}, "sort": {"datedesc"},
		"maxrecords":    {strconv.Itoa(MaxRecords)},
		"startdatetime": {from.UTC().Format(rangeLayout)}, "enddatetime": {to.UTC().Format(rangeLayout)}}
	return c.BaseURL + "?" + v.Encode()
}

// get fetches u and rejects GDELT's plain-text error answers.
func (c *Connector) get(ctx context.Context, u string) ([]byte, error) {
	b, err := c.Fetch.Get(ctx, u)
	if err != nil {
		return nil, err
	}
	if err := checkJSON(b); err != nil {
		return nil, err
	}
	return b, nil
}

func checkJSON(b []byte) error {
	t := bytes.TrimSpace(b)
	if len(t) == 0 || t[0] != '{' {
		msg := string(t)
		if len(msg) > 200 {
			msg = msg[:200]
		}
		return fmt.Errorf("gdelt: non-JSON answer: %q", msg)
	}
	return nil
}

// Poll fetches the latest articles per query.
func (c *Connector) Poll(ctx context.Context, _ connector.PollRequest) ([]connector.RawMessage, error) {
	var out []connector.RawMessage
	var firstErr error
	for _, q := range c.queries {
		u := c.URL(q)
		b, err := c.get(ctx, u)
		if err != nil {
			if ctx.Err() != nil {
				return out, ctx.Err()
			}
			if firstErr == nil {
				firstErr = fmt.Errorf("gdelt %q: %w", q, err)
			}
			continue
		}
		out = append(out, connector.RawMessage{
			SourceID: ID, ReceivedAt: c.deps.Clock.Now(), Bytes: b, ContentType: "application/json",
			TransportMeta: map[string]string{connector.MetaEndpoint: "artlist", connector.MetaURL: u, "query": q},
		})
	}
	if len(out) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// Backfill walks [From, To) in Window-sized requests per query
// (req.VenueSymbol, when set, is the only query).
func (c *Connector) Backfill(ctx context.Context, req connector.BackfillRequest, out chan<- connector.RawMessage) error {
	to := req.To
	if to.IsZero() {
		to = c.deps.Clock.Now()
	}
	from := req.From
	if cur, err := time.Parse(time.RFC3339, req.Cursor); req.Cursor != "" && err == nil && cur.After(from) {
		from = cur
	}
	if floor := to.Add(-MaxBackfillDepth); from.Before(floor) {
		from = floor
	}
	queries := c.queries
	if req.VenueSymbol != "" {
		queries = []string{req.VenueSymbol}
	}
	for _, q := range queries {
		for start := from; start.Before(to); start = start.Add(c.Window) {
			end := start.Add(c.Window)
			if end.After(to) {
				end = to
			}
			u := c.RangeURL(q, start, end)
			b, err := c.get(ctx, u)
			if err != nil {
				return fmt.Errorf("gdelt backfill %q: %w", q, err)
			}
			m := connector.RawMessage{
				SourceID: ID, ReceivedAt: c.deps.Clock.Now(), Bytes: b, ContentType: "application/json",
				TransportMeta: map[string]string{connector.MetaEndpoint: "artlist", connector.MetaURL: u, "query": q, connector.MetaBackfill: "1"},
			}
			select {
			case out <- m:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	return nil
}

type artlist struct {
	Articles []struct {
		URL           string `json:"url"`
		Title         string `json:"title"`
		SeenDate      string `json:"seendate"`
		Domain        string `json:"domain"`
		Language      string `json:"language"`
		SourceCountry string `json:"sourcecountry"`
	} `json:"articles"`
}

// Normalize maps one artlist answer to NewsItems. Live answers skip articles
// already published (seen set + cursor floor); backfill answers emit all.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	if err := checkJSON(raw.Bytes); err != nil {
		return nil, nil, err
	}
	var a artlist
	if err := json.Unmarshal(raw.Bytes, &a); err != nil {
		return nil, nil, fmt.Errorf("gdelt: decode: %w", err)
	}
	query := raw.TransportMeta["query"]
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"
	c.mu.Lock()
	defer c.mu.Unlock()
	newest := c.cursor
	var envs []*fav1.Envelope
	for _, art := range a.Articles {
		if art.URL == "" {
			continue
		}
		key := DedupKey(art.URL)
		at, err := time.Parse(seenLayout, art.SeenDate)
		if err != nil {
			at = raw.ReceivedAt
		}
		at = at.UTC()
		if !backfill {
			if !c.floor.IsZero() && !at.After(c.floor) {
				c.seen.Add(key)
				continue
			}
			if !c.seen.Add(key) {
				continue
			}
		}
		var ents []string
		if query != "" {
			ents = append(ents, "query:"+query)
		}
		if art.SourceCountry != "" {
			ents = append(ents, "country:"+art.SourceCountry)
		}
		envs = append(envs, &fav1.Envelope{
			EventId:       ulid.Make().String(),
			SourceId:      ID,
			EventTime:     timestamppb.New(at),
			PublishTime:   timestamppb.New(at),
			IngestTime:    timestamppb.New(raw.ReceivedAt),
			SchemaVersion: 1,
			Payload: &fav1.Envelope_News{News: &fav1.NewsItem{
				Title:       cleanTitle(art.Title),
				Url:         art.URL,
				Language:    Language(art.Language),
				Outlet:      art.Domain,
				EntitiesRaw: ents,
				DedupKey:    key,
			}},
		})
		if !backfill && at.After(newest) {
			newest = at
		}
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

// NormalizeURL canonicalizes an article URL for de-duplication.
func NormalizeURL(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Host == "" {
		return strings.TrimSpace(raw)
	}
	u.Scheme = strings.ToLower(u.Scheme)
	u.Host = strings.TrimPrefix(strings.ToLower(u.Host), "www.")
	u.Fragment, u.RawFragment = "", ""
	if u.RawQuery != "" {
		q := u.Query()
		for k := range q {
			if strings.HasPrefix(strings.ToLower(k), "utm_") {
				q.Del(k)
			}
		}
		u.RawQuery = q.Encode()
	}
	u.Path = strings.TrimSuffix(u.Path, "/")
	u.RawPath = ""
	return u.String()
}

// DedupKey is "url:" + hex(sha1(NormalizeURL(u))).
func DedupKey(u string) string {
	h := sha1.Sum([]byte(NormalizeURL(u)))
	return "url:" + hex.EncodeToString(h[:])
}

// GDELT tokenizes titles with spaces around punctuation ("Why wont Korea ? ").
var titleFix = strings.NewReplacer(" ,", ",", " .", ".", " ?", "?", " !", "!", " :", ":", " ;", ";")

func cleanTitle(s string) string { return strings.TrimSpace(titleFix.Replace(strings.TrimSpace(s))) }

var languages = map[string]string{
	"english": "en", "spanish": "es", "french": "fr", "german": "de", "italian": "it", "portuguese": "pt",
	"dutch": "nl", "russian": "ru", "chinese": "zh", "japanese": "ja", "korean": "ko", "arabic": "ar",
	"turkish": "tr", "polish": "pl", "swedish": "sv", "norwegian": "no", "danish": "da", "finnish": "fi",
	"greek": "el", "hindi": "hi", "indonesian": "id", "hebrew": "he", "czech": "cs", "hungarian": "hu",
	"romanian": "ro", "ukrainian": "uk", "vietnamese": "vi", "thai": "th",
}

// Language maps a GDELT language name to ISO 639-1, else the lower-cased name.
func Language(name string) string {
	n := strings.ToLower(strings.TrimSpace(name))
	if iso, ok := languages[n]; ok {
		return iso
	}
	return n
}
