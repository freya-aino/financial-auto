// Package googlenews is the Google News RSS connector for catalog id "google_news" (specs 10, 11).
//
// It polls, per query in the catalog entry's `config.queries`,
//
//	https://news.google.com/rss/search?q=<query>&hl=en-US&gl=US&ceid=US:en
//
// and maps every <item> to a NewsItem on `news.item.google_news` with
// dedup_key = the feed guid. De-duplication is by guid in memory plus a
// persisted cursor (ingest.backfill_cursor, scope "items") holding the newest
// pubDate published; after a restart items at or before the cursor are
// skipped. Article links are Google redirect URLs (kept as given, spec 10 §6.5
// leaves URL normalization downstream). Poll only: the feed has no history.
//
// Verified live 2026-11-10: 200, 100 items, RSS 2.0 with <guid isPermaLink="false">,
// <pubDate> RFC 1123 and <source url=...>. The feed's copyright notice limits
// use to personal, non-commercial readers (catalog tos_notes).
package googlenews

import (
	"context"
	"fmt"
	"html"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/fey/financial-auto/go/connectors/rss"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const (
	ID             = "google_news"
	DefaultBaseURL = "https://news.google.com/rss/search"
	cursorScope    = "items"
)

// DefaultQueries apply when the catalog entry has no config.queries.
var DefaultQueries = []string{"stock market"}

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector implements Connector, Poller and Normalizer.
type Connector struct {
	BaseURL string
	Fetch   *connector.Fetcher
	// Locale parameters (defaults hl=en-US, gl=US, ceid=US:en).
	HL, GL, CEID string

	cfg     connector.SourceConfig
	deps    connector.Deps
	queries []string
	seen    *connector.Seen

	mu sync.Mutex
	// floor is the cursor loaded at Init (what earlier runs already published);
	// cursor is the newest time published by this run, persisted for the next.
	floor, cursor time.Time
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Events: []connector.EventKind{connector.KindNews}}
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
	if c.HL == "" {
		c.HL, c.GL, c.CEID = "en-US", "US", "US:en"
	}
	c.queries = cfg.Source.Params().Queries
	if len(c.queries) == 0 {
		c.queries = DefaultQueries
	}
	c.seen = connector.NewSeen(20_000)
	if deps.Cursors != nil {
		if t, ok, err := deps.Cursors.GetCursor(ctx, ID, "", cursorScope); err != nil {
			return fmt.Errorf("googlenews: cursor: %w", err)
		} else if ok {
			c.floor, c.cursor = t, t
		}
	}
	return nil
}

func (c *Connector) Schedule() connector.PollSchedule {
	return connector.PollSchedule{Interval: 5 * time.Minute}
}

// URL builds the RSS search URL for a query.
func (c *Connector) URL(query string) string {
	v := url.Values{"q": {query}, "hl": {c.HL}, "gl": {c.GL}, "ceid": {c.CEID}}
	return c.BaseURL + "?" + v.Encode()
}

// Poll fetches one feed per query.
func (c *Connector) Poll(ctx context.Context, _ connector.PollRequest) ([]connector.RawMessage, error) {
	var out []connector.RawMessage
	var firstErr error
	for _, q := range c.queries {
		u := c.URL(q)
		b, err := c.Fetch.Get(ctx, u)
		if err != nil {
			if ctx.Err() != nil {
				return out, ctx.Err()
			}
			if firstErr == nil {
				firstErr = fmt.Errorf("googlenews %q: %w", q, err)
			}
			continue
		}
		out = append(out, connector.RawMessage{
			SourceID: ID, ReceivedAt: c.deps.Clock.Now(), Bytes: b, ContentType: "application/rss+xml",
			TransportMeta: map[string]string{connector.MetaEndpoint: "rss", connector.MetaURL: u, "query": q},
		})
	}
	if len(out) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// Normalize maps a feed to NewsItems, skipping guids already published.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	feed, err := rss.Parse(raw.Bytes) // shared RSS/Atom parser (spec 11 §6.3)
	if err != nil {
		return nil, nil, fmt.Errorf("googlenews: %w", err)
	}
	lang := strings.ToLower(strings.SplitN(feed.Language, "-", 2)[0])
	if lang == "" {
		lang = "en"
	}
	query := raw.TransportMeta["query"]
	c.mu.Lock()
	defer c.mu.Unlock()
	newest := c.cursor
	var envs []*fav1.Envelope
	for _, it := range feed.Items {
		key := it.Key()
		at := it.Published
		if at.IsZero() {
			at = raw.ReceivedAt
		}
		if !c.floor.IsZero() && !at.After(c.floor) {
			c.seen.Add(key)
			continue // a previous run already published up to floor
		}
		if !c.seen.Add(key) {
			continue
		}
		title, outlet := splitOutlet(html.UnescapeString(it.Title), it.Source)
		var ents []string
		if query != "" {
			ents = []string{"query:" + query}
		}
		envs = append(envs, &fav1.Envelope{
			EventId:       ulid.Make().String(),
			SourceId:      ID,
			EventTime:     timestamppb.New(at),
			PublishTime:   timestamppb.New(at),
			IngestTime:    timestamppb.New(raw.ReceivedAt),
			SchemaVersion: 1,
			Payload: &fav1.Envelope_News{News: &fav1.NewsItem{
				Title: title, Url: it.Link, Language: lang, Outlet: outlet, EntitiesRaw: ents, DedupKey: key,
			}},
		})
		if at.After(newest) {
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

// splitOutlet strips the " - Outlet" suffix Google appends to titles.
func splitOutlet(title, source string) (string, string) {
	if source != "" {
		if t, ok := strings.CutSuffix(title, " - "+source); ok {
			return t, source
		}
		return title, source
	}
	if i := strings.LastIndex(title, " - "); i > 0 {
		return title[:i], title[i+3:]
	}
	return title, ""
}
