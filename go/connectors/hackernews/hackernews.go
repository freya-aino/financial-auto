// Package hackernews is the Hacker News connector for catalog id "hackernews" (specs 10, 11).
//
// Live polling uses the official Firebase API (no auth):
//
//	GET https://hacker-news.firebaseio.com/v0/topstories.json   → [id, ...]
//	GET https://hacker-news.firebaseio.com/v0/item/<id>.json     → story
//
// Each not-yet-seen story becomes a NewsItem on `news.item.hackernews` with
// dedup_key "hn:<id>". De-duplication is by id in memory plus a persisted
// cursor (ingest.backfill_cursor, scope "items", instrument null) holding the
// newest story time seen; after a restart stories older than the cursor are
// skipped. Backfill uses the Algolia HN search API (search_by_date with
// created_at_i filters), which the same Normalize understands.
//
// Verified live 2026-11-10: topstories (500 ids), item/8863 and Algolia all 200.
package hackernews

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"regexp"
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
	ID                = "hackernews"
	Outlet            = "news.ycombinator.com"
	DefaultBaseURL    = "https://hacker-news.firebaseio.com/v0"
	DefaultAlgoliaURL = "https://hn.algolia.com/api/v1"
	cursorScope       = "items"
	// MaxNewPerPoll bounds item fetches per poll (each is one request).
	MaxNewPerPoll = 30
)

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector implements Connector, Poller, Backfiller and Normalizer.
type Connector struct {
	BaseURL    string
	AlgoliaURL string
	Fetch      *connector.Fetcher
	// Lists are the Firebase story lists polled (default newstories, topstories).
	Lists []string

	cfg  connector.SourceConfig
	deps connector.Deps
	// fetched de-dups list ids across polls; seen de-dups published stories.
	fetched *connector.Seen
	seen    *connector.Seen

	mu sync.Mutex
	// floor is the cursor loaded at Init (what earlier runs already published);
	// cursor is the newest time published by this run, persisted for the next.
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
	if c.AlgoliaURL == "" {
		c.AlgoliaURL = DefaultAlgoliaURL
	}
	if c.Fetch == nil {
		c.Fetch = connector.NewFetcher(deps.Limiter)
	}
	if len(c.Lists) == 0 {
		c.Lists = []string{"newstories", "topstories"} // newest first so the per-poll budget reaches fresh stories
	}
	c.fetched = connector.NewSeen(20_000)
	c.seen = connector.NewSeen(20_000)
	if deps.Cursors != nil {
		if t, ok, err := deps.Cursors.GetCursor(ctx, ID, "", cursorScope); err != nil {
			return fmt.Errorf("hackernews: cursor: %w", err)
		} else if ok {
			c.floor, c.cursor = t, t
		}
	}
	return nil
}

func (c *Connector) Schedule() connector.PollSchedule {
	return connector.PollSchedule{Interval: time.Minute}
}

// Poll lists stories and fetches every unseen id (bounded).
func (c *Connector) Poll(ctx context.Context, _ connector.PollRequest) ([]connector.RawMessage, error) {
	var ids []int64
	for _, list := range c.Lists {
		b, err := c.Fetch.Get(ctx, c.BaseURL+"/"+list+".json")
		if err != nil {
			return nil, fmt.Errorf("hackernews %s: %w", list, err)
		}
		var l []int64
		if err := json.Unmarshal(b, &l); err != nil {
			return nil, fmt.Errorf("hackernews %s: decode: %w", list, err)
		}
		ids = append(ids, l...)
	}
	var out []connector.RawMessage
	for _, id := range ids {
		if len(out) >= MaxNewPerPoll {
			break
		}
		key := strconv.FormatInt(id, 10)
		if c.fetched.Has(key) {
			continue
		}
		u := fmt.Sprintf("%s/item/%s.json", c.BaseURL, key)
		b, err := c.Fetch.Get(ctx, u)
		if err != nil {
			if ctx.Err() != nil {
				return out, ctx.Err()
			}
			continue // one bad item must not stall the list
		}
		c.fetched.Add(key) // Normalize de-dups published stories on the payload id
		if string(b) == "null" {
			continue
		}
		out = append(out, connector.RawMessage{
			SourceID: ID, ReceivedAt: c.deps.Clock.Now(), Bytes: b, ContentType: "application/json",
			TransportMeta: map[string]string{connector.MetaEndpoint: "item", connector.MetaURL: u},
		})
	}
	return out, nil
}

// Backfill pages Algolia search_by_date over [From, To).
func (c *Connector) Backfill(ctx context.Context, req connector.BackfillRequest, out chan<- connector.RawMessage) error {
	to := req.To
	if to.IsZero() {
		to = c.deps.Clock.Now()
	}
	for page := 0; page < 1000; page++ {
		u := fmt.Sprintf("%s/search_by_date?tags=story&hitsPerPage=100&page=%d&numericFilters=created_at_i%%3E%%3D%d,created_at_i%%3C%d",
			c.AlgoliaURL, page, req.From.Unix(), to.Unix())
		b, err := c.Fetch.Get(ctx, u)
		if err != nil {
			return fmt.Errorf("hackernews backfill: %w", err)
		}
		var probe struct {
			NbPages int   `json:"nbPages"`
			Hits    []any `json:"hits"`
		}
		if err := json.Unmarshal(b, &probe); err != nil {
			return fmt.Errorf("hackernews backfill: decode: %w", err)
		}
		if len(probe.Hits) == 0 {
			return nil
		}
		m := connector.RawMessage{
			SourceID: ID, ReceivedAt: c.deps.Clock.Now(), Bytes: b, ContentType: "application/json",
			TransportMeta: map[string]string{connector.MetaEndpoint: "algolia", connector.MetaURL: u, connector.MetaBackfill: "1"},
		}
		select {
		case out <- m:
		case <-ctx.Done():
			return ctx.Err()
		}
		if page+1 >= probe.NbPages {
			return nil
		}
	}
	return nil
}

type item struct {
	ID    int64  `json:"id"`
	Type  string `json:"type"`
	By    string `json:"by"`
	Time  int64  `json:"time"`
	Title string `json:"title"`
	Text  string `json:"text"`
	URL   string `json:"url"`
	Score int    `json:"score"`
	Dead  bool   `json:"dead"`
	Del   bool   `json:"deleted"`
}

type algolia struct {
	Hits []struct {
		ObjectID  string `json:"objectID"`
		Title     string `json:"title"`
		URL       string `json:"url"`
		Author    string `json:"author"`
		StoryText string `json:"story_text"`
		CreatedAt int64  `json:"created_at_i"`
	} `json:"hits"`
}

var tagRe = regexp.MustCompile(`<[^>]+>`)

func stripHTML(s string) string {
	s = strings.NewReplacer("<p>", "\n", "<br>", "\n").Replace(s)
	return strings.TrimSpace(html.UnescapeString(tagRe.ReplaceAllString(s, "")))
}

// Normalize maps one Firebase item or one Algolia page to NewsItem envelopes.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var items []item
	switch raw.TransportMeta[connector.MetaEndpoint] {
	case "algolia":
		var a algolia
		if err := json.Unmarshal(raw.Bytes, &a); err != nil {
			return nil, nil, fmt.Errorf("hackernews: decode algolia: %w", err)
		}
		for _, h := range a.Hits {
			id, _ := strconv.ParseInt(h.ObjectID, 10, 64)
			items = append(items, item{ID: id, Type: "story", By: h.Author, Time: h.CreatedAt, Title: h.Title, Text: h.StoryText, URL: h.URL})
		}
	default:
		var it item
		if err := json.Unmarshal(raw.Bytes, &it); err != nil {
			return nil, nil, fmt.Errorf("hackernews: decode item: %w", err)
		}
		items = append(items, it)
	}
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"
	var envs []*fav1.Envelope
	c.mu.Lock()
	defer c.mu.Unlock()
	newest := c.cursor
	for _, it := range items {
		if it.ID == 0 || it.Dead || it.Del || (it.Type != "" && it.Type != "story" && it.Type != "job") {
			continue
		}
		key := strconv.FormatInt(it.ID, 10)
		at := time.Unix(it.Time, 0).UTC()
		if !backfill {
			if !c.floor.IsZero() && !at.After(c.floor) {
				c.seen.Add(key)
				continue // a previous run already published up to floor
			}
			if !c.seen.Add(key) {
				continue
			}
		}
		url := it.URL
		if url == "" {
			url = "https://news.ycombinator.com/item?id=" + key
		}
		envs = append(envs, &fav1.Envelope{
			EventId:       ulid.Make().String(),
			SourceId:      ID,
			EventTime:     timestamppb.New(at),
			PublishTime:   timestamppb.New(at),
			IngestTime:    timestamppb.New(raw.ReceivedAt),
			SchemaVersion: 1,
			Payload: &fav1.Envelope_News{News: &fav1.NewsItem{
				Title:       html.UnescapeString(it.Title),
				Body:        stripHTML(it.Text),
				Url:         url,
				Language:    "en",
				Outlet:      Outlet,
				EntitiesRaw: entities(it.By),
				DedupKey:    "hn:" + key,
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

func entities(by string) []string {
	if by == "" {
		return nil
	}
	return []string{"author:" + by}
}
