// Package rss is the generic RSS 2.0 / Atom adapter (spec 11 §6.3): any
// catalog entry with `adapter: rss` and feed URLs in `config.extra.feeds`
// (comma- or space-separated) runs on it, polling every feed and mapping each
// item to a NewsItem on `news.item.<source>`. Google News (`google_news`)
// shares the feed parser (Parse) but keeps its own connector for the search
// URL. No catalog entry uses the adapter yet (outlet feeds would be new
// sources).
//
// config.extra:
//
//	feeds          feed URLs (required)
//	poll_interval  Go duration, default 5m
//	outlet         outlet name when the feed has none (default: feed title)
//	language       ISO 639-1 when the feed has none (default "en")
//
// De-duplication: in memory by guid (else link) plus one persisted cursor per
// feed (ingest.backfill_cursor, scope "feed:<url>") holding the newest item
// time published; after a restart items at or before it are skipped. The
// pipeline additionally de-duplicates across sources by the normalized URL
// (spec 10 §6.5). Poll only: feeds carry no history.
package rss

import (
	"context"
	"encoding/xml"
	"fmt"
	"html"
	"strings"
	"sync"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// Adapter is the catalog `adapter:` name.
const Adapter = "rss"

func init() {
	connector.RegisterAdapter(Adapter, func(id string) connector.Connector { return &Connector{SourceID: id} })
}

// Item is one feed entry, format-independent.
type Item struct {
	Title, Link, GUID, Description string
	// Published is the item's pubDate / published / updated time (zero when absent or unparsed).
	Published time.Time
	// Source is the RSS <source> outlet name, if any.
	Source string
}

// Feed is a parsed RSS 2.0 channel or Atom feed.
type Feed struct {
	Title, Language string
	Items           []Item
}

type rssDoc struct {
	XMLName xml.Name
	Channel struct {
		Title    string `xml:"title"`
		Language string `xml:"language"`
		Items    []struct {
			Title       string `xml:"title"`
			Link        string `xml:"link"`
			GUID        string `xml:"guid"`
			PubDate     string `xml:"pubDate"`
			Date        string `xml:"http://purl.org/dc/elements/1.1/ date"`
			Description string `xml:"description"`
			Source      struct {
				Name string `xml:",chardata"`
			} `xml:"source"`
		} `xml:"item"`
	} `xml:"channel"`
}

type atomDoc struct {
	Title   string `xml:"title"`
	Lang    string `xml:"http://www.w3.org/XML/1998/namespace lang,attr"`
	Entries []struct {
		Title string `xml:"title"`
		ID    string `xml:"id"`
		Links []struct {
			Href string `xml:"href,attr"`
			Rel  string `xml:"rel,attr"`
		} `xml:"link"`
		Published string `xml:"published"`
		Updated   string `xml:"updated"`
		Summary   string `xml:"summary"`
		Source    struct {
			Title string `xml:"title"`
		} `xml:"source"`
	} `xml:"entry"`
}

// Parse reads an RSS 2.0 or Atom document.
func Parse(b []byte) (Feed, error) {
	var probe struct{ XMLName xml.Name }
	if err := xml.Unmarshal(b, &probe); err != nil {
		return Feed{}, fmt.Errorf("rss: decode: %w", err)
	}
	switch probe.XMLName.Local {
	case "rss", "RDF":
		var d rssDoc
		if err := xml.Unmarshal(b, &d); err != nil {
			return Feed{}, fmt.Errorf("rss: decode: %w", err)
		}
		f := Feed{Title: strings.TrimSpace(d.Channel.Title), Language: strings.TrimSpace(d.Channel.Language)}
		for _, it := range d.Channel.Items {
			date := it.PubDate
			if date == "" {
				date = it.Date
			}
			pub, _ := ParseDate(date)
			f.Items = append(f.Items, Item{Title: strings.TrimSpace(it.Title), Link: strings.TrimSpace(it.Link), GUID: strings.TrimSpace(it.GUID),
				Description: it.Description, Published: pub, Source: strings.TrimSpace(it.Source.Name)})
		}
		return f, nil
	case "feed":
		var d atomDoc
		if err := xml.Unmarshal(b, &d); err != nil {
			return Feed{}, fmt.Errorf("atom: decode: %w", err)
		}
		f := Feed{Title: strings.TrimSpace(d.Title), Language: d.Lang}
		for _, e := range d.Entries {
			link := ""
			for _, l := range e.Links {
				if l.Rel == "" || l.Rel == "alternate" {
					link = l.Href
					break
				}
			}
			if link == "" && len(e.Links) > 0 {
				link = e.Links[0].Href
			}
			date := e.Published
			if date == "" {
				date = e.Updated
			}
			pub, _ := ParseDate(date)
			f.Items = append(f.Items, Item{Title: strings.TrimSpace(e.Title), Link: strings.TrimSpace(link), GUID: strings.TrimSpace(e.ID),
				Description: e.Summary, Published: pub, Source: strings.TrimSpace(e.Source.Title)})
		}
		return f, nil
	}
	return Feed{}, fmt.Errorf("rss: unknown root element <%s>", probe.XMLName.Local)
}

// ParseDate accepts the RFC 822/1123 dates of RSS and the RFC 3339 dates of Atom.
func ParseDate(s string) (time.Time, error) {
	s = strings.TrimSpace(s)
	for _, layout := range []string{time.RFC1123, time.RFC1123Z, time.RFC822, time.RFC822Z, time.RFC3339, time.RFC3339Nano,
		"Mon, 2 Jan 2006 15:04:05 MST", "Mon, 2 Jan 2006 15:04:05 -0700", "2006-01-02T15:04:05Z07:00"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("unparsed date %q", s)
}

// Key is the per-feed de-duplication key of an item: guid, else link.
func (it Item) Key() string {
	if it.GUID != "" {
		return it.GUID
	}
	return it.Link
}

// Connector is the generic adapter (Connector, Poller, Normalizer).
type Connector struct {
	SourceID string
	Fetch    *connector.Fetcher

	deps     connector.Deps
	feeds    []string
	interval time.Duration
	outlet   string
	lang     string
	seen     *connector.Seen

	mu            sync.Mutex
	floor, cursor map[string]time.Time
}

func (c *Connector) ID() string { return c.SourceID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Events: []connector.EventKind{connector.KindNews}}
}

// Feeds splits config.extra.feeds.
func Feeds(extra map[string]string) []string {
	return strings.FieldsFunc(extra["feeds"], func(r rune) bool { return r == ',' || r == ' ' || r == '\n' || r == '\t' })
}

func (c *Connector) Init(ctx context.Context, cfg connector.SourceConfig, deps connector.Deps) error {
	if deps.Clock == nil {
		deps.Clock = connector.SystemClock{}
	}
	if c.SourceID == "" {
		c.SourceID = cfg.Source.ID
	}
	c.deps = deps
	if c.Fetch == nil {
		c.Fetch = connector.NewFetcher(deps.Limiter)
	}
	extra := cfg.Source.Params().Extra
	c.feeds = Feeds(extra)
	if len(c.feeds) == 0 {
		return fmt.Errorf("rss %s: config.extra.feeds is empty", c.SourceID)
	}
	c.interval = 5 * time.Minute
	if v := extra["poll_interval"]; v != "" {
		d, err := time.ParseDuration(v)
		if err != nil || d < 10*time.Second {
			return fmt.Errorf("rss %s: poll_interval %q (Go duration ≥ 10s)", c.SourceID, v)
		}
		c.interval = d
	}
	c.outlet, c.lang = extra["outlet"], extra["language"]
	c.seen = connector.NewSeen(20_000)
	c.floor, c.cursor = map[string]time.Time{}, map[string]time.Time{}
	if deps.Cursors != nil {
		for _, f := range c.feeds {
			if t, ok, err := deps.Cursors.GetCursor(ctx, c.SourceID, "", scope(f)); err != nil {
				return fmt.Errorf("rss %s: cursor: %w", c.SourceID, err)
			} else if ok {
				c.floor[f], c.cursor[f] = t, t
			}
		}
	}
	return nil
}

func scope(feed string) string { return "feed:" + feed }

func (c *Connector) Schedule() connector.PollSchedule {
	return connector.PollSchedule{Interval: c.interval}
}

// Poll fetches every feed.
func (c *Connector) Poll(ctx context.Context, _ connector.PollRequest) ([]connector.RawMessage, error) {
	var out []connector.RawMessage
	var firstErr error
	for _, f := range c.feeds {
		b, err := c.Fetch.Get(ctx, f)
		if err != nil {
			if ctx.Err() != nil {
				return out, ctx.Err()
			}
			if firstErr == nil {
				firstErr = fmt.Errorf("rss %s %s: %w", c.SourceID, f, err)
			}
			continue
		}
		out = append(out, connector.RawMessage{SourceID: c.SourceID, ReceivedAt: c.deps.Clock.Now(), Bytes: b, ContentType: "application/rss+xml",
			TransportMeta: map[string]string{connector.MetaEndpoint: "feed", connector.MetaURL: f}})
	}
	if len(out) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// Normalize maps a feed to NewsItems, skipping items already published.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	feed, err := Parse(raw.Bytes)
	if err != nil {
		return nil, nil, err
	}
	url := raw.TransportMeta[connector.MetaURL]
	lang := strings.ToLower(strings.SplitN(feed.Language, "-", 2)[0])
	if lang == "" {
		lang = c.lang
	}
	if lang == "" {
		lang = "en"
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	floor := c.floor[url]
	newest := c.cursor[url]
	var envs []*fav1.Envelope
	for _, it := range feed.Items {
		key := it.Key()
		if key == "" {
			continue
		}
		at := it.Published
		if at.IsZero() {
			at = raw.ReceivedAt
		}
		if !floor.IsZero() && !at.After(floor) {
			c.seen.Add(key)
			continue
		}
		if !c.seen.Add(key) {
			continue
		}
		outlet := it.Source
		if outlet == "" {
			outlet = c.outlet
		}
		if outlet == "" {
			outlet = feed.Title
		}
		envs = append(envs, &fav1.Envelope{
			SourceId:      c.SourceID,
			EventTime:     timestamppb.New(at),
			PublishTime:   timestamppb.New(at),
			IngestTime:    timestamppb.New(raw.ReceivedAt),
			SchemaVersion: 1,
			Payload: &fav1.Envelope_News{News: &fav1.NewsItem{
				Title: html.UnescapeString(it.Title), Url: it.Link, Language: lang, Outlet: outlet, DedupKey: key,
			}},
		})
		if at.After(newest) {
			newest = at
		}
	}
	if newest.After(c.cursor[url]) {
		c.cursor[url] = newest
		if c.deps.Cursors != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = c.deps.Cursors.SetCursor(ctx, c.SourceID, "", scope(url), newest)
			cancel()
		}
	}
	return envs, nil, nil
}
