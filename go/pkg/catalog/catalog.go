// Package catalog loads and validates the source catalog (config/sources.yaml, spec 11).
package catalog

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"go.yaml.in/yaml/v3"
)

// Allowed enum values, mirroring the header of config/sources.yaml.
var (
	Statuses     = []string{"planned", "stub", "live", "disabled"}
	Capabilities = []string{"data", "execution"}
	Transports   = []string{"rest", "websocket", "fix", "grpc", "graphql", "sse", "socket", "file", "sdk", "stream"}
	Auths        = []string{"none", "api_key", "oauth2", "hmac", "session", "wallet", "gateway"}
	// Timeframes a catalog entry may ask a connector to poll (bar subjects, spec 21).
	Timeframes = []string{"1m", "5m", "15m", "1h", "1d"}
)

// RateLimit is an optional per-source token bucket (`rate_limit:` in
// sources.yaml); sources without one fall back to DefaultRateLimit.
type RateLimit struct {
	PerSecond float64 `yaml:"per_second"`
	Burst     int     `yaml:"burst"`
}

// DefaultRateLimit applies when a source declares none (placeholder, spec 10 tunables).
var DefaultRateLimit = RateLimit{PerSecond: 5, Burst: 10}

// Params is the optional free-form connector configuration of a catalog
// entry (`config:`). Secrets never live here (spec 11 §4, D-020).
type Params struct {
	// Symbols are venue symbols the connector polls/streams (tickers, ISINs, FX pairs).
	Symbols []string `yaml:"symbols,omitempty" json:"symbols,omitempty"`
	// Queries are search terms for text sources (Google News).
	Queries []string `yaml:"queries,omitempty" json:"queries,omitempty"`
	// Timeframe is the bar timeframe polled live (default per connector, e.g. "1m").
	Timeframe string `yaml:"timeframe,omitempty" json:"timeframe,omitempty"`
	// Extra holds connector-specific string settings (e.g. lemon `mic`, oanda `env`).
	Extra map[string]string `yaml:"extra,omitempty" json:"extra,omitempty"`
}

// Source is one catalog entry.
type Source struct {
	ID           string     `yaml:"id" json:"id"`
	Name         string     `yaml:"name" json:"name"`
	Category     string     `yaml:"category" json:"category"`
	Capabilities []string   `yaml:"capabilities" json:"capabilities"`
	AssetClasses []string   `yaml:"asset_classes" json:"asset_classes"`
	Transport    []string   `yaml:"transport" json:"transport"`
	Auth         string     `yaml:"auth" json:"auth"`
	Status       string     `yaml:"status" json:"status"`
	TosNotes     string     `yaml:"tos_notes" json:"tos_notes"`
	Notes        string     `yaml:"notes" json:"notes"`
	RateLimit    *RateLimit `yaml:"rate_limit,omitempty" json:"rate_limit,omitempty"`
	Config       *Params    `yaml:"config,omitempty" json:"config,omitempty"`
	// Adapter names a generic connector (Adapters: rss, sdmx) that serves this
	// entry when no connector is registered under its id (spec 11 §6.3).
	Adapter string `yaml:"adapter,omitempty" json:"adapter,omitempty"`
	// Backfill holds replay/backfill facts (D-079 knowledge delay).
	Backfill *BackfillInfo `yaml:"backfill,omitempty" json:"backfill,omitempty"`
}

// BackfillInfo is the `backfill:` block of a catalog entry.
type BackfillInfo struct {
	// KnowledgeDelaySecs is the measured delay until an event is on the bus (D-079);
	// the ingestor's clock-skew monitor compares the live ingest lag against it.
	KnowledgeDelaySecs int `yaml:"knowledge_delay_secs,omitempty" json:"knowledge_delay_secs,omitempty"`
}

// Adapters are the generic connectors a catalog entry may name in `adapter:`.
var Adapters = []string{"rss", "sdmx"}

// HasCapability reports whether the entry declares cap (data | execution).
func (s Source) HasCapability(c string) bool { return slices.Contains(s.Capabilities, c) }

// Override is a runtime edit of a source's rate limit and connector config
// (ctl.ingest.configure, spec 11 §4 PATCH). A nil field leaves the YAML value.
type Override struct {
	RateLimit *RateLimit `json:"rate_limit,omitempty"`
	Config    *Params    `json:"config,omitempty"`
}

// IsZero reports an empty override.
func (o Override) IsZero() bool { return o.RateLimit == nil && o.Config == nil }

// Validate checks an override with the catalog rules for rate_limit and config.
func (o Override) Validate() error {
	s := Source{ID: "override", Name: "override", Status: "planned", Auth: "none", RateLimit: o.RateLimit, Config: o.Config}
	return (&Catalog{Sources: []Source{s}}).Validate()
}

// Params returns the entry's connector configuration or an empty one.
func (s Source) Params() Params {
	if s.Config != nil {
		return *s.Config
	}
	return Params{}
}

// Limit returns the source's rate limit or the default.
func (s Source) Limit() RateLimit {
	if s.RateLimit != nil {
		return *s.RateLimit
	}
	return DefaultRateLimit
}

// Catalog is the parsed file. Status edits at runtime (ctl.ingest.enable|disable,
// D-054) go through SetStatus so readers and the ingestor's control handlers may
// share one instance.
type Catalog struct {
	Version int      `yaml:"version"`
	Sources []Source `yaml:"sources"`

	mu sync.RWMutex
}

// Load reads and validates a catalog file.
func Load(path string) (*Catalog, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(b)
}

// Parse decodes and validates catalog YAML.
func Parse(b []byte) (*Catalog, error) {
	var c Catalog
	if err := yaml.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("catalog: %w", err)
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

// Validate checks ids are unique and enum fields hold allowed values.
func (c *Catalog) Validate() error {
	var errs []error
	seen := map[string]bool{}
	for i, s := range c.Sources {
		where := fmt.Sprintf("sources[%d] (%s)", i, s.ID)
		if s.ID == "" || s.Name == "" {
			errs = append(errs, fmt.Errorf("%s: id and name are required", where))
		}
		if seen[s.ID] {
			errs = append(errs, fmt.Errorf("%s: duplicate id", where))
		}
		seen[s.ID] = true
		if !slices.Contains(Statuses, s.Status) {
			errs = append(errs, fmt.Errorf("%s: invalid status %q", where, s.Status))
		}
		if !slices.Contains(Auths, s.Auth) {
			errs = append(errs, fmt.Errorf("%s: invalid auth %q", where, s.Auth))
		}
		for _, v := range s.Capabilities {
			if !slices.Contains(Capabilities, v) {
				errs = append(errs, fmt.Errorf("%s: invalid capability %q", where, v))
			}
		}
		for _, v := range s.Transport {
			if !slices.Contains(Transports, v) {
				errs = append(errs, fmt.Errorf("%s: invalid transport %q", where, v))
			}
		}
		if s.Adapter != "" && !slices.Contains(Adapters, s.Adapter) {
			errs = append(errs, fmt.Errorf("%s: invalid adapter %q", where, s.Adapter))
		}
		if s.RateLimit != nil && (s.RateLimit.PerSecond <= 0 || s.RateLimit.Burst <= 0) {
			errs = append(errs, fmt.Errorf("%s: rate_limit needs per_second > 0 and burst > 0", where))
		}
		if s.Config != nil {
			for _, sym := range s.Config.Symbols {
				if sym == "" || strings.ContainsAny(sym, " \t,") {
					errs = append(errs, fmt.Errorf("%s: invalid symbol %q", where, sym))
				}
			}
			for _, q := range s.Config.Queries {
				if strings.TrimSpace(q) == "" {
					errs = append(errs, fmt.Errorf("%s: empty query", where))
				}
			}
			if tf := s.Config.Timeframe; tf != "" && !slices.Contains(Timeframes, tf) {
				errs = append(errs, fmt.Errorf("%s: invalid timeframe %q", where, tf))
			}
		}
	}
	return errors.Join(errs...)
}

// ByID returns the entry with the given id.
func (c *Catalog) ByID(id string) (Source, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, s := range c.Sources {
		if s.ID == id {
			return s, true
		}
	}
	return Source{}, false
}

// ByCapability returns the entries declaring capability c (data | execution), in file order.
func (c *Catalog) ByCapability(capability string) []Source {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var out []Source
	for _, s := range c.Sources {
		if s.HasCapability(capability) {
			out = append(out, s)
		}
	}
	return out
}

// ByCategory returns the entries of one category (broker, macro, news, ...), in file order.
func (c *Catalog) ByCategory(category string) []Source {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var out []Source
	for _, s := range c.Sources {
		if s.Category == category {
			out = append(out, s)
		}
	}
	return out
}

// All returns a copy of every entry (safe to range while the catalog is reloaded).
func (c *Catalog) All() []Source {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return slices.Clone(c.Sources)
}

// ApplyOverride replaces an entry's rate limit and/or config in memory (the
// Postgres row is the caller's job). It reports false for an unknown id.
func (c *Catalog) ApplyOverride(id string, o Override) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i := range c.Sources {
		if c.Sources[i].ID == id {
			if o.RateLimit != nil {
				rl := *o.RateLimit
				c.Sources[i].RateLimit = &rl
			}
			if o.Config != nil {
				p := *o.Config
				c.Sources[i].Config = &p
			}
			return true
		}
	}
	return false
}

// Replace swaps in the entries of a freshly loaded catalog (ctl.ingest.reload,
// catalog.reload_on_change), keeping the current status of ids present in
// both (runtime status wins over the YAML seed, spec 11 §8). It returns the
// ids whose rate limit, config or adapter changed, and the added and removed ids.
func (c *Catalog) Replace(next *Catalog) (changed, added, removed []string) {
	nextSources := next.All()
	c.mu.Lock()
	defer c.mu.Unlock()
	old := map[string]Source{}
	for _, s := range c.Sources {
		old[s.ID] = s
	}
	seen := map[string]bool{}
	for i, s := range nextSources {
		seen[s.ID] = true
		prev, ok := old[s.ID]
		if !ok {
			added = append(added, s.ID)
			continue
		}
		nextSources[i].Status = prev.Status
		if !RuntimeEqual(prev, s) {
			changed = append(changed, s.ID)
		}
	}
	for _, s := range c.Sources {
		if !seen[s.ID] {
			removed = append(removed, s.ID)
		}
	}
	c.Version = next.Version
	c.Sources = nextSources
	return changed, added, removed
}

// RuntimeEqual reports whether two entries configure a connector the same way
// (rate limit, config and adapter; descriptive fields and status are ignored).
func RuntimeEqual(a, b Source) bool {
	if a.Limit() != b.Limit() || a.Adapter != b.Adapter {
		return false
	}
	pa, pb := a.Params(), b.Params()
	if !slices.Equal(pa.Symbols, pb.Symbols) || !slices.Equal(pa.Queries, pb.Queries) || pa.Timeframe != pb.Timeframe || len(pa.Extra) != len(pb.Extra) {
		return false
	}
	for k, v := range pa.Extra {
		if w, ok := pb.Extra[k]; !ok || w != v {
			return false
		}
	}
	return true
}

// SetStatus changes an entry's status in memory (the Postgres row is the
// caller's job). It reports false for an unknown id or an invalid status.
func (c *Catalog) SetStatus(id, status string) bool {
	if !slices.Contains(Statuses, status) {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for i := range c.Sources {
		if c.Sources[i].ID == id {
			c.Sources[i].Status = status
			return true
		}
	}
	return false
}

// ApplyStatuses overwrites statuses from a map (Postgres wins over the YAML
// seed after the first sync, spec 11 §8 / D-054). Unknown ids are ignored;
// returns the ids whose status changed.
func (c *Catalog) ApplyStatuses(statuses map[string]string) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var changed []string
	for i := range c.Sources {
		if st, ok := statuses[c.Sources[i].ID]; ok && st != c.Sources[i].Status && slices.Contains(Statuses, st) {
			c.Sources[i].Status = st
			changed = append(changed, c.Sources[i].ID)
		}
	}
	return changed
}

// BackfillDepth is the default history depth per timescale family (spec 10 §3:
// seconds ≈ weeks, minutes ≈ 1 year, hours ≈ 5 years, days = maximum
// available, capped here at 50 years). Unknown timescales get the minutes depth.
func BackfillDepth(timescale string) time.Duration {
	const day = 24 * time.Hour
	switch {
	case strings.HasSuffix(timescale, "s"):
		return 4 * 7 * day
	case strings.HasSuffix(timescale, "m"):
		return 365 * day
	case strings.HasSuffix(timescale, "h"):
		return 5 * 365 * day
	case strings.HasSuffix(timescale, "d"), strings.HasSuffix(timescale, "w"):
		return 50 * 365 * day
	}
	return 365 * day
}
