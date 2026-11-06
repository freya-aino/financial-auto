// Package catalog loads and validates the source catalog (config/sources.yaml, spec 11).
package catalog

import (
	"errors"
	"fmt"
	"os"
	"slices"

	"go.yaml.in/yaml/v3"
)

// Allowed enum values, mirroring the header of config/sources.yaml.
var (
	Statuses     = []string{"planned", "stub", "live", "disabled"}
	Capabilities = []string{"data", "execution"}
	Transports   = []string{"rest", "websocket", "fix", "grpc", "graphql", "sse", "socket", "file", "sdk", "stream"}
	Auths        = []string{"none", "api_key", "oauth2", "hmac", "session", "wallet", "gateway"}
)

// RateLimit is an optional per-source token bucket. Not yet present in
// sources.yaml; connectors fall back to DefaultRateLimit.
type RateLimit struct {
	PerSecond float64 `yaml:"per_second"`
	Burst     int     `yaml:"burst"`
}

// DefaultRateLimit applies when a source declares none (placeholder, spec 10 tunables).
var DefaultRateLimit = RateLimit{PerSecond: 5, Burst: 10}

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
}

// Limit returns the source's rate limit or the default.
func (s Source) Limit() RateLimit {
	if s.RateLimit != nil {
		return *s.RateLimit
	}
	return DefaultRateLimit
}

// Catalog is the parsed file.
type Catalog struct {
	Version int      `yaml:"version"`
	Sources []Source `yaml:"sources"`
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
	}
	return errors.Join(errs...)
}

// ByID returns the entry with the given id.
func (c *Catalog) ByID(id string) (Source, bool) {
	for _, s := range c.Sources {
		if s.ID == id {
			return s, true
		}
	}
	return Source{}, false
}
