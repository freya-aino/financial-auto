package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/fey/financial-auto/go/pkg/catalog"
)

// CatalogStore is the Postgres side of the catalog (refdata.Store).
type CatalogStore interface {
	SyncCatalog(ctx context.Context, cat *catalog.Catalog) error
	Statuses(ctx context.Context) (map[string]string, error)
	Overrides(ctx context.Context) (map[string]catalog.Override, error)
	SetOverride(ctx context.Context, id string, o catalog.Override, clear bool) error
}

// Reloader applies catalog changes at runtime (spec 11 §4, §7; spec 10 §4.3):
//
//   - `ctl.ingest.reload [source]`: re-read the YAML, sync it into
//     catalog.source, apply Postgres statuses and overrides, update every
//     source's token bucket and restart running connectors whose config or
//     adapter changed (all of them, or only `source`).
//   - `ctl.ingest.configure {source, rate_limit_per_second, rate_limit_burst,
//     config (JSON), clear}`: the PATCH of spec 11 §4 — persist an override of
//     the rate limit and/or connector config in catalog.source, apply it, and
//     restart the connector when its config changed. `clear=1` drops the
//     stored override (the YAML applies again).
//   - Watch: `catalog.reload_on_change` — the YAML file is polled and reloaded
//     when its modification time or size changes.
//
// Each applied change publishes `catalog.changed.<id>` (change `config`).
type Reloader struct {
	Path  string
	Cat   *catalog.Catalog
	Sup   *Supervisor
	Store CatalogStore // nil: YAML only, overrides in memory
	// SetLimit pushes a source's current rate limit into its token bucket.
	SetLimit func(src catalog.Source)
	// Changed publishes catalog.changed.<id>.
	Changed func(id, change string, attrs map[string]string)
	Log     func(msg string, args ...any)

	mu    sync.Mutex
	mtime time.Time
	size  int64
}

// ReloadResult is the JSON reply of ctl.ingest.reload.
type ReloadResult struct {
	Changed   []string `json:"changed"`
	Added     []string `json:"added,omitempty"`
	Removed   []string `json:"removed,omitempty"`
	Restarted []string `json:"restarted,omitempty"`
	Sources   int      `json:"sources"`
}

// Reload re-reads the YAML (only = "" → every source).
func (r *Reloader) Reload(ctx context.Context, only string) (*ReloadResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	next, err := catalog.Load(r.Path)
	if err != nil {
		return nil, fmt.Errorf("%w: catalog %s: %v", ErrInvalid, r.Path, err)
	}
	if only != "" {
		if _, ok := next.ByID(only); !ok {
			return nil, fmt.Errorf("%w: source %q not in %s", ErrNotFound, only, r.Path)
		}
	}
	var statuses map[string]string
	if r.Store != nil {
		if err := r.Store.SyncCatalog(ctx, next); err != nil {
			return nil, err
		}
		if statuses, err = r.Store.Statuses(ctx); err != nil {
			return nil, err
		}
		if ov, err := r.Store.Overrides(ctx); err != nil {
			r.log("catalog overrides not applied", "err", err)
		} else {
			for id, o := range ov {
				next.ApplyOverride(id, o)
			}
		}
	}
	old := map[string]catalog.Source{}
	for _, s := range r.Cat.All() {
		old[s.ID] = s
	}
	changed, added, removed := r.Cat.Replace(next)
	if statuses != nil {
		r.Cat.ApplyStatuses(statuses)
	}
	res := &ReloadResult{Changed: nonNil(changed), Added: added, Removed: removed, Sources: len(next.Sources)}
	for _, id := range changed {
		if only != "" && id != only {
			continue
		}
		restarted, err := r.applyLocked(ctx, old[id], id)
		if err != nil {
			return res, err
		}
		if restarted {
			res.Restarted = append(res.Restarted, id)
		}
	}
	if st, err := os.Stat(r.Path); err == nil {
		r.mtime, r.size = st.ModTime(), st.Size()
	}
	r.log("catalog reloaded", "changed", res.Changed, "added", added, "removed", removed, "restarted", res.Restarted)
	return res, nil
}

// applyLocked pushes a changed entry into the limiter and restarts its connector when needed.
func (r *Reloader) applyLocked(ctx context.Context, before catalog.Source, id string) (bool, error) {
	src, ok := r.Cat.ByID(id)
	if !ok {
		return false, nil
	}
	if r.SetLimit != nil {
		r.SetLimit(src)
	}
	restarted := false
	limitOnly := before.ID != "" && before.Adapter == src.Adapter && catalog.RuntimeEqual(withLimit(before, src.Limit()), src)
	if !limitOnly && r.Sup != nil {
		var err error
		if restarted, err = r.Sup.Restart(ctx, id); err != nil {
			return false, fmt.Errorf("restart %s: %w", id, err)
		}
	}
	if r.Changed != nil {
		lim := src.Limit()
		r.Changed(id, "config", map[string]string{
			"rate_limit_per_second": strconv.FormatFloat(lim.PerSecond, 'g', -1, 64),
			"rate_limit_burst":      strconv.Itoa(lim.Burst),
			"restarted":             strconv.FormatBool(restarted),
		})
	}
	return restarted, nil
}

func withLimit(s catalog.Source, l catalog.RateLimit) catalog.Source {
	s.RateLimit = &l
	return s
}

// ConfigureResult is the JSON reply of ctl.ingest.configure.
type ConfigureResult struct {
	Source    string            `json:"source"`
	RateLimit catalog.RateLimit `json:"rate_limit"`
	Config    catalog.Params    `json:"config"`
	Persisted bool              `json:"persisted"`
	Restarted bool              `json:"restarted"`
	Cleared   bool              `json:"cleared,omitempty"`
}

// ParseOverride reads ctl.ingest.configure args into an override.
func ParseOverride(args map[string]string) (catalog.Override, bool, error) {
	var o catalog.Override
	ps, pb := strings.TrimSpace(args["rate_limit_per_second"]), strings.TrimSpace(args["rate_limit_burst"])
	if ps != "" || pb != "" {
		if ps == "" || pb == "" {
			return o, false, fmt.Errorf("%w: rate_limit_per_second and rate_limit_burst go together", ErrInvalid)
		}
		f, err := strconv.ParseFloat(ps, 64)
		if err != nil {
			return o, false, fmt.Errorf("%w: rate_limit_per_second: %v", ErrInvalid, err)
		}
		b, err := strconv.Atoi(pb)
		if err != nil {
			return o, false, fmt.Errorf("%w: rate_limit_burst: %v", ErrInvalid, err)
		}
		o.RateLimit = &catalog.RateLimit{PerSecond: f, Burst: b}
	}
	if c := strings.TrimSpace(args["config"]); c != "" {
		var p catalog.Params
		dec := json.NewDecoder(strings.NewReader(c))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&p); err != nil {
			return o, false, fmt.Errorf("%w: config: %v", ErrInvalid, err)
		}
		o.Config = &p
	}
	clear := args["clear"] == "1" || args["clear"] == "true"
	if o.IsZero() && !clear {
		return o, false, fmt.Errorf("%w: nothing to configure (rate_limit_per_second+rate_limit_burst, config or clear=1)", ErrInvalid)
	}
	if err := o.Validate(); err != nil {
		return o, false, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	return o, clear, nil
}

// Configure applies a runtime override to one source.
func (r *Reloader) Configure(ctx context.Context, args map[string]string) (*ConfigureResult, error) {
	id := strings.TrimSpace(args["source"])
	if _, ok := r.Cat.ByID(id); !ok {
		return nil, fmt.Errorf("%w: source %q is not in the catalog", ErrNotFound, id)
	}
	o, clear, err := ParseOverride(args)
	if err != nil {
		return nil, err
	}
	res := &ConfigureResult{Source: id, Cleared: clear}
	if clear {
		if r.Store != nil {
			if err := r.Store.SetOverride(ctx, id, o, true); err != nil {
				return nil, err
			}
			res.Persisted = true
		}
		// Back to the YAML entry plus whatever was given with clear.
		rr, err := r.Reload(ctx, id)
		if err != nil {
			return nil, err
		}
		res.Restarted = slices.Contains(rr.Restarted, id)
	} else {
		if r.Store != nil {
			if err := r.Store.SetOverride(ctx, id, o, false); err != nil {
				return nil, err
			}
			res.Persisted = true
		}
		r.mu.Lock()
		before, _ := r.Cat.ByID(id)
		r.Cat.ApplyOverride(id, o)
		after, _ := r.Cat.ByID(id)
		if !catalog.RuntimeEqual(before, after) {
			res.Restarted, err = r.applyLocked(ctx, before, id)
		}
		r.mu.Unlock()
		if err != nil {
			return nil, err
		}
	}
	src, _ := r.Cat.ByID(id)
	res.RateLimit, res.Config = src.Limit(), src.Params()
	return res, nil
}

// ApplyStoredOverrides re-reads catalog.source overrides (a catalog.changed
// notice from another instance or a manual edit) and applies the ones that
// differ from memory. Idempotent: an unchanged override does nothing.
func (r *Reloader) ApplyStoredOverrides(ctx context.Context) ([]string, error) {
	if r.Store == nil {
		return nil, nil
	}
	ov, err := r.Store.Overrides(ctx)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	var applied []string
	for id, o := range ov {
		before, ok := r.Cat.ByID(id)
		if !ok {
			continue
		}
		r.Cat.ApplyOverride(id, o)
		after, _ := r.Cat.ByID(id)
		if catalog.RuntimeEqual(before, after) {
			continue
		}
		if _, err := r.applyLocked(ctx, before, id); err != nil {
			return applied, err
		}
		applied = append(applied, id)
	}
	return applied, nil
}

// Watch polls the YAML every interval and reloads when it changed
// (catalog.reload_on_change, spec 11 §7). interval ≤ 0 disables.
func (r *Reloader) Watch(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		return
	}
	if st, err := os.Stat(r.Path); err == nil {
		r.mu.Lock()
		r.mtime, r.size = st.ModTime(), st.Size()
		r.mu.Unlock()
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		st, err := os.Stat(r.Path)
		if err != nil {
			continue
		}
		r.mu.Lock()
		same := st.ModTime().Equal(r.mtime) && st.Size() == r.size
		r.mu.Unlock()
		if same {
			continue
		}
		if _, err := r.Reload(ctx, ""); err != nil {
			r.log("catalog reload on change failed", "err", err)
			r.mu.Lock()
			r.mtime, r.size = st.ModTime(), st.Size() // do not retry a broken file every tick
			r.mu.Unlock()
		}
	}
}

func (r *Reloader) log(msg string, args ...any) {
	if r.Log != nil {
		r.Log(msg, args...)
	}
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}
