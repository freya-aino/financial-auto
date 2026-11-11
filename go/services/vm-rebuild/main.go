// Command vm-rebuild rebuilds VictoriaMetrics market series from the Garage raw
// archive into a scratch VictoriaMetrics and compares them with the live one
// (spec 20 §6.6 / criterion 5: VM is not backed up, it is rebuilt from the raw
// archive). It is the drill behind scripts/vm-rebuild-drill.sh.
//
// Path per source: archive.Replay (the reader behind ctl.ingest.replay) →
// the connector's own Normalizer (fresh in-memory cursors, as the replay does)
// → tsdb.Writer.AddEnvelope (the ingestor's write path, same-millisecond trade
// VWAP included) → -target. Nothing is published on NATS and nothing is
// written to Garage or Postgres: symbols resolve through refdata Lookup only
// (never provisioning) on a read-only Postgres session. The live VM (-compare)
// is only read (/api/v1/export).
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/fey/financial-auto/go/pkg/archive"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
	"github.com/fey/financial-auto/go/pkg/refdata"
	"github.com/fey/financial-auto/go/pkg/tsdb"

	_ "github.com/fey/financial-auto/go/connectors/kraken"
	_ "github.com/fey/financial-auto/go/connectors/yahoo"
)

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// marketSelector restricts both sides to the series the ingestor's tsdb writer produces.
const marketSelector = `__name__=~"fa_(trade|quote|bar|book)_.*"`

func main() {
	sources := flag.String("sources", "kraken,yahoo", "comma-separated sources to rebuild (connectors linked: kraken, yahoo)")
	from := flag.String("from", "-2h", "window start: RFC 3339 or a negative duration relative to now (-2h)")
	to := flag.String("to", "-10m", "window end: RFC 3339 or a negative duration (default leaves the last 10 min, whose raw objects may not be uploaded yet)")
	target := flag.String("target", env("FA_VM_DRILL_URL", "http://127.0.0.1:18428"), "SCRATCH VictoriaMetrics to rebuild into")
	compare := flag.String("compare", env("FA_TSDB_URL", "http://127.0.0.1:8428"), "live VictoriaMetrics to compare with (read only; empty skips)")
	garage := flag.String("garage", env("FA_GARAGE_URL", "http://127.0.0.1:3900"), "Garage S3 endpoint (read only)")
	secretsPath := flag.String("secrets", env("FA_SECRETS", "config/secrets.toml"), "secrets file with [garage] credentials")
	catalogPath := flag.String("catalog", env("FA_CATALOG", "config/sources.yaml"), "source catalog")
	pgURL := flag.String("postgres", env("FA_POSTGRES_URL", "postgres://fa:fa@127.0.0.1:5432/fa"), "Postgres for read-only symbol lookup (empty: unresolved symbols are skipped)")
	tail := flag.Duration("tail", 5*time.Minute, "read archive payloads received up to this long after -to: events inside the window are received later by up to the source's knowledge delay (Kraken bars ≈ 2 min)")
	minMatch := flag.Float64("min-match", 0.99, "fail (exit 2) when a source's matched samples are below this share of the live samples, or any value differs")
	jsonOut := flag.Bool("json", false, "print the report as JSON")
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil)).With("svc", "vm-rebuild")

	if err := run(log, opts{
		sources: splitList(*sources), from: *from, to: *to, target: *target, compare: *compare,
		garage: *garage, secrets: *secretsPath, catalog: *catalogPath, pg: *pgURL, json: *jsonOut, tail: *tail, minMatch: *minMatch,
	}); err != nil {
		log.Error("drill failed", "err", err)
		os.Exit(1)
	}
}

type opts struct {
	sources                                                 []string
	from, to, target, compare, garage, secrets, catalog, pg string
	json                                                    bool
	tail                                                    time.Duration
	minMatch                                                float64
}

func splitList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// parseTime accepts RFC 3339 or a duration relative to now.
func parseTime(s string, now time.Time) (time.Time, error) {
	if d, err := time.ParseDuration(s); err == nil {
		return now.Add(d), nil
	}
	return time.Parse(time.RFC3339, s)
}

// SourceReport is the per-source drill result.
type SourceReport struct {
	Source        string              `json:"source"`
	Read          archive.ReplayStats `json:"archive"`
	Envelopes     int                 `json:"envelopes"`
	NormalizeErrs int                 `json:"normalize_errors"`
	Unresolved    int                 `json:"unresolved_symbols"`
	Written       uint64              `json:"samples_written"`
	Cmp           *Comparison         `json:"comparison,omitempty"`
}

func run(log *slog.Logger, o opts) error {
	now := time.Now().UTC()
	from, err := parseTime(o.from, now)
	if err != nil {
		return fmt.Errorf("-from: %w", err)
	}
	to, err := parseTime(o.to, now)
	if err != nil {
		return fmt.Errorf("-to: %w", err)
	}
	if !from.Before(to) {
		return errors.New("need -from before -to")
	}
	if sameURL(o.target, o.compare) || isLiveVM(o.target) {
		return fmt.Errorf("-target %s looks like the live VictoriaMetrics; the drill only writes to a scratch instance", o.target)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	if err := get(ctx, o.target+"/health", nil); err != nil {
		return fmt.Errorf("scratch VM %s: %w", o.target, err)
	}

	cat, err := catalog.Load(o.catalog)
	if err != nil {
		return err
	}
	secrets, err := connector.LoadSecrets(o.secrets)
	if err != nil {
		return err
	}
	store, err := archive.NewS3(archive.Options{Endpoint: o.garage, AccessKey: secrets.Get("garage", "access_key_id"), SecretKey: secrets.Get("garage", "secret_access_key")})
	if err != nil {
		return err
	}
	res := &lookupResolver{}
	if o.pg != "" {
		st, err := refdata.Open(ctx, readOnlyURL(o.pg))
		if err != nil {
			return err
		}
		defer st.Close()
		res.r = refdata.NewResolver(st, nil, nil)
	}

	var reports []SourceReport
	for _, source := range o.sources {
		rep, err := rebuild(ctx, log, cat, secrets, store, res, source, from, to.Add(o.tail), o.target)
		if err != nil {
			return fmt.Errorf("%s: %w", source, err)
		}
		reports = append(reports, *rep)
	}
	_ = get(ctx, o.target+"/internal/force_flush", nil)
	if o.compare != "" {
		for i := range reports {
			c, err := compareSource(ctx, o.compare, o.target, reports[i].Source, from, to)
			if err != nil {
				return fmt.Errorf("compare %s: %w", reports[i].Source, err)
			}
			reports[i].Cmp = c
		}
	}
	failed := false
	for _, r := range reports {
		if c := r.Cmp; c != nil && (c.ValueDiff > 0 || float64(c.Matched) < o.minMatch*float64(c.LiveSamples)) {
			failed = true
		}
	}
	defer func() {
		if failed {
			fmt.Println("\nVM REBUILD DRILL FAILED (below -min-match or value mismatches)")
			os.Exit(2)
		}
	}()
	if o.json {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(map[string]any{"from": from, "to": to, "target": o.target, "compare": o.compare, "sources": reports})
	}
	fmt.Printf("VM rebuild drill %s .. %s (scratch %s, live %s)\n", from.Format(time.RFC3339), to.Format(time.RFC3339), o.target, o.compare)
	for _, r := range reports {
		fmt.Printf("\n[%s] archive objects=%d (no index %d, errors %d) payloads=%d → envelopes=%d (normalize errors %d, unresolved symbols %d) → samples written %d\n",
			r.Source, r.Read.Objects, r.Read.MissingIndex, r.Read.Errors, r.Read.Messages, r.Envelopes, r.NormalizeErrs, r.Unresolved, r.Written)
		if c := r.Cmp; c != nil {
			fmt.Print(c.Text())
		}
	}
	return nil
}

func rebuild(ctx context.Context, log *slog.Logger, cat *catalog.Catalog, secrets *connector.Secrets, store archive.ReadStore,
	res *lookupResolver, source string, from, to time.Time, target string) (*SourceReport, error) {
	c, err := connector.New(source)
	if err != nil {
		return nil, err
	}
	src, ok := cat.ByID(source)
	if !ok {
		src = catalog.Source{ID: source, Name: source, Status: "live", Auth: "none"}
	}
	res.unresolved = map[string]bool{}
	deps := connector.Deps{
		Limiter:  unlimited{},
		Resolver: res,
		Archive:  connector.NopArchive{},
		Clock:    connector.SystemClock{},
		Cursors:  connector.NewMemCursors(), // as ctl.ingest.replay: no live watermark
	}
	if err := c.Init(ctx, connector.SourceConfig{Source: src, Secrets: secrets.ForSource(source)}, deps); err != nil {
		return nil, fmt.Errorf("init: %w", err)
	}
	norm, ok := c.(connector.Normalizer)
	if !ok {
		return nil, errors.New("connector has no normalizer")
	}
	items, st, err := archive.Replay(ctx, store, "", source, from, to)
	if err != nil {
		return nil, err
	}
	venue := c.Capabilities().Venue
	w := tsdb.New(target, tsdb.Options{BatchSize: 5000, MaxBuffer: 10_000_000})
	rep := &SourceReport{Source: source, Read: st}
	for _, it := range items {
		envs, _, err := norm.Normalize(it.Raw)
		if err != nil {
			rep.NormalizeErrs++
			if rep.NormalizeErrs <= 5 {
				log.Warn("normalize", "source", source, "received_at", it.Raw.ReceivedAt, "key", it.Ref.Key, "endpoint", it.Raw.TransportMeta[connector.MetaEndpoint], "bytes", len(it.Raw.Bytes), "err", err)
			}
			continue
		}
		for _, e := range envs {
			rep.Envelopes++
			w.AddEnvelope(e, venue) // the ingestor's pipeline call (pipeline.go)
		}
	}
	if err := w.Flush(ctx); err != nil {
		return nil, fmt.Errorf("write scratch VM: %w", err)
	}
	stats := w.Stats()
	rep.Written, rep.Unresolved = stats.Written, len(res.unresolved)
	log.Info("rebuilt", "source", source, "payloads", st.Messages, "envelopes", rep.Envelopes, "samples", stats.Written, "dropped", stats.Dropped)
	return rep, nil
}

// lookupResolver resolves symbols through refdata Lookup only: the drill must
// never provision instruments.
type lookupResolver struct {
	r          *refdata.Resolver
	cache      map[string]string
	unresolved map[string]bool
}

func (l *lookupResolver) Resolve(sourceID, venueSymbol string) (string, bool) {
	key := sourceID + "\x00" + venueSymbol
	if l.cache == nil {
		l.cache = map[string]string{}
	}
	if id, ok := l.cache[key]; ok {
		return id, id != ""
	}
	id := ""
	if l.r != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if got, ok, err := l.r.Lookup(ctx, sourceID, venueSymbol); err == nil && ok {
			id = got
		}
		cancel()
	}
	l.cache[key] = id
	if id == "" {
		l.unresolved[key] = true
	}
	return id, id != ""
}

type unlimited struct{}

func (unlimited) Wait(context.Context) error { return nil }

// readOnlyURL appends default_transaction_read_only=on (pgx passes unknown
// query parameters as runtime parameters), so any accidental write fails.
func readOnlyURL(u string) string {
	sep := "?"
	if strings.Contains(u, "?") {
		sep = "&"
	}
	return u + sep + "default_transaction_read_only=on"
}

func sameURL(a, b string) bool {
	return a != "" && strings.TrimRight(a, "/") == strings.TrimRight(b, "/")
}

func isLiveVM(u string) bool {
	p, err := url.Parse(u)
	if err != nil {
		return true
	}
	return p.Port() == "8428" && (p.Hostname() == "127.0.0.1" || p.Hostname() == "localhost" || p.Hostname() == "victoriametrics")
}

// ---- comparison ----

// series is one exported series: timestamp (ms) → value.
type series map[int64]float64

// Comparison of one source's market series between live and rebuilt.
type Comparison struct {
	LiveSeries    int `json:"live_series"`
	RebuiltSeries int `json:"rebuilt_series"`
	// IdenticalSeries have exactly the same timestamps and values on both sides.
	IdenticalSeries int `json:"identical_series"`
	LiveOnlySeries  int `json:"live_only_series"`
	ExtraSeries     int `json:"rebuilt_only_series"`
	LiveSamples     int `json:"live_samples"`
	RebuiltSamples  int `json:"rebuilt_samples"`
	Matched         int `json:"matched_samples"`
	Missing         int `json:"missing_samples"` // in live, not rebuilt
	Extra           int `json:"extra_samples"`   // rebuilt, not in live
	ValueDiff       int `json:"value_mismatches"`
	// PerMetric splits matched/missing/extra by metric name.
	PerMetric map[string]*[4]int `json:"per_metric"` // matched, missing, extra, value_diff
	Examples  []string           `json:"examples,omitempty"`
}

func (c *Comparison) Text() string {
	var b strings.Builder
	pct := func(n, d int) float64 {
		if d == 0 {
			return 0
		}
		return 100 * float64(n) / float64(d)
	}
	fmt.Fprintf(&b, "  series: live %d, rebuilt %d, identical %d, live-only %d, rebuilt-only %d\n",
		c.LiveSeries, c.RebuiltSeries, c.IdenticalSeries, c.LiveOnlySeries, c.ExtraSeries)
	fmt.Fprintf(&b, "  samples: live %d, rebuilt %d, matched %d (%.1f%% of live), missing %d, extra %d, value mismatches %d\n",
		c.LiveSamples, c.RebuiltSamples, c.Matched, pct(c.Matched, c.LiveSamples), c.Missing, c.Extra, c.ValueDiff)
	names := make([]string, 0, len(c.PerMetric))
	for n := range c.PerMetric {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		m := c.PerMetric[n]
		fmt.Fprintf(&b, "    %-18s matched %6d  missing %6d  extra %6d  value-diff %6d\n", n, m[0], m[1], m[2], m[3])
	}
	for _, e := range c.Examples {
		fmt.Fprintf(&b, "    e.g. %s\n", e)
	}
	return b.String()
}

func compareSource(ctx context.Context, live, rebuilt, source string, from, to time.Time) (*Comparison, error) {
	sel := fmt.Sprintf(`{%s,src=%q}`, marketSelector, source)
	l, err := export(ctx, live, sel, from, to)
	if err != nil {
		return nil, fmt.Errorf("live: %w", err)
	}
	r, err := export(ctx, rebuilt, sel, from, to)
	if err != nil {
		return nil, fmt.Errorf("rebuilt: %w", err)
	}
	return Compare(l, r), nil
}

// Compare diffs exported series sets keyed by their canonical label string.
func Compare(live, rebuilt map[string]series) *Comparison {
	c := &Comparison{LiveSeries: len(live), RebuiltSeries: len(rebuilt), PerMetric: map[string]*[4]int{}}
	metricOf := func(key string) string { return strings.SplitN(key, "{", 2)[0] }
	pm := func(key string) *[4]int {
		m := metricOf(key)
		if c.PerMetric[m] == nil {
			c.PerMetric[m] = &[4]int{}
		}
		return c.PerMetric[m]
	}
	example := func(s string) {
		if len(c.Examples) < 8 {
			c.Examples = append(c.Examples, s)
		}
	}
	keys := map[string]bool{}
	for k := range live {
		keys[k] = true
	}
	for k := range rebuilt {
		keys[k] = true
	}
	sorted := make([]string, 0, len(keys))
	for k := range keys {
		sorted = append(sorted, k)
	}
	sort.Strings(sorted)
	for _, k := range sorted {
		ls, rs := live[k], rebuilt[k]
		c.LiveSamples += len(ls)
		c.RebuiltSamples += len(rs)
		switch {
		case rs == nil:
			c.LiveOnlySeries++
			example(fmt.Sprintf("live-only series %s (%d samples)", k, len(ls)))
		case ls == nil:
			c.ExtraSeries++
			example(fmt.Sprintf("rebuilt-only series %s (%d samples)", k, len(rs)))
		}
		identical := ls != nil && rs != nil && len(ls) == len(rs)
		m := pm(k)
		for ts, lv := range ls {
			rv, ok := rs[ts]
			switch {
			case !ok:
				c.Missing++
				m[1]++
				identical = false
				example(fmt.Sprintf("missing %s @%s (live %g)", k, time.UnixMilli(ts).UTC().Format(time.RFC3339Nano), lv))
			case !approxEqual(lv, rv):
				c.ValueDiff++
				m[3]++
				identical = false
				example(fmt.Sprintf("%s @%s live %g rebuilt %g", k, time.UnixMilli(ts).UTC().Format(time.RFC3339Nano), lv, rv))
			default:
				c.Matched++
				m[0]++
			}
		}
		for ts := range rs {
			if _, ok := ls[ts]; !ok {
				c.Extra++
				m[2]++
				identical = false
			}
		}
		if identical {
			c.IdenticalSeries++
		}
	}
	return c
}

func approxEqual(a, b float64) bool {
	return a == b || math.Abs(a-b) <= 1e-9*math.Max(math.Abs(a), math.Abs(b))
}

// export reads every series matching sel in [from, to) via /api/v1/export (JSON lines).
func export(ctx context.Context, base, sel string, from, to time.Time) (map[string]series, error) {
	q := url.Values{"match[]": {sel}, "start": {fmt.Sprint(from.UnixMilli() / 1000)}, "end": {fmt.Sprint((to.UnixMilli() - 1) / 1000)}}
	out := map[string]series{}
	err := get(ctx, base+"/api/v1/export?"+q.Encode(), func(body io.Reader) error {
		sc := bufio.NewScanner(body)
		sc.Buffer(make([]byte, 0, 1<<20), 256<<20)
		for sc.Scan() {
			var l struct {
				Metric     map[string]string `json:"metric"`
				Values     []float64         `json:"values"`
				Timestamps []int64           `json:"timestamps"`
			}
			if err := json.Unmarshal(sc.Bytes(), &l); err != nil {
				return err
			}
			key := seriesKey(l.Metric)
			s := out[key]
			if s == nil {
				s = series{}
				out[key] = s
			}
			for i, ts := range l.Timestamps {
				if ts >= from.UnixMilli() && ts < to.UnixMilli() && i < len(l.Values) {
					s[ts] = l.Values[i]
				}
			}
		}
		return sc.Err()
	})
	for k, s := range out {
		if len(s) == 0 {
			delete(out, k)
		}
	}
	return out, err
}

// seriesKey renders name{k="v",...} with sorted labels.
func seriesKey(m map[string]string) string {
	ks := make([]string, 0, len(m))
	for k := range m {
		if k != "__name__" {
			ks = append(ks, k)
		}
	}
	sort.Strings(ks)
	var b bytes.Buffer
	b.WriteString(m["__name__"])
	b.WriteByte('{')
	for i, k := range ks {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%s=%q", k, m[k])
	}
	b.WriteByte('}')
	return b.String()
}

func get(ctx context.Context, u string, read func(io.Reader) error) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	res, err := (&http.Client{Timeout: 5 * time.Minute}).Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode/100 != 2 {
		msg, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return fmt.Errorf("%s: %s", res.Status, strings.TrimSpace(string(msg)))
	}
	if read == nil {
		_, _ = io.Copy(io.Discard, res.Body)
		return nil
	}
	return read(res.Body)
}
