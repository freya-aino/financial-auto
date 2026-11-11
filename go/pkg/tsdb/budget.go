package tsdb

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Series budget (spec 20 §5.3, §8 "VM cardinality explosion"): VictoriaMetrics counts a series
// as active while it received a sample within the last hour. The writer admits a point of a
// series it has not written within ActiveWindow only while the active count stays below
// MaxActiveSeries; over budget, an instrument's tick-level series (trades, quotes, book) are
// demoted — every further tick point of that (inst, venue, src) is dropped and only its bars
// are kept — and a new bar series is rejected. The active count is the larger of the series
// this writer saw within ActiveWindow and VictoriaMetrics' own hour count
// (vm_cache_entries{type="storage/hour_metric_ids"} on /metrics, refreshed every BudgetPoll), so
// series written by other processes count too.

// DefaultMaxActiveSeries is spec 20 §7 `vm.max_active_series`.
const DefaultMaxActiveSeries = 5_000_000

// vmActiveMetric is VictoriaMetrics' count of series that received samples in the current hour.
const vmActiveMetric = `vm_cache_entries{type="storage/hour_metric_ids"}`

// isBarMetric: bars are kept for demoted instruments (they are the "bars-only" resolution).
func isBarMetric(m string) bool { return strings.HasPrefix(m, "fa_bar_") }

func instKey(labels map[string]string) string {
	return labels["inst"] + "|" + labels["venue"] + "|" + labels["src"]
}

type seriesBudget struct {
	max     int // <= 0: disabled
	window  time.Duration
	seen    map[string]time.Time // series key → last admitted point (wall clock)
	demoted map[string]time.Time // instKey → demotion time
	vm      int                  // last VictoriaMetrics hour count (0 = unknown)
}

func newSeriesBudget(max int, window time.Duration) *seriesBudget {
	return &seriesBudget{max: max, window: window, seen: map[string]time.Time{}, demoted: map[string]time.Time{}}
}

func (b *seriesBudget) active() int {
	if b.vm > len(b.seen) {
		return b.vm
	}
	return len(b.seen)
}

// verdict of admit.
type verdict int

const (
	admitted verdict = iota
	droppedDemoted
	droppedNewDemotion // the point demoted its instrument
	rejectedBar
)

// admit decides whether p may be written at wall time now.
func (b *seriesBudget) admit(p Point, now time.Time) verdict {
	if b.max <= 0 {
		return admitted
	}
	bar := isBarMetric(p.Metric)
	ik := instKey(p.Labels)
	if !bar {
		if _, ok := b.demoted[ik]; ok {
			return droppedDemoted
		}
	}
	key := seriesKey(p)
	if _, ok := b.seen[key]; ok {
		b.seen[key] = now
		return admitted
	}
	if b.active()+1 > b.max {
		if bar {
			return rejectedBar
		}
		b.demoted[ik] = now
		return droppedNewDemotion
	}
	b.seen[key] = now
	if b.vm > 0 {
		b.vm++ // a new series raises VM's count too, until the next poll
	}
	return admitted
}

// expire forgets series idle for longer than the window (VM stops counting them after an hour).
func (b *seriesBudget) expire(now time.Time) {
	for k, t := range b.seen {
		if now.Sub(t) > b.window {
			delete(b.seen, k)
		}
	}
}

// pollVMActive reads the hour-active series count from VictoriaMetrics' /metrics.
func pollVMActive(ctx context.Context, client *http.Client, url string) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url+"/metrics", nil)
	if err != nil {
		return 0, err
	}
	res, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer res.Body.Close()
	if res.StatusCode/100 != 2 {
		return 0, fmt.Errorf("vm /metrics: %s", res.Status)
	}
	return parseVMActive(bufio.NewScanner(res.Body))
}

func parseVMActive(sc *bufio.Scanner) (int, error) {
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if rest, ok := strings.CutPrefix(line, vmActiveMetric+" "); ok {
			v, err := strconv.ParseFloat(strings.TrimSpace(rest), 64)
			if err != nil {
				return 0, err
			}
			return int(v), nil
		}
	}
	return 0, sc.Err()
}

// Demoted lists the instruments ("inst|venue|src") currently kept at bar resolution only.
func (w *Writer) Demoted() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]string, 0, len(w.budget.demoted))
	for k := range w.budget.demoted {
		out = append(out, k)
	}
	return out
}

// Promote lifts every demotion (operator action after raising the budget or after the
// over-budget series expired); returns how many instruments were promoted.
func (w *Writer) Promote() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	n := len(w.budget.demoted)
	w.budget.demoted = map[string]time.Time{}
	return n
}

// refreshBudget expires idle series and, every BudgetPoll, re-reads VM's active count.
func (w *Writer) refreshBudget(ctx context.Context) {
	if w.budget.max <= 0 {
		return
	}
	now := w.now()
	w.mu.Lock()
	w.budget.expire(now)
	due := now.Sub(w.lastPoll) >= w.pollEvery
	if due {
		w.lastPoll = now
	}
	w.mu.Unlock()
	if !due || w.pollEvery < 0 {
		return
	}
	pctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	n, err := pollVMActive(pctx, w.client, w.url)
	if err != nil {
		return // keep the last value; the local count still bounds this writer
	}
	w.mu.Lock()
	w.budget.vm = n
	w.mu.Unlock()
}
