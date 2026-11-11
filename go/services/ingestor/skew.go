package main

import (
	"context"
	"slices"
	"sync"
	"time"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// Clock-skew monitor tunables (spec 10 §7 `clock.future_tolerance`, `clock.window`).
const (
	// FutureTolerance: an event stamped more than this after its ingest time
	// means one of the two clocks is wrong (sources stamp trades/bars at or
	// before the moment we receive them).
	FutureTolerance = 2 * time.Second
	// SkewWindow is the number of recent lags kept per source.
	SkewWindow = 256
)

// SkewMonitor tracks the ingest-vs-event-time drift of live market data
// (spec 10 §8 "clock skew"): per source the lag ingest_time − event_time of
// the last SkewWindow live trades, quotes, bars and book snapshots, exported
// as heartbeat counters (→ fa_heartbeat_counter, scraped into
// VictoriaMetrics):
//
//	ingest_lag_p50_s.<source>, ingest_lag_p95_s.<source>  seconds
//	ingest_lag_ratio.<source>   p50 / catalog backfill.knowledge_delay_secs (D-079), when set
//	clock.future_events.<source>  events stamped > FutureTolerance after ingest (counter)
//
// vmalert rules IngestClockSkew (future events) and IngestLagOverBudget
// (ratio > 1 for 15 min) turn them into alerts. News, macro and filings are
// excluded: their lag is the source's publication delay, not clock drift.
type SkewMonitor struct {
	Counters *bus.Counters
	// Budget returns the source's knowledge delay (0 = none).
	Budget func(source string) time.Duration

	mu  sync.Mutex
	per map[string]*skewState
}

type skewState struct {
	ring []float64
	idx  int
}

// NewSkewMonitor builds a monitor.
func NewSkewMonitor(counters *bus.Counters, budget func(string) time.Duration) *SkewMonitor {
	return &SkewMonitor{Counters: counters, Budget: budget, per: map[string]*skewState{}}
}

func marketPayload(e *fav1.Envelope) bool {
	switch e.Payload.(type) {
	case *fav1.Envelope_Trade, *fav1.Envelope_Quote, *fav1.Envelope_Bar, *fav1.Envelope_Book:
		return true
	}
	return false
}

// Observe records one published envelope (backfill and gap-fill are ignored).
func (m *SkewMonitor) Observe(source string, e *fav1.Envelope) {
	if m == nil || e.GetBackfill() || e.GetGapFilled() || !marketPayload(e) || e.GetEventTime() == nil || e.GetIngestTime() == nil {
		return
	}
	lag := e.GetIngestTime().AsTime().Sub(e.GetEventTime().AsTime())
	if lag < -FutureTolerance && m.Counters != nil {
		m.Counters.Add("clock.future_events."+source, 1)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	st := m.per[source]
	if st == nil {
		st = &skewState{}
		m.per[source] = st
	}
	if len(st.ring) < SkewWindow {
		st.ring = append(st.ring, lag.Seconds())
	} else {
		st.ring[st.idx] = lag.Seconds()
		st.idx = (st.idx + 1) % SkewWindow
	}
}

// SkewStats is one source's window.
type SkewStats struct {
	N        int
	P50, P95 float64
	Ratio    float64 // p50 / budget, 0 without a budget
}

// Stats computes the current window of every source.
func (m *SkewMonitor) Stats() map[string]SkewStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := map[string]SkewStats{}
	for src, st := range m.per {
		if len(st.ring) == 0 {
			continue
		}
		v := slices.Clone(st.ring)
		slices.Sort(v)
		s := SkewStats{N: len(v), P50: quantile(v, 0.5), P95: quantile(v, 0.95)}
		if m.Budget != nil {
			if b := m.Budget(src); b > 0 {
				s.Ratio = s.P50 / b.Seconds()
			}
		}
		out[src] = s
	}
	return out
}

func quantile(sorted []float64, q float64) float64 {
	i := int(q * float64(len(sorted)-1))
	return sorted[i]
}

// Export writes the gauges once.
func (m *SkewMonitor) Export() {
	if m.Counters == nil {
		return
	}
	for src, s := range m.Stats() {
		m.Counters.Set("ingest_lag_p50_s."+src, s.P50)
		m.Counters.Set("ingest_lag_p95_s."+src, s.P95)
		if s.Ratio > 0 {
			m.Counters.Set("ingest_lag_ratio."+src, s.Ratio)
		}
	}
}

// Run exports every interval until ctx is done.
func (m *SkewMonitor) Run(ctx context.Context, interval time.Duration) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			m.Export()
		}
	}
}
