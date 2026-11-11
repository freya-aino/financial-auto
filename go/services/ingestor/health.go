package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// Health subjects (spec 21 §4.1, spec 10 §6.7). Payloads are the fa.v1 messages
// IngestHealth / IngestGap / IngestUnresolved; until 2026-11-11 they were JSON
// (D-054) and `-health-json` still selects that legacy encoding. Consumers
// (risk-engine, control-api via fa_bus::ingest) accept both.
func healthSubject(source string) string     { return bus.IngestHealth(source) }
func gapSubject(source string) string        { return bus.IngestGap(source) }
func unresolvedSubject(source string) string { return bus.IngestUnresolved(source) }

// HaltScope is the `risk.halt.<scope>` token for a source (D-042 family
// `source`, qualifier after the colon; fa_risk::HaltScope::parse maps
// `source:<id>` to HaltScope::Source). NATS allows ':' inside a token.
func HaltScope(source string) string { return "source:" + source }

const (
	// StreamPeriod is the expected activity period of a stream connector
	// (spec 10 §7 `staleness.stream` 30 s); pollers use their schedule.
	StreamPeriod = 30 * time.Second
	// DownAfterPeriods × expected period without activity → `down`.
	DownAfterPeriods = 3
	// DegradedSaturation: limiter saturation above this marks `degraded`.
	DegradedSaturation = 0.8
	// HealthInterval between `ingest.health.<source>` frames.
	HealthInterval = 5 * time.Second
)

// Frame is one `ingest.health.<source>` payload.
type Frame struct {
	Source  string    `json:"source"`
	Status  string    `json:"status"` // ok | degraded | down
	Running bool      `json:"running"`
	Mode    string    `json:"mode"` // stream | poll | none
	At      time.Time `json:"at"`
	// Reason explains degraded/down (empty when ok).
	Reason string `json:"reason,omitempty"`
	// Halted is true while a `risk.halt.source:<id>` is outstanding.
	Halted bool `json:"halted"`

	StartedAt       *time.Time `json:"started_at,omitempty"`
	ExpectedPeriodS float64    `json:"expected_period_s"`
	// LastActivityAt: last successful poll (even empty) or stream message.
	LastActivityAt   *time.Time `json:"last_activity_at,omitempty"`
	LastActivityAgeS *float64   `json:"last_activity_age_s,omitempty"`
	// LastEventAt: last canonical event published.
	LastEventAt   *time.Time `json:"last_event_at,omitempty"`
	LastEventAgeS *float64   `json:"last_event_age_s,omitempty"`

	EventsTotal int64 `json:"events_total"`
	ErrorsTotal int64 `json:"errors_total"`
	// Window stats since the previous frame.
	WindowS      float64 `json:"window_s"`
	EventsWindow int64   `json:"events_window"`
	ErrorsWindow int64   `json:"errors_window"`
	// ErrorRate is errors per minute over the window.
	ErrorRate float64 `json:"error_rate"`

	LimiterSaturation float64 `json:"limiter_saturation"`
	LimiterRequests   int64   `json:"limiter_requests"`

	LastError   string     `json:"last_error,omitempty"`
	LastErrorAt *time.Time `json:"last_error_at,omitempty"`
}

// Gap is one `ingest.gap.<source>` payload: consecutive bars of one
// instrument/timeframe more than one period apart (also the first bar after a
// restart compared with the persisted cursor).
type Gap struct {
	Source         string    `json:"source"`
	InstrumentID   string    `json:"instrument_id"`
	Timeframe      string    `json:"timeframe"`
	From           time.Time `json:"from"`
	To             time.Time `json:"to"`
	MissingPeriods int64     `json:"missing_periods"`
	At             time.Time `json:"at"`
}

// UnresolvedEvent is one `ingest.unresolved.<source>` payload: either a
// provisioned instrument (D-050, `instrument_id` set) or a symbol that could
// not be resolved at all.
type UnresolvedEvent struct {
	Source       string    `json:"source"`
	VenueSymbol  string    `json:"venue_symbol"`
	InstrumentID string    `json:"instrument_id,omitempty"`
	Reason       string    `json:"reason"`
	At           time.Time `json:"at"`
}

type sourceHealth struct {
	id, mode     string
	period       time.Duration
	limiter      *connector.MeteredLimiter
	startedAt    time.Time
	lastActivity time.Time
	lastEvent    time.Time
	events       int64
	errors       int64
	wEvents      int64
	wErrors      int64
	wSince       time.Time
	lastErr      string
	lastErrAt    time.Time
	running      bool
	stopReason   string // set when the connector ended: "stopped by operator" | "connector exited: ..."
	halted       bool
	wasDown      bool
	frames       int64
}

// Tracker keeps per-source health and publishes frames, halts and resumes.
type Tracker struct {
	pub   *bus.Publisher
	clock func() time.Time
	log   func(msg string, args ...any)

	mu      sync.Mutex
	sources map[string]*sourceHealth
	// halts remembers outstanding halts of sources no longer tracked (they
	// exited on their own), so a restart publishes the matching resume.
	halts map[string]bool
	// Published counts frames/halts/resumes for tests.
	Published struct{ Frames, Halts, Resumes int64 }
	// JSON selects the legacy D-054 JSON payloads instead of fa.v1 protos.
	JSON bool
}

// NewTracker builds a tracker; pub may be nil (frames are still computed).
func NewTracker(pub *bus.Publisher, logf func(msg string, args ...any)) *Tracker {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &Tracker{pub: pub, clock: func() time.Time { return time.Now().UTC() }, log: logf, sources: map[string]*sourceHealth{}, halts: map[string]bool{}}
}

// Register starts tracking a running connector.
func (t *Tracker) Register(c connector.Connector, limiter *connector.MeteredLimiter) {
	mode, period := "none", StreamPeriod
	caps := c.Capabilities()
	switch {
	case caps.Stream:
		mode = "stream"
	case caps.Poll:
		mode = "poll"
		if p, ok := c.(connector.Poller); ok && p.Schedule().Interval > 0 {
			period = p.Schedule().Interval
		}
	}
	now := t.clock()
	t.mu.Lock()
	defer t.mu.Unlock()
	prev := t.sources[c.ID()]
	h := &sourceHealth{id: c.ID(), mode: mode, period: period, limiter: limiter, startedAt: now, lastActivity: now, wSince: now, running: true, halted: t.halts[c.ID()]}
	if prev != nil { // keep lifetime totals and the halt flag across restarts
		h.events, h.errors, h.lastEvent = prev.events, prev.errors, prev.lastEvent
		h.halted = h.halted || prev.halted
	}
	t.sources[c.ID()] = h
}

// Unregister marks the connector stopped. operator=true means a deliberate
// stop (no halt is raised); otherwise the connector exited on its own.
func (t *Tracker) Unregister(id string, operator bool, err error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	h, ok := t.sources[id]
	if !ok {
		return
	}
	h.running = false
	if operator {
		h.stopReason = "stopped by operator"
	} else {
		h.stopReason = "connector exited"
		if err != nil {
			h.stopReason += ": " + err.Error()
		}
	}
}

// Activity records a successful poll or stream message.
func (t *Tracker) Activity(id string) {
	t.mu.Lock()
	if h, ok := t.sources[id]; ok {
		h.lastActivity = t.clock()
	}
	t.mu.Unlock()
}

// Events records n published canonical events.
func (t *Tracker) Events(id string, n int) {
	if n <= 0 {
		return
	}
	t.mu.Lock()
	if h, ok := t.sources[id]; ok {
		now := t.clock()
		h.lastActivity, h.lastEvent = now, now
		h.events += int64(n)
		h.wEvents += int64(n)
	}
	t.mu.Unlock()
}

// Error records a connector error.
func (t *Tracker) Error(id string, err error) {
	t.mu.Lock()
	if h, ok := t.sources[id]; ok {
		h.errors++
		h.wErrors++
		h.lastErrAt = t.clock()
		if err != nil {
			h.lastErr = err.Error()
		}
	}
	t.mu.Unlock()
}

// Frames returns the current frame of every tracked source (sorted by id)
// without publishing. The window is not reset.
func (t *Tracker) Frames() []Frame {
	t.mu.Lock()
	defer t.mu.Unlock()
	ids := make([]string, 0, len(t.sources))
	for id := range t.sources {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]Frame, 0, len(ids))
	for _, id := range ids {
		out = append(out, t.frameLocked(t.sources[id], false))
	}
	return out
}

func (t *Tracker) frameLocked(h *sourceHealth, reset bool) Frame {
	now := t.clock()
	f := Frame{Source: h.id, Running: h.running, Mode: h.mode, At: now, ExpectedPeriodS: h.period.Seconds(),
		EventsTotal: h.events, ErrorsTotal: h.errors, Halted: h.halted}
	started := h.startedAt
	f.StartedAt = &started
	if !h.lastActivity.IsZero() {
		a := h.lastActivity
		age := now.Sub(a).Seconds()
		f.LastActivityAt, f.LastActivityAgeS = &a, &age
	}
	if !h.lastEvent.IsZero() {
		e := h.lastEvent
		age := now.Sub(e).Seconds()
		f.LastEventAt, f.LastEventAgeS = &e, &age
	}
	if !h.lastErrAt.IsZero() {
		at := h.lastErrAt
		f.LastError, f.LastErrorAt = h.lastErr, &at
	}
	window := now.Sub(h.wSince)
	f.WindowS = window.Seconds()
	f.EventsWindow, f.ErrorsWindow = h.wEvents, h.wErrors
	if window > 0 {
		f.ErrorRate = float64(h.wErrors) / window.Minutes()
	}
	if h.limiter != nil {
		st := h.limiter.Snapshot(reset)
		f.LimiterSaturation, f.LimiterRequests = st.Saturation, st.Requests
	}
	stale := DownAfterPeriods * h.period
	switch {
	case !h.running:
		f.Status, f.Reason = "down", h.stopReason
	case now.Sub(h.lastActivity) > stale:
		f.Status = "down"
		f.Reason = fmt.Sprintf("no activity for %s (> %d × %s)", now.Sub(h.lastActivity).Truncate(time.Second), DownAfterPeriods, h.period)
	case h.wErrors > 0:
		f.Status, f.Reason = "degraded", fmt.Sprintf("%d error(s) in the last %s: %s", h.wErrors, window.Truncate(time.Second), h.lastErr)
	case f.LimiterSaturation > DegradedSaturation:
		f.Status, f.Reason = "degraded", fmt.Sprintf("rate limiter saturated (%.0f%%)", f.LimiterSaturation*100)
	default:
		f.Status = "ok"
	}
	if reset {
		h.wSince, h.wEvents, h.wErrors = now, 0, 0
		h.frames++
	}
	return f
}

// Tick publishes one frame per tracked source, raises `risk.halt.source:<id>`
// when a source goes down unexpectedly and `risk.resume.source:<id>` when it
// comes back; operator stops publish a final `down` frame and are forgotten.
func (t *Tracker) Tick(ctx context.Context) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for id, h := range t.sources {
		f := t.frameLocked(h, true)
		t.publishFrame(healthSubject(id), f, HealthProto(f))
		t.Published.Frames++
		down := f.Status == "down"
		deliberate := !h.running && h.stopReason == "stopped by operator"
		switch {
		case down && !deliberate && !h.halted:
			h.halted = true
			t.publishProto(bus.RiskHalt(HaltScope(id)), &fav1.Halt{Scope: HaltScope(id), Reason: "ingest: " + id + " " + f.Reason, Since: timestamppb.New(f.At)})
			t.Published.Halts++
			t.log("source down, halt raised", "source", id, "reason", f.Reason)
		case !down && h.halted:
			h.halted = false
			t.publishProto(bus.RiskResume(HaltScope(id)), &fav1.Resume{Scope: HaltScope(id), Reason: "ingest: " + id + " recovered", At: timestamppb.New(f.At), Actor: "system:ingestor"})
			t.Published.Resumes++
			t.log("source recovered, halt cleared", "source", id)
		}
		t.halts[id] = h.halted
		if !h.running {
			delete(t.sources, id) // final frame sent; a restart re-registers
		}
		_ = ctx
	}
}

// Run ticks every interval until ctx is done, then publishes final frames.
func (t *Tracker) Run(ctx context.Context, interval time.Duration) {
	tk := time.NewTicker(interval)
	defer tk.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tk.C:
			t.Tick(ctx)
		}
	}
}

// PublishGap emits an `ingest.gap.<source>` frame.
func (t *Tracker) PublishGap(g Gap) { t.publishFrame(gapSubject(g.Source), g, GapProto(g)) }

// PublishUnresolved emits an `ingest.unresolved.<source>` frame.
func (t *Tracker) PublishUnresolved(u UnresolvedEvent) {
	t.publishFrame(unresolvedSubject(u.Source), u, UnresolvedProto(u))
}

// publishFrame sends the proto message, or v as JSON when the tracker is in legacy mode.
func (t *Tracker) publishFrame(subject string, v any, m proto.Message) {
	if t.pub == nil {
		return
	}
	var (
		b   []byte
		err error
	)
	if t.JSON {
		b, err = json.Marshal(v)
	} else {
		b, err = proto.Marshal(m)
	}
	if err != nil {
		return
	}
	if err := t.pub.NC.Publish(subject, b); err != nil {
		t.log("publish failed", "subject", subject, "err", err)
	}
}

func tsOrNil(t *time.Time) *timestamppb.Timestamp {
	if t == nil || t.IsZero() {
		return nil
	}
	return timestamppb.New(*t)
}

// HealthStatus maps the frame status string onto the proto enum.
func HealthStatus(s string) fav1.HealthStatus {
	switch s {
	case "ok":
		return fav1.HealthStatus_HEALTH_STATUS_OK
	case "degraded":
		return fav1.HealthStatus_HEALTH_STATUS_DEGRADED
	case "down":
		return fav1.HealthStatus_HEALTH_STATUS_DOWN
	}
	return fav1.HealthStatus_HEALTH_STATUS_UNSPECIFIED
}

// HealthProto converts a frame into its fa.v1 message (ages are derived from `at` by consumers).
func HealthProto(f Frame) *fav1.IngestHealth {
	return &fav1.IngestHealth{
		Source: f.Source, Status: HealthStatus(f.Status), Running: f.Running, Mode: f.Mode,
		At: timestamppb.New(f.At), Reason: f.Reason, Halted: f.Halted,
		StartedAt: tsOrNil(f.StartedAt), ExpectedPeriodS: f.ExpectedPeriodS,
		LastActivityAt: tsOrNil(f.LastActivityAt), LastEventAt: tsOrNil(f.LastEventAt),
		EventsTotal: f.EventsTotal, ErrorsTotal: f.ErrorsTotal,
		WindowS: f.WindowS, EventsWindow: f.EventsWindow, ErrorsWindow: f.ErrorsWindow, ErrorRate: f.ErrorRate,
		LimiterSaturation: f.LimiterSaturation, LimiterRequests: f.LimiterRequests,
		LastError: f.LastError, LastErrorAt: tsOrNil(f.LastErrorAt),
	}
}

// GapProto converts a gap into its fa.v1 message.
func GapProto(g Gap) *fav1.IngestGap {
	return &fav1.IngestGap{Source: g.Source, InstrumentId: g.InstrumentID, Timeframe: g.Timeframe,
		From: timestamppb.New(g.From), To: timestamppb.New(g.To), MissingPeriods: g.MissingPeriods, At: timestamppb.New(g.At)}
}

// UnresolvedProto converts an unresolved-symbol event into its fa.v1 message.
func UnresolvedProto(u UnresolvedEvent) *fav1.IngestUnresolved {
	return &fav1.IngestUnresolved{Source: u.Source, VenueSymbol: u.VenueSymbol, InstrumentId: u.InstrumentID,
		Reason: u.Reason, At: timestamppb.New(u.At)}
}

func (t *Tracker) publishProto(subject string, m proto.Message) {
	if t.pub == nil {
		return
	}
	if err := t.pub.Publish(subject, m); err != nil {
		t.log("publish failed", "subject", subject, "err", err)
	}
}

// GapDetector finds bar gaps per (instrument, timeframe) for one source.
type GapDetector struct {
	source string
	emit   func(Gap)
	clock  func() time.Time

	mu   sync.Mutex
	last map[string]time.Time // inst|tf → last bar close
}

// NewGapDetector seeds the detector with persisted watermarks (inst|tf → cursor).
func NewGapDetector(source string, seed map[string]time.Time, emit func(Gap)) *GapDetector {
	d := &GapDetector{source: source, emit: emit, clock: func() time.Time { return time.Now().UTC() }, last: map[string]time.Time{}}
	for k, v := range seed {
		d.last[k] = v
	}
	return d
}

// GapKey is the detector/seed key for an instrument and timeframe.
func GapKey(instrumentID, tf string) string { return instrumentID + "|" + tf }

// Period parses a bar timeframe ("1m", "15m", "1h", "1d", "1w") into a duration.
func Period(tf string) (time.Duration, bool) {
	if len(tf) < 2 {
		return 0, false
	}
	n := 0
	for _, r := range tf[:len(tf)-1] {
		if r < '0' || r > '9' {
			return 0, false
		}
		n = n*10 + int(r-'0')
	}
	if n == 0 {
		return 0, false
	}
	switch tf[len(tf)-1] {
	case 's':
		return time.Duration(n) * time.Second, true
	case 'm':
		return time.Duration(n) * time.Minute, true
	case 'h':
		return time.Duration(n) * time.Hour, true
	case 'd':
		return time.Duration(n) * 24 * time.Hour, true
	case 'w':
		return time.Duration(n) * 7 * 24 * time.Hour, true
	}
	return 0, false
}

// Observe records a bar close and emits a Gap when it is more than one
// period after the previous one. Bars arriving out of order are ignored.
func (d *GapDetector) Observe(instrumentID, tf string, closeAt time.Time) {
	period, ok := Period(tf)
	if !ok || instrumentID == "" {
		return
	}
	key := GapKey(instrumentID, tf)
	d.mu.Lock()
	defer d.mu.Unlock()
	prev, seen := d.last[key]
	if seen && !closeAt.After(prev) {
		return
	}
	d.last[key] = closeAt
	if !seen {
		return
	}
	if delta := closeAt.Sub(prev); delta > period+period/2 {
		missing := int64(delta/period) - 1
		if d.emit != nil {
			d.emit(Gap{Source: d.source, InstrumentID: instrumentID, Timeframe: tf, From: prev, To: closeAt, MissingPeriods: missing, At: d.clock()})
		}
	}
}
