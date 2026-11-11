package bus

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"strings"

	"github.com/nats-io/nats.go"
)

// Trace ids in NATS headers (spec 60 §6.5), the same W3C `traceparent` header as the Rust
// side (`crates/fa-bus/src/trace.rs`): `00-<32 hex trace id>-<16 hex span id>-01`.
// No OpenTelemetry exporter: the trace id is a log field (`trace_id`).

// TraceHeader is the W3C trace-context header name.
const TraceHeader = "traceparent"

// Trace is one hop of a trace.
type Trace struct {
	TraceID string // 32 lowercase hex
	SpanID  string // 16 lowercase hex
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// NewTrace starts a new trace.
func NewTrace() Trace { return Trace{TraceID: randHex(16), SpanID: randHex(8)} }

// Child is the next hop of the same trace.
func (t Trace) Child() Trace { return Trace{TraceID: t.TraceID, SpanID: randHex(8)} }

// Header renders the traceparent value.
func (t Trace) Header() string { return "00-" + t.TraceID + "-" + t.SpanID + "-01" }

func isHex(s string, n int) bool {
	if len(s) != n || strings.Trim(s, "0") == "" {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// ParseTrace parses a traceparent value.
func ParseTrace(v string) (Trace, bool) {
	p := strings.Split(strings.TrimSpace(v), "-")
	if len(p) != 4 || len(p[0]) != 2 || !isHex(strings.ToLower(p[1]), 32) || !isHex(strings.ToLower(p[2]), 16) {
		return Trace{}, false
	}
	return Trace{TraceID: strings.ToLower(p[1]), SpanID: strings.ToLower(p[2])}, true
}

// TraceOf returns the trace carried by a received message.
func TraceOf(m *nats.Msg) (Trace, bool) {
	if m == nil || m.Header == nil {
		return Trace{}, false
	}
	return ParseTrace(m.Header.Get(TraceHeader))
}

type traceKey struct{}

// WithTrace stores t in ctx; Publisher.PublishCtx continues it.
func WithTrace(ctx context.Context, t Trace) context.Context {
	return context.WithValue(ctx, traceKey{}, t)
}

// TraceFrom returns the trace stored in ctx.
func TraceFrom(ctx context.Context) (Trace, bool) {
	t, ok := ctx.Value(traceKey{}).(Trace)
	return t, ok
}

// Traced reports whether messages on subject carry a trace header: not market data
// (`md.*`) and not heartbeats, which are high-rate and start no chain worth correlating.
func Traced(subject string) bool {
	return !strings.HasPrefix(subject, "md.") && !strings.HasPrefix(subject, "svc.heartbeat.")
}

// Consumed logs a consumed message at debug with its trace id and returns a logger carrying
// `trace_id` (the input logger when the message has no trace) plus a context holding the trace
// for publishes made while handling it.
func Consumed(ctx context.Context, log *slog.Logger, m *nats.Msg) (context.Context, *slog.Logger) {
	t, ok := TraceOf(m)
	if !ok {
		return ctx, log
	}
	l := log.With("trace_id", t.TraceID)
	l.Debug("consume", "subject", m.Subject)
	return WithTrace(ctx, t), l
}
