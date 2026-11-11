package bus

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"

	"github.com/nats-io/nats.go"
)

func TestTraceparentRoundTrip(t *testing.T) {
	tr := NewTrace()
	got, ok := ParseTrace(tr.Header())
	if !ok || got != tr {
		t.Fatalf("round trip: %v %v", got, ok)
	}
	if c := tr.Child(); c.TraceID != tr.TraceID || c.SpanID == tr.SpanID {
		t.Fatalf("child: %v", c)
	}
	// W3C example; also produced by the Rust side
	w3c, ok := ParseTrace("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
	if !ok || w3c.TraceID != "0af7651916cd43dd8448eb211c80319c" {
		t.Fatalf("w3c: %v", w3c)
	}
	for _, bad := range []string{"", "00-abc-def-01", "00-00000000000000000000000000000000-0000000000000001-01", "x"} {
		if _, ok := ParseTrace(bad); ok {
			t.Fatalf("accepted %q", bad)
		}
	}
	if Traced("md.trade.sim.x") || Traced("svc.heartbeat.ingestor") || !Traced("ctl.evo.start") {
		t.Fatal("Traced")
	}
}

func TestTraceHeaderOnMessages(t *testing.T) {
	m := nats.NewMsg("ctl.evo.start")
	if _, ok := TraceOf(m); ok {
		t.Fatal("no header yet")
	}
	tr := NewTrace()
	setTrace(m, context.Background())
	first, ok := TraceOf(m)
	if !ok {
		t.Fatal("publish path sets a header")
	}
	m2 := nats.NewMsg("jobs.result.x")
	setTrace(m2, WithTrace(context.Background(), tr))
	if got, _ := TraceOf(m2); got.TraceID != tr.TraceID || got.SpanID == tr.SpanID {
		t.Fatalf("continue trace: %v vs %v", got, tr)
	}
	if first.TraceID == tr.TraceID {
		t.Fatal("background context starts a new trace")
	}
	md := nats.NewMsg("md.trade.sim.x")
	setTrace(md, context.Background())
	if md.Header.Get(TraceHeader) != "" {
		t.Fatal("market data is not traced")
	}

	var buf bytes.Buffer
	log := newLogger(&buf, "test", "json", "debug")
	ctx, l := Consumed(context.Background(), log, m2)
	l.Info("handled")
	if got, ok := TraceFrom(ctx); !ok || got.TraceID != tr.TraceID {
		t.Fatalf("ctx trace %v", got)
	}
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("lines: %q", buf.String())
	}
	var rec map[string]any
	if err := json.Unmarshal([]byte(lines[1]), &rec); err != nil {
		t.Fatal(err)
	}
	if rec["trace_id"] != tr.TraceID || rec["svc"] != "test" || rec["msg"] != "handled" {
		t.Fatalf("record %v", rec)
	}
}

func TestLoggerFormat(t *testing.T) {
	var buf bytes.Buffer
	newLogger(&buf, "svc1", "text", "").Info("hi", "k", 1)
	if !strings.Contains(buf.String(), "svc=svc1") || strings.HasPrefix(buf.String(), "{") {
		t.Fatalf("text: %q", buf.String())
	}
	buf.Reset()
	newLogger(&buf, "svc1", "JSON", "").Debug("hidden")
	if buf.Len() != 0 {
		t.Fatal("debug hidden at default level")
	}
	newLogger(&buf, "svc1", "JSON", "").Info("hi")
	var rec map[string]any
	if err := json.Unmarshal(buf.Bytes(), &rec); err != nil || rec["svc"] != "svc1" {
		t.Fatalf("json: %q %v", buf.String(), err)
	}
	_ = slog.LevelInfo
}
