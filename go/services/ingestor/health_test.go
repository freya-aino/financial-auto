package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/connectors/heartbeat"
	"github.com/fey/financial-auto/go/connectors/yahoo"
	"github.com/fey/financial-auto/go/pkg/connector"
)

func TestTrackerStatusTransitionsAndHalts(t *testing.T) {
	now := time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)
	tr := NewTracker(nil, nil)
	tr.clock = func() time.Time { return now }
	y := &yahoo.Connector{}
	tr.Register(y, connector.NewMeteredLimiter(nil)) // poll, period 1m
	tr.Register(&heartbeat.Connector{}, nil)         // stream, period 30 s

	fs := tr.Frames()
	if len(fs) != 2 || fs[0].Source != "heartbeat" || fs[1].Source != "yahoo" {
		t.Fatalf("frames %+v", fs)
	}
	if fs[1].Status != "ok" || fs[1].Mode != "poll" || fs[1].ExpectedPeriodS != 60 || fs[0].ExpectedPeriodS != 30 {
		t.Fatalf("initial %+v", fs[1])
	}
	var f Frame

	tr.Events("yahoo", 4)
	tr.Error("yahoo", errTest("boom"))
	f = tr.Frames()[1]
	if f.Status != "degraded" || f.EventsTotal != 4 || f.ErrorsWindow != 1 || f.LastError != "boom" || !strings.Contains(f.Reason, "boom") {
		t.Fatalf("degraded %+v", f)
	}
	tr.Tick(context.Background()) // resets the window
	if f = tr.Frames()[1]; f.Status != "ok" || f.ErrorsWindow != 0 || f.ErrorsTotal != 1 {
		t.Fatalf("after tick %+v", f)
	}

	// No activity for > 3 periods → down + halt, once.
	now = now.Add(3*time.Minute + time.Second)
	tr.Tick(context.Background())
	tr.Tick(context.Background())
	f = tr.Frames()[1]
	if f.Status != "down" || !f.Halted || tr.Published.Halts != 2 || !strings.Contains(f.Reason, "no activity") {
		t.Fatalf("down %+v halts=%d", f, tr.Published.Halts)
	}
	if hb := tr.Frames()[0]; hb.Status != "down" || !hb.Halted {
		t.Fatalf("heartbeat %+v", hb)
	}

	// Activity resumes → ok + resume.
	tr.Activity("yahoo")
	tr.Tick(context.Background())
	f = tr.Frames()[1]
	if f.Status != "ok" || f.Halted || tr.Published.Resumes != 1 {
		t.Fatalf("recovered %+v resumes=%d", f, tr.Published.Resumes)
	}

	// Operator stop: final down frame, no halt, then forgotten.
	tr.Unregister("yahoo", true, nil)
	halts := tr.Published.Halts
	tr.Tick(context.Background())
	if tr.Published.Halts != halts {
		t.Fatal("operator stop must not raise a halt")
	}
	for _, fr := range tr.Frames() {
		if fr.Source == "yahoo" {
			t.Fatal("stopped source still tracked")
		}
	}

	// Self-exit: halt raised; re-register keeps the halt flag so recovery resumes it.
	tr.Register(y, nil)
	tr.Unregister("yahoo", false, errTest("socket closed"))
	tr.Tick(context.Background())
	if tr.Published.Halts != halts+1 {
		t.Fatalf("self exit must halt (%d → %d)", halts, tr.Published.Halts)
	}
	tr.Register(y, nil)
	if f = tr.Frames()[1]; !f.Halted || f.Status != "ok" {
		t.Fatalf("re-registered %+v", f)
	}
	tr.Tick(context.Background())
	if tr.Published.Resumes != 2 {
		t.Fatalf("resumes %d", tr.Published.Resumes)
	}
}

type errTest string

func (e errTest) Error() string { return string(e) }

func TestGapDetector(t *testing.T) {
	var gaps []Gap
	base := time.Date(2026, 11, 10, 14, 0, 0, 0, time.UTC)
	seed := map[string]time.Time{GapKey("i1", "1m"): base}
	d := NewGapDetector("yahoo", seed, func(g Gap) { gaps = append(gaps, g) })
	d.Observe("i1", "1m", base.Add(time.Minute)) // consecutive
	d.Observe("i1", "1m", base.Add(2*time.Minute))
	d.Observe("i1", "1m", base.Add(time.Minute)) // out of order, ignored
	if len(gaps) != 0 {
		t.Fatalf("unexpected gaps %+v", gaps)
	}
	d.Observe("i1", "1m", base.Add(6*time.Minute)) // 3 bars missing
	if len(gaps) != 1 || gaps[0].MissingPeriods != 3 || !gaps[0].From.Equal(base.Add(2*time.Minute)) || gaps[0].Timeframe != "1m" {
		t.Fatalf("gaps %+v", gaps)
	}
	d.Observe("i2", "1h", base) // first sight of an unseeded key: baseline only
	d.Observe("i2", "1h", base.Add(time.Hour))
	d.Observe("i2", "1h", base.Add(3*time.Hour))
	if len(gaps) != 2 || gaps[1].MissingPeriods != 1 || gaps[1].InstrumentID != "i2" {
		t.Fatalf("gaps %+v", gaps)
	}
	d.Observe("", "1m", base) // no instrument → ignored
	d.Observe("i3", "weird", base)
	if len(gaps) != 2 {
		t.Fatal("ignored keys produced gaps")
	}
	if p, ok := Period("15m"); !ok || p != 15*time.Minute {
		t.Fatal("period parse")
	}
	if _, ok := Period("m"); ok {
		t.Fatal("bad period accepted")
	}
}

func TestParseBackfillSpec(t *testing.T) {
	now := time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)
	spec, start, end, err := parseBackfillSpec([]byte(`{"source":" yahoo ","instruments":["AAPL"," ","MSFT"],"timescale":"1m","start":"2026-11-09T00:00:00Z","end":"2026-11-10T00:00:00Z"}`), now)
	if err != nil || spec.Source != "yahoo" || len(spec.Instruments) != 2 || !end.Equal(now.Add(-12*time.Hour)) || !start.Equal(end.Add(-24*time.Hour)) {
		t.Fatalf("%+v %v %v %v", spec, start, end, err)
	}
	// Defaults: end = now, start = now - depth(1h) = 5 years, timescale default 1m.
	spec, start, end, err = parseBackfillSpec([]byte(`{"source":"yahoo","timescale":"1h"}`), now)
	if err != nil || !end.Equal(now) || !start.Equal(now.Add(-5*365*24*time.Hour)) || spec.End != now.Format(time.RFC3339) {
		t.Fatalf("defaults %+v %v %v %v", spec, start, end, err)
	}
	if spec, _, _, err := parseBackfillSpec([]byte(`{"source":"yahoo"}`), now); err != nil || spec.Timescale != "1m" {
		t.Fatalf("timescale default %+v %v", spec, err)
	}
	for _, bad := range []string{`{}`, `{"source":"x","timescale":"fast"}`, `{"source":"x","start":"2026-11-11T00:00:00Z","end":"2026-11-10T00:00:00Z"}`, `{"source":"x","start":"yesterday"}`, `nope`} {
		if _, _, _, err := parseBackfillSpec([]byte(bad), now); err == nil || !strings.HasPrefix(err.Error(), "invalid") {
			t.Fatalf("%s: want invalid error, got %v", bad, err)
		}
	}
}

type memStatuses map[string]string

func (m memStatuses) SetStatus(_ context.Context, id, status string) error {
	m[id] = status
	return nil
}

func TestSupervisorEnableDisable(t *testing.T) {
	s := testSupervisor(t)
	st := memStatuses{}
	s.Statuses = st
	s.Health = NewTracker(nil, nil)
	ctx := context.Background()

	stopped, err := s.Disable(ctx, "yahoo") // stub, not running
	if err != nil || stopped {
		t.Fatalf("disable idle: %v %v", stopped, err)
	}
	if src, _ := s.cat.ByID("yahoo"); src.Status != "disabled" || st["yahoo"] != "disabled" {
		t.Fatalf("status not applied: %+v %v", src, st)
	}
	if err := s.Start(ctx, "yahoo"); err == nil {
		t.Fatal("disabled source must not start")
	}
	started, err := s.Enable(ctx, "yahoo")
	if err != nil || !started || !s.IsRunning("yahoo") || st["yahoo"] != "live" {
		t.Fatalf("enable: %v %v running=%v st=%v", started, err, s.IsRunning("yahoo"), st)
	}
	if f := s.Health.Frames(); len(f) != 1 || f[0].Source != "yahoo" || !f[0].Running {
		t.Fatalf("health after enable %+v", f)
	}
	if started, err := s.Enable(ctx, "yahoo"); err != nil || started {
		t.Fatalf("enable running: %v %v", started, err)
	}
	stopped, err = s.Disable(ctx, "yahoo")
	if err != nil || !stopped || s.IsRunning("yahoo") {
		t.Fatalf("disable running: %v %v", stopped, err)
	}
	if _, err := s.Enable(ctx, "nocode"); err == nil || !strings.HasPrefix(err.Error(), "conflict") {
		t.Fatalf("enable without connector: %v", err)
	}
	if _, err := s.Enable(ctx, "nope"); err == nil || !strings.HasPrefix(err.Error(), "not_found") {
		t.Fatalf("enable unknown: %v", err)
	}
	if err := s.SetStatus(ctx, "yahoo", "bogus"); err == nil {
		t.Fatal("bogus status accepted")
	}
	s.Wait()
}
