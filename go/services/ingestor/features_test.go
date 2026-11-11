package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/nats-io/nats.go"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/fey/financial-auto/go/connectors/yahoo"
	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// fakeConn is a NATSConn whose connectivity and flush can be toggled.
type fakeConn struct {
	mu        sync.Mutex
	up        bool
	failFlush bool
	pending   []*nats.Msg
	delivered []*nats.Msg
}

func (f *fakeConn) IsConnected() bool { f.mu.Lock(); defer f.mu.Unlock(); return f.up }
func (f *fakeConn) PublishMsg(m *nats.Msg) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.up {
		return nats.ErrConnectionClosed
	}
	f.pending = append(f.pending, m)
	return nil
}
func (f *fakeConn) FlushTimeout(time.Duration) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.up || f.failFlush {
		f.pending = nil // lost in flight
		return nats.ErrTimeout
	}
	f.delivered = append(f.delivered, f.pending...)
	f.pending = nil
	return nil
}
func (f *fakeConn) ids() []string {
	f.FlushTimeout(0)
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	for _, m := range f.delivered {
		out = append(out, m.Header.Get(nats.MsgIdHdr))
	}
	return out
}

func tenv(id string) *fav1.Envelope {
	return &fav1.Envelope{EventId: id, SourceId: "t", Payload: &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: 1}}}
}

func TestBusBufferSpoolsWhileDownAndReplaysInOrder(t *testing.T) {
	dir := t.TempDir()
	nc := &fakeConn{up: true}
	b, err := OpenBusBuffer(nc, dir, 1<<20, bus.NewCounters())
	if err != nil {
		t.Fatal(err)
	}
	drained := 0
	b.OnDrained = func() { drained++ }
	b.Batch = 2
	must := func(id string) {
		t.Helper()
		if err := b.PublishWithID("md.trade.x.y", id, tenv(id)); err != nil {
			t.Fatal(err)
		}
	}
	must("a")
	nc.FlushTimeout(0) // delivered while connected
	nc.mu.Lock()
	nc.up = false
	nc.mu.Unlock()
	must("b")
	must("c")
	must("d")
	if b.Records() != 3 {
		t.Fatalf("records %d", b.Records())
	}
	nc.mu.Lock()
	nc.up = true
	nc.mu.Unlock()
	must("e") // buffer not empty: appended behind b, c, d to keep order
	// a flush failure keeps the unconfirmed batch
	nc.mu.Lock()
	nc.failFlush = true
	nc.mu.Unlock()
	if n, err := b.Drain(); err == nil || n != 0 || b.Records() != 4 {
		t.Fatalf("failed drain: n=%d err=%v records=%d", n, err, b.Records())
	}
	nc.mu.Lock()
	nc.failFlush = false
	nc.mu.Unlock()
	if n, err := b.Drain(); err != nil || n != 4 || b.Records() != 0 || drained != 1 {
		t.Fatalf("drain: n=%d err=%v records=%d drained=%d", n, err, b.Records(), drained)
	}
	got := nc.ids()
	want := []string{"a", "b", "c", "d", "e"}
	if len(got) != len(want) {
		t.Fatalf("delivered %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("delivered %v, want %v", got, want)
		}
	}
	var e fav1.Envelope
	if err := proto.Unmarshal(nc.delivered[2].Data, &e); err != nil || e.EventId != "c" {
		t.Fatalf("payload %v %v", e.EventId, err)
	}
	if st, _ := os.Stat(filepath.Join(dir, "nats-envelopes.spool")); st.Size() != 0 {
		t.Fatalf("spool not truncated: %d", st.Size())
	}
}

func TestBusBufferFullAndCrashRecovery(t *testing.T) {
	dir := t.TempDir()
	nc := &fakeConn{}
	b, err := OpenBusBuffer(nc, dir, 200, nil)
	if err != nil {
		t.Fatal(err)
	}
	var full error
	for i := range 20 {
		if err := b.PublishWithID("s", string(rune('a'+i)), tenv("x")); err != nil {
			full = err
			break
		}
	}
	if !errors.Is(full, ErrBusBufferFull) {
		t.Fatalf("want ErrBusBufferFull, got %v", full)
	}
	n := b.Records()
	_ = b.Close()
	// torn tail: half a record appended by a crash
	f, _ := os.OpenFile(filepath.Join(dir, "nats-envelopes.spool"), os.O_APPEND|os.O_WRONLY, 0)
	f.Write(encodeRecord("s", "torn", []byte("0123456789"))[:7])
	f.Close()
	b2, err := OpenBusBuffer(nc, dir, 1<<20, nil)
	if err != nil {
		t.Fatal(err)
	}
	if b2.Records() != n {
		t.Fatalf("recovered %d records, want %d", b2.Records(), n)
	}
	nc.up = true
	if k, err := b2.Drain(); err != nil || int64(k) != n {
		t.Fatalf("drain %d %v", k, err)
	}
}

func TestPipelineStopsConnectorWhenBusBufferFull(t *testing.T) {
	p := &Pipeline{Counters: bus.NewCounters(), Archive: connector.NopArchive{}, Out: fullPub{}}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err := p.Run(ctx, slog.Default(), &streamConn{})
	if !errors.Is(err, ErrBusBufferFull) {
		t.Fatalf("run ended with %v", err)
	}
	if got := p.TakeBusStopped(); len(got) != 1 || got[0] != "streamy" {
		t.Fatalf("bus stopped %v", got)
	}
}

type fullPub struct{}

func (fullPub) PublishWithID(string, string, proto.Message) error { return ErrBusBufferFull }

// streamConn emits one trade per 10 ms until cancelled.
type streamConn struct{}

func (*streamConn) ID() string { return "streamy" }
func (*streamConn) Capabilities() connector.Capabilities {
	return connector.Capabilities{Stream: true, Venue: "v", Events: []connector.EventKind{connector.KindTrade}}
}
func (*streamConn) Init(context.Context, connector.SourceConfig, connector.Deps) error { return nil }
func (*streamConn) Stream(ctx context.Context, _ []connector.Subscription, out chan<- connector.RawMessage) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case out <- connector.RawMessage{SourceID: "streamy", ReceivedAt: time.Now()}:
			time.Sleep(10 * time.Millisecond)
		}
	}
}
func (*streamConn) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	e := tenv("")
	e.SourceId, e.InstrumentId = "streamy", "01900000-0000-7000-8000-000000000001"
	e.EventTime, e.IngestTime = timestamppb.New(raw.ReceivedAt), timestamppb.New(raw.ReceivedAt)
	return []*fav1.Envelope{e}, nil, nil
}

func TestSkewMonitorLagFutureAndRatio(t *testing.T) {
	c := bus.NewCounters()
	m := NewSkewMonitor(c, func(string) time.Duration { return 100 * time.Second })
	now := time.Now()
	for i := range 10 {
		e := tenv("")
		e.EventTime, e.IngestTime = timestamppb.New(now.Add(-time.Duration(40+i)*time.Second)), timestamppb.New(now)
		m.Observe("kraken", e)
	}
	fut := tenv("")
	fut.EventTime, fut.IngestTime = timestamppb.New(now.Add(10*time.Second)), timestamppb.New(now)
	m.Observe("kraken", fut)
	bf := tenv("")
	bf.Backfill = true
	bf.EventTime, bf.IngestTime = timestamppb.New(now.Add(-time.Hour)), timestamppb.New(now)
	m.Observe("kraken", bf) // ignored
	news := &fav1.Envelope{Payload: &fav1.Envelope_News{News: &fav1.NewsItem{}}, EventTime: timestamppb.New(now.Add(-time.Hour)), IngestTime: timestamppb.New(now)}
	m.Observe("kraken", news) // ignored
	m.Export()
	st := m.Stats()["kraken"]
	if st.N != 11 || st.P50 < 40 || st.P50 > 50 || st.Ratio < 0.4 || st.Ratio > 0.5 {
		t.Fatalf("stats %+v", st)
	}
	snap := c.Snapshot()
	if snap["clock.future_events.kraken"] != 1 || snap["ingest_lag_p50_s.kraken"] != st.P50 || snap["ingest_lag_ratio.kraken"] == 0 {
		t.Fatalf("counters %v", snap)
	}
}

type capturePub struct {
	mu   sync.Mutex
	envs []*fav1.Envelope
}

func (c *capturePub) PublishWithID(_, _ string, m proto.Message) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.envs = append(c.envs, m.(*fav1.Envelope))
	return nil
}

func yahooFixtureServer(t *testing.T, hits *int) *httptest.Server {
	t.Helper()
	fixture, err := os.ReadFile("../../connectors/yahoo/testdata/chart_aapl_1m.json")
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		*hits++
		mu.Unlock()
		w.Write(fixture)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestGapFillMarksEnvelopesGapFilled(t *testing.T) {
	hits := 0
	srv := yahooFixtureServer(t, &hits)
	out := &capturePub{}
	c := bus.NewCounters()
	p := &Pipeline{Counters: c, Archive: connector.NopArchive{}, Out: out}
	res := connector.NewStaticResolver()
	res.Add("yahoo", "AAPL", "01900000-0000-7000-8000-00000000aaaa")
	g := &GapFiller{Pipe: p, Counters: c, Instance: func(ctx context.Context, _ string) (connector.Connector, error) {
		y := &yahoo.Connector{BaseURL: srv.URL}
		src := catalog.Source{ID: "yahoo", Config: &catalog.Params{Symbols: []string{"AAPL"}}}
		return y, y.Init(ctx, connector.SourceConfig{Source: src}, connector.Deps{Resolver: res, Clock: connector.SystemClock{}})
	}}
	from := time.Unix(1794234600, 0).UTC()
	j := GapJob{Source: "yahoo", Symbol: "AAPL", InstrumentID: "01900000-0000-7000-8000-00000000aaaa", Timeframe: "1m", From: from, To: from.Add(5 * time.Minute), Reason: "bar_gap"}
	if !g.Enqueue(j) || g.Enqueue(j) {
		t.Fatal("enqueue: first must be accepted, duplicate refused")
	}
	if g.Enqueue(GapJob{Source: "yahoo", Symbol: "AAPL", Timeframe: "1m", From: from, To: from.Add(48 * time.Hour)}) {
		t.Fatal("a 48 h gap must be left to a backfill job")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { g.Run(ctx); close(done) }()
	deadline := time.Now().Add(5 * time.Second)
	for c.Snapshot()["gapfill.done.yahoo"] == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	cancel()
	<-done
	if len(out.envs) == 0 || hits != 1 {
		t.Fatalf("published %d, hits %d, counters %v", len(out.envs), hits, c.Snapshot())
	}
	for _, e := range out.envs {
		if !e.GapFilled || !e.Backfill || e.EventId == "" {
			t.Fatalf("envelope not marked: %+v", e)
		}
	}
}

func TestBarGapEnqueuesFill(t *testing.T) {
	var got []GapJob
	var mu sync.Mutex
	h := NewTracker(nil, nil)
	p := &Pipeline{Counters: bus.NewCounters(), Health: h}
	g := &GapFiller{Pipe: p}
	p.GapFill = g
	d := p.gaps(context.Background(), "yahoo")
	t0 := time.Unix(1794234600, 0).UTC()
	d.ObserveSymbol("inst", "AAPL", "1m", t0)
	d.ObserveSymbol("inst", "AAPL", "1m", t0.Add(5*time.Minute))
	g.init()
	select {
	case j := <-g.jobs:
		mu.Lock()
		got = append(got, j)
		mu.Unlock()
	default:
	}
	if len(got) != 1 || got[0].Symbol != "AAPL" || !got[0].From.Equal(t0) || got[0].Reason != "bar_gap" {
		t.Fatalf("jobs %+v", got)
	}
}

func TestAutoDisablerAfterPersistentDown(t *testing.T) {
	now := time.Unix(1_800_000_000, 0).UTC()
	var disabled []string
	a := &AutoDisabler{After: 24 * time.Hour, Now: func() time.Time { return now },
		InCatalog: func(id string) bool { return id != "heartbeat" },
		Disable:   func(_ context.Context, id, note string) error { disabled = append(disabled, id+": "+note); return nil }}
	a.Observe(Frame{Source: "gdelt", Status: "down", Reason: "no activity"}, false)
	a.Observe(Frame{Source: "kraken", Status: "down"}, false)
	a.Observe(Frame{Source: "heartbeat", Status: "down"}, false)
	a.Observe(Frame{Source: "yahoo", Status: "down"}, true) // operator stop never counts
	now = now.Add(23 * time.Hour)
	a.Observe(Frame{Source: "kraken", Status: "ok"}, false) // recovered: clock reset
	if got := a.Check(context.Background()); len(got) != 0 {
		t.Fatalf("disabled early: %v", got)
	}
	now = now.Add(time.Hour)
	if got := a.Check(context.Background()); len(got) != 1 || got[0] != "gdelt" {
		t.Fatalf("disabled %v (%v)", got, disabled)
	}
	if len(disabled) != 1 || !contains(disabled[0], "no activity") {
		t.Fatalf("note %v", disabled)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(sub) == 0 || indexOf(s, sub) >= 0)
}
func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func TestReloaderConfigureReloadAndWatch(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sources.yaml")
	write := func(body string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(`version: 1
sources:
  - {id: yahoo, name: Yahoo, status: live, auth: none, rate_limit: {per_second: 1, burst: 3}, config: {symbols: [AAPL]}}
`)
	cat, err := catalog.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	starts := 0
	var smu sync.Mutex
	run := func(ctx context.Context, _ *slog.Logger, _ connector.Connector) error {
		smu.Lock()
		starts++
		smu.Unlock()
		<-ctx.Done()
		return nil
	}
	sup := NewSupervisor(cat, slog.Default(), bus.NewCounters(), func(catalog.Source) connector.Deps { return connector.Deps{Clock: connector.SystemClock{}} }, run)
	ctx, cancel := context.WithCancel(context.Background())
	defer func() { cancel(); sup.Wait() }()
	if err := sup.Start(ctx, "yahoo"); err != nil {
		t.Fatal(err)
	}
	limits := map[string]catalog.RateLimit{}
	var changes []string
	r := &Reloader{Path: path, Cat: cat, Sup: sup,
		SetLimit: func(s catalog.Source) { limits[s.ID] = s.Limit() },
		Changed:  func(id, change string, _ map[string]string) { changes = append(changes, id+":"+change) }}

	// rate limit only: bucket updated, connector not restarted
	res, err := r.Configure(ctx, map[string]string{"source": "yahoo", "rate_limit_per_second": "0.5", "rate_limit_burst": "1"})
	if err != nil || res.Restarted || res.RateLimit.PerSecond != 0.5 || limits["yahoo"].Burst != 1 {
		t.Fatalf("configure limit: %+v %v %v", res, err, limits)
	}
	// config: restart
	res, err = r.Configure(ctx, map[string]string{"source": "yahoo", "config": `{"symbols":["AAPL","MSFT"],"timeframe":"5m"}`})
	if err != nil || !res.Restarted || len(res.Config.Symbols) != 2 {
		t.Fatalf("configure config: %+v %v", res, err)
	}
	for _, bad := range []map[string]string{
		{"source": "yahoo"},
		{"source": "yahoo", "rate_limit_per_second": "1"},
		{"source": "yahoo", "rate_limit_per_second": "0", "rate_limit_burst": "1"},
		{"source": "yahoo", "config": `{"symbolz":["X"]}`},
	} {
		if _, err := r.Configure(ctx, bad); !errors.Is(err, ErrInvalid) {
			t.Fatalf("%v: want invalid, got %v", bad, err)
		}
	}
	if _, err := r.Configure(ctx, map[string]string{"source": "nope", "clear": "1"}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown source: %v", err)
	}

	// YAML edit picked up by the watcher (reload_on_change); YAML now differs from memory
	go r.Watch(ctx, 20*time.Millisecond)
	time.Sleep(50 * time.Millisecond)
	write(`version: 1
sources:
  - {id: yahoo, name: Yahoo, status: live, auth: none, rate_limit: {per_second: 2, burst: 4}, config: {symbols: [SPY]}}
  - {id: kraken, name: Kraken, status: planned, auth: none}
`)
	deadline := time.Now().Add(3 * time.Second)
	for {
		if s, _ := cat.ByID("yahoo"); s.Limit().PerSecond == 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("watcher did not reload")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, ok := cat.ByID("kraken"); !ok {
		t.Fatal("added source missing after reload")
	}
	time.Sleep(50 * time.Millisecond)
	smu.Lock()
	n := starts
	smu.Unlock()
	if n != 3 || !sup.IsRunning("yahoo") || len(changes) < 3 {
		t.Fatalf("starts %d running %v changes %v", n, sup.IsRunning("yahoo"), changes)
	}
}
