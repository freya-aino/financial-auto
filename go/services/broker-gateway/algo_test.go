package main

import (
	"math"
	"strings"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// clock is a settable engine clock.
type clock struct{ t time.Time }

func (c *clock) now() time.Time { return c.t }

func clocked(m *CostModel) (*PaperEngine, *clock) {
	c := &clock{t: t0}
	p := NewPaperEngine(m)
	p.now = c.now
	p.Counters = bus.NewCounters()
	return p, c
}

func algoIntent(id, algo string, qty float64) *fav1.OrderIntent {
	in := intent(id)
	in.ExecAlgo, in.Quantity = algo, qty
	return in
}

func sumQty(evs []Published) float64 {
	var s float64
	for _, e := range evs {
		s += e.Event.FillQuantity
	}
	return s
}

func TestParseAlgo(t *testing.T) {
	a, err := ParseAlgo("")
	if err != nil || a.Name != "market" {
		t.Fatal(a, err)
	}
	a, err = ParseAlgo("twap:dur_s=60,slices=4")
	if err != nil || a.Name != "twap" || a.DurS != 60 || a.Slices != 4 {
		t.Fatal(a, err)
	}
	a, err = ParseAlgo(" POV : rate=0.2 ")
	if err != nil || a.Name != "pov" || a.Rate != 0.2 || a.MaxDurS != 900 {
		t.Fatal(a, err)
	}
	for _, bad := range []string{"iceberg", "twap:slices=0", "twap:slices=2.5", "pov:rate=2", "twap:foo=1",
		"smart_limit:timeout_s=x", "vwap:dur_s=-1", "twap:dur_s", "market:dur_s=1"} {
		if _, err := ParseAlgo(bad); err == nil {
			t.Errorf("%q must be rejected", bad)
		}
	}
}

func TestUnknownAlgoIsRejected(t *testing.T) {
	p, _ := clocked(nil)
	p.ObserveTrade(tradeSub, 100, 1, t0)
	ev := p.Execute(algoIntent("x", "iceberg", 1))
	if ev.Kind != fav1.OrderEvent_KIND_REJECTED || !strings.Contains(ev.Reason, "unknown exec algo") {
		t.Fatalf("%v", ev)
	}
}

func TestTwapSlicesEvenlyAndFinishes(t *testing.T) {
	p, c := clocked(free().Costs)
	p.ObserveTrade(tradeSub, 100, 1, t0)
	ack := p.Execute(algoIntent("tw", "twap:dur_s=40,slices=4", 8))
	if ack.Kind != fav1.OrderEvent_KIND_ACK || ack.ArrivalPrice != 100 || p.Working() != 1 {
		t.Fatalf("ack %v", ack)
	}
	var evs []Published
	evs = append(evs, p.Tick()...) // slice 1 at acceptance
	if len(evs) != 1 || evs[0].Event.Kind != fav1.OrderEvent_KIND_PARTIAL_FILL || evs[0].Event.FillQuantity != 2 ||
		evs[0].Event.VenueOrderId != "paper-tw-1" {
		t.Fatalf("slice 1 %v", evs)
	}
	c.t = t0.Add(5 * time.Second)
	if out := p.Tick(); len(out) != 0 {
		t.Fatalf("nothing due mid-slice: %v", out)
	}
	c.t = t0.Add(10 * time.Second)
	evs = append(evs, p.ObserveTrade(tradeSub, 102, 1, c.t)...) // a trade drives the schedule too
	if len(evs) != 2 {
		t.Fatalf("slice 2 on the trade: %v", evs)
	}
	evs = append(evs, p.Tick()...)
	c.t = t0.Add(20 * time.Second)
	evs = append(evs, p.Tick()...)
	c.t = t0.Add(30 * time.Second)
	last := p.Tick()
	evs = append(evs, last...)
	if len(evs) != 4 || last[0].Event.Kind != fav1.OrderEvent_KIND_FILL || last[0].Event.VenueOrderId != "paper-tw-4" {
		t.Fatalf("want 4 children ending in FILL: %v", evs)
	}
	if !near(sumQty(evs), 8, 1e-9) || p.Working() != 0 {
		t.Fatalf("qty %v working %d", sumQty(evs), p.Working())
	}
	// implementation shortfall vs the parent arrival: later children at 102 → +200 bps
	if e := last[0].Event; e.ArrivalPrice != 100 || !near(e.SlippageBps, 200, 1e-9) || e.FillPrice != 102 {
		t.Fatalf("child vs parent arrival: %v", e)
	}
	snap := p.Counters.Snapshot()
	if snap["algo.twap.done"] != 1 || snap["algo.twap.working"] != 0 {
		t.Fatalf("algo counters %v", snap)
	}
}

func TestVwapFollowsLearnedProfile(t *testing.T) {
	p, c := clocked(free().Costs)
	// learn a profile: the bucket of t0+5min carries 3× the volume of t0's bucket
	day := t0.Add(-24 * time.Hour)
	p.ObserveTrade(tradeSub, 100, 1, day)
	p.ObserveTrade(tradeSub, 100, 3, day.Add(5*time.Minute))
	p.ObserveTrade(tradeSub, 100, 1, t0)
	p.Execute(algoIntent("vw", "vwap:dur_s=600,slices=2", 8))
	first := p.Tick()
	// weights: slice 1 at t0+2.5min (bucket of t0: 1+1 = 2 units incl. today's trade), slice 2
	// at t0+7.5min (bucket t0+5min: 3) → 8·2/5 = 3.2 first
	if len(first) != 1 || !near(first[0].Event.FillQuantity, 3.2, 1e-9) {
		t.Fatalf("vwap first slice %v", first)
	}
	c.t = t0.Add(5 * time.Minute)
	rest := p.Tick()
	if len(rest) != 1 || !near(rest[0].Event.FillQuantity, 4.8, 1e-9) || rest[0].Event.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("vwap second slice %v", rest)
	}
	// unknown profile → uniform
	cum := vwapWeights(nil, t0, AlgoSpec{Slices: 4, DurS: 60})
	if !near(cum[0], 0.25, 1e-12) || cum[3] != 1 {
		t.Fatal(cum)
	}
}

func TestPovParticipatesInObservedVolumeThenCrosses(t *testing.T) {
	p, c := clocked(free().Costs)
	p.ObserveTrade(tradeSub, 100, 1, t0)
	p.Execute(algoIntent("pv", "pov:rate=0.5,max_dur_s=60", 10))
	if out := p.Tick(); len(out) != 0 {
		t.Fatalf("no volume yet: %v", out)
	}
	c.t = t0.Add(time.Second)
	out := p.ObserveTrade(tradeSub, 100, 4, c.t) // 0.5 × 4 = 2
	if len(out) != 1 || !near(out[0].Event.FillQuantity, 2, 1e-9) || out[0].Event.Kind != fav1.OrderEvent_KIND_PARTIAL_FILL {
		t.Fatalf("pov child %v", out)
	}
	c.t = t0.Add(61 * time.Second)
	end := p.Tick()
	if len(end) != 1 || !near(end[0].Event.FillQuantity, 8, 1e-9) || end[0].Event.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("remainder crossed at max_dur: %v", end)
	}
}

func TestSmartLimitPassiveThenTimeoutCross(t *testing.T) {
	m := DefaultCostModel()
	p, c := clocked(m)
	p.ObserveTrade(tradeSub, 100, 1, t0)
	p.ObserveQuote(quoteSub, 99.8, 100.2)
	if ack := p.Execute(algoIntent("sl", "smart_limit:timeout_s=10", 1)); ack.Kind != fav1.OrderEvent_KIND_ACK {
		t.Fatal(ack)
	}
	// trade at the bid → passive fill at 99.8, commission only (min 1 EUR)
	out := p.ObserveTrade(tradeSub, 99.8, 1, t0.Add(time.Second))
	if len(out) != 1 || out[0].Event.FillPrice != 99.8 || out[0].Event.Kind != fav1.OrderEvent_KIND_FILL ||
		out[0].Event.Fee != 1 || out[0].Event.VenueOrderId != "paper-sl" {
		t.Fatalf("passive %v", out)
	}
	// second order never crossed → taker at the timeout
	p.Execute(algoIntent("sl2", "smart_limit:timeout_s=10,offset_bps=50", 1))
	c.t = t0.Add(5 * time.Second)
	if out := p.Tick(); len(out) != 0 {
		t.Fatal(out)
	}
	c.t = t0.Add(11 * time.Second)
	out = p.Tick()
	if len(out) != 1 || out[0].Event.FillPrice <= 99.8 || out[0].Event.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("timeout cross %v", out)
	}
}

func TestCancelWorkingAlgoCancelsRemainder(t *testing.T) {
	p, c := clocked(free().Costs)
	p.ObserveTrade(tradeSub, 100, 1, t0)
	p.Execute(algoIntent("tc", "twap:dur_s=40,slices=4", 8))
	p.Tick()
	c.t = t0.Add(time.Second)
	cn := p.CancelResting("tc")
	if cn == nil || cn.Event.Kind != fav1.OrderEvent_KIND_CANCELED || cn.Event.FillQuantity != 0 ||
		!strings.Contains(cn.Event.Reason, "filled 2 of 8") {
		t.Fatalf("cancel %v", cn)
	}
	c.t = t0.Add(time.Minute)
	if out := p.Tick(); len(out) != 0 || p.Working() != 0 {
		t.Fatal("cancelled algo must not trade")
	}
	if p.Counters.Snapshot()["algo.twap.cancelled"] != 1 {
		t.Fatal(p.Counters.Snapshot())
	}
}

func TestAutoChoosesByParticipationAndSpread(t *testing.T) {
	cases := []struct {
		notional, adv, spread float64
		want                  string
	}{
		{1_000, 10_000_000, 5, "market"},
		{1_000, 0, 5, "market"},
		{1_000, 10_000_000, 30, "smart_limit"},
		{50_000, 10_000_000, 5, "twap"},
		{500_000, 10_000_000, 5, "pov"},
	}
	for _, c := range cases {
		if got := chooseAlgo(c.notional, c.adv, c.spread).Name; got != c.want {
			t.Errorf("%+v: got %s", c, got)
		}
	}
	p, _ := clocked(free().Costs)
	p.ObserveTrade(tradeSub, 100, 1, t0)
	if ev := p.Execute(algoIntent("au", "auto", 1)); ev.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("small auto order fills at market: %v", ev)
	}
}

func TestDecodeCancelAcceptsBothPayloads(t *testing.T) {
	b, _ := proto.Marshal(&fav1.OrderCancel{IntentId: "c1", AccountId: "paper", Reason: "ui", At: timestamppb.New(t0), Actor: "ui"})
	c, err := decodeCancel(b)
	if err != nil || c.IntentId != "c1" || c.Actor != "ui" {
		t.Fatal(c, err)
	}
	px := 5.0
	legacy, _ := proto.Marshal(&fav1.OrderIntent{IntentId: "c2", AccountId: "paper", InstrumentId: inst, Side: fav1.Side_SIDE_SELL,
		Quantity: 3, ExecAlgo: "limit", LimitPrice: &px, StrategyIds: []string{"s"}, CreatedAt: timestamppb.New(t0)})
	c, err = decodeCancel(legacy)
	if err != nil || c.IntentId != "c2" || c.AccountId != "paper" {
		t.Fatal(c, err)
	}
	if _, err := decodeCancel(nil); err == nil {
		t.Fatal("empty cancel must fail")
	}
}

func TestCalibrationFitShrinkAndClamp(t *testing.T) {
	c := NewCalibrator()
	// y = 0.3 · x exactly, 50 samples, prior 0.1, n0 50 → w 0.5 → k 0.2
	var s []slipSample
	for i := 1; i <= 50; i++ {
		s = append(s, slipSample{x: float64(i), y: 0.3 * float64(i)})
	}
	kFit, w, k, clamped := c.FitK(s, 0.1)
	if !near(kFit, 0.3, 1e-12) || !near(w, 0.5, 1e-12) || !near(k, 0.2, 1e-12) || clamped {
		t.Fatal(kFit, w, k, clamped)
	}
	// strong evidence of k = 1 → clamped at 2 × prior
	s = s[:0]
	for i := 1; i <= 500; i++ {
		s = append(s, slipSample{x: 1, y: 1})
	}
	if _, _, k, clamped := c.FitK(s, 0.1); !near(k, 0.2, 1e-12) || !clamped {
		t.Fatal(k)
	}
	// negative drift → clamped at 0.5 × prior
	for i := range s {
		s[i].y = -1
	}
	if _, _, k, clamped := c.FitK(s, 0.1); !near(k, 0.05, 1e-12) || !clamped {
		t.Fatal(k)
	}
}

func TestCalibrateFromAlgoFillsExcludesModelEcho(t *testing.T) {
	m := DefaultCostModel()
	m.Default.Default.SpreadBps = 0
	p, c := clocked(m)
	p.Calib = NewCalibrator()
	p.Calib.MinSamples = 5
	// market fills are the model echoing itself: excluded
	p.ObserveTrade("md.trade.yahoo.01900000000070008000000000000001", 100, 1, t0)
	for i := 0; i < 10; i++ {
		p.Execute(algoIntent("m"+string(rune('a'+i)), "market", 10))
	}
	rep, out := p.Calib.Calibrate(m, t0)
	if out != nil || len(rep.Entries) != 1 || rep.Entries[0].Echo != 10 || rep.Entries[0].Status != "insufficient" {
		t.Fatalf("%+v", rep)
	}
	// TWAP children after an adverse drift carry information → k rises (clamped ≤ 2×)
	for i := 0; i < 10; i++ {
		id := "t" + string(rune('a'+i))
		c.t = t0.Add(time.Duration(i) * time.Hour)
		p.ObserveTrade("md.trade.yahoo.01900000000070008000000000000001", 100, 1, c.t)
		p.Execute(algoIntent(id, "twap:dur_s=2,slices=2", 1000))
		p.Tick()
		c.t = c.t.Add(time.Second)
		p.ObserveTrade("md.trade.yahoo.01900000000070008000000000000001", 100.5, 1, c.t)
	}
	rep, out = p.Calib.Calibrate(m, t0)
	e := rep.Entries[0]
	if out == nil || e.Status != "calibrated" || e.Informative != 10 || e.Echo != 20 || e.KNew <= 0.1 || e.KNew > 0.2+1e-12 {
		t.Fatalf("%+v", e)
	}
	s := out.Schedule("yahoo", "equity")
	if s.ImpactK == nil || *s.ImpactK != e.KNew || out.Schedule("sim", "equity").ImpactK != nil {
		t.Fatalf("calibrated model: %+v", s)
	}
	if err := out.Validate(); err != nil {
		t.Fatal(err)
	}
	path := t.TempDir() + "/costs.calibrated.yaml"
	if err := WriteCalibrated(path, out, rep); err != nil {
		t.Fatal(err)
	}
	back, err := LoadCostModel(path)
	if err != nil || *back.Schedule("yahoo", "equity").ImpactK != e.KNew {
		t.Fatal(back, err)
	}
	snap := p.Counters.Snapshot()
	if snap["slippage.yahoo.equity.n"] != 30 || math.IsNaN(snap["slippage.yahoo.equity.bps_mean"]) ||
		snap["costs.k.yahoo.equity"] != 0.1 || snap["exec_slippage_bps.twap"] <= 0 {
		t.Fatalf("metrics %v", snap)
	}
}
