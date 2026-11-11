package main

import (
	"math"
	"testing"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

const (
	inst     = "01900000-0000-7000-8000-000000000001"
	tradeSub = "md.trade.sim.01900000000070008000000000000001"
	quoteSub = "md.quote.sim.01900000000070008000000000000001"
)

var t0 = time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)

func intent(id string) *fav1.OrderIntent {
	return &fav1.OrderIntent{IntentId: id, AccountId: "paper", InstrumentId: inst, Side: fav1.Side_SIDE_BUY,
		Quantity: 2, Mode: fav1.TradingMode_TRADING_MODE_PAPER}
}

func limit(id string, side fav1.Side, px float64) *fav1.OrderIntent {
	in := intent(id)
	in.Side, in.LimitPrice, in.ExecAlgo = side, &px, "limit"
	return in
}

// costless engine: fills at the last price exactly.
func free() *PaperEngine {
	m := DefaultCostModel()
	m.Default.Default = &ClassCosts{}
	m.DefaultK = 0
	p := NewPaperEngine(m)
	p.now = func() time.Time { return t0 }
	return p
}

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestPaperFillsAtLastPriceWhenCostless(t *testing.T) {
	p := free()
	if ev := p.Execute(intent("a")); ev.Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatalf("expected reject without price, got %v", ev.Kind)
	}
	p.ObserveTrade(tradeSub, 101.5, 1, t0)
	ev := p.Execute(intent("b"))
	if ev.Kind != fav1.OrderEvent_KIND_FILL || ev.FillPrice != 101.5 || ev.FillQuantity != 2 || ev.Fee != 0 {
		t.Fatalf("bad fill: %v", ev)
	}
	if ev.ArrivalPrice != 101.5 || ev.SlippageBps != 0 {
		t.Fatalf("arrival/slippage: %v", ev)
	}
	if p.Execute(intent("b")) != nil {
		t.Fatal("duplicate intent must be ignored")
	}
}

func TestPaperRejectsLiveAndBadInputs(t *testing.T) {
	p := free()
	p.ObserveTrade(tradeSub, 100, 1, t0)
	live := intent("l")
	live.Mode = fav1.TradingMode_TRADING_MODE_LIVE
	if p.Execute(live).Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatal("live must be rejected")
	}
	zero := intent("z")
	zero.Quantity = 0
	if p.Execute(zero).Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatal("zero quantity must be rejected")
	}
	if p.Execute(limit("neg", fav1.Side_SIDE_BUY, -1)).Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatal("negative limit must be rejected")
	}
}

func TestMarketOrderPaysSpreadImpactAndCommission(t *testing.T) {
	m := DefaultCostModel() // 5 bps commission, min 1 EUR, 5 bps spread, k 0.1, σ 2 %, ADV 10 M
	p := NewPaperEngine(m)
	p.now = func() time.Time { return t0 }
	p.ObserveTrade(tradeSub, 100, 0, t0) // size 0: no volume observed → default ADV
	in := intent("mkt")
	in.Quantity = 1000 // notional 100 000 → sqrt(0.01) = 0.1 → impact 100·0.1·0.02·0.1 = 0.02
	ev := p.Execute(in)
	if ev.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("want fill, got %v", ev)
	}
	// half spread 100·5/20000 = 0.025; fill = 100.045
	if !near(ev.FillPrice, 100.045, 1e-9) {
		t.Fatalf("fill price %v", ev.FillPrice)
	}
	if !near(ev.Fee, 100045.0*5/10_000, 1e-9) { // 50.0225
		t.Fatalf("fee %v", ev.Fee)
	}
	if !near(ev.SlippageBps, 4.5, 1e-9) || ev.ArrivalPrice != 100 {
		t.Fatalf("slippage %v arrival %v", ev.SlippageBps, ev.ArrivalPrice)
	}
	// sells are symmetric
	s := intent("sell")
	s.Side, s.Quantity = fav1.Side_SIDE_SELL, 1000
	es := p.Execute(s)
	if !near(es.FillPrice, 99.955, 1e-9) || !near(es.SlippageBps, 4.5, 1e-9) {
		t.Fatalf("sell fill %v slippage %v", es.FillPrice, es.SlippageBps)
	}
	// minimum commission floors small fills
	small := intent("small")
	small.Quantity = 0.01
	if e := p.Execute(small); e.Fee != 1 {
		t.Fatalf("want min fee 1, got %v", e.Fee)
	}
}

func TestQuoteSpreadReplacesModelledSpread(t *testing.T) {
	m := DefaultCostModel()
	m.DefaultK = 0
	p := NewPaperEngine(m)
	p.now = func() time.Time { return t0 }
	p.ObserveTrade(tradeSub, 100, 1, t0)
	p.ObserveQuote(quoteSub, 99.9, 100.3) // spread 0.4 → half 0.2
	ev := p.Execute(intent("q"))
	if !near(ev.FillPrice, 100.2, 1e-9) {
		t.Fatalf("fill %v", ev.FillPrice)
	}
	p.ObserveQuote(quoteSub, 100.3, 99.9) // crossed quote ignored
	if ev := p.Execute(intent("q2")); !near(ev.FillPrice, 100.2, 1e-9) {
		t.Fatalf("crossed quote must be ignored: %v", ev.FillPrice)
	}
}

func TestLimitOrdersRestUntilMarketableThenFillPassively(t *testing.T) {
	m := DefaultCostModel()
	p := NewPaperEngine(m)
	p.now = func() time.Time { return t0 }
	p.ObserveTrade(tradeSub, 100, 1, t0)
	// marketable buy limit (last ≤ limit) fills immediately as a taker at last ± costs
	if ev := p.Execute(limit("m", fav1.Side_SIDE_BUY, 101)); ev.Kind != fav1.OrderEvent_KIND_FILL || ev.FillPrice <= 100 {
		t.Fatalf("marketable limit must fill as taker: %v", ev)
	}
	// non-marketable buy rests
	ack := p.Execute(limit("r", fav1.Side_SIDE_BUY, 99))
	if ack.Kind != fav1.OrderEvent_KIND_ACK || ack.ArrivalPrice != 100 || p.Resting() != 1 {
		t.Fatalf("want ACK + resting, got %v (resting %d)", ack, p.Resting())
	}
	if out := p.ObserveTrade(tradeSub, 99.5, 1, t0.Add(time.Second)); len(out) != 0 {
		t.Fatalf("not yet marketable: %v", out)
	}
	out := p.ObserveTrade(tradeSub, 98.5, 1, t0.Add(2*time.Second))
	if len(out) != 1 || out[0].Account != "paper" {
		t.Fatalf("want one fill, got %v", out)
	}
	ev := out[0].Event
	// passive: at the limit, commission only, slippage vs arrival negative (price improvement)
	if ev.Kind != fav1.OrderEvent_KIND_FILL || ev.FillPrice != 99 || ev.Fee != 1 || ev.ArrivalPrice != 100 {
		t.Fatalf("passive fill: %v", ev)
	}
	if !near(ev.SlippageBps, -100, 1e-9) {
		t.Fatalf("slippage %v", ev.SlippageBps)
	}
	if p.Resting() != 0 {
		t.Fatal("resting order must be gone")
	}
	// sell limit: rests while last < limit, fills when last ≥ limit
	if ev := p.Execute(limit("s", fav1.Side_SIDE_SELL, 105)); ev.Kind != fav1.OrderEvent_KIND_ACK {
		t.Fatalf("sell above market must rest: %v", ev)
	}
	if out := p.ObserveTrade(tradeSub, 105, 1, t0.Add(3*time.Second)); len(out) != 1 || out[0].Event.FillPrice != 105 {
		t.Fatalf("sell limit fill: %v", out)
	}
}

func TestCancelRemovesRestingOrder(t *testing.T) {
	p := free()
	p.ObserveTrade(tradeSub, 100, 1, t0)
	p.Execute(limit("r", fav1.Side_SIDE_BUY, 90))
	if p.CancelResting("nope") != nil {
		t.Fatal("unknown id must not cancel anything")
	}
	c := p.CancelResting("r")
	if c == nil || c.Event.Kind != fav1.OrderEvent_KIND_CANCELED || c.Account != "paper" || p.Resting() != 0 {
		t.Fatalf("cancel: %v", c)
	}
	if out := p.ObserveTrade(tradeSub, 80, 1, t0.Add(time.Second)); len(out) != 0 {
		t.Fatal("cancelled order must not fill")
	}
	if p.Execute(limit("r", fav1.Side_SIDE_BUY, 90)) != nil {
		t.Fatal("intent id stays consumed after cancel (idempotency)")
	}
}

func TestVolatilityAndVolumeEstimates(t *testing.T) {
	p := free()
	for i := 0; i < 20; i++ {
		px := 100.0
		if i%2 == 1 {
			px = 101
		}
		p.ObserveTrade(tradeSub, px, 10, t0.Add(time.Duration(i)*time.Second))
	}
	m := p.markets["01900000000070008000000000000001"]
	if m.sigma() <= 0 {
		t.Fatal("sigma must be estimated after enough returns")
	}
	// ~1 % per second alternating → daily σ is large; just check scale monotonicity
	if m.sigma() < 1 {
		t.Fatalf("sigma %v unexpectedly small", m.sigma())
	}
	adv := m.adv(t0.Add(19*time.Second), time.Hour)
	// 20 trades × ~1 005 EUR over 19 s → ×(86400/19)
	want := (10*100.0*10 + 10*101.0*10) * 86_400 / 19
	if !near(adv, want, 1) {
		t.Fatalf("adv %v want %v", adv, want)
	}
	if (&market{}).adv(t0, time.Hour) != 0 || (&market{}).sigma() != 0 {
		t.Fatal("unknown market must report 0")
	}
}

func TestVenueAdapterInterface(t *testing.T) {
	var v VenueAdapter = free()
	if v.Venue() != "paper" {
		t.Fatal(v.Venue())
	}
	if err := v.Cancel(t.Context(), "none"); err == nil {
		t.Fatal("cancel of unknown must error")
	}
}

// spec 42 §5: fills priced in the same millisecond (here: a frozen clock) still get distinct
// venue fill ids, and every fill names its instrument and venue symbol (orphan identification).
func TestFillIdsAreUniqueWithinOneMillisecond(t *testing.T) {
	p := free()
	p.ObserveTrade(tradeSub, 100, 1, t0)
	a, b := p.Execute(intent("x1")), p.Execute(intent("x2"))
	if a.VenueFillId == "" || a.VenueFillId == b.VenueFillId {
		t.Fatalf("fill ids not unique: %q %q", a.VenueFillId, b.VenueFillId)
	}
	if a.InstrumentId != inst || a.VenueSymbol == "" {
		t.Fatalf("fill does not identify its instrument: %v", a)
	}
}
