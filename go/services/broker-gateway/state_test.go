package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// A TWAP and a resting limit survive a gateway restart: the new process loads orders.json,
// finishes the TWAP schedule from where it stopped and still fills the limit when crossed; a
// redelivered intent stays a duplicate.
func TestWorkingOrdersSurviveARestart(t *testing.T) {
	dir := t.TempDir()
	store := &StateStore{Path: filepath.Join(dir, "orders.json")}
	p, c := clocked(DefaultCostModel())
	p.ObserveTrade(tradeSub, 100, 1, t0)
	if ev := p.Execute(algoIntent("tw", "twap:dur_s=40,slices=4", 8)); ev.Kind != fav1.OrderEvent_KIND_ACK {
		t.Fatal(ev)
	}
	first := p.Tick()
	if len(first) != 1 || first[0].Event.FillQuantity != 2 {
		t.Fatalf("first slice at acceptance: %v", first)
	}
	if ev := p.Execute(limit("lim", fav1.Side_SIDE_BUY, 95)); ev.Kind != fav1.OrderEvent_KIND_ACK {
		t.Fatal(ev)
	}
	if ok, err := store.SaveIfChanged(p, false); !ok || err != nil {
		t.Fatal(ok, err)
	}
	if ok, _ := store.SaveIfChanged(p, false); ok {
		t.Fatal("unchanged state must not be rewritten")
	}

	// "restart": a fresh engine from the file, 25 s later
	st, err := store.Load()
	if err != nil || st == nil || len(st.Working) != 1 || len(st.Resting) != 1 {
		t.Fatalf("load: %v %+v", err, st)
	}
	q, c2 := clocked(DefaultCostModel())
	if err := q.Restore(st); err != nil {
		t.Fatal(err)
	}
	if q.Working() != 1 || q.Resting() != 1 {
		t.Fatal("orders not restored")
	}
	if q.Execute(algoIntent("tw", "twap:dur_s=40,slices=4", 8)) != nil {
		t.Fatal("redelivered intent must stay a duplicate across the restart")
	}
	c2.t = t0.Add(25 * time.Second)
	if out := q.Tick(); len(out) != 0 {
		t.Fatal("no market after the restart yet: nothing may fill blind")
	}
	out := q.ObserveTrade(tradeSub, 101, 1, c2.t)
	if sumQty(out) != 4 { // slices 2 and 3 fell due (t = 10 s, 20 s)
		t.Fatalf("catch-up children: %v", out)
	}
	c2.t = t0.Add(31 * time.Second)
	out = q.Tick()
	if len(out) != 1 || out[0].Event.Kind != fav1.OrderEvent_KIND_FILL || out[0].Event.FillQuantity != 2 {
		t.Fatalf("last slice: %v", out)
	}
	if out[0].Event.VenueOrderId != "paper-tw-3" || out[0].Event.ArrivalPrice != 100 {
		t.Fatalf("child numbering / parent arrival kept: %v", out[0].Event)
	}
	out = q.ObserveTrade(tradeSub, 94, 1, c2.t)
	if len(out) != 1 || out[0].Event.IntentId != "lim" {
		t.Fatalf("restored limit must fill when crossed: %v", out)
	}
	_ = c
	// positions carried over: 8 (TWAP) + 2 (limit) bought on account paper
	snap, _ := q.Snapshot()
	if got := snap.Positions[posKey("paper", inst)]; !near(got, 10, 1e-9) {
		t.Fatalf("position %v", got)
	}
}

func TestCorruptStateRefusesToLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "orders.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := (&StateStore{Path: path}).Load(); err == nil {
		t.Fatal("corrupt state must be an error, not an empty book")
	}
	if st, err := (&StateStore{Path: filepath.Join(t.TempDir(), "absent.json")}).Load(); st != nil || err != nil {
		t.Fatal("absent state is a clean start")
	}
}

// Without a fresh risk-engine heartbeat only risk-reducing intents execute (spec 40 §8).
func TestFailClosedWithoutRiskHeartbeat(t *testing.T) {
	p, c := clocked(DefaultCostModel())
	p.Guard = &RiskGuard{Timeout: 15 * time.Second}
	p.ObserveTrade(tradeSub, 100, 1, t0)
	ev := p.Execute(intent("a"))
	if ev.Kind != fav1.OrderEvent_KIND_REJECTED || !strings.Contains(ev.Reason, "fail-closed") {
		t.Fatalf("no heartbeat ever seen: %v", ev)
	}
	p.Guard.Beat(c.t)
	if ev := p.Execute(intent("b")); ev.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("fresh heartbeat: %v", ev)
	}
	// the risk-engine dies: 20 s without a heartbeat
	c.t = t0.Add(20 * time.Second)
	if ev := p.Execute(intent("c")); ev.Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatalf("adding to the long must be rejected: %v", ev)
	}
	sell := intent("d")
	sell.Side, sell.Quantity = fav1.Side_SIDE_SELL, 1.5
	if ev := p.Execute(sell); ev.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("a reduction executes: %v", ev)
	}
	flip := intent("e")
	flip.Side, flip.Quantity = fav1.Side_SIDE_SELL, 1
	if ev := p.Execute(flip); ev.Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatalf("selling through zero opens a short: %v", ev)
	}
	other := intent("f")
	other.AccountId, other.Side, other.Quantity = "paper-test", fav1.Side_SIDE_SELL, 0.5
	if ev := p.Execute(other); ev.Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatalf("positions are per account: %v", ev)
	}
	close := intent("g")
	close.Side, close.Quantity = fav1.Side_SIDE_SELL, 0.5
	if ev := p.Execute(close); ev.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("closing to zero executes: %v", ev)
	}
	if got := p.Counters.Snapshot()["failclosed.rejected"]; got != 4 {
		t.Fatalf("failclosed.rejected = %v", got)
	}
	// the heartbeat returns
	p.Guard.Beat(c.t)
	if ev := p.Execute(intent("h")); ev.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("after recovery: %v", ev)
	}
	var off *RiskGuard
	if !off.Alive(c.t) || !(&RiskGuard{}).Alive(c.t) {
		t.Fatal("nil / zero-timeout guard is disabled")
	}
	_ = bus.SubjectToken
}

func TestMergeCalibratedKeepsTheCurrentSchedules(t *testing.T) {
	base, err := LoadCostModel("costs.yaml")
	if err != nil {
		t.Fatal(err)
	}
	old := DefaultCostModel() // a calibrated file written from an older base
	k := 0.15
	old.Venues = map[string]VenueCosts{"yahoo": {Classes: map[string]ClassCosts{"equity": {CommissionBps: 99, ImpactK: &k}}}}
	m := MergeCalibrated(base, old)
	s := m.Schedule("yahoo", "equity")
	if s.ImpactK == nil || *s.ImpactK != 0.15 || s.CommissionBps != base.Schedule("yahoo", "equity").CommissionBps {
		t.Fatalf("merged schedule %+v", s)
	}
	if m.ClassOf("850b1098-d844-59e3-a7eb-1fb84638c1bf") != "crypto" || m.Schedule("kraken", "crypto").CommissionBps != 40 {
		t.Fatal("instrument classes and kraken schedule come from the current costs.yaml")
	}
	if base.Schedule("yahoo", "equity").ImpactK != nil {
		t.Fatal("base must not be mutated")
	}
}

// Kraken BTC/ETH and Yahoo BTC are crypto (no 1 EUR equity minimum); Kraken charges its taker fee.
func TestCryptoInstrumentsArePricedAsCrypto(t *testing.T) {
	m, err := LoadCostModel("costs.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"850b1098-d844-59e3-a7eb-1fb84638c1bf", "88649f5f-667f-5a21-aee9-7bc843b60fc1", "9daa8cea-67a6-5f6b-8347-f9e5a30f7463"} {
		if m.ClassOf(id) != "crypto" {
			t.Fatalf("%s class %s", id, m.ClassOf(id))
		}
	}
	r := m.Price(CostInputs{Venue: "kraken", Class: "crypto", Side: "buy", Price: 90_000, Qty: 0.001})
	if !near(r.Fee, r.Notional*40/10_000, 1e-9) || r.Fee >= 1 {
		t.Fatalf("90 EUR of BTC on kraken: fee %v (40 bp taker, no minimum)", r.Fee)
	}
	if s := m.Schedule("kraken", "equity"); s.CommissionMinEUR != 0 || s.CommissionBps != 40 {
		t.Fatalf("unmapped kraken pairs use the kraken default: %+v", s)
	}
}
