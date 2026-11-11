package main

import (
	"testing"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// carried: costless fills, funding 1 bp/day on any position, borrow 2 bp/day on shorts.
func carried(now *time.Time) *PaperEngine {
	p := free()
	p.Costs.Default.Default = &ClassCosts{FundingBpsDay: 1, BorrowBpsDay: 2}
	p.now = func() time.Time { return *now }
	return p
}

func byKind(cs []CarryCharge, end time.Time) map[string]CarryCharge {
	out := map[string]CarryCharge{}
	for _, c := range cs {
		if c.PeriodEnd.Equal(end) {
			out[c.Kind] = c
		}
	}
	return out
}

func TestCarryChargesFundingOnAnyAndBorrowOnShortsPerPeriod(t *testing.T) {
	now := t0 // 12:00 UTC
	p := carried(&now)
	p.ObserveTrade(tradeSub, 100, 1, now)
	if cs := p.TickCarry(); cs != nil {
		t.Fatalf("first tick only starts the clock: %v", cs)
	}
	// long 2 @ 100 for 30 min, then short 2 for 30 min
	if ev := p.Execute(intent("long")); ev.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("fill: %v", ev)
	}
	now = t0.Add(30 * time.Minute)
	if cs := p.TickCarry(); cs != nil {
		t.Fatalf("no period closed yet: %v", cs)
	}
	sell := intent("short")
	sell.Side, sell.Quantity = fav1.Side_SIDE_SELL, 4
	if ev := p.Execute(sell); ev.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("fill: %v", ev)
	}
	now = t0.Add(time.Hour)
	cs := p.TickCarry()
	end := t0.Add(time.Hour)
	got := byKind(cs, end)
	halfDay := 200.0 * (30.0 / 1440.0) // 200 EUR notional for half an hour, in notional-days
	if f := got["funding"]; !near(f.Amount, -2*halfDay*1e-4, 1e-12) || f.Reference == "" || f.Account != "paper" {
		t.Fatalf("funding: %+v (want %g)", f, -2*halfDay*1e-4)
	}
	if b := got["borrow"]; !near(b.Amount, -halfDay*2e-4, 1e-12) || !near(b.NotionalDays, halfDay, 1e-9) {
		t.Fatalf("borrow on the short half hour only: %+v", b)
	}
	if got["funding"].Reference == got["borrow"].Reference {
		t.Fatal("funding and borrow need distinct booking references")
	}
	// next period: the closed one is re-announced with the same references
	now = t0.Add(2 * time.Hour)
	again := p.TickCarry()
	if r := byKind(again, end); r["funding"].Reference != got["funding"].Reference || r["funding"].Amount != got["funding"].Amount {
		t.Fatalf("previous period not re-announced identically: %+v", r)
	}
	if b := byKind(again, t0.Add(2*time.Hour))["borrow"]; !near(b.Amount, -200.0/24*2e-4, 1e-12) {
		t.Fatalf("full hour short: %+v", b)
	}
}

func TestCarrySurvivesARestartWithoutRepeatingAPeriod(t *testing.T) {
	now := t0
	p := carried(&now)
	p.ObserveTrade(tradeSub, 100, 1, now)
	p.TickCarry()
	sell := intent("s")
	sell.Side = fav1.Side_SIDE_SELL
	p.Execute(sell) // short 2 @ 100
	now = t0.Add(20 * time.Minute)
	p.TickCarry()
	st, err := p.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	// restart: a new engine restored from the snapshot, down for 10 minutes
	q := carried(&now)
	if err := q.Restore(st); err != nil {
		t.Fatal(err)
	}
	q.ObserveTrade(tradeSub, 100, 1, now)
	now = t0.Add(time.Hour)
	got := byKind(q.TickCarry(), t0.Add(time.Hour))
	if b := got["borrow"]; !near(b.Amount, -200.0/24*2e-4, 1e-12) {
		t.Fatalf("the whole hour (incl. the downtime) is charged once: %+v", b)
	}
}

func TestCarryDisabledAndFlatChargeNothing(t *testing.T) {
	now := t0
	p := carried(&now)
	p.ObserveTrade(tradeSub, 100, 1, now)
	p.TickCarry()
	now = t0.Add(2 * time.Hour)
	if cs := p.TickCarry(); len(cs) != 0 {
		t.Fatalf("flat book: %v", cs)
	}
	p.CarryPeriod = 0
	p.Execute(intent("x"))
	now = t0.Add(5 * time.Hour)
	if cs := p.TickCarry(); cs != nil {
		t.Fatalf("disabled: %v", cs)
	}
}
