package main

import (
	"testing"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

const inst = "01900000-0000-7000-8000-000000000001"

func intent(id string) *fav1.OrderIntent {
	return &fav1.OrderIntent{IntentId: id, AccountId: "paper", InstrumentId: inst, Side: fav1.Side_SIDE_BUY,
		Quantity: 2, Mode: fav1.TradingMode_TRADING_MODE_PAPER}
}

func TestPaperFillsAtLastPrice(t *testing.T) {
	p := NewPaperEngine()
	if ev := p.Execute(intent("a")); ev.Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatalf("expected reject without price, got %v", ev.Kind)
	}
	p.ObserveTrade("md.trade.sim.01900000000070008000000000000001", 101.5)
	ev := p.Execute(intent("b"))
	if ev.Kind != fav1.OrderEvent_KIND_FILL || ev.FillPrice != 101.5 || ev.FillQuantity != 2 {
		t.Fatalf("bad fill: %v", ev)
	}
	if p.Execute(intent("b")) != nil {
		t.Fatal("duplicate intent must be ignored")
	}
}

func TestPaperRejectsLiveAndUnmarketableLimit(t *testing.T) {
	p := NewPaperEngine()
	p.ObserveTrade("md.trade.sim.01900000000070008000000000000001", 100)
	live := intent("l")
	live.Mode = fav1.TradingMode_TRADING_MODE_LIVE
	if p.Execute(live).Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatal("live must be rejected")
	}
	lim := intent("m")
	lp := 99.0
	lim.LimitPrice = &lp
	if p.Execute(lim).Kind != fav1.OrderEvent_KIND_CANCELED {
		t.Fatal("unmarketable buy limit must cancel")
	}
}
