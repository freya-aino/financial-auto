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

func TestPaperFeeModel(t *testing.T) {
	p := NewPaperEngine()
	p.ObserveTrade("md.trade.sim.01900000000070008000000000000001", 100)
	if ev := p.Execute(intent("zero")); ev.Fee != 0 || ev.FeeCurrency != "EUR" {
		t.Fatalf("default must be zero-fee: %v", ev)
	}
	p.Fees = BpsFee{Bps: 10, Min: 1} // 10 bps of 200 notional = 0.2 → floored to 1
	if ev := p.Execute(intent("bps")); ev.Fee != 1 {
		t.Fatalf("want min fee 1, got %v", ev.Fee)
	}
	p.Fees = BpsFee{Bps: 100}
	if ev := p.Execute(intent("bps2")); ev.Fee != 2 {
		t.Fatalf("want 1%% of 200 = 2, got %v", ev.Fee)
	}
}
