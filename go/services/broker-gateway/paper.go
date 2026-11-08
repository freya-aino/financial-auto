package main

import (
	"context"
	"strings"
	"sync"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// VenueAdapter executes approved intents at a real venue (spec 40). Live
// adapters (lemon, oanda, …) are not implemented yet.
type VenueAdapter interface {
	Venue() string
	Submit(ctx context.Context, intent *fav1.OrderIntent) ([]*fav1.OrderEvent, error)
	Cancel(ctx context.Context, intentID string) error
}

// FeeModel prices the cost of a fill (spec 41 cost model hook). The paper
// engine charges ZeroFee unless configured otherwise; slippage stays zero.
type FeeModel interface {
	// Fee returns the fee and its currency for a fill of qty at price.
	Fee(in *fav1.OrderIntent, price, qty float64) (fee float64, currency string)
}

// ZeroFee is the default: no costs.
type ZeroFee struct{}

func (ZeroFee) Fee(*fav1.OrderIntent, float64, float64) (float64, string) { return 0, "EUR" }

// BpsFee charges a flat basis-point fee on notional, floored at Min.
type BpsFee struct {
	Bps float64
	Min float64
}

func (f BpsFee) Fee(_ *fav1.OrderIntent, price, qty float64) (float64, string) {
	return max(f.Min, price*qty*f.Bps/10_000), "EUR"
}

// PaperEngine fills intents immediately at the last seen trade price.
// Slippage is zero; fees come from Fees (ZeroFee by default).
type PaperEngine struct {
	mu        sync.Mutex
	lastPrice map[string]float64 // subject token -> price
	seen      map[string]bool    // intent ids, for idempotency (spec 42)
	now       func() time.Time
	Fees      FeeModel
}

func NewPaperEngine() *PaperEngine {
	return &PaperEngine{lastPrice: map[string]float64{}, seen: map[string]bool{}, now: time.Now, Fees: ZeroFee{}}
}

// ObserveTrade records the latest price from an md.trade.<venue>.<token> subject.
func (p *PaperEngine) ObserveTrade(subject string, price float64) {
	parts := strings.Split(subject, ".")
	if len(parts) != 4 || price <= 0 {
		return
	}
	p.mu.Lock()
	p.lastPrice[parts[3]] = price
	p.mu.Unlock()
}

// Execute returns the order event for an intent, or nil if it was already handled.
func (p *PaperEngine) Execute(in *fav1.OrderIntent) *fav1.OrderEvent {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.seen[in.IntentId] {
		return nil
	}
	p.seen[in.IntentId] = true

	ev := &fav1.OrderEvent{IntentId: in.IntentId, VenueOrderId: "paper-" + in.IntentId, EventTime: timestamppb.New(p.now())}
	switch {
	case in.Mode == fav1.TradingMode_TRADING_MODE_LIVE:
		ev.Kind, ev.Reason = fav1.OrderEvent_KIND_REJECTED, "live venue adapters not implemented"
		return ev
	case in.Quantity <= 0:
		ev.Kind, ev.Reason = fav1.OrderEvent_KIND_REJECTED, "non-positive quantity"
		return ev
	}
	price, ok := p.lastPrice[bus.SubjectToken(in.InstrumentId)]
	if !ok {
		ev.Kind, ev.Reason = fav1.OrderEvent_KIND_REJECTED, "no market price observed"
		return ev
	}
	if in.LimitPrice != nil {
		lp := *in.LimitPrice
		if (in.Side == fav1.Side_SIDE_BUY && price > lp) || (in.Side == fav1.Side_SIDE_SELL && price < lp) {
			ev.Kind, ev.Reason = fav1.OrderEvent_KIND_CANCELED, "limit not marketable (paper engine has no resting orders)"
			return ev
		}
	}
	ev.Kind = fav1.OrderEvent_KIND_FILL
	ev.FillPrice, ev.FillQuantity = price, in.Quantity
	ev.Fee, ev.FeeCurrency = p.Fees.Fee(in, price, in.Quantity)
	return ev
}
