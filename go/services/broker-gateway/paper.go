package main

import (
	"context"
	"math"
	"strings"
	"sync"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// VenueAdapter executes approved intents at a real venue (spec 40). Live
// adapters (lemon, oanda, …) are not implemented yet; the paper engine is the
// only implementation and shares this interface so live differs in `Submit` only.
type VenueAdapter interface {
	Venue() string
	Submit(ctx context.Context, intent *fav1.OrderIntent) ([]*fav1.OrderEvent, error)
	Cancel(ctx context.Context, intentID string) error
}

// Published is an order event bound for `order.event.<account>`.
type Published struct {
	Account string
	Event   *fav1.OrderEvent
}

// market is what the engine knows about one instrument token.
type market struct {
	venue    string
	last     float64
	lastAt   time.Time
	bid, ask float64
	// EWMA variance of log returns per second → daily σ = sqrt(var · 86400).
	varPerSec float64
	nReturns  int
	// Rolling traded notional for the ADV estimate.
	vol      []volPoint
	firstVol time.Time
}

type volPoint struct {
	at       time.Time
	notional float64
}

type resting struct {
	intent  *fav1.OrderIntent
	arrival float64
	token   string
}

// PaperEngine fills approved intents against the live tape with the cost model
// (spec 40 §6.7): market orders fill at last ± half spread ± square-root impact
// with the schedule's commission; limit orders fill when marketable (at the limit,
// commission only) or rest as ACKED until crossed or cancelled.
type PaperEngine struct {
	mu      sync.Mutex
	markets map[string]*market // subject token -> market
	seen    map[string]bool    // intent ids, for idempotency (spec 42)
	resting map[string]*resting
	now     func() time.Time
	Costs   *CostModel
	// Rolling volume window; the sum is extrapolated to 24 h for the ADV.
	VolumeWindow time.Duration
	// EWMA decay of the return variance.
	SigmaDecay float64
}

func NewPaperEngine(costs *CostModel) *PaperEngine {
	if costs == nil {
		costs = DefaultCostModel()
	}
	return &PaperEngine{
		markets: map[string]*market{}, seen: map[string]bool{}, resting: map[string]*resting{},
		now: time.Now, Costs: costs, VolumeWindow: time.Hour, SigmaDecay: 0.97,
	}
}

func (p *PaperEngine) Venue() string { return "paper" }

// Submit implements VenueAdapter for the paper venue.
func (p *PaperEngine) Submit(_ context.Context, in *fav1.OrderIntent) ([]*fav1.OrderEvent, error) {
	if ev := p.Execute(in); ev != nil {
		return []*fav1.OrderEvent{ev}, nil
	}
	return nil, nil
}

// Cancel implements VenueAdapter: cancels a resting limit order.
func (p *PaperEngine) Cancel(_ context.Context, intentID string) error {
	if p.CancelResting(intentID) == nil {
		return errNotResting
	}
	return nil
}

type gatewayError string

func (e gatewayError) Error() string { return string(e) }

const errNotResting = gatewayError("not_found: no resting order with that intent id")

func (p *PaperEngine) market(token string) *market {
	m, ok := p.markets[token]
	if !ok {
		m = &market{}
		p.markets[token] = m
	}
	return m
}

func subjectParts(subject string) (venue, token string, ok bool) {
	parts := strings.Split(subject, ".")
	if len(parts) != 4 {
		return "", "", false
	}
	return parts[2], parts[3], true
}

// ObserveTrade records a trade from `md.trade.<venue>.<token>` (price, size, event time): the
// mark, the volatility estimate and the volume window. Resting limit orders that became
// marketable are filled and returned for publishing.
func (p *PaperEngine) ObserveTrade(subject string, price, size float64, at time.Time) []Published {
	venue, token, ok := subjectParts(subject)
	if !ok || price <= 0 || math.IsNaN(price) || math.IsInf(price, 0) {
		return nil
	}
	if at.IsZero() {
		at = p.now()
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	m := p.market(token)
	if m.last > 0 && at.After(m.lastAt) {
		dt := at.Sub(m.lastAt).Seconds()
		if dt > 0 {
			r := math.Log(price / m.last)
			x := r * r / dt
			if m.nReturns == 0 {
				m.varPerSec = x
			} else {
				m.varPerSec = p.SigmaDecay*m.varPerSec + (1-p.SigmaDecay)*x
			}
			m.nReturns++
		}
	}
	m.venue, m.last, m.lastAt = venue, price, at
	if size > 0 {
		if m.firstVol.IsZero() {
			m.firstVol = at
		}
		m.vol = append(m.vol, volPoint{at: at, notional: price * size})
		cut := at.Add(-p.VolumeWindow)
		i := 0
		for i < len(m.vol) && m.vol[i].at.Before(cut) {
			i++
		}
		m.vol = m.vol[i:]
	}
	var out []Published
	for id, r := range p.resting {
		if r.token != token || !marketable(r.intent, price) {
			continue
		}
		delete(p.resting, id)
		out = append(out, Published{Account: r.intent.AccountId, Event: p.fill(r.intent, m, r.arrival, true)})
	}
	return out
}

// ObserveQuote records the best bid/ask from `md.quote.<venue>.<token>`.
func (p *PaperEngine) ObserveQuote(subject string, bid, ask float64) {
	_, token, ok := subjectParts(subject)
	if !ok || bid <= 0 || ask <= 0 || ask < bid {
		return
	}
	p.mu.Lock()
	m := p.market(token)
	m.bid, m.ask = bid, ask
	p.mu.Unlock()
}

// Sigma returns the daily volatility estimate for a token (0 when unknown).
func (m *market) sigma() float64 {
	if m.nReturns < 5 || m.varPerSec <= 0 {
		return 0
	}
	return math.Sqrt(m.varPerSec * 86_400)
}

// adv extrapolates the window's notional to 24 h (0 when nothing observed).
func (m *market) adv(now time.Time, window time.Duration) float64 {
	if len(m.vol) == 0 {
		return 0
	}
	var sum float64
	for _, v := range m.vol {
		sum += v.notional
	}
	elapsed := now.Sub(m.firstVol)
	if elapsed < time.Second {
		elapsed = time.Second
	}
	if elapsed > window {
		elapsed = window
	}
	return sum * (24 * time.Hour).Seconds() / elapsed.Seconds()
}

func marketable(in *fav1.OrderIntent, last float64) bool {
	if in.LimitPrice == nil {
		return true
	}
	lp := *in.LimitPrice
	if in.Side == fav1.Side_SIDE_SELL {
		return last >= lp
	}
	return last <= lp
}

func side(in *fav1.OrderIntent) string {
	if in.Side == fav1.Side_SIDE_SELL {
		return "sell"
	}
	return "buy"
}

// fill prices a fill with the cost model. Passive fills (crossed resting limits) fill at the
// limit price with commission only; taker fills pay half spread + impact.
func (p *PaperEngine) fill(in *fav1.OrderIntent, m *market, arrival float64, passive bool) *fav1.OrderEvent {
	now := p.now()
	var spread float64
	if m.ask > m.bid && m.bid > 0 {
		spread = m.ask - m.bid
	}
	price := arrival
	if passive && in.LimitPrice != nil {
		price = *in.LimitPrice
	}
	res := p.Costs.Price(CostInputs{
		Venue: m.venue, Class: p.Costs.ClassOf(in.InstrumentId), Side: side(in),
		Price: price, Qty: in.Quantity, Spread: spread, Sigma: m.sigma(), ADV: m.adv(now, p.VolumeWindow),
		Passive: passive,
	})
	var bps float64
	if arrival > 0 {
		dir := 1.0
		if in.Side == fav1.Side_SIDE_SELL {
			dir = -1
		}
		bps = dir * (res.FillPrice - arrival) / arrival * 10_000
	}
	return &fav1.OrderEvent{
		IntentId: in.IntentId, VenueOrderId: "paper-" + in.IntentId, Kind: fav1.OrderEvent_KIND_FILL,
		FillPrice: res.FillPrice, FillQuantity: in.Quantity, Fee: res.Fee, FeeCurrency: res.FeeCurrency,
		EventTime: timestamppb.New(now), ArrivalPrice: arrival, SlippageBps: bps,
	}
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
	case in.Quantity <= 0 || math.IsNaN(in.Quantity) || math.IsInf(in.Quantity, 0):
		ev.Kind, ev.Reason = fav1.OrderEvent_KIND_REJECTED, "non-positive quantity"
		return ev
	case in.LimitPrice != nil && *in.LimitPrice <= 0:
		ev.Kind, ev.Reason = fav1.OrderEvent_KIND_REJECTED, "non-positive limit price"
		return ev
	}
	token := bus.SubjectToken(in.InstrumentId)
	m, ok := p.markets[token]
	if !ok || m.last <= 0 {
		ev.Kind, ev.Reason = fav1.OrderEvent_KIND_REJECTED, "no market price observed"
		return ev
	}
	if !marketable(in, m.last) {
		p.resting[in.IntentId] = &resting{intent: in, arrival: m.last, token: token}
		ev.Kind, ev.Reason, ev.ArrivalPrice = fav1.OrderEvent_KIND_ACK, "limit not marketable; resting", m.last
		return ev
	}
	return p.fill(in, m, m.last, false)
}

// CancelResting cancels a resting limit order; nil when there is none for the id.
func (p *PaperEngine) CancelResting(intentID string) *Published {
	p.mu.Lock()
	defer p.mu.Unlock()
	r, ok := p.resting[intentID]
	if !ok {
		return nil
	}
	delete(p.resting, intentID)
	return &Published{Account: r.intent.AccountId, Event: &fav1.OrderEvent{
		IntentId: intentID, VenueOrderId: "paper-" + intentID, Kind: fav1.OrderEvent_KIND_CANCELED,
		EventTime: timestamppb.New(p.now()), Reason: "cancelled by request", ArrivalPrice: r.arrival,
	}}
}

// Resting returns the number of resting limit orders (heartbeat counter).
func (p *PaperEngine) Resting() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.resting)
}
