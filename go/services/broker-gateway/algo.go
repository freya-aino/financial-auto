package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// Execution algorithms (spec 40 §6.4). `OrderIntent.exec_algo` is `name` or
// `name:k=v,k=v`; names: market | limit | smart_limit | twap | vwap | pov | auto.
//
//	smart_limit timeout_s=30 offset_bps=0   passive at the touch (bid for buys, ask for sells,
//	                                        last without a quote) ∓ offset; crossed as a taker
//	                                        at the timeout
//	twap        dur_s=300 slices=10         equal child slices every dur/slices
//	vwap        dur_s=300 slices=10         child slices weighted by the learned time-of-day
//	                                        volume profile (uniform when unknown)
//	pov         rate=0.1 max_dur_s=900      children = rate × traded size observed since the
//	                                        last child; remainder crossed at max_dur
//	auto                                    chosen by chooseAlgo from notional / ADV and spread
//
// Algo orders are ACKed first; every child is priced by the cost model at the then-last price
// with the child quantity and published as PARTIAL_FILL (`paper-<intent>-<n>`), the last child
// as FILL; arrival price and slippage refer to the parent arrival (implementation shortfall).

type AlgoSpec struct {
	Name      string
	TimeoutS  float64
	OffsetBps float64
	DurS      float64
	Slices    int
	Rate      float64
	MaxDurS   float64
}

var algoParams = map[string][]string{
	"market": {}, "limit": {}, "auto": {},
	"smart_limit": {"timeout_s", "offset_bps"},
	"twap":        {"dur_s", "slices"},
	"vwap":        {"dur_s", "slices"},
	"pov":         {"rate", "max_dur_s"},
}

// ParseAlgo parses the exec_algo grammar; empty means market.
func ParseAlgo(s string) (AlgoSpec, error) {
	s = strings.TrimSpace(s)
	name, params, _ := strings.Cut(s, ":")
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		name = "market"
	}
	a := AlgoSpec{Name: name, TimeoutS: 30, DurS: 300, Slices: 10, Rate: 0.1, MaxDurS: 900}
	allowed, ok := algoParams[name]
	if !ok {
		return a, fmt.Errorf("unknown exec algo %q", name)
	}
	if strings.TrimSpace(params) == "" {
		return a, nil
	}
	for _, kv := range strings.Split(params, ",") {
		k, v, ok := strings.Cut(strings.TrimSpace(kv), "=")
		k = strings.TrimSpace(k)
		if !ok || k == "" {
			return a, fmt.Errorf("exec algo %s: bad parameter %q", name, kv)
		}
		known := false
		for _, p := range allowed {
			known = known || p == k
		}
		if !known {
			return a, fmt.Errorf("exec algo %s: unknown parameter %q", name, k)
		}
		f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
			return a, fmt.Errorf("exec algo %s: %s=%q is not a number", name, k, v)
		}
		switch k {
		case "timeout_s":
			a.TimeoutS = f
		case "offset_bps":
			a.OffsetBps = f
		case "dur_s":
			a.DurS = f
		case "slices":
			if f != math.Trunc(f) {
				return a, fmt.Errorf("exec algo %s: slices must be an integer", name)
			}
			a.Slices = int(f)
		case "rate":
			a.Rate = f
		case "max_dur_s":
			a.MaxDurS = f
		}
	}
	switch {
	case a.TimeoutS < 0 || a.TimeoutS > 86_400:
		return a, fmt.Errorf("exec algo %s: timeout_s out of [0, 86400]", name)
	case a.OffsetBps < 0 || a.OffsetBps > 1_000:
		return a, fmt.Errorf("exec algo %s: offset_bps out of [0, 1000]", name)
	case a.DurS <= 0 || a.DurS > 86_400:
		return a, fmt.Errorf("exec algo %s: dur_s out of (0, 86400]", name)
	case a.Slices < 1 || a.Slices > 1_000:
		return a, fmt.Errorf("exec algo %s: slices out of [1, 1000]", name)
	case a.Rate <= 0 || a.Rate > 1:
		return a, fmt.Errorf("exec algo %s: rate out of (0, 1]", name)
	case a.MaxDurS <= 0 || a.MaxDurS > 86_400:
		return a, fmt.Errorf("exec algo %s: max_dur_s out of (0, 86400]", name)
	}
	return a, nil
}

// Thresholds of the `auto` choice (participation = order notional / learned ADV).
const (
	autoMarketMaxPart = 0.001 // ≤ 0.1 % of a day's volume and a tight spread → market
	autoSpreadBps     = 10    // spread above this → passive smart_limit while small
	autoTwapMaxPart   = 0.01  // ≤ 1 % → TWAP over 5 min; above → POV at 10 %
)

// chooseAlgo is the gateway's fallback choice for `auto` (spec 40 §6.4 selection inputs:
// order notional vs horizon volume, spread). Without an observed ADV the order is treated as
// small (the model default ADV applies for pricing anyway).
func chooseAlgo(notional, adv, spreadBps float64) AlgoSpec {
	a, _ := ParseAlgo("market")
	part := 0.0
	if adv > 0 {
		part = notional / adv
	}
	switch {
	case part <= autoMarketMaxPart && spreadBps <= autoSpreadBps:
		return a
	case part <= autoMarketMaxPart:
		a, _ = ParseAlgo("smart_limit")
	case part <= autoTwapMaxPart:
		a, _ = ParseAlgo("twap")
	default:
		a, _ = ParseAlgo("pov")
	}
	return a
}

// working is one algo order in progress.
type working struct {
	intent  *fav1.OrderIntent
	algo    AlgoSpec
	token   string
	arrival float64
	start   time.Time
	filled  float64
	seq     int
	// vwap: cumulative target fraction after each slice.
	cum []float64
	// pov: quantity allowed by observed volume, not yet executed.
	allowance float64
	// smart_limit: the passive price.
	limit float64
}

const qtyEps = 1e-12

func (w *working) remaining() float64 { return w.intent.Quantity - w.filled }

// profileBuckets is the time-of-day resolution of the learned volume profile.
const profileBuckets = 288 // 5 min

func bucketOf(t time.Time) int {
	t = t.UTC()
	s := t.Hour()*3600 + t.Minute()*60 + t.Second()
	return s * profileBuckets / 86_400
}

// vwapWeights returns cumulative target fractions per slice from the token's profile.
func vwapWeights(profile []float64, start time.Time, a AlgoSpec) []float64 {
	n := a.Slices
	w := make([]float64, n)
	var sum float64
	step := time.Duration(a.DurS / float64(n) * float64(time.Second))
	for i := range w {
		if profile != nil {
			w[i] = profile[bucketOf(start.Add(time.Duration(i)*step+step/2))]
		}
		sum += w[i]
	}
	if !(sum > 0) {
		for i := range w {
			w[i] = 1
		}
		sum = float64(n)
	}
	cum := make([]float64, n)
	var acc float64
	for i := range w {
		acc += w[i]
		cum[i] = acc / sum
	}
	cum[n-1] = 1
	return cum
}

// target returns the cumulative quantity the schedule wants executed by `now`, and whether the
// schedule is complete (everything due).
func (w *working) target(now time.Time) (float64, bool) {
	q := w.intent.Quantity
	el := now.Sub(w.start).Seconds()
	switch w.algo.Name {
	case "twap", "vwap":
		n := w.algo.Slices
		due := int(el/(w.algo.DurS/float64(n))) + 1 // first slice at acceptance
		if due >= n {
			return q, true
		}
		if w.algo.Name == "twap" {
			return q * float64(due) / float64(n), false
		}
		return q * w.cum[due-1], false
	case "pov":
		if el >= w.algo.MaxDurS {
			return q, true
		}
		return math.Min(q, w.filled+w.allowance), false
	}
	return q, true
}

// startAlgo registers a working algo order; caller holds p.mu. Returns the ACK.
func (p *PaperEngine) startAlgo(in *fav1.OrderIntent, a AlgoSpec, m *market, token string) *fav1.OrderEvent {
	now := p.now()
	w := &working{intent: in, algo: a, token: token, arrival: m.last, start: now}
	switch a.Name {
	case "vwap":
		w.cum = vwapWeights(p.profiles[token], now, a)
	case "smart_limit":
		dir := 1.0
		ref := m.last
		if in.Side == fav1.Side_SIDE_SELL {
			dir = -1
			if m.ask > 0 && m.ask >= m.bid {
				ref = m.ask
			}
		} else if m.bid > 0 && m.ask >= m.bid {
			ref = m.bid
		}
		w.limit = ref * (1 - dir*a.OffsetBps/10_000)
	}
	p.working[in.IntentId] = w
	p.touch()
	p.algoCount(a.Name, "working", 1)
	return &fav1.OrderEvent{
		IntentId: in.IntentId, VenueOrderId: "paper-" + in.IntentId, Kind: fav1.OrderEvent_KIND_ACK,
		EventTime: timestamppb.New(now), ArrivalPrice: m.last,
		Reason: fmt.Sprintf("working %s", a.Name),
	}
}

// child prices and returns one child fill of qty; caller holds p.mu.
func (p *PaperEngine) child(w *working, m *market, qty float64, passive bool, final bool) Published {
	c := proto.Clone(w.intent).(*fav1.OrderIntent)
	c.Quantity = qty
	if passive {
		lp := w.limit
		c.LimitPrice = &lp
	}
	ev := p.fill(c, m, m.last, w.arrival, passive, w.algo.Name)
	w.filled += qty
	w.seq++
	if w.algo.Name != "smart_limit" {
		ev.VenueOrderId = fmt.Sprintf("paper-%s-%d", w.intent.IntentId, w.seq)
	}
	if final {
		ev.Kind = fav1.OrderEvent_KIND_FILL
	} else {
		ev.Kind = fav1.OrderEvent_KIND_PARTIAL_FILL
	}
	return Published{Account: w.intent.AccountId, Event: ev}
}

// progress advances every working order (of `token`, or all when empty); caller holds p.mu.
func (p *PaperEngine) progress(token string, now time.Time) []Published {
	ids := make([]string, 0, len(p.working))
	for id, w := range p.working {
		if token == "" || w.token == token {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids) // deterministic publish order
	var out []Published
	for _, id := range ids {
		w := p.working[id]
		m := p.markets[w.token]
		if m == nil || m.last <= 0 {
			continue
		}
		done := false
		if w.algo.Name == "smart_limit" {
			switch {
			case marketableAt(w.intent.Side, w.limit, m.last):
				out = append(out, p.child(w, m, w.remaining(), true, true))
				done = true
			case now.Sub(w.start).Seconds() >= w.algo.TimeoutS:
				out = append(out, p.child(w, m, w.remaining(), false, true))
				done = true
			}
		} else {
			tgt, complete := w.target(now)
			qty := tgt - w.filled
			if complete {
				qty = w.remaining()
			}
			if qty > qtyEps*math.Max(1, w.intent.Quantity) {
				final := complete || w.remaining()-qty <= qtyEps*math.Max(1, w.intent.Quantity)
				if final {
					qty = w.remaining()
				}
				if w.algo.Name == "pov" {
					w.allowance = math.Max(0, w.allowance-qty)
				}
				out = append(out, p.child(w, m, qty, false, final))
				done = final
			}
		}
		if done {
			delete(p.working, id)
			p.touch()
			p.algoCount(w.algo.Name, "working", -1)
			p.algoCount(w.algo.Name, "done", 1)
		}
	}
	return out
}

func marketableAt(side fav1.Side, limit, last float64) bool {
	if side == fav1.Side_SIDE_SELL {
		return last >= limit
	}
	return last <= limit
}

// Tick drives the time-based schedules (TWAP/VWAP slices, smart-limit timeouts, POV end).
func (p *PaperEngine) Tick() []Published {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.progress("", p.now())
}

// Working returns the number of algo orders in progress.
func (p *PaperEngine) Working() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.working)
}

func (p *PaperEngine) algoCount(name, what string, d float64) {
	if p.Counters != nil {
		p.Counters.Add("algo."+name+"."+what, d)
	}
}
