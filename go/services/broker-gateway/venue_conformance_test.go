// One conformance suite for every execution venue (spec 40 §10, D-121).
//
// The suite's premise is that paper and real money are the same execution path, so the contract the
// gateway, the ledger and the risk engine depend on must hold for the paper engine and for each
// live adapter identically. Those invariants are what `conformanceCases` checks:
//
//  1. a market order on a priced instrument fills exactly its quantity, once;
//  2. polling a venue that has not changed produces no further events (no double-counted fill);
//  3. cumulative fills never exceed the intent;
//  4. an order ends in exactly one terminal event, and nothing follows it;
//  5. a refusal is a rejection with a reason and no quantity;
//  6. the same intent submitted twice creates one order;
//  7. every fill carries what the ledger needs: instrument, a unique venue fill id, a fee with a
//     currency, and a slippage consistent with its side;
//  8. an adapter restarted mid-order re-attaches it and emits the increment, not the whole fill.
//
// What the suite deliberately does **not** assert is that the venues behave alike — they do not.
// The paper engine fills a market order from the tape in one event; Kraken acknowledges and fills
// asynchronously; OANDA fills a market order inside the submit response; lemon creates an order
// inactive and must activate it. Each venue is therefore driven through a small scripted surface
// (`scriptedVenue`) that says *what the venue did*, and the assertions are about the events that
// come out.
//
// The live adapters run against stub servers serving the fixtures in `testdata/venues/` — the
// shapes from each venue's published reference, never a capture from a live venue (see that
// directory's README). This proves the mapping, not the venue.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// --- the scripted surface ---------------------------------------------------------------------

// scriptedVenue is one venue under test plus the means to say what it did.
type scriptedVenue interface {
	Name() string
	Adapter() VenueAdapter
	// Mark gives an instrument a tradable price. The paper engine needs a tape; a live stub uses
	// it as the price its fills come back at.
	Mark(instrument, symbol string, price float64)
	// FillTo scripts a cumulative filled quantity at a price (venues report cumulatively).
	FillTo(intentID string, cum, price, fee float64)
	// CancelAtVenue scripts the venue reporting the order cancelled.
	CancelAtVenue(intentID string)
	// RejectNext makes the next submit be refused with this reason.
	RejectNext(reason string)
	// Poll collects whatever the venue produced since the last call.
	Poll(t *testing.T) []Published
	// Can reports whether a behaviour is scriptable for this venue.
	Can(feature string) bool
	// Restarted returns the same venue behind a freshly built adapter that has been handed the old
	// one's persisted state — a gateway restart.
	Restarted(t *testing.T) scriptedVenue
}

// Features a venue may or may not be able to script.
const (
	featPartial   = "partial"   // the venue can report a partial fill
	featAsync     = "async"     // fills arrive on a poll rather than inside the submit
	featRestart   = "restart"   // the adapter persists its open orders
	featDuplicate = "duplicate" // the venue refuses a duplicate client order id
)

const (
	testInstrument = "01900000-0000-7000-8000-000000000042"
	testAccount    = "live"
)

// confIntent is one intent in the mode that venue trades in.
//
// The mode is not a free choice: the paper engine refuses a `LIVE` intent on purpose
// (`TestPaperRejectsLiveAndBadInputs`) and a live adapter is only ever handed one, because the
// router is what decides which of the two an intent reaches. The invariants below are about the
// events that come back, which are the same either way — that is the whole claim of one execution
// path.
func confIntent(v scriptedVenue, id string, qty float64, limit *float64) *fav1.OrderIntent {
	mode := fav1.TradingMode_TRADING_MODE_LIVE
	if v.Name() == "paper" {
		mode = fav1.TradingMode_TRADING_MODE_PAPER
	}
	return &fav1.OrderIntent{
		IntentId: id, AccountId: testAccount, InstrumentId: testInstrument,
		Side: fav1.Side_SIDE_BUY, Quantity: qty, ExecAlgo: algoOf(limit), LimitPrice: limit,
		Mode: mode,
	}
}

func algoOf(limit *float64) string {
	if limit != nil {
		return "limit"
	}
	return "market"
}

func confPrice(f float64) *float64 { return &f }

// --- the cases --------------------------------------------------------------------------------

func TestVenueConformance(t *testing.T) {
	venues := []struct {
		name  string
		build func(t *testing.T) scriptedVenue
	}{
		{"paper", newPaperScript},
		{"kraken", newKrakenScript},
		{"lemon", newLemonScript},
		{"oanda", newOandaScript},
	}
	for _, v := range venues {
		t.Run(v.name, func(t *testing.T) {
			for _, c := range conformanceCases {
				t.Run(c.name, func(t *testing.T) {
					s := v.build(t)
					if c.needs != "" && !s.Can(c.needs) {
						t.Skipf("%s cannot script %q", s.Name(), c.needs)
					}
					c.run(t, s)
				})
			}
		})
	}
}

var conformanceCases = []struct {
	name  string
	needs string
	run   func(t *testing.T, v scriptedVenue)
}{
	{name: "a market order fills its whole quantity once", run: func(t *testing.T, v scriptedVenue) {
		v.Mark(testInstrument, symbolFor(v.Name()), 100)
		in := confIntent(v, "01MARKET0000000000000000A1", 2, nil)
		evs := confSubmit(t, v, in)
		if v.Can(featAsync) {
			v.FillTo(in.IntentId, 2, 100, 0.2)
			evs = append(evs, v.Poll(t)...)
		}
		filled, terminal := confSummarise(t, evs, in)
		if math.Abs(filled-2) > 1e-9 {
			t.Fatalf("filled %g, want 2 (events %s)", filled, confRender(evs))
		}
		if terminal != fav1.OrderEvent_KIND_FILL {
			t.Fatalf("terminal %v, want FILL (events %s)", terminal, confRender(evs))
		}
	}},

	{name: "an unchanged venue produces no further events", needs: featAsync, run: func(t *testing.T, v scriptedVenue) {
		v.Mark(testInstrument, symbolFor(v.Name()), 100)
		// A limit order, because that is the one shape every venue holds and reports progress on:
		// a market order is filled inside the submit response at OANDA and from the tape on paper.
		in := confIntent(v, "01IDEMPOTENT000000000000A2", 2, confPrice(100))
		confSubmit(t, v, in)
		v.FillTo(in.IntentId, 2, 100, 0.1)
		first := v.Poll(t)
		if len(first) == 0 {
			t.Fatal("the first poll after a fill must report it")
		}
		// Same cumulative quantity, polled twice: a venue repeating itself must not double-count.
		if again := v.Poll(t); len(again) != 0 {
			t.Fatalf("a repeated poll produced %d more events: %s", len(again), confRender(again))
		}
	}},

	{name: "fills never exceed the intent", needs: featPartial, run: func(t *testing.T, v scriptedVenue) {
		v.Mark(testInstrument, symbolFor(v.Name()), 100)
		in := confIntent(v, "01PARTIAL00000000000000A3", 10, confPrice(100))
		evs := confSubmit(t, v, in)
		v.FillTo(in.IntentId, 4, 100, 0.4)
		evs = append(evs, v.Poll(t)...)
		v.FillTo(in.IntentId, 10, 101, 1.0)
		evs = append(evs, v.Poll(t)...)
		filled, terminal := confSummarise(t, evs, in)
		if math.Abs(filled-10) > 1e-9 {
			t.Fatalf("filled %g, want 10: %s", filled, confRender(evs))
		}
		if terminal != fav1.OrderEvent_KIND_FILL {
			t.Fatalf("terminal %v, want FILL: %s", terminal, confRender(evs))
		}
		// The second event prices only the new quantity: 6 units at (10·101 − 4·100)/6.
		var partials []*fav1.OrderEvent
		for _, e := range evs {
			if e.Event.Kind == fav1.OrderEvent_KIND_PARTIAL_FILL || e.Event.Kind == fav1.OrderEvent_KIND_FILL {
				partials = append(partials, e.Event)
			}
		}
		if len(partials) < 2 {
			t.Fatalf("want a partial then a fill, got %s", confRender(evs))
		}
		last := partials[len(partials)-1]
		want := (10*101 - 4*100) / 6.0
		if math.Abs(last.FillPrice-want) > 1e-6 {
			t.Fatalf("increment priced %g, want %g (the average of the new fills only)", last.FillPrice, want)
		}
	}},

	{name: "a cancelled order ends in one terminal event", run: func(t *testing.T, v scriptedVenue) {
		v.Mark(testInstrument, symbolFor(v.Name()), 100)
		in := confIntent(v, "01CANCEL0000000000000000A4", 2, confPrice(50)) // far from the mark: it rests
		evs := confSubmit(t, v, in)
		v.CancelAtVenue(in.IntentId)
		evs = append(evs, v.Poll(t)...)
		_, terminal := confSummarise(t, evs, in)
		if terminal != fav1.OrderEvent_KIND_CANCELED {
			t.Fatalf("terminal %v, want CANCELED: %s", terminal, confRender(evs))
		}
		// Nothing after the cancellation, however often the venue is asked again.
		if again := v.Poll(t); len(again) != 0 {
			t.Fatalf("events after the terminal one: %s", confRender(again))
		}
	}},

	{name: "a refusal is a rejection with a reason", run: func(t *testing.T, v scriptedVenue) {
		v.Mark(testInstrument, symbolFor(v.Name()), 100)
		v.RejectNext("insufficient funds")
		in := confIntent(v, "01REJECT0000000000000000A5", 2, nil)
		evs := confSubmit(t, v, in)
		filled, terminal := confSummarise(t, evs, in)
		if terminal != fav1.OrderEvent_KIND_REJECTED {
			t.Fatalf("terminal %v, want REJECTED: %s", terminal, confRender(evs))
		}
		if filled != 0 {
			t.Fatalf("a rejected order filled %g", filled)
		}
		if r := evs[len(evs)-1].Event.Reason; strings.TrimSpace(r) == "" {
			t.Fatal("a rejection must say why")
		}
	}},

	{name: "the same intent twice creates one order", needs: featDuplicate, run: func(t *testing.T, v scriptedVenue) {
		v.Mark(testInstrument, symbolFor(v.Name()), 100)
		in := confIntent(v, "01DUPLICATE00000000000A6", 2, confPrice(50))
		confSubmit(t, v, in)
		second := confSubmit(t, v, in)
		for _, e := range second {
			if k := e.Event.Kind; k == fav1.OrderEvent_KIND_FILL || k == fav1.OrderEvent_KIND_PARTIAL_FILL {
				t.Fatalf("the duplicate filled: %s", confRender(second))
			}
		}
	}},

	{name: "fills carry what the ledger needs", run: func(t *testing.T, v scriptedVenue) {
		v.Mark(testInstrument, symbolFor(v.Name()), 100)
		in := confIntent(v, "01LEDGER0000000000000000A7", 2, nil)
		evs := confSubmit(t, v, in)
		if v.Can(featAsync) {
			v.FillTo(in.IntentId, 2, 100, 0.2)
			evs = append(evs, v.Poll(t)...)
		}
		seen := map[string]bool{}
		fills := 0
		for _, e := range evs {
			ev := e.Event
			if ev.Kind != fav1.OrderEvent_KIND_FILL && ev.Kind != fav1.OrderEvent_KIND_PARTIAL_FILL {
				continue
			}
			fills++
			if ev.FillPrice <= 0 {
				t.Fatalf("fill at %g", ev.FillPrice)
			}
			if ev.Fee < 0 {
				t.Fatalf("negative fee %g", ev.Fee)
			}
			if ev.Fee > 0 && ev.FeeCurrency == "" {
				t.Fatal("a fee without a currency cannot be booked")
			}
			if ev.VenueFillId == "" {
				t.Fatal("a fill needs a venue fill id (spec 42 §5)")
			}
			if seen[ev.VenueFillId] {
				t.Fatalf("venue fill id %q repeated", ev.VenueFillId)
			}
			seen[ev.VenueFillId] = true
			if ev.EventTime == nil || ev.EventTime.AsTime().IsZero() {
				t.Fatal("a fill needs an event time")
			}
			// Slippage is signed against the taker; with an arrival price it must agree with it.
			if ev.ArrivalPrice > 0 {
				want := (ev.FillPrice - ev.ArrivalPrice) / ev.ArrivalPrice * 10_000
				if math.Abs(ev.SlippageBps-want) > 1e-6 {
					t.Fatalf("slippage %g, want %g for a buy", ev.SlippageBps, want)
				}
			}
		}
		if fills == 0 {
			t.Fatalf("no fill to check: %s", confRender(evs))
		}
	}},

	{name: "a restart re-attaches an open order and emits only the increment", needs: featRestart,
		run: func(t *testing.T, v scriptedVenue) {
			v.Mark(testInstrument, symbolFor(v.Name()), 100)
			in := confIntent(v, "01RESTART000000000000A8", 10, confPrice(100))
			confSubmit(t, v, in)
			// Where the venue reports progress, publish some of it before the restart: the point is
			// that the restarted adapter continues from what was published, not from zero.
			published := 0.0
			if v.Can(featPartial) {
				v.FillTo(in.IntentId, 4, 100, 0.4)
				before := v.Poll(t)
				if confSumFilled(before) != 4 {
					t.Fatalf("before the restart %g filled: %s", confSumFilled(before), confRender(before))
				}
				published = 4
			}
			w := v.Restarted(t)
			// The venue has not changed: the restarted adapter must not re-publish what was.
			if again := w.Poll(t); len(again) != 0 {
				t.Fatalf("the restarted adapter republished: %s", confRender(again))
			}
			w.FillTo(in.IntentId, 10, 100, 1.0)
			after := w.Poll(t)
			if want := 10 - published; math.Abs(confSumFilled(after)-want) > 1e-9 {
				t.Fatalf("after the restart %g filled, want the %g increment: %s",
					confSumFilled(after), want, confRender(after))
			}
			// And the increment is attributed to the intent that created the order, which is the
			// thing a venue cannot tell a restarted gateway.
			for _, e := range after {
				if e.Event.IntentId != in.IntentId || e.Account != in.AccountId {
					t.Fatalf("event re-attached to %q/%q, want %q/%q",
						e.Event.IntentId, e.Account, in.IntentId, in.AccountId)
				}
			}
		}},
}

// --- helpers ----------------------------------------------------------------------------------

func confSubmit(t *testing.T, v scriptedVenue, in *fav1.OrderIntent) []Published {
	t.Helper()
	evs, err := v.Adapter().Submit(context.Background(), in)
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	out := make([]Published, 0, len(evs))
	for _, e := range evs {
		if e == nil {
			t.Fatal("a nil event")
		}
		out = append(out, Published{Account: in.AccountId, Event: e})
	}
	return out
}

// summarise checks the ordering invariants and returns the filled quantity and terminal kind.
func confSummarise(t *testing.T, evs []Published, in *fav1.OrderIntent) (float64, fav1.OrderEvent_Kind) {
	t.Helper()
	if len(evs) == 0 {
		t.Fatal("a submitted intent must produce at least one event")
	}
	var filled float64
	terminal := fav1.OrderEvent_KIND_UNSPECIFIED
	for i, e := range evs {
		ev := e.Event
		if ev.IntentId != in.IntentId {
			t.Fatalf("event %d belongs to %q, not %q", i, ev.IntentId, in.IntentId)
		}
		if terminal != fav1.OrderEvent_KIND_UNSPECIFIED {
			t.Fatalf("event %d (%v) follows the terminal %v: %s", i, ev.Kind, terminal, confRender(evs))
		}
		switch ev.Kind {
		case fav1.OrderEvent_KIND_PARTIAL_FILL:
			filled += ev.FillQuantity
		case fav1.OrderEvent_KIND_FILL:
			filled += ev.FillQuantity
			terminal = ev.Kind
		case fav1.OrderEvent_KIND_CANCELED, fav1.OrderEvent_KIND_REJECTED:
			terminal = ev.Kind
		}
		if filled > in.Quantity+1e-9 {
			t.Fatalf("filled %g of an intent for %g: %s", filled, in.Quantity, confRender(evs))
		}
	}
	return filled, terminal
}

func confSumFilled(evs []Published) float64 {
	var f float64
	for _, e := range evs {
		if k := e.Event.Kind; k == fav1.OrderEvent_KIND_FILL || k == fav1.OrderEvent_KIND_PARTIAL_FILL {
			f += e.Event.FillQuantity
		}
	}
	return f
}

func confRender(evs []Published) string {
	parts := make([]string, 0, len(evs))
	for _, e := range evs {
		parts = append(parts, fmt.Sprintf("{%v q=%g p=%g fee=%g %q}",
			e.Event.Kind, e.Event.FillQuantity, e.Event.FillPrice, e.Event.Fee, e.Event.Reason))
	}
	return "[" + strings.Join(parts, " ") + "]"
}

func symbolFor(venue string) string {
	switch venue {
	case "kraken":
		return "XBTEUR"
	case "lemon":
		return "US0378331005"
	case "oanda":
		return "EUR_USD"
	}
	return "sim.TEST"
}

// fixture renders a testdata file with {{PLACEHOLDER}} substitutions.
func confFixture(t *testing.T, venue, file string, vals map[string]string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "venues", venue, file))
	if err != nil {
		t.Fatalf("fixture %s/%s: %v", venue, file, err)
	}
	out := string(raw)
	for k, v := range vals {
		out = strings.ReplaceAll(out, "{{"+k+"}}", v)
	}
	if i := strings.Index(out, "{{"); i >= 0 {
		t.Fatalf("fixture %s/%s has an unfilled placeholder at %d: %s", venue, file, i, out[i:min(i+40, len(out))])
	}
	// A fixture that is not valid JSON would make every case fail with a decode error instead of
	// the assertion it is about.
	var probe any
	if err := json.Unmarshal([]byte(out), &probe); err != nil {
		t.Fatalf("fixture %s/%s is not JSON after substitution: %v", venue, file, err)
	}
	return out
}

func confNum(f float64) string { return strconv.FormatFloat(f, 'f', -1, 64) }

// --- the paper venue --------------------------------------------------------------------------

type paperScript struct {
	engine *PaperEngine
	token  string
	queued []Published
}

func newPaperScript(t *testing.T) scriptedVenue {
	e := NewPaperEngine(DefaultCostModel())
	e.Guard = &RiskGuard{Timeout: 0} // the fail-closed guard is state_test.go's subject, not this one
	return &paperScript{engine: e, token: bus.SubjectToken(testInstrument)}
}

func (p *paperScript) Name() string          { return "paper" }
func (p *paperScript) Adapter() VenueAdapter { return p.engine }

func (p *paperScript) Mark(_, _ string, price float64) {
	p.queued = append(p.queued, p.engine.ObserveTrade("md.trade.sim."+p.token, price, 1, time.Now())...)
	p.engine.ObserveQuote("md.trade.sim."+p.token, price*0.999, price*1.001)
}

// The paper engine fills from the tape, so a scripted fill is a price that crosses.
func (p *paperScript) FillTo(_ string, _, price, _ float64) {
	p.queued = append(p.queued, p.engine.ObserveTrade("md.trade.sim."+p.token, price, 1, time.Now())...)
}

func (p *paperScript) CancelAtVenue(intentID string) {
	if c := p.engine.CancelResting(intentID); c != nil {
		p.queued = append(p.queued, *c)
	}
}

// Nothing refuses a paper order from outside, so the rejection the suite needs is produced the way
// the engine produces one: an unpriced instrument.
func (p *paperScript) RejectNext(string) { p.engine = freshPaper(p.engine) }

func freshPaper(old *PaperEngine) *PaperEngine {
	e := NewPaperEngine(old.Costs)
	e.Guard = old.Guard
	return e
}

func (p *paperScript) Poll(*testing.T) []Published {
	out := append(p.queued, p.engine.Tick()...)
	p.queued = nil
	return out
}

func (p *paperScript) Can(feature string) bool {
	// Paper fills synchronously from the tape, and its partials come from execution algorithms
	// (algo.go), not from a venue: those are algo_test.go's subject.
	return feature == "" || feature == featDuplicate
}

func (p *paperScript) Restarted(t *testing.T) scriptedVenue {
	t.Skip("the paper engine's restart is state_test.go's subject (StateStore round trip)")
	return p
}

// --- a scripted live venue --------------------------------------------------------------------

// stubOrder is the venue's own view of one order.
type stubOrder struct {
	id       string
	clOrdID  string
	symbol   string
	side     string
	qty      float64
	limit    float64
	cum      float64
	price    float64
	fee      float64
	canceled bool
}

// venueStub is the scripted state behind a stub HTTP server: the orders the venue holds, keyed by
// its own id and by the client id, so the duplicate-intent case can be answered the way a venue
// answers it.
type venueStub struct {
	mu         sync.Mutex
	venue      string
	orders     map[string]*stubOrder
	byClient   map[string]string
	rejectNext string
	seq        int
	t          *testing.T
}

func newVenueStub(t *testing.T, venue string) *venueStub {
	return &venueStub{venue: venue, orders: map[string]*stubOrder{}, byClient: map[string]string{}, t: t}
}

func (s *venueStub) create(clOrdID, symbol, side string, qty, limit float64) (*stubOrder, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if id, dup := s.byClient[clOrdID]; dup {
		return s.orders[id], false
	}
	s.seq++
	o := &stubOrder{id: fmt.Sprintf("%s-ORD-%d", strings.ToUpper(s.venue), s.seq), clOrdID: clOrdID,
		symbol: symbol, side: side, qty: qty, limit: limit}
	s.orders[o.id], s.byClient[clOrdID] = o, o.id
	return o, true
}

func (s *venueStub) byIntent(intentID string) *stubOrder {
	s.mu.Lock()
	defer s.mu.Unlock()
	if id, ok := s.byClient[intentID]; ok {
		return s.orders[id]
	}
	return nil
}

func (s *venueStub) fillTo(intentID string, cum, price, fee float64) {
	if o := s.byIntent(intentID); o != nil {
		s.mu.Lock()
		o.cum, o.price, o.fee = cum, price, fee
		s.mu.Unlock()
	}
}

func (s *venueStub) cancel(intentID string) {
	if o := s.byIntent(intentID); o != nil {
		s.mu.Lock()
		o.canceled = true
		s.mu.Unlock()
	}
}

func (s *venueStub) takeReject() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.rejectNext
	s.rejectNext = ""
	return r
}

// liveScript drives one live adapter against its stub.
type liveScript struct {
	name     string
	features map[string]bool
	stub     *venueStub
	server   *httptest.Server
	adapter  LiveVenue
	mark     float64
	rebuild  func(stub *venueStub, url string) LiveVenue
}

func (l *liveScript) Name() string          { return l.name }
func (l *liveScript) Adapter() VenueAdapter { return l.adapter }

func (l *liveScript) Mark(_, _ string, price float64) { l.mark = price }

func (l *liveScript) FillTo(intentID string, cum, price, fee float64) {
	l.stub.fillTo(intentID, cum, price, fee)
}

func (l *liveScript) CancelAtVenue(intentID string) { l.stub.cancel(intentID) }

func (l *liveScript) RejectNext(reason string) {
	l.stub.mu.Lock()
	l.stub.rejectNext = reason
	l.stub.mu.Unlock()
}

func (l *liveScript) Poll(t *testing.T) []Published {
	out, err := l.adapter.Poll(context.Background())
	if err != nil && len(out) == 0 {
		// A poll error with no events is reported, not failed: the cases that care assert on the
		// events, and a venue that is briefly unhappy is a normal condition.
		t.Logf("%s poll: %v", l.name, err)
	}
	return out
}

func (l *liveScript) Can(feature string) bool {
	return feature == "" || l.features[feature]
}

func (l *liveScript) Restarted(t *testing.T) scriptedVenue {
	saved := l.adapter.Snapshot()
	if len(saved) == 0 {
		t.Fatal("an open order must be in the snapshot")
	}
	// The snapshot survives a serialisation round trip, because that is how it reaches the next
	// process (`orders.json`).
	raw, err := json.Marshal(saved)
	if err != nil {
		t.Fatalf("marshal snapshot: %v", err)
	}
	var back []LiveOrderState
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal snapshot: %v", err)
	}
	next := l.rebuild(l.stub, l.server.URL)
	next.Restore(back)
	return &liveScript{name: l.name, features: l.features, stub: l.stub, server: l.server, adapter: next,
		mark: l.mark, rebuild: l.rebuild}
}

// fastFetcher is the adapters' HTTP client with the retry backoff removed: a stub answering 4xx
// should fail a test in milliseconds, not after three waits.
func fastFetcher() *connector.Fetcher {
	f := connector.NewFetcher(nil)
	f.Retries, f.Backoff = 1, time.Millisecond
	return f
}

// --- kraken ------------------------------------------------------------------------------------

func newKrakenScript(t *testing.T) scriptedVenue {
	stub := newVenueStub(t, "kraken")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		krakenStubHandler(t, stub, w, r)
	}))
	t.Cleanup(srv.Close)
	build := func(stub *venueStub, url string) LiveVenue {
		k := NewKraken(VenueCreds{Key: "test-key", Secret: "c2VjcmV0"}, nil) // base64 "secret"
		k.BaseURL, k.Fetch = url, fastFetcher()
		k.Symbols[testInstrument] = "XBTEUR"
		return k
	}
	return &liveScript{name: "kraken", stub: stub, server: srv, adapter: build(stub, srv.URL), rebuild: build,
		features: map[string]bool{featAsync: true, featPartial: true, featRestart: true, featDuplicate: true}}
}

func krakenStubHandler(t *testing.T, stub *venueStub, w http.ResponseWriter, r *http.Request) {
	t.Helper()
	if got := r.Header.Get("API-Sign"); got == "" {
		t.Error("kraken: every private call must be signed")
	}
	if err := r.ParseForm(); err != nil {
		t.Fatalf("kraken: body: %v", err)
	}
	if r.PostForm.Get("nonce") == "" {
		t.Error("kraken: every private call needs a nonce")
	}
	write := func(s string) { w.Header().Set("Content-Type", "application/json"); _, _ = w.Write([]byte(s)) }
	switch r.URL.Path {
	case "/0/private/AddOrder":
		if reason := stub.takeReject(); reason != "" {
			write(confFixture(t, "kraken", "error.json", map[string]string{"ERROR": "EOrder:" + reason}))
			return
		}
		cl := r.PostForm.Get("cl_ord_id")
		limit, _ := strconv.ParseFloat(r.PostForm.Get("price"), 64)
		qty, _ := strconv.ParseFloat(r.PostForm.Get("volume"), 64)
		o, fresh := stub.create(cl, r.PostForm.Get("pair"), r.PostForm.Get("type"), qty, limit)
		if !fresh {
			// Kraken refuses a repeated client order id instead of opening a second order.
			write(confFixture(t, "kraken", "error.json", map[string]string{"ERROR": "EOrder:Duplicate cl_ord_id"}))
			return
		}
		write(confFixture(t, "kraken", "add_order.json", map[string]string{
			"TXID": o.id, "DESCR": fmt.Sprintf("%s %g %s", o.side, o.qty, o.symbol),
		}))
	case "/0/private/QueryOrders":
		ids := strings.Split(r.PostForm.Get("txid"), ",")
		stub.mu.Lock()
		defer stub.mu.Unlock()
		// One entry per requested id, merged into one result object.
		var entries []string
		for _, id := range ids {
			o := stub.orders[id]
			if o == nil {
				continue
			}
			status := "open"
			switch {
			case o.canceled:
				status = "canceled"
			case o.cum >= o.qty-1e-9 && o.cum > 0:
				status = "closed"
			}
			one := confFixture(t, "kraken", "query_orders.json", map[string]string{
				"TXID": o.id, "STATUS": status, "PAIR": o.symbol, "SIDE": o.side,
				"ORDERTYPE": krakenOrderType(o), "LIMIT": confNum(o.limit), "VOL": confNum(o.qty),
				"VOL_EXEC": confNum(o.cum), "COST": confNum(o.cum * o.price), "FEE": confNum(o.fee),
				"PRICE": confNum(o.price), "REASON": "",
			})
			var parsed struct {
				Result map[string]json.RawMessage `json:"result"`
			}
			if err := json.Unmarshal([]byte(one), &parsed); err != nil {
				t.Fatalf("kraken fixture: %v", err)
			}
			entries = append(entries, fmt.Sprintf("%q:%s", o.id, string(parsed.Result[o.id])))
		}
		write(fmt.Sprintf(`{"error":[],"result":{%s}}`, strings.Join(entries, ",")))
	case "/0/private/CancelOrder":
		stub.mu.Lock()
		if o := stub.orders[r.PostForm.Get("txid")]; o != nil {
			o.canceled = true
		}
		stub.mu.Unlock()
		write(confFixture(t, "kraken", "cancel_order.json", nil))
	case "/0/private/Balance":
		write(confFixture(t, "kraken", "balance.json", nil))
	default:
		t.Fatalf("kraken stub: unexpected path %s", r.URL.Path)
	}
}

func krakenOrderType(o *stubOrder) string {
	if o.limit > 0 {
		return "limit"
	}
	return "market"
}

// --- lemon -------------------------------------------------------------------------------------

func newLemonScript(t *testing.T) scriptedVenue {
	stub := newVenueStub(t, "lemon")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lemonStubHandler(t, stub, w, r)
	}))
	t.Cleanup(srv.Close)
	build := func(stub *venueStub, url string) LiveVenue {
		l := NewLemon(VenueCreds{Key: "test-key"}, nil)
		l.BaseURL, l.Fetch = url+"/v1", fastFetcher()
		l.Symbols[testInstrument] = "US0378331005"
		return l
	}
	return &liveScript{name: "lemon", stub: stub, server: srv, adapter: build(stub, srv.URL), rebuild: build,
		features: map[string]bool{featAsync: true, featPartial: true, featRestart: true, featDuplicate: true}}
}

func lemonStubHandler(t *testing.T, stub *venueStub, w http.ResponseWriter, r *http.Request) {
	t.Helper()
	if got := r.Header.Get("Authorization"); !strings.HasPrefix(got, "Bearer ") {
		t.Errorf("lemon: bearer auth expected, got %q", got)
	}
	write := func(code int, s string) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		_, _ = w.Write([]byte(s))
	}
	path := strings.TrimPrefix(r.URL.Path, "/v1")
	switch {
	case path == "/orders" && r.Method == http.MethodPost:
		if reason := stub.takeReject(); reason != "" {
			write(http.StatusBadRequest, confFixture(t, "lemon", "error.json",
				map[string]string{"CODE": "order_rejected", "ERROR": reason}))
			return
		}
		var body struct {
			ISIN        string  `json:"isin"`
			Side        string  `json:"side"`
			Quantity    float64 `json:"quantity"`
			LimitPrice  float64 `json:"limit_price"`
			Idempotency string  `json:"idempotency"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("lemon: body: %v", err)
		}
		o, fresh := stub.create(body.Idempotency, body.ISIN, body.Side, body.Quantity, body.LimitPrice/100)
		if !fresh {
			write(http.StatusBadRequest, confFixture(t, "lemon", "error.json",
				map[string]string{"CODE": "order_duplicate", "ERROR": "an order with this idempotency key exists"}))
			return
		}
		write(http.StatusOK, confFixture(t, "lemon", "create_order.json", map[string]string{
			"ORDER_ID": o.id, "ISIN": o.symbol, "SIDE": o.side, "QTY": confNum(o.qty),
			"EST_PRICE": confNum(o.limit * 100),
		}))
	case strings.HasSuffix(path, "/activate") && r.Method == http.MethodPost:
		write(http.StatusOK, confFixture(t, "lemon", "activate_order.json", nil))
	case strings.HasPrefix(path, "/orders/") && r.Method == http.MethodGet:
		id := strings.TrimPrefix(path, "/orders/")
		stub.mu.Lock()
		o := stub.orders[id]
		stub.mu.Unlock()
		if o == nil {
			write(http.StatusNotFound, `{"status":"error","error_message":"not found"}`)
			return
		}
		status := "activated"
		execAt := ""
		switch {
		case o.canceled:
			status = "canceled"
		case o.cum >= o.qty-1e-9 && o.cum > 0:
			status, execAt = "executed", time.Now().UTC().Format(time.RFC3339)
		case o.cum > 0:
			status, execAt = "activated", time.Now().UTC().Format(time.RFC3339)
		}
		write(http.StatusOK, confFixture(t, "lemon", "get_order.json", map[string]string{
			"ORDER_ID": o.id, "ISIN": o.symbol, "STATUS": status, "SIDE": o.side, "QTY": confNum(o.qty),
			"EXEC_QTY": confNum(o.cum), "EXEC_PRICE": confNum(o.price * 100), "EXEC_AT": execAt,
			"CHARGE": confNum(o.fee * 100), "ERROR": "",
		}))
	case strings.HasPrefix(path, "/orders/") && r.Method == http.MethodDelete:
		stub.mu.Lock()
		if o := stub.orders[strings.TrimPrefix(path, "/orders/")]; o != nil {
			o.canceled = true
		}
		stub.mu.Unlock()
		write(http.StatusOK, confFixture(t, "lemon", "activate_order.json", nil))
	case path == "/positions":
		write(http.StatusOK, confFixture(t, "lemon", "positions.json", nil))
	case path == "/account":
		write(http.StatusOK, confFixture(t, "lemon", "account.json", nil))
	default:
		t.Fatalf("lemon stub: unexpected %s %s", r.Method, r.URL.Path)
	}
}

// --- oanda -------------------------------------------------------------------------------------

func newOandaScript(t *testing.T) scriptedVenue {
	stub := newVenueStub(t, "oanda")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		oandaStubHandler(t, stub, w, r)
	}))
	t.Cleanup(srv.Close)
	build := func(stub *venueStub, url string) LiveVenue {
		o := NewOanda(VenueCreds{Key: "test-key", Account: "001-004-1-001", Env: "practice"}, nil)
		o.BaseURL, o.Fetch = url, fastFetcher()
		o.Symbols[testInstrument] = "EUR_USD"
		return o
	}
	// v20 fills an order in one transaction and reports the order as FILLED or not at all: a
	// partial fill of a resting limit arrives as its own trade, not as progress on the order. So
	// the partial-increment case does not apply to OANDA, and saying so here is better than a
	// stub that pretends otherwise.
	return &liveScript{name: "oanda", stub: stub, server: srv, adapter: build(stub, srv.URL), rebuild: build,
		features: map[string]bool{featAsync: true, featRestart: true, featDuplicate: true}}
}

func oandaStubHandler(t *testing.T, stub *venueStub, w http.ResponseWriter, r *http.Request) {
	t.Helper()
	if got := r.Header.Get("Authorization"); !strings.HasPrefix(got, "Bearer ") {
		t.Errorf("oanda: bearer auth expected, got %q", got)
	}
	const account = "001-004-1-001"
	write := func(code int, s string) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		_, _ = w.Write([]byte(s))
	}
	path := strings.TrimPrefix(r.URL.Path, "/v3/accounts/"+account)
	switch {
	case path == "/orders" && r.Method == http.MethodPost:
		if reason := stub.takeReject(); reason != "" {
			write(http.StatusBadRequest, confFixture(t, "oanda", "reject.json", map[string]string{"REASON": reason}))
			return
		}
		var body struct {
			Order struct {
				Type             string `json:"type"`
				Instrument       string `json:"instrument"`
				Units            string `json:"units"`
				Price            string `json:"price"`
				ClientExtensions struct {
					ID string `json:"id"`
				} `json:"clientExtensions"`
			} `json:"order"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("oanda: body: %v", err)
		}
		units := mustFloat(body.Order.Units)
		side := "buy"
		if units < 0 {
			side = "sell"
		}
		o, fresh := stub.create(body.Order.ClientExtensions.ID, body.Order.Instrument, side,
			absFloat(units), mustFloat(body.Order.Price))
		if !fresh {
			write(http.StatusBadRequest, confFixture(t, "oanda", "reject.json",
				map[string]string{"REASON": "CLIENT_ORDER_ID_ALREADY_EXISTS"}))
			return
		}
		if body.Order.Type == "MARKET" {
			// v20 fills a market order inside the submit response.
			stub.mu.Lock()
			o.cum, o.price, o.fee = o.qty, mustFloatOr(body.Order.Price, stubMark), 0.02
			stub.mu.Unlock()
			write(http.StatusCreated, confFixture(t, "oanda", "market_fill.json", map[string]string{
				"ORDER_ID": o.id, "FILL_ID": o.id + "-F", "ACCOUNT": account, "SYMBOL": o.symbol,
				"UNITS": confNum(units), "FILLED_UNITS": confNum(units), "PRICE": confNum(o.price),
				"COMMISSION": confNum(o.fee), "TIME": time.Now().UTC().Format(time.RFC3339),
			}))
			return
		}
		write(http.StatusCreated, confFixture(t, "oanda", "limit_create.json", map[string]string{
			"ORDER_ID": o.id, "SYMBOL": o.symbol, "UNITS": confNum(units), "LIMIT": confNum(o.limit),
			"TIME": time.Now().UTC().Format(time.RFC3339),
		}))
	case strings.HasPrefix(path, "/orders/") && strings.HasSuffix(path, "/cancel"):
		id := strings.TrimSuffix(strings.TrimPrefix(path, "/orders/"), "/cancel")
		stub.mu.Lock()
		if o := stub.orders[id]; o != nil {
			o.canceled = true
		}
		stub.mu.Unlock()
		write(http.StatusOK, `{"orderCancelTransaction":{"id":"9"},"lastTransactionID":"9"}`)
	case strings.HasPrefix(path, "/orders/") && r.Method == http.MethodGet:
		id := strings.TrimPrefix(path, "/orders/")
		stub.mu.Lock()
		o := stub.orders[id]
		stub.mu.Unlock()
		if o == nil {
			write(http.StatusNotFound, `{"errorMessage":"no such order"}`)
			return
		}
		state, filled, cancelled := "PENDING", "", ""
		switch {
		case o.canceled:
			state, cancelled = "CANCELLED", time.Now().UTC().Format(time.RFC3339)
		case o.cum >= o.qty-1e-9 && o.cum > 0:
			state, filled = "FILLED", time.Now().UTC().Format(time.RFC3339)
		}
		units := o.qty
		if o.side == "sell" {
			units = -units
		}
		write(http.StatusOK, confFixture(t, "oanda", "order.json", map[string]string{
			"ORDER_ID": o.id, "SYMBOL": o.symbol, "UNITS": confNum(units), "STATE": state,
			"TRADE_ID": o.id + "-T", "FILLED_TIME": filled, "CANCELLED_TIME": cancelled,
		}))
	case strings.HasPrefix(path, "/trades/"):
		id := strings.TrimSuffix(strings.TrimPrefix(path, "/trades/"), "-T")
		stub.mu.Lock()
		o := stub.orders[id]
		stub.mu.Unlock()
		if o == nil {
			write(http.StatusNotFound, `{"errorMessage":"no such trade"}`)
			return
		}
		write(http.StatusOK, confFixture(t, "oanda", "trade.json", map[string]string{
			"TRADE_ID": o.id + "-T", "SYMBOL": o.symbol, "PRICE": confNum(o.price),
			"UNITS": confNum(o.cum), "FINANCING": confNum(o.fee),
		}))
	case path == "/openPositions":
		write(http.StatusOK, confFixture(t, "oanda", "open_positions.json", nil))
	case path == "/summary":
		write(http.StatusOK, confFixture(t, "oanda", "summary.json", map[string]string{"ACCOUNT": account}))
	default:
		t.Fatalf("oanda stub: unexpected %s %s", r.Method, r.URL.Path)
	}
}

// stubMark is the price a stub fills a market order at when the order carried none.
const stubMark = 100.0

func mustFloatOr(s string, fallback float64) float64 {
	if f := mustFloat(s); f > 0 {
		return f
	}
	return fallback
}
