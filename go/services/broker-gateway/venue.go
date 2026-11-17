// Venue adapters: the one execution path for paper and real money (spec 40, D-121).
//
// `VenueAdapter` was declared when only the paper engine existed, with the note that "live differs
// in `Submit` only". This file makes that true. The paper engine and every live venue implement the
// same interface, the gateway routes an intent to one of them by the intent's own
// `TradingMode`, and the same conformance suite runs against all of them
// (`venue_conformance_test.go`). Switching the platform to real money therefore changes which
// adapter an intent reaches — not how orders are made, netted, accounted or audited.
//
// What a live adapter must do that paper does not:
//
//   - **Say whether it can trade at all.** `Ready` fails when the venue's credentials are missing
//     or mis-shaped, and an intent routed to an unready venue is rejected rather than dropped.
//     This is what lets the mode switch refuse to flip (the risk engine asks before stamping
//     `LIVE` on anything).
//   - **Poll.** A venue answers a submit with an acknowledgement; the fills arrive afterwards. Each
//     adapter keeps the orders it has open and turns the venue's view of them into `OrderEvent`s on
//     the gateway's tick, emitting one event per *increase* in cumulative filled quantity so a
//     repeated poll cannot double-count a fill.
//   - **Report its own positions and cash**, so reconciliation compares the live book against the
//     venue rather than against itself (`bins/risk-engine/src/recon.rs`).
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// LiveVenue is a VenueAdapter that talks to a real venue.
type LiveVenue interface {
	VenueAdapter
	// Ready reports whether this venue could accept an order right now: credentials present and
	// shaped as the venue needs them. It performs no network call — a venue being briefly
	// unreachable is not the same as being unconfigured, and only the second one may block a mode
	// switch.
	Ready() error
	// Poll turns the venue's view of this adapter's open orders into the events that have not been
	// published yet.
	Poll(ctx context.Context) ([]Published, error)
	// Positions and Balances are the venue's own books, for reconciliation.
	Positions(ctx context.Context) ([]VenuePosition, error)
	Balances(ctx context.Context) ([]VenueBalance, error)
	// Symbol maps an internal instrument id to the venue's symbol, when the adapter knows it.
	Symbol(instrumentID string) (string, bool)
	// Snapshot and Restore carry this adapter's open orders across a gateway restart.
	//
	// They are needed because the venue cannot supply what they hold: a venue knows its own order
	// id and the client id we sent, but not which account or instrument of ours the order belongs
	// to. A restarted gateway that could not re-attach an open live order would keep trading
	// around a position it no longer knows it has — so the open orders are persisted next to the
	// paper engine's own state (`state.go`) and handed back here at start.
	Snapshot() []LiveOrderState
	Restore(orders []LiveOrderState)
	// Revision changes whenever the open orders did, so the state file is written when a live
	// order moves and not only when the paper engine does something.
	Revision() uint64
}

// LiveOrderState is one open live order, as persisted.
type LiveOrderState struct {
	IntentID   string  `json:"intent_id"`
	VenueID    string  `json:"venue_id"`
	Account    string  `json:"account"`
	Instrument string  `json:"instrument"`
	Symbol     string  `json:"symbol"`
	Side       int32   `json:"side"`
	Quantity   float64 `json:"quantity"`
	Filled     float64 `json:"filled"`
	Notional   float64 `json:"notional"`
	Fees       float64 `json:"fees"`
	Arrival    float64 `json:"arrival"`
	Seq        int     `json:"seq"`
}

// VenuePosition is one open position as the venue reports it.
type VenuePosition struct {
	Venue    string  `json:"venue"`
	Symbol   string  `json:"symbol"`
	Quantity float64 `json:"quantity"`
	AvgPrice float64 `json:"avg_price"`
}

// VenueBalance is one cash balance as the venue reports it.
type VenueBalance struct {
	Venue    string  `json:"venue"`
	Currency string  `json:"currency"`
	Amount   float64 `json:"amount"`
}

// VenueCreds are one venue's credentials, from `[venues.<id>]` in config/secrets.toml or
// `FA_VENUES_<ID>_<KEY>` (D-121). The values are never logged and never leave this process.
type VenueCreds struct {
	Key     string
	Secret  string
	Account string
	// Env selects a venue's practice or production hosts where it has both.
	Env string
}

// LoadVenueCreds reads `[venues.<id>]` from a secrets file plus the environment overrides.
func LoadVenueCreds(s *connector.Secrets, id string) VenueCreds {
	section := "venues." + id
	get := func(key string) string {
		if v := s.Get(section, key); v != "" {
			return v
		}
		// `[sources.<id>]` is the data connector's credential for the same venue. Falling back to
		// it would mean a read-only data key silently becoming a trading key, so it is not read
		// here: trading credentials are entered deliberately, under `[venues.<id>]`.
		return ""
	}
	return VenueCreds{Key: get("api_key"), Secret: get("api_secret"), Account: get("account_id"), Env: get("env")}
}

// Redacted is what a credential set may be printed as.
func (c VenueCreds) Redacted() string {
	mark := func(v string) string {
		if v == "" {
			return "unset"
		}
		return "set"
	}
	return fmt.Sprintf("key=%s secret=%s account=%s env=%s", mark(c.Key), mark(c.Secret), mark(c.Account), c.Env)
}

// errNoCredentials is what Ready returns when the operator has entered nothing.
type venueError struct {
	venue  string
	detail string
}

func (e *venueError) Error() string { return e.venue + ": " + e.detail }

func missingCreds(venue string, fields ...string) error {
	return &venueError{venue: venue, detail: "missing credentials: " + strings.Join(fields, ", ")}
}

// VenueStatus is what `ctl.exec.venues` reports: enough for the mode switch to decide, and nothing
// that could leak a credential.
type VenueStatus struct {
	Venue      string `json:"venue"`
	Configured bool   `json:"configured"`
	Detail     string `json:"detail,omitempty"`
	Open       int    `json:"open_orders"`
}

// Router sends an intent to the paper engine or to a live venue, by the intent's own mode.
//
// The mode travels with the intent rather than being a setting of this process. That is
// deliberate: the risk engine decides what is real money, one intent at a time, and a gateway
// restart cannot silently change the meaning of intents already in flight.
type Router struct {
	Paper *PaperEngine
	live  map[string]LiveVenue
	// VenueOf resolves the venue an instrument trades at (the market-data venue it was priced
	// from). Nil means no live routing is possible.
	VenueOf  func(instrumentID string) (string, bool)
	Counters *bus.Counters
	now      func() time.Time
}

func NewRouter(paper *PaperEngine) *Router {
	return &Router{Paper: paper, live: map[string]LiveVenue{}, now: time.Now}
}

// Add registers a live venue. Adapters are added whether or not they have credentials: an
// unconfigured venue must be able to say so (a rejected intent naming the venue is far easier to
// act on than an intent that vanished).
func (r *Router) Add(v LiveVenue) {
	r.live[v.Venue()] = v
}

func (r *Router) Live(venue string) (LiveVenue, bool) {
	v, ok := r.live[venue]
	return v, ok
}

// Venues reports every live venue's readiness, sorted by id.
func (r *Router) Venues() []VenueStatus {
	out := make([]VenueStatus, 0, len(r.live))
	for id, v := range r.live {
		st := VenueStatus{Venue: id, Configured: true}
		if err := v.Ready(); err != nil {
			st.Configured, st.Detail = false, err.Error()
		}
		if o, ok := v.(interface{ OpenOrders() int }); ok {
			st.Open = o.OpenOrders()
		}
		out = append(out, st)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Venue < out[j].Venue })
	return out
}

// Ready reports whether any live venue could trade, with the per-venue detail.
func (r *Router) Ready() ([]VenueStatus, bool) {
	st := r.Venues()
	for _, s := range st {
		if s.Configured {
			return st, true
		}
	}
	return st, false
}

// Execute routes one approved intent.
//
// A `LIVE` intent that cannot be routed is **rejected with the reason**, never passed to the paper
// engine: filling real-money intent on the paper book would make the two ledgers disagree in the
// one direction nobody checks.
func (r *Router) Execute(ctx context.Context, in *fav1.OrderIntent) []Published {
	if in.Mode != fav1.TradingMode_TRADING_MODE_LIVE {
		if ev := r.Paper.Execute(in); ev != nil {
			return []Published{{Account: in.AccountId, Event: ev}}
		}
		return nil
	}
	reject := func(reason string) []Published {
		if r.Counters != nil {
			r.Counters.Add("live.rejected", 1)
		}
		return []Published{{Account: in.AccountId, Event: &fav1.OrderEvent{
			IntentId: in.IntentId, Kind: fav1.OrderEvent_KIND_REJECTED, Reason: reason,
			InstrumentId: in.InstrumentId, EventTime: timestamppb.New(r.now()),
		}}}
	}
	if r.VenueOf == nil {
		return reject("live: no venue resolver configured")
	}
	venue, ok := r.VenueOf(in.InstrumentId)
	if !ok {
		return reject("live: no venue known for this instrument (no market data seen)")
	}
	adapter, ok := r.live[venue]
	if !ok {
		return reject("live: no adapter for venue " + venue)
	}
	if err := adapter.Ready(); err != nil {
		return reject("live: " + err.Error())
	}
	evs, err := adapter.Submit(ctx, in)
	if err != nil {
		return reject("live: " + err.Error())
	}
	out := make([]Published, 0, len(evs))
	for _, ev := range evs {
		out = append(out, Published{Account: in.AccountId, Event: ev})
	}
	if r.Counters != nil {
		r.Counters.Add("live.submitted", 1)
	}
	return out
}

// Cancel routes a cancel to whichever adapter holds the order.
func (r *Router) Cancel(ctx context.Context, intentID string) []Published {
	if c := r.Paper.CancelResting(intentID); c != nil {
		return []Published{*c}
	}
	for _, v := range r.live {
		if h, ok := v.(interface{ Holds(string) bool }); ok && !h.Holds(intentID) {
			continue
		}
		if err := v.Cancel(ctx, intentID); err == nil {
			return nil // the cancellation shows up as an event on the next poll
		}
	}
	return nil
}

// Poll collects events from every live venue.
func (r *Router) Poll(ctx context.Context) []Published {
	var out []Published
	for id, v := range r.live {
		if v.Ready() != nil {
			continue
		}
		evs, err := v.Poll(ctx)
		if err != nil {
			if r.Counters != nil {
				r.Counters.Add("live.poll_errors", 1)
			}
			continue
		}
		if len(evs) > 0 && r.Counters != nil {
			r.Counters.Add("live.events."+id, float64(len(evs)))
		}
		out = append(out, evs...)
	}
	return out
}

// Snapshot returns every live venue's open orders, by venue id.
func (r *Router) Snapshot() map[string][]LiveOrderState {
	out := map[string][]LiveOrderState{}
	for id, v := range r.live {
		if s := v.Snapshot(); len(s) > 0 {
			out[id] = s
		}
	}
	return out
}

// Restore hands each venue back its own open orders. A venue in the file that is no longer
// registered is reported rather than dropped silently: an open order at a venue the gateway no
// longer talks to is exactly the situation an operator must know about.
func (r *Router) Restore(saved map[string][]LiveOrderState) []string {
	var orphaned []string
	for id, orders := range saved {
		v, ok := r.live[id]
		if !ok {
			orphaned = append(orphaned, fmt.Sprintf("%s (%d orders)", id, len(orders)))
			continue
		}
		v.Restore(orders)
	}
	sort.Strings(orphaned)
	return orphaned
}

// Revision changes whenever any live venue's open orders changed.
func (r *Router) Revision() uint64 {
	var sum uint64
	for _, v := range r.live {
		sum += v.Revision()
	}
	return sum
}

// --- shared live-order bookkeeping ------------------------------------------------------------

// liveOrder is one order an adapter has at a venue.
type liveOrder struct {
	intentID   string
	venueID    string
	account    string
	instrument string
	symbol     string
	side       fav1.Side
	quantity   float64
	// Cumulative filled quantity already published, and the notional behind it, so each poll
	// emits only the increment.
	filled   float64
	notional float64
	fees     float64
	arrival  float64
	seq      int
	done     bool
}

// liveBook is the set of open orders of one adapter. Every adapter shares it, which is what keeps
// the fill-increment rule (and therefore the no-double-count property) in one place.
type liveBook struct {
	mu     sync.Mutex
	venue  string
	orders map[string]*liveOrder
	rev    uint64
	now    func() time.Time
}

func newLiveBook(venue string) *liveBook {
	return &liveBook{venue: venue, orders: map[string]*liveOrder{}, now: time.Now}
}

func (b *liveBook) put(o *liveOrder) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.orders[o.intentID] = o
	b.rev++
}

func (b *liveBook) Revision() uint64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.rev
}

func (b *liveBook) get(intentID string) (*liveOrder, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	o, ok := b.orders[intentID]
	return o, ok
}

func (b *liveBook) Holds(intentID string) bool {
	_, ok := b.get(intentID)
	return ok
}

func (b *liveBook) OpenOrders() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := 0
	for _, o := range b.orders {
		if !o.done {
			n++
		}
	}
	return n
}

func (b *liveBook) open() []*liveOrder {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]*liveOrder, 0, len(b.orders))
	for _, o := range b.orders {
		if !o.done {
			out = append(out, o)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].intentID < out[j].intentID })
	return out
}

// Snapshot returns the open orders of this book, oldest intent id first.
func (b *liveBook) Snapshot() []LiveOrderState {
	out := make([]LiveOrderState, 0)
	for _, o := range b.open() {
		out = append(out, LiveOrderState{
			IntentID: o.intentID, VenueID: o.venueID, Account: o.account, Instrument: o.instrument,
			Symbol: o.symbol, Side: int32(o.side), Quantity: o.quantity, Filled: o.filled,
			Notional: o.notional, Fees: o.fees, Arrival: o.arrival, Seq: o.seq,
		})
	}
	return out
}

// Restore re-attaches persisted orders. The published progress (`Filled`, `Notional`, `Fees`,
// `Seq`) comes back with them, which is what makes the first poll after a restart emit the
// increment since the restart rather than the whole fill again.
func (b *liveBook) Restore(orders []LiveOrderState) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.rev++
	for _, s := range orders {
		if s.IntentID == "" || s.VenueID == "" {
			continue
		}
		b.orders[s.IntentID] = &liveOrder{
			intentID: s.IntentID, venueID: s.VenueID, account: s.Account, instrument: s.Instrument,
			symbol: s.Symbol, side: fav1.Side(s.Side), quantity: s.Quantity, filled: s.Filled,
			notional: s.Notional, fees: s.Fees, arrival: s.Arrival, seq: s.Seq,
		}
	}
}

// venueState is what a poll found about one order at the venue, in venue-neutral terms.
type venueState struct {
	// Cumulative filled quantity and the volume-weighted average price of those fills.
	Filled   float64
	AvgPrice float64
	Fees     float64
	Currency string
	// Terminal states.
	Canceled bool
	Rejected bool
	Reason   string
	At       time.Time
}

// reconcile turns a venue's view of one order into the events that have not been published.
//
// The increment rule: an event is emitted for the *new* quantity only, priced at the average of the
// new fills (backed out of the cumulative average), so a venue that reports the same cumulative
// fill twice produces one event. A terminal state closes the order.
func (b *liveBook) reconcile(o *liveOrder, st venueState) []Published {
	b.mu.Lock()
	defer b.mu.Unlock()
	at := st.At
	if at.IsZero() {
		at = b.now()
	}
	var out []Published
	if st.Filled > o.filled+1e-12 {
		newQty := st.Filled - o.filled
		newNotional := st.Filled*st.AvgPrice - o.notional
		price := st.AvgPrice
		if newQty > 0 && newNotional > 0 {
			price = newNotional / newQty
		}
		fee := st.Fees - o.fees
		if fee < 0 {
			fee = 0
		}
		o.seq++
		kind := fav1.OrderEvent_KIND_PARTIAL_FILL
		if st.Filled >= o.quantity-1e-9 {
			kind = fav1.OrderEvent_KIND_FILL
			o.done = true
		}
		ev := &fav1.OrderEvent{
			IntentId: o.intentID, VenueOrderId: o.venueID, Kind: kind,
			FillPrice: price, FillQuantity: newQty, Fee: fee, FeeCurrency: st.Currency,
			EventTime: timestamppb.New(at), ArrivalPrice: o.arrival,
			SlippageBps:  slippageBps(o.side, o.arrival, price),
			VenueFillId:  fmt.Sprintf("%s-%s-%d", b.venue, o.venueID, o.seq),
			InstrumentId: o.instrument, VenueSymbol: o.symbol,
		}
		o.filled, o.notional, o.fees = st.Filled, st.Filled*st.AvgPrice, st.Fees
		out = append(out, Published{Account: o.account, Event: ev})
	}
	switch {
	case st.Rejected && !o.done:
		o.done = true
		out = append(out, Published{Account: o.account, Event: &fav1.OrderEvent{
			IntentId: o.intentID, VenueOrderId: o.venueID, Kind: fav1.OrderEvent_KIND_REJECTED,
			Reason: st.Reason, EventTime: timestamppb.New(at), InstrumentId: o.instrument,
			VenueSymbol: o.symbol, ArrivalPrice: o.arrival,
		}})
	case st.Canceled && !o.done:
		o.done = true
		reason := st.Reason
		if reason == "" {
			reason = fmt.Sprintf("cancelled at the venue; filled %g of %g", o.filled, o.quantity)
		}
		out = append(out, Published{Account: o.account, Event: &fav1.OrderEvent{
			IntentId: o.intentID, VenueOrderId: o.venueID, Kind: fav1.OrderEvent_KIND_CANCELED,
			Reason: reason, EventTime: timestamppb.New(at), InstrumentId: o.instrument,
			VenueSymbol: o.symbol, ArrivalPrice: o.arrival,
		}})
	}
	if o.done {
		delete(b.orders, o.intentID)
	}
	if len(out) > 0 {
		b.rev++
	}
	return out
}

// slippageBps is the signed cost against the taker, in basis points of the arrival price — the
// same convention the paper engine uses, so execution quality is comparable across venues.
func slippageBps(side fav1.Side, arrival, fill float64) float64 {
	if arrival <= 0 || fill <= 0 || math.IsNaN(arrival) || math.IsNaN(fill) {
		return 0
	}
	s := (fill - arrival) / arrival * 10_000
	if side == fav1.Side_SIDE_SELL {
		s = -s
	}
	return s
}

// ackEvent is the acknowledgement a live venue's submit produces.
func ackEvent(o *liveOrder, at time.Time, reason string) *fav1.OrderEvent {
	return &fav1.OrderEvent{
		IntentId: o.intentID, VenueOrderId: o.venueID, Kind: fav1.OrderEvent_KIND_ACK,
		Reason: reason, EventTime: timestamppb.New(at), ArrivalPrice: o.arrival,
		InstrumentId: o.instrument, VenueSymbol: o.symbol,
	}
}

// rejectEvent is a venue refusing the order outright (a 4xx on submit).
func rejectEvent(in *fav1.OrderIntent, symbol string, at time.Time, reason string) *fav1.OrderEvent {
	return &fav1.OrderEvent{
		IntentId: in.IntentId, Kind: fav1.OrderEvent_KIND_REJECTED, Reason: reason,
		EventTime: timestamppb.New(at), InstrumentId: in.InstrumentId, VenueSymbol: symbol,
	}
}

// clientOrderID is the venue-facing idempotency key: the intent id, which is already a ULID and
// already the platform's idempotency key (spec 42). Sending the same intent twice therefore cannot
// create two orders at a venue that honours the field.
func clientOrderID(intentID string) string {
	return intentID
}

// decodeJSON reads a venue response, keeping the body in the error so a shape change is debuggable
// without re-running it.
func decodeJSON(body []byte, into any) error {
	if err := json.Unmarshal(body, into); err != nil {
		snippet := string(body)
		if len(snippet) > 200 {
			snippet = snippet[:200] + "…"
		}
		return fmt.Errorf("decoding venue response: %w (%s)", err, snippet)
	}
	return nil
}

// asVenueError recognises a venue *refusing* something, as opposed to the network failing.
//
// This distinction decides whether a strategy learns that its order did not happen: a refusal
// becomes a REJECTED event, a transport failure is retried. The shared `connector.Fetcher` already
// turns a non-2xx into `*connector.HTTPError` after its retries, so both shapes arrive here and
// both mean "the venue answered, and the answer was no".
func asVenueError(venue string, err error) (*venueError, bool) {
	var ve *venueError
	if errors.As(err, &ve) {
		return ve, true
	}
	var he *connector.HTTPError
	if errors.As(err, &he) {
		// 5xx is the venue failing, not refusing: retrying that is right, rejecting the order is not.
		if he.Status >= 500 {
			return nil, false
		}
		return &venueError{venue: venue, detail: he.Error()}, true
	}
	return nil, false
}

// httpStatusError reports a non-2xx as something an operator can read.
func httpStatusError(venue string, resp *http.Response, body []byte) error {
	snippet := strings.TrimSpace(string(body))
	if len(snippet) > 300 {
		snippet = snippet[:300] + "…"
	}
	return &venueError{venue: venue, detail: fmt.Sprintf("HTTP %d: %s", resp.StatusCode, snippet)}
}
