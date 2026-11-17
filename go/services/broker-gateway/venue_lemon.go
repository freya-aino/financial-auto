// The lemon.markets execution adapter (spec 40, D-121).
//
// Implemented against the published trading reference (https://docs.lemon.markets):
//
//	POST   /v1/orders               {isin, side, quantity, venue, expires_at} → order id, status "inactive"
//	POST   /v1/orders/{id}/activate                                           → the order goes live
//	GET    /v1/orders/{id}          → status, executed_quantity, executed_price, executed_at
//	DELETE /v1/orders/{id}          → cancel
//	GET    /v1/positions            → per-ISIN quantity and average buy price
//	GET    /v1/account              → cash balance
//
// Two things are specific to lemon and easy to get wrong:
//
//   - **An order is created inactive and must be activated.** A created-but-not-activated order is
//     not at the market, so this adapter treats a failed activation as a rejection rather than an
//     acknowledgement — otherwise the platform would believe it has a working order that nobody
//     will ever fill.
//   - **Every monetary value is an integer number of cents.** Prices are divided by 100 on the way
//     in; an adapter that forgot this would report fills at 100× the price, which is exactly the
//     sort of error the shared conformance suite exists to catch.
//
// Status: implemented against the reference and verified by `venue_conformance_test.go` against
// reference fixtures. **Not verified against the live venue** (no key).
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const lemonVenue = "lemon"

// LemonDefaultBaseURL is the production trading host (the paper host is `paper-trading…`, which is
// deliberately not used: paper trading in this platform is the paper engine, not a broker's
// sandbox, so that one execution path is exercised rather than two).
const LemonDefaultBaseURL = "https://trading.lemon.markets/v1"

// LemonOrderTTL is how long a created order stays valid at the venue.
const LemonOrderTTL = 24 * time.Hour

// Lemon is the lemon.markets execution adapter.
type Lemon struct {
	BaseURL string
	Creds   VenueCreds
	Fetch   *connector.Fetcher
	// MIC of the exchange to route to (lemon's `venue`).
	MIC string
	// Symbols maps internal instrument ids to ISINs.
	Symbols map[string]string

	book *liveBook
	now  func() time.Time
}

func NewLemon(creds VenueCreds, lim connector.RateLimiter) *Lemon {
	return &Lemon{
		BaseURL: LemonDefaultBaseURL, Creds: creds, Fetch: connector.NewFetcher(lim), MIC: "XMUN",
		Symbols: map[string]string{}, book: newLiveBook(lemonVenue), now: time.Now,
	}
}

func (l *Lemon) Venue() string { return lemonVenue }

func (l *Lemon) Ready() error {
	if l.Creds.Key == "" {
		return missingCreds(lemonVenue, "api_key")
	}
	return nil
}

func (l *Lemon) Holds(intentID string) bool { return l.book.Holds(intentID) }
func (l *Lemon) OpenOrders() int            { return l.book.OpenOrders() }

func (l *Lemon) Symbol(instrumentID string) (string, bool) {
	s, ok := l.Symbols[instrumentID]
	return s, ok
}

func (l *Lemon) Instrument(symbol string) (string, bool) { return instrumentOf(l.Symbols, symbol) }

func (l *Lemon) do(ctx context.Context, method, path string, body any) ([]byte, error) {
	if err := l.Ready(); err != nil {
		return nil, err
	}
	var rdr *bytes.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		rdr = bytes.NewReader(b)
	} else {
		rdr = bytes.NewReader(nil)
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimSuffix(l.BaseURL, "/")+path, rdr)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+l.Creds.Key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", connector.UserAgent)
	out, resp, err := l.Fetch.Do(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		return nil, httpStatusError(lemonVenue, resp, out)
	}
	return out, nil
}

// lemonOrder is the subset of lemon's order object this adapter reads.
type lemonOrder struct {
	Status  string `json:"status"`
	Results struct {
		ID               string  `json:"id"`
		Status           string  `json:"status"`
		ISIN             string  `json:"isin"`
		Quantity         float64 `json:"quantity"`
		ExecutedQuantity float64 `json:"executed_quantity"`
		ExecutedPrice    float64 `json:"executed_price"`
		ExecutedAt       string  `json:"executed_at"`
		EstimatedPrice   float64 `json:"estimated_price"`
		ChargeableAt     string  `json:"chargeable_at"`
		Charge           float64 `json:"charge"`
		RegulatoryFee    float64 `json:"regulatory_fee"`
		Currency         string  `json:"currency"`
		Error            string  `json:"error_message"`
	} `json:"results"`
}

func (l *Lemon) Submit(ctx context.Context, in *fav1.OrderIntent) ([]*fav1.OrderEvent, error) {
	isin, ok := l.Symbol(in.InstrumentId)
	if !ok {
		return []*fav1.OrderEvent{rejectEvent(in, "", l.now(), "lemon: no ISIN mapped for this instrument")}, nil
	}
	side := "buy"
	if in.Side == fav1.Side_SIDE_SELL {
		side = "sell"
	}
	// lemon trades whole shares; a fractional intent would be silently floored at the venue.
	if in.Quantity != float64(int64(in.Quantity)) {
		return []*fav1.OrderEvent{rejectEvent(in, isin, l.now(),
			fmt.Sprintf("lemon: whole shares only, got %g", in.Quantity))}, nil
	}
	body := map[string]any{
		"isin":       isin,
		"side":       side,
		"quantity":   int64(in.Quantity),
		"venue":      l.MIC,
		"expires_at": l.now().Add(LemonOrderTTL).Format(time.RFC3339),
		// lemon echoes this back on every read of the order, which is how a restarted gateway can
		// re-attach an order to the intent that created it.
		"idempotency": clientOrderID(in.IntentId),
	}
	if in.LimitPrice != nil && *in.LimitPrice > 0 {
		body["stop_price"] = nil
		body["limit_price"] = int64(*in.LimitPrice * 100) // cents
	}
	raw, err := l.do(ctx, http.MethodPost, "/orders", body)
	if err != nil {
		if ve, refused := asVenueError(lemonVenue, err); refused {
			return []*fav1.OrderEvent{rejectEvent(in, isin, l.now(), ve.Error())}, nil
		}
		return nil, err
	}
	var created lemonOrder
	if err := decodeJSON(raw, &created); err != nil {
		return nil, err
	}
	if created.Results.ID == "" {
		return []*fav1.OrderEvent{rejectEvent(in, isin, l.now(), "lemon: no order id in the reply")}, nil
	}
	// An order lemon created but did not activate is not at the market: report the rejection.
	if _, err := l.do(ctx, http.MethodPost, "/orders/"+created.Results.ID+"/activate", map[string]any{}); err != nil {
		return []*fav1.OrderEvent{rejectEvent(in, isin, l.now(),
			"lemon: order created but not activated: "+err.Error())}, nil
	}
	o := &liveOrder{
		intentID: in.IntentId, venueID: created.Results.ID, account: in.AccountId,
		instrument: in.InstrumentId, symbol: isin, side: in.Side, quantity: in.Quantity,
		arrival: limitOrZero(in),
	}
	l.book.put(o)
	return []*fav1.OrderEvent{ackEvent(o, l.now(), "activated at lemon as "+created.Results.ID)}, nil
}

func (l *Lemon) Cancel(ctx context.Context, intentID string) error {
	o, ok := l.book.get(intentID)
	if !ok {
		return errNotResting
	}
	_, err := l.do(ctx, http.MethodDelete, "/orders/"+o.venueID, nil)
	return err
}

// Poll reads each open order; lemon has no batch order endpoint, so this is one call per order and
// the adapter is correspondingly conservative about how many it keeps open.
func (l *Lemon) Poll(ctx context.Context) ([]Published, error) {
	var out []Published
	var firstErr error
	for _, o := range l.book.open() {
		raw, err := l.do(ctx, http.MethodGet, "/orders/"+o.venueID, nil)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		var got lemonOrder
		if err := decodeJSON(raw, &got); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		out = append(out, l.book.reconcile(o, lemonState(got, l.now()))...)
	}
	if len(out) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

func lemonState(got lemonOrder, now time.Time) venueState {
	r := got.Results
	st := venueState{
		Filled:   r.ExecutedQuantity,
		AvgPrice: r.ExecutedPrice / 100, // cents
		Fees:     (r.Charge + r.RegulatoryFee) / 100,
		Currency: strings.ToUpper(r.Currency),
		At:       now,
	}
	if st.Currency == "" {
		st.Currency = "EUR"
	}
	if t, err := time.Parse(time.RFC3339, r.ExecutedAt); err == nil && !t.IsZero() {
		st.At = t
	}
	switch strings.ToLower(r.Status) {
	case "canceled", "cancelled", "expired":
		st.Canceled, st.Reason = true, r.Error
	case "rejected":
		st.Rejected, st.Reason = true, firstNonEmpty(r.Error, "rejected by lemon")
	}
	return st
}

func (l *Lemon) Positions(ctx context.Context) ([]VenuePosition, error) {
	raw, err := l.do(ctx, http.MethodGet, "/positions", nil)
	if err != nil {
		return nil, err
	}
	var got struct {
		Results []struct {
			ISIN        string  `json:"isin"`
			Quantity    float64 `json:"quantity"`
			BuyPriceAvg float64 `json:"buy_price_avg"`
		} `json:"results"`
	}
	if err := decodeJSON(raw, &got); err != nil {
		return nil, err
	}
	out := make([]VenuePosition, 0, len(got.Results))
	for _, p := range got.Results {
		out = append(out, VenuePosition{
			Venue: lemonVenue, Symbol: p.ISIN, Quantity: p.Quantity, AvgPrice: p.BuyPriceAvg / 100,
		})
	}
	return out, nil
}

func (l *Lemon) Balances(ctx context.Context) ([]VenueBalance, error) {
	raw, err := l.do(ctx, http.MethodGet, "/account", nil)
	if err != nil {
		return nil, err
	}
	var got struct {
		Results struct {
			Balance  float64 `json:"balance"`
			Currency string  `json:"currency"`
		} `json:"results"`
	}
	if err := decodeJSON(raw, &got); err != nil {
		return nil, err
	}
	cur := strings.ToUpper(got.Results.Currency)
	if cur == "" {
		cur = "EUR"
	}
	return []VenueBalance{{Venue: lemonVenue, Currency: cur, Amount: got.Results.Balance / 100}}, nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// Snapshot and Restore persist this adapter's open orders across a restart (see LiveVenue).
func (l *Lemon) Snapshot() []LiveOrderState  { return l.book.Snapshot() }
func (l *Lemon) Restore(os []LiveOrderState) { l.book.Restore(os) }

func (l *Lemon) Revision() uint64 { return l.book.Revision() }

var _ LiveVenue = (*Lemon)(nil)
