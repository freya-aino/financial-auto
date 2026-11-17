// The OANDA v20 execution adapter (spec 40, D-121).
//
// Implemented against the published v20 reference (https://developer.oanda.com/rest-live-v20/):
//
//	POST /v3/accounts/{id}/orders                 {order: {type, instrument, units, …}}
//	GET  /v3/accounts/{id}/orders/{specifier}     → state PENDING | FILLED | CANCELLED
//	PUT  /v3/accounts/{id}/orders/{specifier}/cancel
//	GET  /v3/accounts/{id}/trades/{tradeID}       → the fill's price and units
//	GET  /v3/accounts/{id}/openPositions, /summary
//
// Two v20 specifics:
//
//   - **Units are signed**, not sided: a sell is negative units. The side is therefore encoded in
//     the number and must be read back out of it when reconciling.
//   - **A market order fills inside its own submit response** (`orderFillTransaction`), so this
//     adapter publishes the fill immediately instead of waiting for a poll. A limit order is
//     acknowledged and polled, and the fill price comes from the trade the order opened.
//
// `clientExtensions.id` carries the intent id, which is what makes a resubmitted intent idempotent
// at the venue.
//
// Status: implemented against the reference and verified by `venue_conformance_test.go` against
// reference fixtures. **Not verified against the live venue** (no key).
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const oandaVenue = "oanda"

// OandaHosts are the practice and production REST hosts. `env` in the credentials picks one;
// "practice" is the default, because an adapter that defaults to real money is a mistake waiting
// for a missing config line.
var OandaHosts = map[string]string{
	"practice": "https://api-fxpractice.oanda.com",
	"live":     "https://api-fxtrade.oanda.com",
}

// Oanda is the OANDA v20 execution adapter.
type Oanda struct {
	BaseURL string
	Creds   VenueCreds
	Fetch   *connector.Fetcher
	// Symbols maps internal instrument ids to v20 instrument names (EUR_USD).
	Symbols map[string]string

	book *liveBook
	now  func() time.Time
}

func NewOanda(creds VenueCreds, lim connector.RateLimiter) *Oanda {
	env := creds.Env
	if env == "" {
		env = "practice"
	}
	host, ok := OandaHosts[env]
	if !ok {
		host = OandaHosts["practice"]
	}
	return &Oanda{
		BaseURL: host, Creds: creds, Fetch: connector.NewFetcher(lim), Symbols: map[string]string{},
		book: newLiveBook(oandaVenue), now: time.Now,
	}
}

func (o *Oanda) Venue() string { return oandaVenue }

func (o *Oanda) Ready() error {
	var missing []string
	if o.Creds.Key == "" {
		missing = append(missing, "api_key")
	}
	if o.Creds.Account == "" {
		missing = append(missing, "account_id")
	}
	if len(missing) > 0 {
		return missingCreds(oandaVenue, missing...)
	}
	if o.Creds.Env == "live" && !strings.Contains(o.BaseURL, "fxtrade") {
		return &venueError{venue: oandaVenue, detail: "env is `live` but the host is the practice one"}
	}
	return nil
}

func (o *Oanda) Holds(intentID string) bool { return o.book.Holds(intentID) }
func (o *Oanda) OpenOrders() int            { return o.book.OpenOrders() }

func (o *Oanda) Symbol(instrumentID string) (string, bool) {
	s, ok := o.Symbols[instrumentID]
	return s, ok
}

func (o *Oanda) Instrument(symbol string) (string, bool) { return instrumentOf(o.Symbols, symbol) }

func (o *Oanda) path(suffix string) string {
	return strings.TrimSuffix(o.BaseURL, "/") + "/v3/accounts/" + o.Creds.Account + suffix
}

func (o *Oanda) do(ctx context.Context, method, path string, body any) ([]byte, error) {
	if err := o.Ready(); err != nil {
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
	req, err := http.NewRequestWithContext(ctx, method, path, rdr)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+o.Creds.Key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept-Datetime-Format", "RFC3339")
	req.Header.Set("User-Agent", connector.UserAgent)
	out, resp, err := o.Fetch.Do(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		return nil, httpStatusError(oandaVenue, resp, out)
	}
	return out, nil
}

// oandaSubmitReply is the subset of the submit response this adapter reads.
type oandaSubmitReply struct {
	OrderCreateTransaction struct {
		ID string `json:"id"`
	} `json:"orderCreateTransaction"`
	OrderFillTransaction struct {
		ID         string `json:"id"`
		OrderID    string `json:"orderID"`
		Instrument string `json:"instrument"`
		Units      string `json:"units"`
		Price      string `json:"price"`
		Commission string `json:"commission"`
		Time       string `json:"time"`
		AccountID  string `json:"accountID"`
	} `json:"orderFillTransaction"`
	OrderRejectTransaction struct {
		ID           string `json:"id"`
		RejectReason string `json:"rejectReason"`
	} `json:"orderRejectTransaction"`
	ErrorMessage string `json:"errorMessage"`
}

func (o *Oanda) Submit(ctx context.Context, in *fav1.OrderIntent) ([]*fav1.OrderEvent, error) {
	sym, ok := o.Symbol(in.InstrumentId)
	if !ok {
		return []*fav1.OrderEvent{rejectEvent(in, "", o.now(), "oanda: no v20 instrument mapped")}, nil
	}
	units := in.Quantity
	if in.Side == fav1.Side_SIDE_SELL {
		units = -units
	}
	order := map[string]any{
		"type":             "MARKET",
		"instrument":       sym,
		"units":            strconv.FormatFloat(units, 'f', -1, 64),
		"timeInForce":      "FOK",
		"positionFill":     "DEFAULT",
		"clientExtensions": map[string]string{"id": clientOrderID(in.IntentId)},
	}
	if in.LimitPrice != nil && *in.LimitPrice > 0 {
		order["type"] = "LIMIT"
		order["price"] = strconv.FormatFloat(*in.LimitPrice, 'f', -1, 64)
		order["timeInForce"] = "GTC"
	}
	raw, err := o.do(ctx, http.MethodPost, o.path("/orders"), map[string]any{"order": order})
	if err != nil {
		if ve, refused := asVenueError(oandaVenue, err); refused {
			return []*fav1.OrderEvent{rejectEvent(in, sym, o.now(), ve.Error())}, nil
		}
		return nil, err
	}
	var reply oandaSubmitReply
	if err := decodeJSON(raw, &reply); err != nil {
		return nil, err
	}
	if r := reply.OrderRejectTransaction.RejectReason; r != "" {
		return []*fav1.OrderEvent{rejectEvent(in, sym, o.now(), "oanda rejected the order: "+r)}, nil
	}
	rec := &liveOrder{
		intentID: in.IntentId, account: in.AccountId, instrument: in.InstrumentId, symbol: sym,
		side: in.Side, quantity: in.Quantity, arrival: limitOrZero(in),
	}
	// A market order is already filled in this response; publish the fill rather than waiting a
	// tick for a poll to notice.
	if f := reply.OrderFillTransaction; f.ID != "" {
		rec.venueID = firstNonEmpty(f.OrderID, reply.OrderCreateTransaction.ID, f.ID)
		at := o.now()
		if t, err := time.Parse(time.RFC3339, f.Time); err == nil {
			at = t
		}
		price := mustFloat(f.Price)
		filled := absFloat(mustFloat(f.Units))
		kind := fav1.OrderEvent_KIND_FILL
		if filled+1e-9 < in.Quantity {
			kind = fav1.OrderEvent_KIND_PARTIAL_FILL
			rec.filled, rec.notional = filled, filled*price
			rec.venueID = firstNonEmpty(reply.OrderCreateTransaction.ID, f.OrderID)
			o.book.put(rec)
		}
		return []*fav1.OrderEvent{{
			IntentId: in.IntentId, VenueOrderId: rec.venueID, Kind: kind,
			FillPrice: price, FillQuantity: filled, Fee: absFloat(mustFloat(f.Commission)),
			FeeCurrency: "EUR", EventTime: timestamppb.New(at), ArrivalPrice: rec.arrival,
			SlippageBps: slippageBps(in.Side, rec.arrival, price),
			VenueFillId: oandaVenue + "-" + f.ID, InstrumentId: in.InstrumentId, VenueSymbol: sym,
		}}, nil
	}
	rec.venueID = reply.OrderCreateTransaction.ID
	if rec.venueID == "" {
		return []*fav1.OrderEvent{rejectEvent(in, sym, o.now(),
			firstNonEmpty(reply.ErrorMessage, "oanda: neither a fill nor an order id in the reply"))}, nil
	}
	o.book.put(rec)
	return []*fav1.OrderEvent{ackEvent(rec, o.now(), "accepted by oanda as "+rec.venueID)}, nil
}

func (o *Oanda) Cancel(ctx context.Context, intentID string) error {
	rec, ok := o.book.get(intentID)
	if !ok {
		return errNotResting
	}
	_, err := o.do(ctx, http.MethodPut, o.path("/orders/"+rec.venueID+"/cancel"), map[string]any{})
	return err
}

// Poll reads each open order, and for a filled one the trade it opened — v20 keeps the price on the
// trade, not on the order.
func (o *Oanda) Poll(ctx context.Context) ([]Published, error) {
	var out []Published
	var firstErr error
	for _, rec := range o.book.open() {
		raw, err := o.do(ctx, http.MethodGet, o.path("/orders/"+rec.venueID), nil)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		var got struct {
			Order struct {
				State         string `json:"state"`
				Units         string `json:"units"`
				TradeOpenedID string `json:"tradeOpenedID"`
				FilledTime    string `json:"filledTime"`
				CancelledTime string `json:"cancelledTime"`
			} `json:"order"`
		}
		if err := decodeJSON(raw, &got); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		st := venueState{Currency: "EUR", At: o.now()}
		switch strings.ToUpper(got.Order.State) {
		case "FILLED":
			trade, terr := o.trade(ctx, got.Order.TradeOpenedID)
			if terr != nil {
				if firstErr == nil {
					firstErr = terr
				}
				continue
			}
			st.Filled, st.AvgPrice, st.Fees = trade.units, trade.price, trade.fees
			if t, err := time.Parse(time.RFC3339, got.Order.FilledTime); err == nil {
				st.At = t
			}
		case "CANCELLED":
			st.Canceled = true
			if t, err := time.Parse(time.RFC3339, got.Order.CancelledTime); err == nil {
				st.At = t
			}
		case "TRIGGERED", "PENDING":
			continue
		}
		out = append(out, o.book.reconcile(rec, st)...)
	}
	if len(out) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

type oandaTrade struct {
	units, price, fees float64
}

func (o *Oanda) trade(ctx context.Context, id string) (oandaTrade, error) {
	if id == "" {
		return oandaTrade{}, &venueError{venue: oandaVenue, detail: "a filled order with no trade id"}
	}
	raw, err := o.do(ctx, http.MethodGet, o.path("/trades/"+id), nil)
	if err != nil {
		return oandaTrade{}, err
	}
	var got struct {
		Trade struct {
			InitialUnits string `json:"initialUnits"`
			Price        string `json:"price"`
			Financing    string `json:"financing"`
		} `json:"trade"`
	}
	if err := decodeJSON(raw, &got); err != nil {
		return oandaTrade{}, err
	}
	return oandaTrade{
		units: absFloat(mustFloat(got.Trade.InitialUnits)),
		price: mustFloat(got.Trade.Price),
		fees:  absFloat(mustFloat(got.Trade.Financing)),
	}, nil
}

func (o *Oanda) Positions(ctx context.Context) ([]VenuePosition, error) {
	raw, err := o.do(ctx, http.MethodGet, o.path("/openPositions"), nil)
	if err != nil {
		return nil, err
	}
	var got struct {
		Positions []struct {
			Instrument string `json:"instrument"`
			Long       struct {
				Units        string `json:"units"`
				AveragePrice string `json:"averagePrice"`
			} `json:"long"`
			Short struct {
				Units        string `json:"units"`
				AveragePrice string `json:"averagePrice"`
			} `json:"short"`
		} `json:"positions"`
	}
	if err := decodeJSON(raw, &got); err != nil {
		return nil, err
	}
	var out []VenuePosition
	for _, p := range got.Positions {
		// v20 reports the long and short legs separately; the platform holds one net position per
		// instrument, so they are netted here and the average price follows the surviving side.
		long, short := mustFloat(p.Long.Units), mustFloat(p.Short.Units)
		net := long + short // short units are negative
		if net == 0 {
			continue
		}
		avg := mustFloat(p.Long.AveragePrice)
		if net < 0 {
			avg = mustFloat(p.Short.AveragePrice)
		}
		out = append(out, VenuePosition{Venue: oandaVenue, Symbol: p.Instrument, Quantity: net, AvgPrice: avg})
	}
	return out, nil
}

func (o *Oanda) Balances(ctx context.Context) ([]VenueBalance, error) {
	raw, err := o.do(ctx, http.MethodGet, o.path("/summary"), nil)
	if err != nil {
		return nil, err
	}
	var got struct {
		Account struct {
			Balance  string `json:"balance"`
			Currency string `json:"currency"`
		} `json:"account"`
	}
	if err := decodeJSON(raw, &got); err != nil {
		return nil, err
	}
	cur := strings.ToUpper(got.Account.Currency)
	if cur == "" {
		cur = "EUR"
	}
	return []VenueBalance{{Venue: oandaVenue, Currency: cur, Amount: mustFloat(got.Account.Balance)}}, nil
}

// mustFloat parses a v20 decimal string; v20 sends every number as a string, and an unparsable one
// is 0 rather than an error because the caller's own emptiness checks are the real guard.
func mustFloat(s string) float64 {
	f, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0
	}
	return f
}

func absFloat(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

// Snapshot and Restore persist this adapter's open orders across a restart (see LiveVenue).
func (o *Oanda) Snapshot() []LiveOrderState  { return o.book.Snapshot() }
func (o *Oanda) Restore(os []LiveOrderState) { o.book.Restore(os) }

func (o *Oanda) Revision() uint64 { return o.book.Revision() }

var _ LiveVenue = (*Oanda)(nil)
