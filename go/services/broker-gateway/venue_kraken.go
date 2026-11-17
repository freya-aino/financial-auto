// The Kraken execution adapter (spec 40, D-121).
//
// Implemented against Kraken's published REST reference (https://docs.kraken.com/api/):
//
//	POST /0/private/AddOrder      {ordertype, type, pair, volume, price?, cl_ord_id}  → txid
//	POST /0/private/CancelOrder   {txid}
//	POST /0/private/QueryOrders   {txid}  → status, vol, vol_exec, price, cost, fee
//	POST /0/private/Balance                → per-currency balances
//
// Auth is Kraken's HMAC scheme: `API-Sign` is
// base64(HMAC-SHA512(base64decode(secret), path + SHA256(nonce + urlencoded body))), with the key
// in `API-Key`. The nonce must increase across requests of one key, so it is a microsecond clock
// that is also monotonic within the process.
//
// `cl_ord_id` carries the intent id, which makes a resubmitted intent idempotent **at the venue**:
// Kraken refuses a second order with the same client id rather than opening one.
//
// Status: implemented against the reference and verified by `venue_conformance_test.go` against
// reference fixtures. **Not verified against the live venue** — that needs the owner's key, and is
// the last step of this work.
package main

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/base64"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const krakenVenue = "kraken"

// KrakenDefaultBaseURL is the production REST host; the conformance suite points at a stub.
const KrakenDefaultBaseURL = "https://api.kraken.com"

// Kraken is the Kraken execution adapter.
type Kraken struct {
	BaseURL string
	Creds   VenueCreds
	Fetch   *connector.Fetcher
	// Symbols maps internal instrument ids to Kraken pairs. Without an entry the adapter refuses
	// the order: guessing a pair from an instrument id is how an order ends up on the wrong asset.
	Symbols map[string]string

	book  *liveBook
	nonce atomic.Int64
	mu    sync.Mutex
	now   func() time.Time
}

func NewKraken(creds VenueCreds, lim connector.RateLimiter) *Kraken {
	k := &Kraken{
		BaseURL: KrakenDefaultBaseURL, Creds: creds, Fetch: connector.NewFetcher(lim),
		Symbols: map[string]string{}, book: newLiveBook(krakenVenue), now: time.Now,
	}
	k.nonce.Store(time.Now().UnixMicro())
	return k
}

func (k *Kraken) Venue() string { return krakenVenue }

func (k *Kraken) Ready() error {
	var missing []string
	if k.Creds.Key == "" {
		missing = append(missing, "api_key")
	}
	if k.Creds.Secret == "" {
		missing = append(missing, "api_secret")
	}
	if len(missing) > 0 {
		return missingCreds(krakenVenue, missing...)
	}
	// A secret that is not base64 cannot sign anything, and finding that out on the first real
	// order is worse than finding it out when the switch is thrown.
	if _, err := base64.StdEncoding.DecodeString(k.Creds.Secret); err != nil {
		return &venueError{venue: krakenVenue, detail: "api_secret is not base64 (Kraken's secrets are)"}
	}
	return nil
}

func (k *Kraken) Holds(intentID string) bool { return k.book.Holds(intentID) }
func (k *Kraken) OpenOrders() int            { return k.book.OpenOrders() }

func (k *Kraken) Symbol(instrumentID string) (string, bool) {
	s, ok := k.Symbols[instrumentID]
	return s, ok
}

// sign builds the signed request for one private endpoint.
func (k *Kraken) sign(path string, form map[string]string) (*http.Request, error) {
	secret, err := base64.StdEncoding.DecodeString(k.Creds.Secret)
	if err != nil {
		return nil, &venueError{venue: krakenVenue, detail: "api_secret is not base64"}
	}
	// A nonce that is both wall-clock and monotonic: restarts move forward, and two requests in
	// the same microsecond still differ.
	nonce := strconv.FormatInt(k.nonce.Add(1), 10)
	// Sorted, so the body a test signs is the body it sends: Kraken signs the exact bytes, and a
	// map iteration order would make the signature irreproducible.
	keys := make([]string, 0, len(form))
	for key := range form {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	values := make([]string, 0, len(form)+1)
	values = append(values, "nonce="+nonce)
	for _, key := range keys {
		values = append(values, key+"="+url.QueryEscape(form[key]))
	}
	body := strings.Join(values, "&")
	sha := sha256.Sum256([]byte(nonce + body))
	mac := hmac.New(sha512.New, secret)
	mac.Write([]byte(path))
	mac.Write(sha[:])
	req, err := http.NewRequest(http.MethodPost, strings.TrimSuffix(k.BaseURL, "/")+path, strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("API-Key", k.Creds.Key)
	req.Header.Set("API-Sign", base64.StdEncoding.EncodeToString(mac.Sum(nil)))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", connector.UserAgent)
	return req, nil
}

// krakenReply is the envelope every Kraken endpoint answers with: errors are in the body at 200.
type krakenReply struct {
	Error  []string       `json:"error"`
	Result map[string]any `json:"result"`
}

func (k *Kraken) call(ctx context.Context, path string, form map[string]string) (map[string]any, error) {
	if err := k.Ready(); err != nil {
		return nil, err
	}
	k.mu.Lock()
	req, err := k.sign(path, form)
	k.mu.Unlock()
	if err != nil {
		return nil, err
	}
	body, resp, err := k.Fetch.Do(ctx, req.WithContext(ctx))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		return nil, httpStatusError(krakenVenue, resp, body)
	}
	var out krakenReply
	if err := decodeJSON(body, &out); err != nil {
		return nil, err
	}
	// Kraken reports business errors at HTTP 200, so this is the real error channel.
	if len(out.Error) > 0 {
		return nil, &venueError{venue: krakenVenue, detail: strings.Join(out.Error, "; ")}
	}
	return out.Result, nil
}

func (k *Kraken) Submit(ctx context.Context, in *fav1.OrderIntent) ([]*fav1.OrderEvent, error) {
	pair, ok := k.Symbol(in.InstrumentId)
	if !ok {
		return []*fav1.OrderEvent{rejectEvent(in, "", k.now(), "kraken: no pair mapped for this instrument")}, nil
	}
	side := "buy"
	if in.Side == fav1.Side_SIDE_SELL {
		side = "sell"
	}
	form := map[string]string{
		"pair":      pair,
		"type":      side,
		"volume":    strconv.FormatFloat(in.Quantity, 'f', -1, 64),
		"cl_ord_id": clientOrderID(in.IntentId),
		"ordertype": "market",
	}
	if in.LimitPrice != nil && *in.LimitPrice > 0 {
		form["ordertype"] = "limit"
		form["price"] = strconv.FormatFloat(*in.LimitPrice, 'f', -1, 64)
	}
	result, err := k.call(ctx, "/0/private/AddOrder", form)
	if err != nil {
		// A venue refusing the order is an event, not a transport failure: the strategy that asked
		// for it must learn that it did not happen.
		if ve, refused := asVenueError(krakenVenue, err); refused {
			return []*fav1.OrderEvent{rejectEvent(in, pair, k.now(), ve.Error())}, nil
		}
		return nil, err
	}
	txids, _ := result["txid"].([]any)
	if len(txids) == 0 {
		return []*fav1.OrderEvent{rejectEvent(in, pair, k.now(), "kraken: no txid in the AddOrder reply")}, nil
	}
	txid, _ := txids[0].(string)
	o := &liveOrder{
		intentID: in.IntentId, venueID: txid, account: in.AccountId, instrument: in.InstrumentId,
		symbol: pair, side: in.Side, quantity: in.Quantity, arrival: limitOrZero(in),
	}
	k.book.put(o)
	return []*fav1.OrderEvent{ackEvent(o, k.now(), "accepted by kraken as "+txid)}, nil
}

func (k *Kraken) Cancel(ctx context.Context, intentID string) error {
	o, ok := k.book.get(intentID)
	if !ok {
		return errNotResting
	}
	_, err := k.call(ctx, "/0/private/CancelOrder", map[string]string{"txid": o.venueID})
	return err
}

// Poll asks Kraken for every open order of this adapter in one QueryOrders call.
func (k *Kraken) Poll(ctx context.Context) ([]Published, error) {
	open := k.book.open()
	if len(open) == 0 {
		return nil, nil
	}
	ids := make([]string, 0, len(open))
	for _, o := range open {
		ids = append(ids, o.venueID)
	}
	// QueryOrders takes up to 50 ids; more than that is split across polls rather than truncated.
	if len(ids) > 50 {
		ids, open = ids[:50], open[:50]
	}
	result, err := k.call(ctx, "/0/private/QueryOrders", map[string]string{"txid": strings.Join(ids, ",")})
	if err != nil {
		return nil, err
	}
	var out []Published
	for _, o := range open {
		raw, ok := result[o.venueID].(map[string]any)
		if !ok {
			continue
		}
		out = append(out, k.book.reconcile(o, krakenState(raw, k.now()))...)
	}
	return out, nil
}

// krakenState reads one QueryOrders entry.
func krakenState(raw map[string]any, now time.Time) venueState {
	st := venueState{
		Filled:   numOf(raw["vol_exec"]),
		AvgPrice: numOf(raw["price"]),
		Fees:     numOf(raw["fee"]),
		Currency: "EUR", // Kraken charges the fee in the pair's quote currency; EUR for the *EUR pairs traded here.
		At:       now,
	}
	// `price` is 0 for an unfilled order and the average fill price once there is volume; `cost`
	// divided by volume is the same number and is used when `price` is absent.
	if st.Filled > 0 && st.AvgPrice == 0 {
		if cost := numOf(raw["cost"]); cost > 0 {
			st.AvgPrice = cost / st.Filled
		}
	}
	switch s, _ := raw["status"].(string); s {
	case "canceled", "expired":
		st.Canceled = true
		st.Reason, _ = raw["reason"].(string)
	case "closed":
		// Terminal with everything filled; `reconcile` turns that into KIND_FILL.
	}
	if r, ok := raw["reason"].(string); ok && strings.Contains(strings.ToLower(r), "insufficient") {
		st.Rejected, st.Reason = true, r
	}
	return st
}

func (k *Kraken) Balances(ctx context.Context) ([]VenueBalance, error) {
	result, err := k.call(ctx, "/0/private/Balance", nil)
	if err != nil {
		return nil, err
	}
	out := make([]VenueBalance, 0, len(result))
	for cur, v := range result {
		amount := numOf(v)
		if amount == 0 {
			continue
		}
		out = append(out, VenueBalance{Venue: krakenVenue, Currency: krakenCurrency(cur), Amount: amount})
	}
	return out, nil
}

// Positions: Kraken spot has no positions, only balances — a holding of XBT *is* the position, and
// reporting it twice (here and as a balance) would double it in reconciliation. Margin positions
// (`OpenPositions`) are not traded by this platform.
func (k *Kraken) Positions(context.Context) ([]VenuePosition, error) { return nil, nil }

// krakenCurrency strips Kraken's X/Z asset-class prefixes (ZEUR → EUR, XXBT → XBT).
func krakenCurrency(code string) string {
	if len(code) == 4 && (code[0] == 'X' || code[0] == 'Z') {
		return code[1:]
	}
	return code
}

// numOf reads a venue number that may arrive as a JSON number or a quoted string (Kraken quotes
// every amount; most venues do not).
func numOf(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case string:
		f, err := strconv.ParseFloat(x, 64)
		if err != nil {
			return 0
		}
		return f
	case int:
		return float64(x)
	}
	return 0
}

// limitOrZero is the arrival price a live order is measured against: its limit when it has one,
// and 0 for a market order, where the venue's own fill is the only price this adapter sees.
func limitOrZero(in *fav1.OrderIntent) float64 {
	if in.LimitPrice != nil {
		return *in.LimitPrice
	}
	return 0
}

// Snapshot and Restore persist this adapter's open orders across a restart (see LiveVenue).
func (k *Kraken) Snapshot() []LiveOrderState  { return k.book.Snapshot() }
func (k *Kraken) Restore(os []LiveOrderState) { k.book.Restore(os) }

func (k *Kraken) Revision() uint64 { return k.book.Revision() }

var _ LiveVenue = (*Kraken)(nil)
