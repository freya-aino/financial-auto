package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// A live intent must never be filled on the paper book: that is the one failure mode of a single
// execution path, because the two ledgers would then disagree in the direction nobody checks.
func TestLiveIntentIsRejectedRatherThanPapered(t *testing.T) {
	engine := free()
	engine.ObserveTrade(tradeSub, 100, 1, t0)
	r := NewRouter(engine)
	r.VenueOf = engine.VenueOf

	in := intent("live-no-adapter")
	in.Mode = fav1.TradingMode_TRADING_MODE_LIVE
	out := r.Execute(context.Background(), in)
	if len(out) != 1 || out[0].Event.Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatalf("want one rejection, got %v", out)
	}
	if out[0].Event.Reason == "" {
		t.Fatal("a rejection must say why")
	}

	// With an adapter that has no credentials, the rejection names the venue's own complaint.
	k := NewKraken(VenueCreds{}, nil)
	r.Add(k)
	in2 := intent("live-unconfigured")
	in2.Mode = fav1.TradingMode_TRADING_MODE_LIVE
	out = r.Execute(context.Background(), in2)
	if len(out) != 1 || out[0].Event.Kind != fav1.OrderEvent_KIND_REJECTED {
		t.Fatalf("want one rejection, got %v", out)
	}
	// `sim` is the venue of the test tape, so the complaint is about the venue, not the key.
	if reason := out[0].Event.Reason; reason == "" {
		t.Fatal("empty reason")
	}

	// And a paper intent still fills, through the same router.
	out = r.Execute(context.Background(), intent("paper-ok"))
	if len(out) != 1 || out[0].Event.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("paper must still fill through the router: %v", out)
	}
}

// The readiness report is what the mode switch consults, so it must be honest about every venue
// and must not contain a credential.
func TestVenueReadinessIsReportedWithoutSecrets(t *testing.T) {
	r := NewRouter(free())
	r.Add(NewKraken(VenueCreds{Key: "k", Secret: "c2VjcmV0"}, nil)) // base64 "secret"
	r.Add(NewLemon(VenueCreds{}, nil))
	r.Add(NewOanda(VenueCreds{Key: "oanda-key", Account: "001-004-1-001"}, nil))

	status, possible := r.Ready()
	if !possible {
		t.Fatal("one configured venue makes live possible")
	}
	if len(status) != 3 {
		t.Fatalf("want all three venues reported, got %d", len(status))
	}
	byID := map[string]VenueStatus{}
	for _, s := range status {
		byID[s.Venue] = s
	}
	if !byID["kraken"].Configured || !byID["oanda"].Configured {
		t.Fatalf("configured venues reported as unconfigured: %+v", status)
	}
	if byID["lemon"].Configured || byID["lemon"].Detail == "" {
		t.Fatalf("lemon has no key and must say so: %+v", byID["lemon"])
	}
	raw, err := json.Marshal(status)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"oanda-key", "c2VjcmV0", "secret"} {
		if strings.Contains(string(raw), secret) {
			t.Fatalf("the readiness report leaked %q: %s", secret, raw)
		}
	}

	// A base64-invalid Kraken secret is caught before an order, not during one.
	bad := NewKraken(VenueCreds{Key: "k", Secret: "not base64!"}, nil)
	if err := bad.Ready(); err == nil {
		t.Fatal("a Kraken secret that cannot sign must fail readiness")
	}
}

// Open live orders survive a restart through `orders.json`, and a venue that is no longer
// registered is reported rather than silently dropped.
func TestLiveOrdersRoundTripThroughTheStateFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "orders.json")

	engine := free()
	r := NewRouter(engine)
	k := NewKraken(VenueCreds{Key: "k", Secret: "c2VjcmV0"}, nil)
	k.Symbols = map[string]string{inst: "XBTEUR"}
	r.Add(k)
	k.book.put(&liveOrder{
		intentID: "i1", venueID: "OABC", account: "live", instrument: inst, symbol: "XBTEUR",
		side: fav1.Side_SIDE_BUY, quantity: 3, filled: 1, notional: 100, fees: 0.1, arrival: 100, seq: 1,
	})

	store := &StateStore{Path: path, Router: r}
	if ok, err := store.SaveIfChanged(engine, false); err != nil || !ok {
		t.Fatalf("save: ok=%v err=%v", ok, err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var st GatewayState
	if err := json.Unmarshal(raw, &st); err != nil {
		t.Fatal(err)
	}
	if len(st.Live["kraken"]) != 1 {
		t.Fatalf("live orders not persisted: %s", raw)
	}
	if got := st.Live["kraken"][0]; got.IntentID != "i1" || got.Filled != 1 || got.Account != "live" {
		t.Fatalf("persisted order %+v", got)
	}

	// A fresh process: the order comes back attached to its intent, with the published progress.
	next := NewRouter(free())
	k2 := NewKraken(VenueCreds{Key: "k", Secret: "c2VjcmV0"}, nil)
	next.Add(k2)
	if orphaned := next.Restore(st.Live); len(orphaned) != 0 {
		t.Fatalf("nothing should be orphaned: %v", orphaned)
	}
	o, ok := k2.book.get("i1")
	if !ok {
		t.Fatal("the order did not come back")
	}
	if o.filled != 1 || o.notional != 100 || o.seq != 1 || o.account != "live" {
		t.Fatalf("restored %+v", *o)
	}

	// Without the venue registered, the orders are reported as orphaned.
	empty := NewRouter(free())
	orphaned := empty.Restore(st.Live)
	if len(orphaned) != 1 {
		t.Fatalf("want one orphaned venue, got %v", orphaned)
	}
}

// The gateway's reconcile rule in isolation: a venue repeating a cumulative fill produces one
// event, and the increment is priced at the average of the new fills only.
func TestFillIncrementsAreNeverDoubleCounted(t *testing.T) {
	b := newLiveBook("test")
	o := &liveOrder{intentID: "i", venueID: "V", account: "live", instrument: inst, symbol: "S",
		side: fav1.Side_SIDE_BUY, quantity: 10, arrival: 100}
	b.put(o)

	first := b.reconcile(o, venueState{Filled: 4, AvgPrice: 100, Fees: 0.4, Currency: "EUR"})
	if len(first) != 1 || first[0].Event.FillQuantity != 4 || first[0].Event.Kind != fav1.OrderEvent_KIND_PARTIAL_FILL {
		t.Fatalf("first: %v", first)
	}
	if again := b.reconcile(o, venueState{Filled: 4, AvgPrice: 100, Fees: 0.4, Currency: "EUR"}); len(again) != 0 {
		t.Fatalf("the same cumulative fill produced %d events", len(again))
	}
	last := b.reconcile(o, venueState{Filled: 10, AvgPrice: 101, Fees: 1.0, Currency: "EUR"})
	if len(last) != 1 || last[0].Event.Kind != fav1.OrderEvent_KIND_FILL {
		t.Fatalf("last: %v", last)
	}
	ev := last[0].Event
	if want := (10*101 - 4*100) / 6.0; !near(ev.FillPrice, want, 1e-9) {
		t.Fatalf("increment priced %g, want %g", ev.FillPrice, want)
	}
	if !near(ev.FillQuantity, 6, 1e-9) {
		t.Fatalf("increment %g, want 6", ev.FillQuantity)
	}
	if !near(ev.Fee, 0.6, 1e-9) {
		t.Fatalf("fee %g, want the 0.6 increment", ev.Fee)
	}
	// Terminal: the order left the book, so nothing further can be published for it.
	if b.OpenOrders() != 0 {
		t.Fatalf("%d orders still open after the terminal fill", b.OpenOrders())
	}
	if more := b.reconcile(o, venueState{Filled: 10, AvgPrice: 101, Fees: 1.0}); len(more) != 0 {
		t.Fatalf("events after the terminal one: %v", more)
	}
}
