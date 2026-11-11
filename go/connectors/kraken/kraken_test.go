package kraken

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const inst = "01900000-0000-7000-8000-00000000bbbb"

type fixedClock struct{ t time.Time }

func (f fixedClock) Now() time.Time { return f.t }

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

type server struct {
	mu   sync.Mutex
	reqs []string
}

func newTest(t *testing.T, now time.Time, cursors connector.CursorStore, symbols ...string) (*Connector, *server) {
	t.Helper()
	ohlc, trades, ticker, bad := fixture(t, "ohlc_xbteur_1m.json"), fixture(t, "trades_xbteur.json"), fixture(t, "ticker_xbteur.json"), fixture(t, "error_unknown_pair.json")
	s := &server{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.reqs = append(s.reqs, r.URL.String())
		s.mu.Unlock()
		if r.URL.Query().Get("pair") != "XBTEUR" {
			w.Write(bad)
			return
		}
		switch r.URL.Path {
		case "/0/public/OHLC":
			w.Write(ohlc)
		case "/0/public/Trades":
			w.Write(trades)
		case "/0/public/Ticker":
			w.Write(ticker)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	res := connector.NewStaticResolver()
	res.Add(ID, "XBTEUR", inst)
	if len(symbols) == 0 {
		symbols = []string{"XBTEUR"}
	}
	if cursors == nil {
		cursors = connector.NewMemCursors()
	}
	c := &Connector{BaseURL: srv.URL}
	src := catalog.Source{ID: ID, Config: &catalog.Params{Symbols: symbols, Timeframe: "1m"}}
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{Resolver: res, Clock: fixedClock{now}, Cursors: cursors}); err != nil {
		t.Fatal(err)
	}
	return c, s
}

// Fixture OHLC rows open at 1794362040..1794362280 (5 × 1m); at "now" the
// 1794362280 row is still forming. Trades ids 110355596..110355600.
var now = time.Unix(1794362290, 0).UTC()

func byEndpoint(t *testing.T, raws []connector.RawMessage) map[string]connector.RawMessage {
	t.Helper()
	m := map[string]connector.RawMessage{}
	for _, r := range raws {
		if r.TransportMeta[connector.MetaSymbol] != "XBTEUR" {
			t.Fatalf("raw symbol: %+v", r.TransportMeta)
		}
		m[r.TransportMeta[connector.MetaEndpoint]] = r
	}
	return m
}

func TestPollNormalize(t *testing.T) {
	cur := connector.NewMemCursors()
	c, s := newTest(t, now, cur)
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 3 {
		t.Fatalf("poll: %v %d", err, len(raws))
	}
	// First poll without cursor asks for one poll interval of trades.
	if want := "since=1794362230000000000"; !strings.Contains(s.reqs[1], want) {
		t.Fatalf("trades request %q lacks %s", s.reqs[1], want)
	}
	m := byEndpoint(t, raws)

	bars, unres, err := c.Normalize(m["ohlc"])
	if err != nil || len(unres) != 0 || len(bars) != 4 {
		t.Fatalf("ohlc: %v %v %d", err, unres, len(bars))
	}
	b := bars[0].GetBar()
	if b == nil || b.Timeframe != "1m" || b.Open != 65917.7 || b.Close != 65912.1 || b.Volume != 0.05440769 || b.TradeCount != 2 || b.GetVwap() != 65917.4 {
		t.Fatalf("bar: %v", bars[0])
	}
	if bars[0].EventTime.AsTime() != time.Unix(1794362100, 0).UTC() || bars[0].InstrumentId != inst || bars[0].PublishTime.AsTime() != now {
		t.Fatalf("bar envelope: %v", bars[0])
	}
	if got, _, _ := cur.GetCursor(context.Background(), ID, inst, "1m"); got != time.Unix(1794362280, 0).UTC() {
		t.Fatalf("bar cursor %v", got)
	}

	trades, _, err := c.Normalize(m["trades"])
	if err != nil || len(trades) != 5 {
		t.Fatalf("trades: %v %d", err, len(trades))
	}
	tr := trades[0].GetTrade()
	if tr.Price != 65846.3 || tr.Size != 0.00008612 || tr.Aggressor != fav1.Side_SIDE_SELL || tr.VenueTradeId != "110355596" {
		t.Fatalf("trade: %v", tr)
	}
	if trades[4].GetTrade().Aggressor != fav1.Side_SIDE_BUY || trades[0].EventTime.AsTime() != time.Unix(1794362222, 196271400).UTC() {
		t.Fatalf("trade 4 / time: %v %v", trades[4], trades[0].EventTime.AsTime())
	}

	quotes, _, err := c.Normalize(m["ticker"])
	if err != nil || len(quotes) != 1 {
		t.Fatalf("ticker: %v %d", err, len(quotes))
	}
	if q := quotes[0].GetQuote(); q.BidPrice != 65845.2 || q.AskPrice != 65845.4 || q.BidSize != 1 || q.AskSize != 1 {
		t.Fatalf("quote: %v", q)
	}

	// Same payloads again: no new bars or trades.
	if e, _, _ := c.Normalize(m["ohlc"]); len(e) != 0 {
		t.Fatalf("bars re-emitted: %d", len(e))
	}
	if e, _, _ := c.Normalize(m["trades"]); len(e) != 0 {
		t.Fatalf("trades re-emitted: %d", len(e))
	}
	// The next poll continues from Kraken's `last` cursor.
	s.reqs = nil
	if _, err := c.Poll(context.Background(), connector.PollRequest{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(s.reqs[1], "since=1794362312664586004") {
		t.Fatalf("trades cursor not advanced: %s", s.reqs[1])
	}

	// A restarted connector resumes from the persisted cursors.
	c2, _ := newTest(t, now, cur)
	if e, _, _ := c2.Normalize(m["ohlc"]); len(e) != 0 {
		t.Fatalf("restart re-emitted %d bars", len(e))
	}
}

func TestDeterministicPayloads(t *testing.T) {
	c1, _ := newTest(t, now, nil)
	c2, _ := newTest(t, now, nil)
	raws, _ := c1.Poll(context.Background(), connector.PollRequest{})
	for _, r := range raws {
		a, _, _ := c1.Normalize(r)
		b, _, _ := c2.Normalize(r)
		if len(a) != len(b) {
			t.Fatalf("%s: %d vs %d", r.TransportMeta[connector.MetaEndpoint], len(a), len(b))
		}
		for i := range a {
			if a[i].String() == "" || a[i].EventTime.AsTime() != b[i].EventTime.AsTime() || a[i].GetPayload() == nil ||
				a[i].GetBar().String() != b[i].GetBar().String() || a[i].GetTrade().String() != b[i].GetTrade().String() || a[i].GetQuote().String() != b[i].GetQuote().String() {
				t.Fatalf("payload differs: %v vs %v", a[i], b[i])
			}
		}
	}
}

func TestUnresolvedAndError(t *testing.T) {
	c, _ := newTest(t, now, nil, "XBTEUR", "ETHEUR")
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 6 {
		t.Fatalf("poll: %v %d", err, len(raws))
	}
	// ETHEUR is served the error payload and is not in the resolver.
	var sawErr bool
	for _, r := range raws[3:] {
		_, _, err := c.Normalize(r)
		if err == nil || !strings.Contains(err.Error(), "Unknown asset pair") {
			t.Fatalf("want kraken error, got %v", err)
		}
		sawErr = true
	}
	if !sawErr {
		t.Fatal("no error payload")
	}
	// A valid payload for an unresolved symbol is quarantined.
	r := raws[0]
	r.TransportMeta = map[string]string{connector.MetaSymbol: "ETHEUR", connector.MetaEndpoint: "ohlc", connector.MetaTf: "1m"}
	envs, unres, err := c.Normalize(r)
	if err != nil || len(envs) != 0 || len(unres) != 1 || unres[0].VenueSymbol != "ETHEUR" {
		t.Fatalf("unresolved: %v %v %v", err, envs, unres)
	}
}

func TestBackfillRange(t *testing.T) {
	c, s := newTest(t, now, nil)
	out := make(chan connector.RawMessage, 4)
	from, to := time.Unix(1794362100, 0).UTC(), time.Unix(1794362220, 0).UTC()
	if err := c.Backfill(context.Background(), connector.BackfillRequest{VenueSymbol: "XBTEUR", Timeframe: "1m", From: from, To: to}, out); err != nil {
		t.Fatal(err)
	}
	close(out)
	if len(s.reqs) != 1 || !strings.Contains(s.reqs[0], "since=1794362099") {
		t.Fatalf("requests: %v", s.reqs)
	}
	raw := <-out
	if raw.TransportMeta[connector.MetaBackfill] != "1" {
		t.Fatal("backfill flag missing")
	}
	envs, _, err := c.Normalize(raw)
	if err != nil || len(envs) != 2 { // opens 1794362100 and 1794362160
		t.Fatalf("backfill: %v %d", err, len(envs))
	}
	if envs[0].EventTime.AsTime() != time.Unix(1794362160, 0).UTC() || envs[1].EventTime.AsTime() != to {
		t.Fatalf("backfill times: %v %v", envs[0].EventTime.AsTime(), envs[1].EventTime.AsTime())
	}
	// Backfill does not move the live watermark.
	if live, _, _ := c.Normalize(connector.RawMessage{SourceID: ID, ReceivedAt: now, Bytes: raw.Bytes,
		TransportMeta: map[string]string{connector.MetaSymbol: "XBTEUR", connector.MetaEndpoint: "ohlc", connector.MetaTf: "1m"}}); len(live) != 4 {
		t.Fatalf("live after backfill: %d", len(live))
	}
	if err := c.Backfill(context.Background(), connector.BackfillRequest{VenueSymbol: "XBTEUR", Timeframe: "7m", From: from, To: to}, make(chan connector.RawMessage, 1)); err == nil {
		t.Fatal("want unsupported timeframe error")
	}
}
