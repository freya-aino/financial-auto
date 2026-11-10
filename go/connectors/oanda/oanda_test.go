package oanda

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

type fixedClock struct{ t time.Time }

func (f fixedClock) Now() time.Time { return f.t }

func newTest(t *testing.T) (*Connector, *atomic.Int32, *[]string) {
	t.Helper()
	var streams atomic.Int32
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.RequestURI())
		if r.Header.Get("Authorization") != "Bearer tok" {
			t.Errorf("auth %q", r.Header.Get("Authorization"))
		}
		switch {
		case strings.Contains(r.URL.Path, "/pricing/stream"):
			if streams.Add(1) > 1 { // second connection: pretend credentials were revoked
				w.WriteHeader(401)
				return
			}
			b, _ := os.ReadFile("testdata/pricing_stream.ndjson")
			w.Write(b)
		case strings.Contains(r.URL.Path, "/candles"):
			b, _ := os.ReadFile("testdata/candles_m1.json")
			w.Write(b)
		default:
			t.Errorf("unexpected %s", r.URL)
		}
	}))
	t.Cleanup(srv.Close)
	res := connector.NewStaticResolver()
	res.Add(ID, "EUR_USD", "01900000-0000-7000-8000-0000000000e0")
	c := &Connector{RESTURL: srv.URL, StreamURL: srv.URL, Reconnect: time.Millisecond}
	cfg := connector.SourceConfig{
		Source:  catalog.Source{ID: ID, Config: &catalog.Params{Symbols: []string{"EUR_USD", "DE30_EUR"}}},
		Secrets: map[string]string{"api_key": "tok", "account_id": "101-004-1"},
	}
	if err := c.Init(context.Background(), cfg, connector.Deps{Resolver: res, Clock: fixedClock{time.Unix(1_800_000_000, 0).UTC()}}); err != nil {
		t.Fatal(err)
	}
	return c, &streams, &paths
}

func TestInitSecrets(t *testing.T) {
	src := catalog.Source{ID: ID, Config: &catalog.Params{Symbols: []string{"EUR_USD"}}}
	if err := (&Connector{}).Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{}); !errors.Is(err, ErrNoAPIKey) {
		t.Fatal(err)
	}
	t.Setenv("FA_OANDA_API_KEY", "k")
	if err := (&Connector{}).Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{}); !errors.Is(err, ErrNoAccount) {
		t.Fatal(err)
	}
	t.Setenv("FA_OANDA_ACCOUNT_ID", "a")
	c := &Connector{}
	if err := c.Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{}); err != nil || c.RESTURL != Hosts["practice"].REST {
		t.Fatal(err, c.RESTURL)
	}
	src.Config.Extra = map[string]string{"env": "nope"}
	if err := (&Connector{}).Init(context.Background(), connector.SourceConfig{Source: src}, connector.Deps{}); err == nil {
		t.Fatal("bad env must error")
	}
}

func TestStreamQuotes(t *testing.T) {
	c, streams, paths := newTest(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out := make(chan connector.RawMessage, 16)
	err := c.Stream(ctx, nil, out)
	if !connector.IsHTTPStatus(err, 401) || streams.Load() != 2 {
		t.Fatalf("stream ended with %v after %d connections", err, streams.Load())
	}
	if !strings.HasPrefix((*paths)[0], "/v3/accounts/101-004-1/pricing/stream?instruments=EUR_USD%2CDE30_EUR") {
		t.Fatalf("url %s", (*paths)[0])
	}
	close(out)
	var raws []connector.RawMessage
	for r := range out {
		raws = append(raws, r)
	}
	if len(raws) != 2 { // heartbeats are dropped
		t.Fatalf("raws %d", len(raws))
	}
	envs, unres, err := c.Normalize(raws[0])
	if err != nil || len(envs) != 0 || len(unres) != 1 || unres[0].VenueSymbol != "DE30_EUR" {
		t.Fatalf("DE30: %v %v %v", envs, unres, err)
	}
	envs, unres, err = c.Normalize(raws[1])
	if err != nil || len(envs) != 1 || len(unres) != 0 {
		t.Fatalf("EUR_USD: %v %v %v", envs, unres, err)
	}
	q := envs[0].GetQuote()
	if q.BidPrice != 1.11722 || q.AskPrice != 1.11735 || q.BidSize != 10000000 || envs[0].InstrumentId != "01900000-0000-7000-8000-0000000000e0" {
		t.Fatalf("quote %v", envs[0])
	}
	if envs[0].EventTime.AsTime() != time.Date(2016, 9, 20, 15, 5, 50, 900152606, time.UTC) {
		t.Fatalf("event time %v", envs[0].EventTime.AsTime())
	}
}

func TestBackfillCandles(t *testing.T) {
	c, _, paths := newTest(t)
	out := make(chan connector.RawMessage, 8)
	from := time.Date(2016, 10, 17, 0, 0, 0, 0, time.UTC)
	// 6 days of M1 at 5000 candles per request → 2 requests.
	req := connector.BackfillRequest{VenueSymbol: "EUR_USD", Timeframe: "1m", From: from, To: from.Add(6 * 24 * time.Hour)}
	if err := c.Backfill(context.Background(), req, out); err != nil {
		t.Fatal(err)
	}
	close(out)
	if len(*paths) != 2 || !strings.Contains((*paths)[0], "/v3/instruments/EUR_USD/candles?granularity=M1&price=M&from=2016-10-17T00%3A00%3A00Z") {
		t.Fatalf("paths %v", *paths)
	}
	raw := <-out
	envs, _, err := c.Normalize(raw)
	if err != nil || len(envs) != 2 { // incomplete candle skipped
		t.Fatalf("%v %v", envs, err)
	}
	b := envs[0].GetBar()
	if b.Timeframe != "1m" || b.Open != 1.09981 || b.Close != 1.09983 || b.Volume != 12 {
		t.Fatalf("bar %v", b)
	}
	if envs[0].EventTime.AsTime() != time.Date(2016, 10, 17, 15, 17, 0, 0, time.UTC) {
		t.Fatalf("bar close %v", envs[0].EventTime.AsTime())
	}
	bad := connector.RawMessage{Bytes: []byte(`{"errorMessage":"Insufficient authorization to perform request."}`), TransportMeta: map[string]string{connector.MetaEndpoint: "candles"}}
	if _, _, err := c.Normalize(bad); err == nil {
		t.Fatal("api error must surface")
	}
}
