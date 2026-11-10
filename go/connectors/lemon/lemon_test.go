package lemon

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

type fixedClock struct{ t time.Time }

func (f fixedClock) Now() time.Time { return f.t }

const tsla, aapl = "US88160R1014", "US0378331005"

func newTest(t *testing.T) (*Connector, *[]string) {
	t.Helper()
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.RequestURI())
		if r.Header.Get("Authorization") != "Bearer k3y" {
			t.Errorf("auth header %q", r.Header.Get("Authorization"))
		}
		var f string
		switch {
		case strings.HasPrefix(r.URL.Path, "/quotes/latest"):
			f = "quotes_latest.json"
		case strings.HasPrefix(r.URL.Path, "/trades/latest"):
			f = "trades_latest.json"
		case strings.HasPrefix(r.URL.Path, "/ohlc/m1"):
			if r.URL.Query().Get("page") == "2" {
				w.Write([]byte(`{"results":[],"next":null,"page":2,"pages":2}`))
				return
			}
			f = "ohlc_m1.json"
		default:
			t.Errorf("unexpected %s", r.URL)
		}
		b, _ := os.ReadFile("testdata/" + f)
		w.Write(b)
	}))
	t.Cleanup(srv.Close)
	res := connector.NewStaticResolver()
	res.Add(ID, tsla, "01900000-0000-7000-8000-000000000011")
	c := &Connector{BaseURL: srv.URL}
	cfg := connector.SourceConfig{
		Source:  catalog.Source{ID: ID, Config: &catalog.Params{Symbols: []string{tsla, aapl}}},
		Secrets: map[string]string{"api_key": "k3y"},
	}
	if err := c.Init(context.Background(), cfg, connector.Deps{Resolver: res, Clock: fixedClock{time.Unix(1_800_000_000, 0).UTC()}}); err != nil {
		t.Fatal(err)
	}
	return c, &paths
}

func TestInitRequiresKey(t *testing.T) {
	err := (&Connector{}).Init(context.Background(), connector.SourceConfig{Source: catalog.Source{ID: ID}}, connector.Deps{})
	if !errors.Is(err, ErrNoAPIKey) {
		t.Fatalf("want ErrNoAPIKey, got %v", err)
	}
	t.Setenv("FA_LEMON_API_KEY", "env")
	err = (&Connector{}).Init(context.Background(), connector.SourceConfig{Source: catalog.Source{ID: ID, Config: &catalog.Params{Symbols: []string{tsla}}}}, connector.Deps{})
	if err != nil {
		t.Fatalf("env key must satisfy Init: %v", err)
	}
}

func TestPollQuotesTradesAndDedup(t *testing.T) {
	c, paths := newTest(t)
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil || len(raws) != 2 {
		t.Fatalf("%v %d", err, len(raws))
	}
	if (*paths)[0] != "/quotes/latest?isin="+tsla+","+aapl+"&mic=XMUN" {
		t.Fatalf("url %s", (*paths)[0])
	}
	envs, unres, err := c.Normalize(raws[0])
	if err != nil || len(envs) != 1 || len(unres) != 1 || unres[0].VenueSymbol != aapl {
		t.Fatalf("quotes: %v %v %v", envs, unres, err)
	}
	q := envs[0].GetQuote()
	if q.BidPrice != 921.1 || q.AskPrice != 921.3 || q.BidSize != 87 || envs[0].InstrumentId != "01900000-0000-7000-8000-000000000011" {
		t.Fatalf("quote %v", envs[0])
	}
	if envs[0].EventTime.AsTime() != time.Date(2021, 10, 28, 8, 51, 3, 669_000_000, time.UTC) {
		t.Fatalf("event time %v", envs[0].EventTime.AsTime())
	}
	envs, _, _ = c.Normalize(raws[1])
	if len(envs) != 1 || envs[0].GetTrade().Price != 274 || envs[0].GetTrade().Size != 2 {
		t.Fatalf("trades: %v", envs)
	}
	// Unchanged "latest" rows are not re-emitted.
	if envs, _, _ := c.Normalize(raws[0]); len(envs) != 0 {
		t.Fatal("dedup failed")
	}
}

func TestBackfillOHLC(t *testing.T) {
	c, paths := newTest(t)
	out := make(chan connector.RawMessage, 4)
	req := connector.BackfillRequest{VenueSymbol: tsla, Timeframe: "1m", From: time.Date(2021, 9, 2, 9, 0, 0, 0, time.UTC), To: time.Date(2021, 9, 2, 10, 0, 0, 0, time.UTC)}
	if err := c.Backfill(context.Background(), req, out); err != nil {
		t.Fatal(err)
	}
	close(out)
	// The fixture's "next" link (data.lemon.markets/v1/ohlc/m1/?...page=2) is followed against BaseURL.
	if len(*paths) != 2 || !strings.HasPrefix((*paths)[0], "/ohlc/m1?isin="+tsla+"&mic=XMUN&from=2021-09-02T09:00:00&to=2021-09-02T10:00:00") ||
		(*paths)[1] != "/ohlc/m1/?isin="+tsla+"&from=2021-09-02T09:02:00&to=2021-09-02T10:00:00&page=2" {
		t.Fatalf("paths %v", *paths)
	}
	raw := <-out
	if empty := <-out; len(empty.Bytes) == 0 {
		t.Fatal("second page missing")
	}
	envs, _, err := c.Normalize(raw)
	if err != nil || len(envs) != 2 {
		t.Fatalf("%v %v", envs, err)
	}
	b := envs[0].GetBar()
	if b.Timeframe != "1m" || b.Open != 777.9 || b.Close != 768.7 || b.Volume != 433 || b.Vwap == nil || *b.Vwap < 770 || *b.Vwap > 771 {
		t.Fatalf("bar %v", b)
	}
	if envs[0].EventTime.AsTime() != time.Date(2021, 9, 2, 9, 1, 0, 0, time.UTC) {
		t.Fatalf("bar close time %v", envs[0].EventTime.AsTime())
	}
	if err := c.Backfill(context.Background(), connector.BackfillRequest{Timeframe: "5m"}, out); err == nil {
		t.Fatal("unsupported timeframe must error")
	}
	bad := connector.RawMessage{Bytes: []byte(`{"status":"error","error_message":"unauthorized"}`), TransportMeta: map[string]string{}}
	if _, _, err := c.Normalize(bad); err == nil || !strings.Contains(err.Error(), "unauthorized") {
		t.Fatalf("api error: %v", err)
	}
}
