package catalog

import (
	"testing"
	"time"
)

func TestRepoCatalogIsValid(t *testing.T) {
	c, err := Load("../../../config/sources.yaml")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(c.Sources) < 70 {
		t.Fatalf("expected the full catalog, got %d sources", len(c.Sources))
	}
	for _, id := range []string{"yahoo", "lemon", "oanda", "hackernews", "google_news"} {
		s, ok := c.ByID(id)
		if !ok {
			t.Fatalf("first-wave source %q missing", id)
		}
		if s.Status != "stub" && s.Status != "live" {
			t.Errorf("%s: status = %q, want stub or live", id, s.Status)
		}
	}
}

func TestValidateRejectsDuplicatesAndBadStatus(t *testing.T) {
	_, err := Parse([]byte(`
version: 1
sources:
  - {id: a, name: A, status: planned, auth: none}
  - {id: a, name: A2, status: bogus, auth: none}
`))
	if err == nil {
		t.Fatal("expected validation error")
	}
}

func TestParamsParseAndValidate(t *testing.T) {
	c, err := Parse([]byte(`
version: 1
sources:
  - {id: a, name: A, status: live, auth: none, config: {symbols: [AAPL, MSFT], timeframe: 1m, extra: {mic: XMUN}}}
`))
	if err != nil {
		t.Fatal(err)
	}
	p := c.Sources[0].Params()
	if len(p.Symbols) != 2 || p.Timeframe != "1m" || p.Extra["mic"] != "XMUN" {
		t.Fatalf("params = %+v", p)
	}
	if (Source{}).Params().Symbols != nil {
		t.Fatal("nil config must yield empty params")
	}
	if _, err := Parse([]byte("version: 1\nsources:\n  - {id: a, name: A, status: live, auth: none, config: {symbols: ['A B'], timeframe: 2m}}\n")); err == nil {
		t.Fatal("expected invalid symbol/timeframe error")
	}
}

func TestSetStatusAndDepth(t *testing.T) {
	c, err := Parse([]byte("version: 1\nsources:\n  - {id: a, name: A, status: stub, auth: none}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if !c.SetStatus("a", "disabled") || c.SetStatus("a", "bogus") || c.SetStatus("zz", "live") {
		t.Fatal("SetStatus results")
	}
	if s, _ := c.ByID("a"); s.Status != "disabled" {
		t.Fatalf("status %q", s.Status)
	}
	if got := c.ApplyStatuses(map[string]string{"a": "live", "zz": "live"}); len(got) != 1 || got[0] != "a" {
		t.Fatalf("ApplyStatuses %v", got)
	}
	if BackfillDepth("1m") != 365*24*time.Hour || BackfillDepth("1h") != 5*365*24*time.Hour || BackfillDepth("1s") != 28*24*time.Hour {
		t.Fatal("depths")
	}
}

// catalogIDs is the fixed id list of spec 11 §6.2: every API from the user's
// list, the first wave and the media sources. Adding a source means adding it
// here; removing one fails the test.
var catalogIDs = []string{
	// brokers / multi-asset
	"interactive_brokers", "saxo", "alpaca", "tastytrade", "schwab", "lemon", "ig", "oanda", "ctrader", "metatrader5",
	// crypto exchanges & DEX
	"bitvavo", "kraken", "coinbase", "binance", "bybit", "okx", "bitget", "bitstamp", "bitpanda", "ccxt", "hyperliquid", "dydx",
	// prediction markets
	"kalshi", "polymarket",
	// market data vendors
	"fmp", "eodhd", "twelve_data", "finnhub", "alpha_vantage", "tiingo", "marketstack", "alpaca_market_data", "yahoo", "dukascopy", "nasdaq_data_link",
	// FX reference
	"frankfurter", "exchangerate_host", "fixer", "currencylayer",
	// crypto data, on-chain, DeFi
	"coingecko", "coinmarketcap", "coindesk_data", "etherscan", "blockscout", "dune", "the_graph", "defillama",
	// macro / statistics
	"fred", "ecb", "eurostat", "cbs_statline", "us_treasury_fiscal", "bls", "world_bank", "imf", "oecd", "dbnomics", "eia", "usda_quickstats", "entsoe",
	// fundamentals / filings / identifiers
	"sec_edgar", "simfin", "openfigi", "gleif",
	// news
	"marketaux", "newsapi", "gdelt", "hackernews", "google_news",
	// broadcast media
	"youtube", "earnings_webcasts", "central_bank_pressers",
}

func TestRepoCatalogMatchesFixedIDList(t *testing.T) {
	c, err := Load("../../../config/sources.yaml")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{}
	for _, id := range catalogIDs {
		if want[id] {
			t.Fatalf("duplicate id %q in the fixed list", id)
		}
		want[id] = true
	}
	have := map[string]bool{}
	for _, s := range c.Sources {
		have[s.ID] = true
		if !want[s.ID] {
			t.Errorf("catalog has %q, which is not in the fixed list of spec 11 §6.2", s.ID)
		}
	}
	for _, id := range catalogIDs {
		if !have[id] {
			t.Errorf("fixed-list id %q missing from config/sources.yaml", id)
		}
	}
	if len(c.Sources) != len(catalogIDs) {
		t.Errorf("catalog has %d entries, fixed list %d", len(c.Sources), len(catalogIDs))
	}
}

func TestByCapabilityAndCategory(t *testing.T) {
	c, err := Load("../../../config/sources.yaml")
	if err != nil {
		t.Fatal(err)
	}
	exec := c.ByCapability("execution")
	data := c.ByCapability("data")
	if len(exec) == 0 || len(data) < len(exec) {
		t.Fatalf("execution %d, data %d", len(exec), len(data))
	}
	for _, s := range exec {
		if !s.HasCapability("execution") {
			t.Fatalf("%s lacks execution", s.ID)
		}
	}
	news := c.ByCategory("news")
	ids := map[string]bool{}
	for _, s := range news {
		ids[s.ID] = true
	}
	for _, id := range []string{"gdelt", "hackernews", "google_news"} {
		if !ids[id] {
			t.Errorf("news category lacks %s", id)
		}
	}
	if len(c.ByCategory("no_such_category")) != 0 {
		t.Fatal("unknown category must be empty")
	}
	if s, _ := c.ByID("kraken"); s.Backfill == nil || s.Backfill.KnowledgeDelaySecs != 120 {
		t.Fatalf("kraken backfill = %+v", s.Backfill)
	}
}

func TestReplaceAndOverride(t *testing.T) {
	c, err := Parse([]byte(`
version: 1
sources:
  - {id: a, name: A, status: stub, auth: none, rate_limit: {per_second: 1, burst: 2}}
  - {id: b, name: B, status: live, auth: none, config: {symbols: [X]}}
  - {id: gone, name: G, status: live, auth: none}
`))
	if err != nil {
		t.Fatal(err)
	}
	c.SetStatus("a", "live") // runtime status must survive the reload
	next, err := Parse([]byte(`
version: 2
sources:
  - {id: a, name: A renamed, status: planned, auth: none, rate_limit: {per_second: 1, burst: 2}}
  - {id: b, name: B, status: live, auth: none, config: {symbols: [X, Y]}}
  - {id: new, name: N, status: planned, auth: none, adapter: rss}
`))
	if err != nil {
		t.Fatal(err)
	}
	changed, added, removed := c.Replace(next)
	if len(changed) != 1 || changed[0] != "b" || len(added) != 1 || added[0] != "new" || len(removed) != 1 || removed[0] != "gone" {
		t.Fatalf("changed %v added %v removed %v", changed, added, removed)
	}
	if s, _ := c.ByID("a"); s.Status != "live" || s.Name != "A renamed" {
		t.Fatalf("a = %+v", s)
	}
	o := Override{RateLimit: &RateLimit{PerSecond: 0.2, Burst: 1}}
	if err := o.Validate(); err != nil {
		t.Fatal(err)
	}
	if (Override{RateLimit: &RateLimit{PerSecond: 0, Burst: 1}}).Validate() == nil {
		t.Fatal("zero rate must be invalid")
	}
	if (Override{Config: &Params{Timeframe: "7m"}}).Validate() == nil {
		t.Fatal("bad timeframe must be invalid")
	}
	if !c.ApplyOverride("a", o) || c.ApplyOverride("zz", o) {
		t.Fatal("ApplyOverride results")
	}
	if s, _ := c.ByID("a"); s.Limit() != (RateLimit{PerSecond: 0.2, Burst: 1}) {
		t.Fatalf("limit %+v", s.Limit())
	}
	if _, err := Parse([]byte("version: 1\nsources:\n  - {id: a, name: A, status: live, auth: none, adapter: ccxt}\n")); err == nil {
		t.Fatal("unknown adapter must be invalid")
	}
}
