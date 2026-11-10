package catalog

import "testing"

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
