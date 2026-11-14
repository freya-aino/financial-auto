package bus

import "testing"

func TestMarketDataModeAdmitsVenues(t *testing.T) {
	m, err := ParseMarketDataMode("")
	if err != nil || m != MarketDataReal {
		t.Fatalf("default = %q, %v", m, err)
	}
	if _, err := ParseMarketDataMode("live"); err == nil {
		t.Fatal("unknown mode accepted")
	}
	cases := []struct {
		mode    MarketDataMode
		subject string
		want    bool
	}{
		{MarketDataReal, "md.trade.kraken.0190abc", true},
		{MarketDataReal, "md.trade.sim.01900000000070008000000000000001", false},
		{MarketDataReal, "md.quote.sim.x", false},
		{MarketDataSynthetic, "md.trade.sim.x", true},
		{MarketDataSynthetic, "md.trade.yahoo.x", false},
		{MarketDataAll, "md.trade.sim.x", true},
		{MarketDataAll, "md.trade.kraken.x", true},
		{MarketDataReal, "md.bar.1m.sim.x", false},
		{MarketDataReal, "md.bar.1m.kraken.x", true},
		{MarketDataReal, "order.approved.paper", true},
	}
	for _, c := range cases {
		if got := c.mode.AdmitsSubject(c.subject); got != c.want {
			t.Errorf("%s admits %s = %v, want %v", c.mode, c.subject, got, c.want)
		}
	}
}
