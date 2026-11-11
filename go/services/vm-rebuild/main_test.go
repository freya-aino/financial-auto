package main

import (
	"testing"
	"time"
)

func TestCompareCountsMatchedMissingExtra(t *testing.T) {
	live := map[string]series{
		`fa_trade_price{inst="a",src="kraken"}`: {1: 10, 2: 11, 3: 12},
		`fa_bar_close{inst="a",src="kraken"}`:   {1: 5},
	}
	rebuilt := map[string]series{
		`fa_trade_price{inst="a",src="kraken"}`: {1: 10, 2: 11.5, 4: 13},
		`fa_bar_close{inst="a",src="kraken"}`:   {1: 5},
		`fa_bar_open{inst="a",src="kraken"}`:    {1: 4},
	}
	c := Compare(live, rebuilt)
	if c.Matched != 2 || c.Missing != 1 || c.Extra != 2 || c.ValueDiff != 1 {
		t.Fatalf("%+v", c)
	}
	if c.IdenticalSeries != 1 || c.ExtraSeries != 1 || c.LiveOnlySeries != 0 {
		t.Fatalf("%+v", c)
	}
	if m := c.PerMetric["fa_trade_price"]; *m != [4]int{1, 1, 1, 1} {
		t.Fatalf("per metric %v", *m)
	}
}

func TestRefusesLiveTarget(t *testing.T) {
	for _, u := range []string{"http://127.0.0.1:8428", "http://localhost:8428/", "http://victoriametrics:8428"} {
		if !isLiveVM(u) {
			t.Fatalf("%s should count as live", u)
		}
	}
	if isLiveVM("http://127.0.0.1:18428") || !sameURL("http://x:1/", "http://x:1") {
		t.Fatal("scratch url misclassified")
	}
}

func TestParseTimeAndKeys(t *testing.T) {
	now := time.Date(2026, 11, 11, 12, 0, 0, 0, time.UTC)
	if got, _ := parseTime("-2h", now); !got.Equal(now.Add(-2 * time.Hour)) {
		t.Fatal(got)
	}
	if got := seriesKey(map[string]string{"__name__": "fa_x", "venue": "k", "inst": "a"}); got != `fa_x{inst="a",venue="k"}` {
		t.Fatal(got)
	}
	if got := readOnlyURL("postgres://a@b/c"); got != "postgres://a@b/c?default_transaction_read_only=on" {
		t.Fatal(got)
	}
}
