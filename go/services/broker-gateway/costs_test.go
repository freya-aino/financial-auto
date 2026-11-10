package main

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// fixtureCases are the inputs exported to testdata/cost_fixture.json for the Rust backtester.
var fixtureCases = map[string]CostInputs{
	"equity_buy_default_venue":      {Venue: "unknown", Class: "equity", Side: "buy", Price: 100, Qty: 1000, Sigma: 0.02, ADV: 10_000_000},
	"equity_sell_default_venue":     {Venue: "unknown", Class: "equity", Side: "sell", Price: 100, Qty: 1000, Sigma: 0.02, ADV: 10_000_000},
	"equity_small_min_commission":   {Venue: "yahoo", Class: "equity", Side: "buy", Price: 50, Qty: 1, Sigma: 0.01, ADV: 5_000_000},
	"equity_quoted_spread":          {Venue: "yahoo", Class: "equity", Side: "buy", Price: 200, Qty: 50, Spread: 0.10, Sigma: 0.015, ADV: 50_000_000},
	"crypto_large_impact":           {Venue: "yahoo", Class: "crypto", Side: "buy", Price: 75_000, Qty: 2, Sigma: 0.04, ADV: 20_000_000},
	"crypto_sell_default_sigma":     {Venue: "yahoo", Class: "crypto", Side: "sell", Price: 3_000, Qty: 10},
	"fx_no_commission":              {Venue: "unknown", Class: "fx", Side: "sell", Price: 1.08, Qty: 100_000, Sigma: 0.006, ADV: 1e9},
	"sim_costless":                  {Venue: "sim", Class: "equity", Side: "buy", Price: 100, Qty: 1000, Sigma: 0.5, ADV: 1_000},
	"passive_limit_commission_only": {Venue: "yahoo", Class: "equity", Side: "buy", Price: 99, Qty: 100, Spread: 1, Sigma: 0.02, ADV: 1_000_000, Passive: true},
	"unknown_class_falls_back":      {Venue: "yahoo", Class: "prediction_market", Side: "buy", Price: 0.6, Qty: 1000, Sigma: 0.1, ADV: 100_000},
}

func TestSquareRootImpactFormula(t *testing.T) {
	m := DefaultCostModel()
	r := m.Price(CostInputs{Side: "buy", Price: 100, Qty: 1000, Sigma: 0.02, ADV: 10_000_000})
	// impact = 100 · 0.1 · 0.02 · sqrt(1e5/1e7) = 0.02; half spread 0.025
	if !near(r.Impact, 0.02, 1e-12) || !near(r.HalfSpread, 0.025, 1e-12) || !near(r.FillPrice, 100.045, 1e-12) {
		t.Fatalf("%+v", r)
	}
	// 4× notional → 2× impact
	r4 := m.Price(CostInputs{Side: "buy", Price: 100, Qty: 4000, Sigma: 0.02, ADV: 10_000_000})
	if !near(r4.Impact, 0.04, 1e-12) {
		t.Fatalf("impact should scale with sqrt(notional): %v", r4.Impact)
	}
	// defaults replace missing σ / ADV
	rd := m.Price(CostInputs{Side: "buy", Price: 100, Qty: 1000})
	if !near(rd.Impact, r.Impact, 1e-12) {
		t.Fatalf("defaults: %v vs %v", rd.Impact, r.Impact)
	}
	// passive: commission only
	rp := m.Price(CostInputs{Side: "buy", Price: 100, Qty: 1000, Passive: true})
	if rp.FillPrice != 100 || rp.Impact != 0 || rp.HalfSpread != 0 || rp.Fee != 50 {
		t.Fatalf("passive: %+v", rp)
	}
}

func TestScheduleLookupOrder(t *testing.T) {
	m, err := LoadCostModel("costs.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if s := m.Schedule("yahoo", "crypto"); s.CommissionBps != 10 || s.SpreadBps != 10 {
		t.Fatalf("venue class: %+v", s)
	}
	if s := m.Schedule("yahoo", "fx"); s.CommissionBps != 0 || s.SpreadBps != 1 {
		t.Fatalf("venue without class → default class: %+v", s)
	}
	if s := m.Schedule("sim", "crypto"); s.CommissionBps != 0 || s.ImpactK == nil || *s.ImpactK != 0 {
		t.Fatalf("venue default: %+v", s)
	}
	if s := m.Schedule("nowhere", "nothing"); s.CommissionBps != 5 || s.SpreadBps != 5 {
		t.Fatalf("global default: %+v", s)
	}
	if m.ClassOf("01900000-0000-7000-8000-000000000001") != "equity" || m.ClassOf("x") != "equity" {
		t.Fatal("class lookup")
	}
	if r := m.Price(CostInputs{Venue: "sim", Class: "equity", Side: "buy", Price: 100, Qty: 10}); r.FillPrice != 100 || r.Fee != 0 {
		t.Fatalf("sim must be costless: %+v", r)
	}
}

func TestLoadRejectsBadModel(t *testing.T) {
	dir := t.TempDir()
	bad := dir + "/bad.yaml"
	if err := os.WriteFile(bad, []byte("default_sigma: -1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCostModel(bad); err == nil {
		t.Fatal("negative sigma must fail validation")
	}
	if err := os.WriteFile(bad, []byte("default:\n  default: {commission_bps: -5}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCostModel(bad); err == nil {
		t.Fatal("negative commission must fail validation")
	}
	if _, err := LoadCostModel(dir + "/missing.yaml"); err == nil {
		t.Fatal("missing file must fail")
	}
}

// The fixture is regenerated with UPDATE_FIXTURE=1; otherwise the committed file must match
// what costs.yaml produces today, so the Rust side can trust it.
func TestFixtureMatchesModel(t *testing.T) {
	m, err := LoadCostModel("costs.yaml")
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.ExportFixture(fixtureCases)
	if err != nil {
		t.Fatal(err)
	}
	const path = "testdata/cost_fixture.json"
	if os.Getenv("UPDATE_FIXTURE") != "" {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(got, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run with UPDATE_FIXTURE=1 to create it)", err)
	}
	if !bytes.Equal(bytes.TrimSpace(want), bytes.TrimSpace(got)) {
		t.Fatal("testdata/cost_fixture.json is stale: run `UPDATE_FIXTURE=1 go test ./services/broker-gateway`")
	}
	var f Fixture
	if err := json.Unmarshal(want, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) != len(fixtureCases) {
		t.Fatalf("cases %d", len(f.Cases))
	}
	for _, c := range f.Cases {
		r := m.Price(c.Inputs)
		if math.Abs(r.Fee-c.Expected.Fee) > 1e-9 || math.Abs(r.SlippageBps-c.Expected.SlippageBps) > 1e-9 {
			t.Fatalf("%s: %+v vs %+v", c.Name, r, c.Expected)
		}
	}
}
