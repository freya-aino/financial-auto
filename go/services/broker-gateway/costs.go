package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"go.yaml.in/yaml/v3"
)

// CostModel is the paper venue's cost + slippage model (spec 41 §6.3, spec 40 §6.7): the same
// numbers the backtester must use, so the Rust side matches `testdata/cost_fixture.json`.
//
// Lookup order for an (venue, asset class) pair: venues[v].classes[c], venues[v].default,
// default.classes[c], default.default — the first table found wins wholly (no field merging).
type CostModel struct {
	// Square-root impact coefficient: slippage = k · σ · sqrt(notional / ADV).
	DefaultK float64 `yaml:"default_k" json:"default_k"`
	// Daily volatility (fraction) used when none can be estimated from trades.
	DefaultSigma float64 `yaml:"default_sigma" json:"default_sigma"`
	// Daily traded notional (EUR) used when none has been observed.
	DefaultADV float64 `yaml:"default_adv_eur" json:"default_adv_eur"`
	// Asset class assumed for instruments missing from `instruments`.
	DefaultClass string `yaml:"default_class" json:"default_class"`
	// Instrument UUID → asset class (the gateway has no reference data of its own).
	Instruments map[string]string     `yaml:"instruments" json:"instruments,omitempty"`
	Default     VenueCosts            `yaml:"default" json:"default"`
	Venues      map[string]VenueCosts `yaml:"venues" json:"venues,omitempty"`
}

type VenueCosts struct {
	Default *ClassCosts           `yaml:"default" json:"default,omitempty"`
	Classes map[string]ClassCosts `yaml:"classes" json:"classes,omitempty"`
}

// ClassCosts is one fee schedule.
type ClassCosts struct {
	CommissionBps    float64 `yaml:"commission_bps" json:"commission_bps"`
	CommissionMinEUR float64 `yaml:"commission_min_eur" json:"commission_min_eur"`
	// Full quoted spread in bps of the price; used only when no `md.quote.>` has been seen.
	SpreadBps float64 `yaml:"spread_bps" json:"spread_bps"`
	// Overrides `default_k` when set.
	ImpactK *float64 `yaml:"impact_k" json:"impact_k,omitempty"`
	// Carry per day of the held notional (spec 41 §6.3): funding on any open position, borrow
	// on shorts; charged by the paper venue per carry period (carry.go) and by the backtest.
	FundingBpsDay float64 `yaml:"funding_bps_day" json:"funding_bps_day"`
	BorrowBpsDay  float64 `yaml:"borrow_bps_day" json:"borrow_bps_day"`
}

// CostInputs describe one taker fill to price.
type CostInputs struct {
	Venue string  `json:"venue"`
	Class string  `json:"class"`
	Side  string  `json:"side"`  // buy | sell
	Price float64 `json:"price"` // arrival price (last trade)
	Qty   float64 `json:"qty"`
	// Absolute quoted spread (ask − bid); 0 → modelled from spread_bps.
	Spread float64 `json:"spread"`
	// Daily volatility (fraction); 0 → model default.
	Sigma float64 `json:"sigma"`
	// Daily traded notional in EUR; 0 → model default.
	ADV float64 `json:"adv_eur"`
	// Passive fill (resting limit crossed): no spread, no impact, commission only.
	Passive bool `json:"passive,omitempty"`
}

// CostResult is the priced fill.
type CostResult struct {
	FillPrice   float64 `json:"fill_price"`
	Fee         float64 `json:"fee"`
	FeeCurrency string  `json:"fee_currency"`
	HalfSpread  float64 `json:"half_spread"`
	Impact      float64 `json:"impact"` // price units
	// Signed cost vs arrival in bps of the arrival price (positive = paid by the taker).
	SlippageBps float64 `json:"slippage_bps"`
	Notional    float64 `json:"notional"` // at the fill price
}

// DefaultCostModel is used when no costs file is given: 5 bps commission, 1 EUR min, 5 bps
// spread, k = 0.1, σ 2 %/day, ADV 10 M EUR.
func DefaultCostModel() *CostModel {
	return &CostModel{
		DefaultK:     0.1,
		DefaultSigma: 0.02,
		DefaultADV:   10_000_000,
		DefaultClass: "equity",
		Default:      VenueCosts{Default: &ClassCosts{CommissionBps: 5, CommissionMinEUR: 1, SpreadBps: 5}},
	}
}

// LoadCostModel reads a YAML costs file and validates it.
func LoadCostModel(path string) (*CostModel, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m := DefaultCostModel()
	if err := yaml.Unmarshal(b, m); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if err := m.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return m, nil
}

func (m *CostModel) Validate() error {
	if m.DefaultK < 0 || m.DefaultSigma <= 0 || m.DefaultADV <= 0 {
		return errors.New("default_k must be ≥ 0, default_sigma and default_adv_eur > 0")
	}
	if m.Default.Default == nil {
		return errors.New("default.default fee schedule is required")
	}
	check := func(where string, c ClassCosts) error {
		if c.CommissionBps < 0 || c.CommissionMinEUR < 0 || c.SpreadBps < 0 || (c.ImpactK != nil && *c.ImpactK < 0) {
			return fmt.Errorf("%s: negative cost", where)
		}
		return nil
	}
	if err := check("default.default", *m.Default.Default); err != nil {
		return err
	}
	for c, cc := range m.Default.Classes {
		if err := check("default.classes."+c, cc); err != nil {
			return err
		}
	}
	for v, vc := range m.Venues {
		if vc.Default != nil {
			if err := check("venues."+v+".default", *vc.Default); err != nil {
				return err
			}
		}
		for c, cc := range vc.Classes {
			if err := check("venues."+v+".classes."+c, cc); err != nil {
				return err
			}
		}
	}
	return nil
}

// ClassOf returns the configured asset class of an instrument.
func (m *CostModel) ClassOf(instrumentID string) string {
	if c, ok := m.Instruments[strings.ToLower(instrumentID)]; ok {
		return c
	}
	return m.DefaultClass
}

// Schedule resolves the fee schedule for a venue / class pair.
func (m *CostModel) Schedule(venue, class string) ClassCosts {
	if v, ok := m.Venues[venue]; ok {
		if c, ok := v.Classes[class]; ok {
			return c
		}
		if v.Default != nil {
			return *v.Default
		}
	}
	if c, ok := m.Default.Classes[class]; ok {
		return c
	}
	return *m.Default.Default
}

// Price computes the fill for a taker (or passive) order.
//
//	half_spread = spread/2 (quote) or price·spread_bps/2e4
//	impact      = price · k · σ · sqrt(notional/ADV)        (square-root impact)
//	fill        = price ± (half_spread + impact)            (+ buy, − sell)
//	fee         = max(commission_min_eur, |qty·fill| · commission_bps/1e4)
func (m *CostModel) Price(in CostInputs) CostResult {
	s := m.Schedule(in.Venue, in.Class)
	k := m.DefaultK
	if s.ImpactK != nil {
		k = *s.ImpactK
	}
	sigma, adv := in.Sigma, in.ADV
	if sigma <= 0 {
		sigma = m.DefaultSigma
	}
	if adv <= 0 {
		adv = m.DefaultADV
	}
	var half, impact float64
	if !in.Passive {
		if in.Spread > 0 {
			half = in.Spread / 2
		} else {
			half = in.Price * s.SpreadBps / 20_000
		}
		notional := math.Abs(in.Price * in.Qty)
		impact = in.Price * k * sigma * math.Sqrt(notional/adv)
	}
	dir := 1.0
	if strings.EqualFold(in.Side, "sell") {
		dir = -1
	}
	fill := in.Price + dir*(half+impact)
	if fill <= 0 {
		fill = in.Price // never cross zero on absurd inputs
	}
	notional := math.Abs(fill * in.Qty)
	fee := math.Max(s.CommissionMinEUR, notional*s.CommissionBps/10_000)
	var bps float64
	if in.Price > 0 {
		bps = dir * (fill - in.Price) / in.Price * 10_000
	}
	return CostResult{FillPrice: fill, Fee: fee, FeeCurrency: "EUR", HalfSpread: half, Impact: impact, SlippageBps: bps, Notional: notional}
}

// Fixture is the JSON export the Rust backtester (`crates/fa-backtest`) replays to match this model.
type Fixture struct {
	Model *CostModel    `json:"model"`
	Cases []FixtureCase `json:"cases"`
}

type FixtureCase struct {
	Name     string     `json:"name"`
	Inputs   CostInputs `json:"inputs"`
	Expected CostResult `json:"expected"`
}

// ExportFixture prices every case with the model and returns the fixture JSON.
func (m *CostModel) ExportFixture(cases map[string]CostInputs) ([]byte, error) {
	names := make([]string, 0, len(cases))
	for n := range cases {
		names = append(names, n)
	}
	sort.Strings(names)
	f := Fixture{Model: m}
	for _, n := range names {
		f.Cases = append(f.Cases, FixtureCase{Name: n, Inputs: cases[n], Expected: m.Price(cases[n])})
	}
	return json.MarshalIndent(f, "", "  ")
}
