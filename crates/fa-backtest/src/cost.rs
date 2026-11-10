//! Execution cost model (spec 41 §6.3): a port of the paper venue's model in
//! `go/services/broker-gateway/costs.go`, loading the same `costs.yaml` and pinned to
//! `go/services/broker-gateway/testdata/cost_fixture.json` so replay and paper fills never
//! drift.
//!
//! ```text
//! half_spread = spread / 2  (quote)  or  price · spread_bps / 2e4
//! impact      = price · k · σ_day · sqrt(notional / ADV)           (square-root impact)
//! fill        = price ± (half_spread + impact)                      (+ buy, − sell)
//! fee         = max(commission_min_eur, |qty · fill| · commission_bps / 1e4)
//! ```
//! Schedule lookup for `(venue, class)`: `venues.<v>.classes.<c>`, `venues.<v>.default`,
//! `default.classes.<c>`, `default.default` — first table found wins wholly. Funding / borrow
//! (per day of |notional|) are carried in the schedule as placeholders; the replay charges them
//! per held step ([`ClassCosts::carry`]), the gateway does not yet.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One fee schedule (`ClassCosts` in Go).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ClassCosts {
    pub commission_bps: f64,
    pub commission_min_eur: f64,
    /// Full quoted spread in bps; used when no quote spread is given.
    pub spread_bps: f64,
    /// Overrides the model's `default_k` when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub impact_k: Option<f64>,
    /// Funding per day in bps of |notional| (perps / FX swaps). Placeholder.
    pub funding_bps_day: f64,
    /// Borrow per day in bps of |notional| of shorts. Placeholder.
    pub borrow_bps_day: f64,
}

impl ClassCosts {
    /// Carry on `signed_notional` held for `days`: funding on any side, borrow on shorts.
    pub fn carry(&self, signed_notional: f64, days: f64) -> f64 {
        let mut bps = self.funding_bps_day;
        if signed_notional < 0.0 {
            bps += self.borrow_bps_day;
        }
        bps * signed_notional.abs() * 1e-4 * days.max(0.0)
    }
}

/// Schedules of one venue (or the model-wide defaults).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct VenueCosts {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<ClassCosts>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub classes: BTreeMap<String, ClassCosts>,
}

/// The cost model (`costs.yaml`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CostModel {
    /// Square-root impact coefficient.
    pub default_k: f64,
    /// Daily volatility (fraction) when none is known.
    pub default_sigma: f64,
    /// Daily traded notional (EUR) when none is known.
    pub default_adv_eur: f64,
    /// Asset class of instruments missing from `instruments`.
    pub default_class: String,
    /// Instrument UUID (dashed, lowercase) → asset class.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub instruments: BTreeMap<String, String>,
    pub default: VenueCosts,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub venues: BTreeMap<String, VenueCosts>,
    /// Name used in replay reports (not part of the Go model; `default` or the file stem).
    #[serde(skip)]
    pub id: String,
}

impl Default for CostModel {
    /// `DefaultCostModel()` of the gateway.
    fn default() -> Self {
        Self {
            default_k: 0.1,
            default_sigma: 0.02,
            default_adv_eur: 10_000_000.0,
            default_class: "equity".into(),
            instruments: BTreeMap::new(),
            default: VenueCosts {
                default: Some(ClassCosts {
                    commission_bps: 5.0,
                    commission_min_eur: 1.0,
                    spread_bps: 5.0,
                    ..Default::default()
                }),
                classes: BTreeMap::new(),
            },
            venues: BTreeMap::new(),
            id: "default".into(),
        }
    }
}

/// One taker (or passive) fill to price (`CostInputs`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CostInputs {
    pub venue: String,
    pub class: String,
    /// `buy` | `sell`
    pub side: String,
    /// Arrival price (last trade / bar close).
    pub price: f64,
    pub qty: f64,
    /// Absolute quoted spread; 0 → modelled from `spread_bps`.
    pub spread: f64,
    /// Daily volatility (fraction); 0 → model default.
    pub sigma: f64,
    /// Daily traded notional in EUR; 0 → model default.
    pub adv_eur: f64,
    pub passive: bool,
}

/// A priced fill (`CostResult`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CostResult {
    pub fill_price: f64,
    pub fee: f64,
    pub fee_currency: String,
    pub half_spread: f64,
    /// Price units.
    pub impact: f64,
    /// Signed slippage vs arrival in bps (positive = paid by the taker).
    pub slippage_bps: f64,
    /// At the fill price.
    pub notional: f64,
}

impl CostResult {
    /// Total explicit + implicit cost in quote currency: `|qty| · (half_spread + impact) + fee`.
    pub fn total_cost(&self, qty: f64) -> f64 {
        qty.abs() * (self.half_spread + self.impact) + self.fee
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CostError {
    #[error("parse: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("invalid cost model: {0}")]
    Invalid(String),
}

impl CostModel {
    /// Parses `costs.yaml` on top of the defaults (as `LoadCostModel`) and validates.
    pub fn from_yaml(src: &str) -> Result<Self, CostError> {
        // Fields absent from the file keep the Go defaults (`default.default` in particular).
        let mut m: CostModel = serde_yaml::from_str(src)?;
        let d = CostModel::default();
        if m.default_k == 0.0 && m.default_sigma == 0.0 && m.default_adv_eur == 0.0 {
            m.default_k = d.default_k;
        }
        if m.default_sigma == 0.0 {
            m.default_sigma = d.default_sigma;
        }
        if m.default_adv_eur == 0.0 {
            m.default_adv_eur = d.default_adv_eur;
        }
        if m.default_class.is_empty() {
            m.default_class = d.default_class;
        }
        if m.default.default.is_none() {
            m.default.default = d.default.default;
        }
        m.id = "costs.yaml".into();
        m.validate()?;
        Ok(m)
    }

    /// Same as `from_yaml` for the JSON form of the fixture's `model`.
    pub fn from_json(v: &serde_json::Value) -> Result<Self, CostError> {
        let mut m: CostModel = serde_json::from_value(v.clone()).map_err(|e| CostError::Invalid(e.to_string()))?;
        if m.default.default.is_none() {
            m.default.default = CostModel::default().default.default;
        }
        m.id = "fixture".into();
        m.validate()?;
        Ok(m)
    }

    pub fn validate(&self) -> Result<(), CostError> {
        if self.default_k < 0.0 || self.default_sigma <= 0.0 || self.default_adv_eur <= 0.0 {
            return Err(CostError::Invalid("default_k must be ≥ 0, default_sigma and default_adv_eur > 0".into()));
        }
        let Some(dd) = self.default.default else {
            return Err(CostError::Invalid("default.default fee schedule is required".into()));
        };
        let check = |where_: String, c: &ClassCosts| {
            if c.commission_bps < 0.0
                || c.commission_min_eur < 0.0
                || c.spread_bps < 0.0
                || c.impact_k.is_some_and(|k| k < 0.0)
            {
                Err(CostError::Invalid(format!("{where_}: negative cost")))
            } else {
                Ok(())
            }
        };
        check("default.default".into(), &dd)?;
        for (c, cc) in &self.default.classes {
            check(format!("default.classes.{c}"), cc)?;
        }
        for (v, vc) in &self.venues {
            if let Some(d) = &vc.default {
                check(format!("venues.{v}.default"), d)?;
            }
            for (c, cc) in &vc.classes {
                check(format!("venues.{v}.classes.{c}"), cc)?;
            }
        }
        Ok(())
    }

    /// Asset class of an instrument (dashed or dashless UUID); unknown → `default_class`.
    pub fn class_of(&self, instrument_id: &str) -> String {
        let want = instrument_id.trim().to_ascii_lowercase().replace('-', "");
        self.instruments
            .iter()
            .find(|(k, _)| k.to_ascii_lowercase().replace('-', "") == want)
            .map_or_else(|| self.default_class.clone(), |(_, c)| c.clone())
    }

    /// Fee schedule for a venue / class pair (see module docs for the lookup order).
    pub fn schedule(&self, venue: &str, class: &str) -> ClassCosts {
        if let Some(v) = self.venues.get(venue) {
            if let Some(c) = v.classes.get(class) {
                return *c;
            }
            if let Some(d) = v.default {
                return d;
            }
        }
        if let Some(c) = self.default.classes.get(class) {
            return *c;
        }
        self.default.default.unwrap_or_default()
    }

    /// Prices a fill exactly as `CostModel.Price` in Go.
    pub fn price(&self, inp: &CostInputs) -> CostResult {
        let s = self.schedule(&inp.venue, &inp.class);
        let k = s.impact_k.unwrap_or(self.default_k);
        let sigma = if inp.sigma > 0.0 { inp.sigma } else { self.default_sigma };
        let adv = if inp.adv_eur > 0.0 { inp.adv_eur } else { self.default_adv_eur };
        let (mut half, mut impact) = (0.0, 0.0);
        if !inp.passive {
            half = if inp.spread > 0.0 { inp.spread / 2.0 } else { inp.price * s.spread_bps / 20_000.0 };
            let notional = (inp.price * inp.qty).abs();
            impact = inp.price * k * sigma * (notional / adv).sqrt();
        }
        let dir = if inp.side.eq_ignore_ascii_case("sell") { -1.0 } else { 1.0 };
        let mut fill = inp.price + dir * (half + impact);
        if fill <= 0.0 {
            fill = inp.price;
        }
        let notional = (fill * inp.qty).abs();
        let fee = s.commission_min_eur.max(notional * s.commission_bps / 10_000.0);
        let bps = if inp.price > 0.0 { dir * (fill - inp.price) / inp.price * 10_000.0 } else { 0.0 };
        CostResult {
            fill_price: fill,
            fee,
            fee_currency: "EUR".into(),
            half_spread: half,
            impact,
            slippage_bps: bps,
            notional,
        }
    }

    /// A model with every proportional cost multiplied by `factor` (the 2×-cost sensitivity).
    pub fn scaled(&self, factor: f64) -> Self {
        let sc = |c: &ClassCosts| ClassCosts {
            commission_bps: c.commission_bps * factor,
            commission_min_eur: c.commission_min_eur * factor,
            spread_bps: c.spread_bps * factor,
            impact_k: c.impact_k.map(|k| k * factor),
            funding_bps_day: c.funding_bps_day * factor,
            borrow_bps_day: c.borrow_bps_day * factor,
        };
        let vc = |v: &VenueCosts| VenueCosts {
            default: v.default.as_ref().map(sc),
            classes: v.classes.iter().map(|(k, c)| (k.clone(), sc(c))).collect(),
        };
        Self {
            default_k: self.default_k * factor,
            default_sigma: self.default_sigma,
            default_adv_eur: self.default_adv_eur,
            default_class: self.default_class.clone(),
            instruments: self.instruments.clone(),
            default: vc(&self.default),
            venues: self.venues.iter().map(|(k, v)| (k.clone(), vc(v))).collect(),
            id: format!("{}x{factor}", self.id),
        }
    }
}

/// Path of the shared cost file relative to the repository root.
pub const COSTS_YAML_PATH: &str = "go/services/broker-gateway/costs.yaml";

#[cfg(test)]
mod tests {
    use super::*;

    const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

    #[test]
    fn default_model_prices_like_the_gateway() {
        let m = CostModel::default();
        let r = m.price(&CostInputs {
            venue: "unknown".into(),
            class: "equity".into(),
            side: "buy".into(),
            price: 100.0,
            qty: 1000.0,
            sigma: 0.02,
            adv_eur: 10_000_000.0,
            ..Default::default()
        });
        // half = 100·5/20000 = 0.025; impact = 100·0.1·0.02·sqrt(1e5/1e7) = 0.02
        assert!((r.half_spread - 0.025).abs() < 1e-12 && (r.impact - 0.02).abs() < 1e-12);
        assert!((r.fill_price - 100.045).abs() < 1e-9);
        assert!((r.fee - 50.0225).abs() < 1e-9);
        assert!((r.total_cost(1000.0) - (45.0 + 50.0225)).abs() < 1e-9);
        let passive = m.price(&CostInputs { passive: true, price: 100.0, qty: 1.0, ..Default::default() });
        assert_eq!((passive.half_spread, passive.impact, passive.fill_price), (0.0, 0.0, 100.0));
        assert_eq!(passive.fee, 1.0, "commission floor");
        let sell = m.price(&CostInputs { side: "sell".into(), price: 100.0, qty: 1.0, ..Default::default() });
        assert!(sell.fill_price < 100.0 && sell.slippage_bps > 0.0);
        assert_eq!(m.class_of("01900000-0000-7000-8000-000000000001"), "equity");
        let twice = m.scaled(2.0);
        assert_eq!(twice.default.default.unwrap().spread_bps, 10.0);
        assert_eq!(twice.default_k, 0.2);
        // borrow only on shorts
        let c = ClassCosts { funding_bps_day: 1.0, borrow_bps_day: 0.5, ..Default::default() };
        assert!((c.carry(-1000.0, 1.0) - 0.15).abs() < 1e-12 && (c.carry(1000.0, 2.0) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn loads_the_gateway_yaml_with_lookup_order() {
        let src = std::fs::read_to_string(format!("{REPO}/{COSTS_YAML_PATH}")).expect("costs.yaml");
        let m = CostModel::from_yaml(&src).unwrap();
        assert_eq!(m.class_of("01900000000070008000000000000001"), "equity");
        assert_eq!(m.schedule("sim", "equity").spread_bps, 0.0, "venue default beats default.classes");
        assert_eq!(m.schedule("yahoo", "crypto").impact_k, Some(0.2));
        assert_eq!(m.schedule("yahoo", "fx").funding_bps_day, 0.5, "falls through to default.classes.fx");
        assert_eq!(m.schedule("nope", "nope").commission_bps, 5.0);
        assert!(CostModel::from_yaml("default_sigma: -1\n").is_err());
        assert!(CostModel::from_yaml("default:\n  classes:\n    fx: {commission_bps: -1}\n").is_err());
    }

    /// Every case of the gateway's exported fixture must price identically here.
    #[test]
    fn matches_broker_gateway_fixture() {
        let path = format!("{REPO}/go/services/broker-gateway/testdata/cost_fixture.json");
        let src = std::fs::read_to_string(&path).expect("cost_fixture.json (exported by broker-gateway tests)");
        let v: serde_json::Value = serde_json::from_str(&src).unwrap();
        let model = CostModel::from_json(&v["model"]).unwrap();
        let cases = v["cases"].as_array().expect("cases");
        assert!(cases.len() >= 3);
        for c in cases {
            let inputs: CostInputs = serde_json::from_value(c["inputs"].clone()).unwrap();
            let want: CostResult = serde_json::from_value(c["expected"].clone()).unwrap();
            let got = model.price(&inputs);
            let name = c["name"].as_str().unwrap_or("?");
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);
            assert!(close(got.fill_price, want.fill_price), "{name}: fill {} != {}", got.fill_price, want.fill_price);
            assert!(close(got.fee, want.fee), "{name}: fee {} != {}", got.fee, want.fee);
            assert!(close(got.half_spread, want.half_spread), "{name}: half_spread");
            assert!(close(got.impact, want.impact), "{name}: impact {} != {}", got.impact, want.impact);
            assert!(close(got.slippage_bps, want.slippage_bps), "{name}: slippage_bps");
            assert!(close(got.notional, want.notional), "{name}: notional");
            assert_eq!(got.fee_currency, want.fee_currency);
        }
    }
}
