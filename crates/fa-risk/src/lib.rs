//! Non-evolvable hard risk limits and the halt state machine (spec 40).
//! Limits come from config only; strategies and evolution can never change them.
//!
//! [`RiskEngine::clip_targets`] applies the caps; [`RiskEngine::gate`] applies the halt
//! semantics on top (kill switch → flatten, daily loss → reduce-only). [`HaltScope`] maps
//! to and from the bus scope strings of `risk.halt.<scope>` (spec 21, D-030).
//! [`RiskConfig::check_tightens`] enforces "runtime changes may only tighten" (spec 40 §6.5).
//! [`strategy`] holds the strategy lifecycle (`active|paused|flattening|shadow`), the
//! per-strategy drawdown tracker and the promotion gate checklist (spec 40 §6.6).

pub mod strategy;

use std::collections::{BTreeMap, BTreeSet};

use fa_core::{AssetClass, InstrumentId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RiskConfig {
    /// Portfolio drawdown from peak equity that triggers flatten + halt.
    pub max_drawdown: f64,
    /// Per-strategy drawdown of attributed P&L, as a fraction of the strategy's sleeve, that
    /// pauses (and flattens) the strategy (spec 40 §6.5).
    pub strategy_max_drawdown: f64,
    /// Daily loss (fraction of start-of-day equity) that stops trading for the day.
    pub max_daily_loss: f64,
    /// Max |position| per instrument as a fraction of deployable capital.
    pub max_position_frac: f64,
    /// Max gross exposure per asset class as a fraction of deployable capital (concentration).
    pub max_class_frac: f64,
    /// Max gross exposure per venue as a fraction of deployable capital (counterparty risk).
    pub max_venue_frac: f64,
    /// Max gross exposure per asset class as a multiple of deployable capital.
    pub leverage_cap: BTreeMap<AssetClass, f64>,
    /// Portfolio-wide gross exposure cap as a multiple of deployable capital.
    pub portfolio_gross_cap: f64,
    /// Liquidity cap: |position| ≤ k × traded volume (EUR) over the horizon.
    pub liquidity_k: f64,
    /// Capital reserved per day for payouts (spec 40 §6.5 liquidity cap net of payouts). The
    /// larger of this and the day's `ledger.payout_quota` row is reserved; both only rise at
    /// runtime (`ctl.risk.limits` upserts the row).
    pub daily_payout_quota_eur: f64,
    /// Position cap for instruments whose reference data is still `provisional` (D-050: symbol
    /// provisioned by the resolver, not yet reviewed), as a fraction of deployable capital.
    /// `None` = half of `max_position_frac`.
    #[serde(default)]
    pub provisional_position_frac: Option<f64>,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_drawdown: 0.15,
            strategy_max_drawdown: 0.05,
            max_daily_loss: 0.03,
            max_position_frac: 0.10,
            max_class_frac: 0.40,
            max_venue_frac: 0.50,
            leverage_cap: BTreeMap::from([
                (AssetClass::Equity, 1.0),
                (AssetClass::Etf, 1.0),
                (AssetClass::Fx, 2.0),
                (AssetClass::Crypto, 1.0),
            ]),
            portfolio_gross_cap: 2.0,
            liquidity_k: 0.01,
            daily_payout_quota_eur: 0.0,
            provisional_position_frac: None,
        }
    }
}

impl RiskConfig {
    /// Effective position cap (fraction of deployable) for a `provisional` instrument.
    pub fn provisional_position_frac(&self) -> f64 {
        self.provisional_position_frac.unwrap_or(self.max_position_frac * 0.5).min(self.max_position_frac)
    }

    /// Checks that `new` is at least as strict as `self` in every field (spec 40 §6.5: runtime
    /// changes may only tighten; loosening needs a config change + restart). Returns the list
    /// of fields that would loosen, empty when the change is acceptable. A leverage class that
    /// is absent has cap 0, so adding a class loosens and removing one tightens.
    pub fn check_tightens(&self, new: &RiskConfig) -> Vec<String> {
        let mut bad = Vec::new();
        let mut le = |name: &str, old: f64, new: f64| {
            if !new.is_finite() || new < 0.0 {
                bad.push(format!("{name}: {new} is not a valid limit"));
            } else if new > old {
                bad.push(format!("{name}: {old} → {new} would loosen"));
            }
        };
        le("max_drawdown", self.max_drawdown, new.max_drawdown);
        le("strategy_max_drawdown", self.strategy_max_drawdown, new.strategy_max_drawdown);
        le("max_daily_loss", self.max_daily_loss, new.max_daily_loss);
        le("max_position_frac", self.max_position_frac, new.max_position_frac);
        le("provisional_position_frac", self.provisional_position_frac(), new.provisional_position_frac());
        le("max_class_frac", self.max_class_frac, new.max_class_frac);
        le("max_venue_frac", self.max_venue_frac, new.max_venue_frac);
        le("portfolio_gross_cap", self.portfolio_gross_cap, new.portfolio_gross_cap);
        le("liquidity_k", self.liquidity_k, new.liquidity_k);
        for (class, cap) in &new.leverage_cap {
            let old = self.leverage_cap.get(class).copied().unwrap_or(0.0);
            le(&format!("leverage_cap.{class:?}").to_lowercase(), old, *cap);
        }
        if !new.daily_payout_quota_eur.is_finite() || new.daily_payout_quota_eur < self.daily_payout_quota_eur {
            bad.push(format!(
                "daily_payout_quota_eur: {} → {} would loosen (less reserved)",
                self.daily_payout_quota_eur, new.daily_payout_quota_eur
            ));
        }
        bad
    }
}

/// Inputs for one pre-trade check.
#[derive(Debug, Clone)]
pub struct MarketInfo {
    pub asset_class: AssetClass,
    /// Venue the instrument trades on (empty when unknown; still capped as one bucket).
    pub venue: String,
    /// Traded volume in EUR over the strategy horizon.
    pub horizon_volume_eur: f64,
    /// Reference data still `provisional` (D-050): the stricter
    /// [`RiskConfig::provisional_position_frac`] applies.
    pub provisional: bool,
}

impl MarketInfo {
    pub fn new(asset_class: AssetClass, venue: &str, horizon_volume_eur: f64) -> Self {
        Self { asset_class, venue: venue.into(), horizon_volume_eur, provisional: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HaltScope {
    Portfolio,
    DailyLoss,
    Source,
    Model,
    Broker,
    /// Accounting FX stale beyond the configured age (spec 42 §8, spec 12 §8): reduce-only, raised
    /// by the risk-engine itself when `fx.halt_on_stale` is set; clears when a fresh rate lands.
    Fx,
}

impl HaltScope {
    /// Bus scope token used in `risk.halt.<scope>` / `risk.resume.<scope>` and `Halt.scope`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Portfolio => "portfolio",
            Self::DailyLoss => "daily_loss",
            Self::Source => "source",
            Self::Model => "model",
            Self::Broker => "broker",
            Self::Fx => "fx",
        }
    }

    /// Parses a bus scope; `source:<id>`, `venue:<id>` style qualifiers map to their family.
    /// Strategy- and asset-level scopes are not portfolio halts and yield `None`.
    pub fn parse(scope: &str) -> Option<Self> {
        let family = scope.split(':').next().unwrap_or(scope);
        Some(match family {
            "portfolio" => Self::Portfolio,
            "daily_loss" => Self::DailyLoss,
            "source" | "feed" => Self::Source,
            "model" | "snapshot" => Self::Model,
            "broker" | "venue" => Self::Broker,
            "fx" => Self::Fx,
            _ => return None,
        })
    }

    /// Halts that imply flattening everything (kill switch and health halts); the daily-loss
    /// and stale-FX halts only forbid risk-increasing orders.
    pub fn flattens(self) -> bool {
        !matches!(self, Self::DailyLoss | Self::Fx)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HaltReason {
    Drawdown(f64),
    DailyLoss(f64),
    Health(String),
    Manual(String),
}

/// Risk state machine: tracks peak equity, start-of-day equity and active halts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEngine {
    pub cfg: RiskConfig,
    peak_equity: f64,
    day_start_equity: f64,
    halts: BTreeMap<HaltScope, HaltReason>,
    /// Today's reserved payout from the ledger (`ledger.payout_quota`, spec 42 §6), if larger
    /// than the configured quota.
    #[serde(default)]
    reserved_payout_eur: f64,
}

impl RiskEngine {
    pub fn new(cfg: RiskConfig, equity: f64) -> Self {
        Self { cfg, peak_equity: equity, day_start_equity: equity, halts: BTreeMap::new(), reserved_payout_eur: 0.0 }
    }

    /// Payout reserved today: the larger of the configured quota and the ledger's row.
    pub fn payout_reserve(&self) -> f64 {
        self.cfg.daily_payout_quota_eur.max(self.reserved_payout_eur).max(0.0)
    }

    /// Sets the ledger-side reserve (`quota − paid` of today's `ledger.payout_quota` row).
    pub fn set_reserved_payout(&mut self, eur: f64) {
        self.reserved_payout_eur = if eur.is_finite() { eur.max(0.0) } else { 0.0 };
    }

    /// Deployable capital = equity − reserved daily payout quota (never negative).
    pub fn deployable(&self, equity: f64) -> f64 {
        (equity - self.payout_reserve()).max(0.0)
    }

    /// New trading day: resets the daily-loss baseline and clears the daily-loss halt.
    pub fn start_day(&mut self, equity: f64) {
        self.day_start_equity = equity;
        self.halts.remove(&HaltScope::DailyLoss);
    }

    /// Updates equity; returns newly triggered halts.
    pub fn on_equity(&mut self, equity: f64) -> Vec<(HaltScope, HaltReason)> {
        self.peak_equity = self.peak_equity.max(equity);
        let mut new = Vec::new();
        let dd = if self.peak_equity > 0.0 { 1.0 - equity / self.peak_equity } else { 0.0 };
        if dd >= self.cfg.max_drawdown && !self.halts.contains_key(&HaltScope::Portfolio) {
            new.push((HaltScope::Portfolio, HaltReason::Drawdown(dd)));
        }
        let dl = if self.day_start_equity > 0.0 { 1.0 - equity / self.day_start_equity } else { 0.0 };
        if dl >= self.cfg.max_daily_loss && !self.halts.contains_key(&HaltScope::DailyLoss) {
            new.push((HaltScope::DailyLoss, HaltReason::DailyLoss(dl)));
        }
        for (s, r) in &new {
            self.halts.insert(*s, r.clone());
        }
        new
    }

    pub fn halt(&mut self, scope: HaltScope, reason: HaltReason) {
        self.halts.insert(scope, reason);
    }

    /// Only manual action may clear the portfolio (kill-switch) halt.
    pub fn clear(&mut self, scope: HaltScope) {
        self.halts.remove(&scope);
    }

    /// Manual re-arm after a kill switch (spec 40 §6.5): clears the portfolio halt and
    /// restarts the drawdown peak from the current equity, so the halt does not re-fire
    /// immediately. Daily-loss and health halts are unaffected.
    pub fn rearm(&mut self, equity: f64) {
        self.halts.remove(&HaltScope::Portfolio);
        self.peak_equity = equity;
    }

    pub fn peak_equity(&self) -> f64 {
        self.peak_equity
    }

    pub fn day_start_equity(&self) -> f64 {
        self.day_start_equity
    }

    pub fn halts(&self) -> &BTreeMap<HaltScope, HaltReason> {
        &self.halts
    }

    pub fn trading_allowed(&self) -> bool {
        self.halts.is_empty()
    }

    /// Only the daily-loss halt is active: existing risk may be reduced, not increased.
    pub fn reduce_only(&self) -> bool {
        !self.halts.is_empty() && self.halts.keys().all(|s| !s.flattens())
    }

    /// Applies the halt semantics to already-clipped targets given current positions (EUR,
    /// signed): any flattening halt → all targets 0; reduce-only → each target is pulled
    /// toward 0 no further than the current position allows (same sign, smaller magnitude);
    /// otherwise targets pass through.
    pub fn gate(
        &self,
        clipped: &BTreeMap<InstrumentId, f64>,
        current: &BTreeMap<InstrumentId, f64>,
    ) -> BTreeMap<InstrumentId, f64> {
        gate_for(self.halts.keys().copied(), clipped, current)
    }

    /// [`Self::gate`] with some scopes left out (the caller handles them per instrument, e.g.
    /// a `source:<id>` halt that only flattens the instruments that source feeds).
    pub fn gate_excluding(
        &self,
        excluded: &[HaltScope],
        clipped: &BTreeMap<InstrumentId, f64>,
        current: &BTreeMap<InstrumentId, f64>,
    ) -> BTreeMap<InstrumentId, f64> {
        gate_for(self.halts.keys().copied().filter(|s| !excluded.contains(s)), clipped, current)
    }

    /// Clips target positions (EUR, signed) to every hard limit. When halted (any scope), all
    /// targets become 0 (flatten); use [`Self::gate`] for the reduce-only daily-loss semantics.
    pub fn clip_targets(
        &self,
        targets: &BTreeMap<InstrumentId, f64>,
        market: &BTreeMap<InstrumentId, MarketInfo>,
        equity: f64,
    ) -> BTreeMap<InstrumentId, f64> {
        if !self.trading_allowed() {
            return targets.keys().map(|k| (*k, 0.0)).collect();
        }
        self.cap_targets(targets, market, equity)
    }

    /// The position, liquidity and leverage caps alone, ignoring halts (combine with
    /// [`Self::gate`] for the full chain).
    pub fn cap_targets(
        &self,
        targets: &BTreeMap<InstrumentId, f64>,
        market: &BTreeMap<InstrumentId, MarketInfo>,
        equity: f64,
    ) -> BTreeMap<InstrumentId, f64> {
        let deployable = self.deployable(equity);
        let pos_cap = self.cfg.max_position_frac * deployable;
        let prov_cap = self.cfg.provisional_position_frac() * deployable;
        // per-instrument caps: position (stricter for provisional reference data) and liquidity
        let mut clipped: BTreeMap<InstrumentId, f64> = targets
            .iter()
            .map(|(i, &t)| {
                let Some(m) = market.get(i) else {
                    return (*i, 0.0);
                }; // unknown market → no position
                let liq_cap = self.cfg.liquidity_k * m.horizon_volume_eur;
                let cap = if m.provisional { prov_cap } else { pos_cap }.min(liq_cap);
                (*i, t.clamp(-cap, cap))
            })
            .collect();
        // per-asset-class gross leverage caps: scale the class down proportionally
        let mut gross: BTreeMap<AssetClass, f64> = BTreeMap::new();
        for (i, v) in &clipped {
            if let Some(m) = market.get(i) {
                *gross.entry(m.asset_class).or_default() += v.abs();
            }
        }
        let classes: BTreeSet<AssetClass> = gross.keys().copied().collect();
        for c in classes {
            // leverage cap and concentration cap per class: the tighter one binds
            let cap = self.cfg.leverage_cap.get(&c).copied().unwrap_or(0.0).min(self.cfg.max_class_frac) * deployable;
            let g = gross[&c];
            if g > cap {
                let scale = if g > 0.0 { cap / g } else { 0.0 };
                for (i, v) in clipped.iter_mut() {
                    if market.get(i).is_some_and(|m| m.asset_class == c) {
                        *v *= scale;
                    }
                }
            }
        }
        // per-venue concentration cap (counterparty risk)
        let mut by_venue: BTreeMap<&str, f64> = BTreeMap::new();
        for (i, v) in &clipped {
            if let Some(m) = market.get(i) {
                *by_venue.entry(m.venue.as_str()).or_default() += v.abs();
            }
        }
        let venue_cap = self.cfg.max_venue_frac * deployable;
        let over: Vec<(String, f64)> =
            by_venue.iter().filter(|(_, g)| **g > venue_cap).map(|(v, g)| ((*v).to_owned(), *g)).collect();
        for (venue, g) in over {
            let scale = if g > 0.0 { venue_cap / g } else { 0.0 };
            for (i, v) in clipped.iter_mut() {
                if market.get(i).is_some_and(|m| m.venue == venue) {
                    *v *= scale;
                }
            }
        }
        // portfolio-wide gross cap
        let total: f64 = clipped.values().map(|v| v.abs()).sum();
        let gross_cap = self.cfg.portfolio_gross_cap * deployable;
        if total > gross_cap {
            let scale = if total > 0.0 { gross_cap / total } else { 0.0 };
            for v in clipped.values_mut() {
                *v *= scale;
            }
        }
        clipped
    }
}

/// Halt semantics for a given set of active scopes (see [`RiskEngine::gate`]).
pub fn gate_for(
    halts: impl IntoIterator<Item = HaltScope>,
    clipped: &BTreeMap<InstrumentId, f64>,
    current: &BTreeMap<InstrumentId, f64>,
) -> BTreeMap<InstrumentId, f64> {
    let halts: Vec<HaltScope> = halts.into_iter().collect();
    if halts.is_empty() {
        return clipped.clone();
    }
    if halts.iter().any(|s| s.flattens()) {
        return clipped.keys().chain(current.keys()).map(|k| (*k, 0.0)).collect();
    }
    clipped
        .iter()
        .map(|(i, &t)| {
            let c = current.get(i).copied().unwrap_or(0.0);
            let v = if c == 0.0 || t.signum() != c.signum() { 0.0 } else { t.abs().min(c.abs()) * c.signum() };
            (*i, v)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> RiskEngine {
        RiskEngine::new(RiskConfig { daily_payout_quota_eur: 1_000.0, ..RiskConfig::default() }, 11_000.0)
    }

    #[test]
    fn deployable_nets_payout_quota() {
        assert_eq!(engine().deployable(11_000.0), 10_000.0);
        assert_eq!(engine().deployable(500.0), 0.0);
        // the ledger's reserved quota binds when larger than the configured one, never smaller
        let mut e = engine();
        e.set_reserved_payout(2_500.0);
        assert_eq!((e.payout_reserve(), e.deployable(11_000.0)), (2_500.0, 8_500.0));
        e.set_reserved_payout(100.0);
        assert_eq!(e.deployable(11_000.0), 10_000.0);
        e.set_reserved_payout(f64::NAN);
        assert_eq!(e.deployable(11_000.0), 10_000.0);
    }

    #[test]
    fn provisional_instruments_get_the_stricter_position_cap() {
        let mut e = engine();
        assert_eq!(e.cfg.provisional_position_frac(), 0.05, "default: half the position cap");
        let (a, b) = (InstrumentId::new(), InstrumentId::new());
        let t = BTreeMap::from([(a, 5_000.0), (b, 5_000.0)]);
        let mut m = BTreeMap::from([
            (a, MarketInfo::new(AssetClass::Equity, "yahoo", 1e9)),
            (b, MarketInfo::new(AssetClass::Equity, "yahoo", 1e9)),
        ]);
        m.get_mut(&b).unwrap().provisional = true;
        let c = e.cap_targets(&t, &m, 11_000.0);
        assert_eq!((c[&a], c[&b]), (1_000.0, 500.0));
        e.cfg.provisional_position_frac = Some(0.02);
        assert_eq!(e.cap_targets(&t, &m, 11_000.0)[&b], 200.0);
        e.cfg.provisional_position_frac = Some(0.5);
        assert_eq!(e.cfg.provisional_position_frac(), 0.10, "never looser than the position cap");
        // tighten-only covers the effective provisional cap
        let old = RiskConfig::default();
        let mut looser = old.clone();
        looser.provisional_position_frac = Some(0.08);
        assert!(old.check_tightens(&looser)[0].starts_with("provisional_position_frac"));
        let mut tighter = old.clone();
        tighter.provisional_position_frac = Some(0.01);
        assert!(old.check_tightens(&tighter).is_empty());
    }

    #[test]
    fn gate_excluding_leaves_a_scope_to_the_caller() {
        let (a, b) = (InstrumentId::new(), InstrumentId::new());
        let targets = BTreeMap::from([(a, 500.0), (b, -300.0)]);
        let current = BTreeMap::from([(a, 200.0), (b, -800.0)]);
        let mut e = RiskEngine::new(RiskConfig::default(), 10_000.0);
        e.halt(HaltScope::Source, HaltReason::Health("yahoo down".into()));
        assert!(e.gate(&targets, &current).values().all(|v| *v == 0.0), "a source halt flattens by default");
        assert_eq!(e.gate_excluding(&[HaltScope::Source], &targets, &current), targets);
        e.halt(HaltScope::Fx, HaltReason::Health("USD stale".into()));
        assert!(!e.reduce_only(), "source + fx: the flattening scope wins");
        let g = e.gate_excluding(&[HaltScope::Source], &targets, &current);
        assert_eq!((g[&a], g[&b]), (200.0, -300.0), "fx alone is reduce-only");
        e.clear(HaltScope::Source);
        assert!(e.reduce_only());
    }

    #[test]
    fn position_and_liquidity_caps() {
        let e = engine();
        let (a, b) = (InstrumentId::new(), InstrumentId::new());
        let t = BTreeMap::from([(a, 5_000.0), (b, -5_000.0)]);
        let m = BTreeMap::from([
            (a, MarketInfo::new(AssetClass::Equity, "x", 1e9)),
            (b, MarketInfo::new(AssetClass::Crypto, "x", 50_000.0)),
        ]);
        let c = e.clip_targets(&t, &m, 11_000.0);
        assert_eq!(c[&a], 1_000.0); // 10% of 10k deployable
        assert_eq!(c[&b], -500.0); // 1% of 50k volume
    }

    #[test]
    fn leverage_cap_scales_class() {
        let mut e = engine();
        e.cfg.max_position_frac = 10.0;
        e.cfg.max_class_frac = 1.0; // isolate the leverage cap from the concentration cap
        e.cfg.leverage_cap.insert(AssetClass::Fx, 0.5);
        let (a, b) = (InstrumentId::new(), InstrumentId::new());
        let t = BTreeMap::from([(a, 4_000.0), (b, -6_000.0)]);
        let m: BTreeMap<_, _> = [a, b].into_iter().map(|i| (i, MarketInfo::new(AssetClass::Fx, "x", 1e12))).collect();
        let c = e.clip_targets(&t, &m, 11_000.0);
        assert!((c[&a].abs() + c[&b].abs() - 5_000.0).abs() < 1e-6);
    }

    #[test]
    fn concentration_caps_per_class_venue_and_portfolio() {
        let mut e = engine();
        e.cfg.max_position_frac = 10.0;
        e.cfg.leverage_cap = BTreeMap::from([(AssetClass::Equity, 5.0), (AssetClass::Crypto, 5.0)]);
        e.cfg.max_class_frac = 0.5; // 5 000 of 10 000 deployable per class
        e.cfg.max_venue_frac = 0.7; // 7 000 per venue
        e.cfg.portfolio_gross_cap = 10.0;
        let (a, b, c) = (InstrumentId::new(), InstrumentId::new(), InstrumentId::new());
        let m = BTreeMap::from([
            (a, MarketInfo::new(AssetClass::Equity, "v1", 1e12)),
            (b, MarketInfo::new(AssetClass::Equity, "v1", 1e12)),
            (c, MarketInfo::new(AssetClass::Crypto, "v1", 1e12)),
        ]);
        let t = BTreeMap::from([(a, 4_000.0), (b, -4_000.0), (c, 4_000.0)]);
        let r = e.cap_targets(&t, &m, 11_000.0);
        // equity class 8 000 → 5 000 (a 2 500, b −2 500); crypto 4 000 stays; venue 9 000 → 7 000
        let gross: f64 = r.values().map(|v| v.abs()).sum();
        assert!((gross - 7_000.0).abs() < 1e-6, "venue cap binds: {r:?}");
        assert!(r[&a] > 0.0 && r[&b] < 0.0 && (r[&a] + r[&b]).abs() < 1e-9, "class scaled proportionally");
        e.cfg.portfolio_gross_cap = 0.3; // 3 000 total
        let r = e.cap_targets(&t, &m, 11_000.0);
        let gross: f64 = r.values().map(|v| v.abs()).sum();
        assert!((gross - 3_000.0).abs() < 1e-6, "portfolio cap binds: {r:?}");
    }

    #[test]
    fn runtime_limit_changes_may_only_tighten() {
        let old = RiskConfig::default();
        assert!(old.check_tightens(&old).is_empty(), "same limits are fine");
        let mut tighter = old.clone();
        tighter.max_drawdown = 0.05;
        tighter.leverage_cap.remove(&AssetClass::Fx);
        tighter.daily_payout_quota_eur = 100.0;
        assert!(old.check_tightens(&tighter).is_empty());
        let mut looser = old.clone();
        looser.max_daily_loss = 0.5;
        looser.leverage_cap.insert(AssetClass::Commodity, 1.0);
        looser.daily_payout_quota_eur = -1.0;
        let bad = old.check_tightens(&looser);
        assert_eq!(bad.len(), 3, "{bad:?}");
        assert!(bad[0].starts_with("max_daily_loss"));
        assert!(bad.iter().any(|b| b.starts_with("leverage_cap.commodity")));
        let mut nan = old.clone();
        nan.liquidity_k = f64::NAN;
        assert_eq!(old.check_tightens(&nan).len(), 1);
    }

    #[test]
    fn scope_strings_round_trip_and_qualify() {
        for s in [
            HaltScope::Portfolio,
            HaltScope::DailyLoss,
            HaltScope::Source,
            HaltScope::Model,
            HaltScope::Broker,
            HaltScope::Fx,
        ] {
            assert_eq!(HaltScope::parse(s.as_str()), Some(s));
        }
        assert_eq!(HaltScope::parse("venue:lemon"), Some(HaltScope::Broker));
        assert_eq!(HaltScope::parse("strategy:s1"), None);
        assert!(HaltScope::Portfolio.flattens() && !HaltScope::DailyLoss.flattens() && !HaltScope::Fx.flattens());
    }

    #[test]
    fn gate_flattens_or_reduces_only() {
        let (a, b, c) = (InstrumentId::new(), InstrumentId::new(), InstrumentId::new());
        let targets = BTreeMap::from([(a, 500.0), (b, -300.0), (c, 200.0)]);
        let current = BTreeMap::from([(a, 200.0), (b, -800.0), (c, -100.0)]);
        let mut e = RiskEngine::new(RiskConfig::default(), 10_000.0);
        assert_eq!(e.gate(&targets, &current), targets);
        e.halt(HaltScope::DailyLoss, HaltReason::DailyLoss(0.05));
        assert!(e.reduce_only());
        let g = e.gate(&targets, &current);
        assert_eq!(g[&a], 200.0, "cannot add to a long");
        assert_eq!(g[&b], -300.0, "may reduce a short");
        assert_eq!(g[&c], 0.0, "flipping sides is closing, not opening");
        e.halt(HaltScope::Portfolio, HaltReason::Manual("ui".into()));
        assert!(!e.reduce_only());
        assert!(e.gate(&targets, &current).values().all(|v| *v == 0.0));
    }

    #[test]
    fn drawdown_and_daily_loss_halt_and_flatten() {
        let mut e = RiskEngine::new(RiskConfig::default(), 10_000.0);
        assert!(e.on_equity(9_800.0).is_empty());
        let h = e.on_equity(9_650.0);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].0, HaltScope::DailyLoss);
        assert!(!e.trading_allowed());
        let i = InstrumentId::new();
        assert_eq!(e.clip_targets(&BTreeMap::from([(i, 100.0)]), &BTreeMap::new(), 9_650.0)[&i], 0.0);
        e.start_day(9_650.0);
        assert!(e.trading_allowed());
        let h = e.on_equity(8_400.0);
        assert!(h.iter().any(|x| x.0 == HaltScope::Portfolio));
        e.start_day(8_400.0);
        assert!(!e.trading_allowed(), "kill switch survives day rollover");
        e.rearm(8_400.0);
        assert!(e.trading_allowed());
        assert_eq!(e.peak_equity(), 8_400.0);
        assert!(e.on_equity(8_300.0).is_empty(), "drawdown measured from the re-armed peak");
    }
}
