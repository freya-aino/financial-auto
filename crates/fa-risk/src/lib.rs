//! Non-evolvable hard risk limits and the halt state machine (spec 40).
//! Limits come from config only; strategies and evolution can never change them.

use std::collections::{BTreeMap, BTreeSet};

use fa_core::{AssetClass, InstrumentId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Portfolio drawdown from peak equity that triggers flatten + halt.
    pub max_drawdown: f64,
    /// Daily loss (fraction of start-of-day equity) that stops trading for the day.
    pub max_daily_loss: f64,
    /// Max |position| per instrument as a fraction of deployable capital.
    pub max_position_frac: f64,
    /// Max gross exposure per asset class as a multiple of deployable capital.
    pub leverage_cap: BTreeMap<AssetClass, f64>,
    /// Liquidity cap: |position| ≤ k × traded volume (EUR) over the horizon.
    pub liquidity_k: f64,
    /// Capital reserved per day for payouts.
    pub daily_payout_quota_eur: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_drawdown: 0.15,
            max_daily_loss: 0.03,
            max_position_frac: 0.10,
            leverage_cap: BTreeMap::from([
                (AssetClass::Equity, 1.0),
                (AssetClass::Etf, 1.0),
                (AssetClass::Fx, 2.0),
                (AssetClass::Crypto, 1.0),
            ]),
            liquidity_k: 0.01,
            daily_payout_quota_eur: 0.0,
        }
    }
}

/// Inputs for one pre-trade check.
#[derive(Debug, Clone)]
pub struct MarketInfo {
    pub asset_class: AssetClass,
    /// Traded volume in EUR over the strategy horizon.
    pub horizon_volume_eur: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HaltScope {
    Portfolio,
    DailyLoss,
    Source,
    Model,
    Broker,
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
}

impl RiskEngine {
    pub fn new(cfg: RiskConfig, equity: f64) -> Self {
        Self { cfg, peak_equity: equity, day_start_equity: equity, halts: BTreeMap::new() }
    }

    /// Deployable capital = equity − daily payout quota (never negative).
    pub fn deployable(&self, equity: f64) -> f64 {
        (equity - self.cfg.daily_payout_quota_eur).max(0.0)
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

    pub fn halts(&self) -> &BTreeMap<HaltScope, HaltReason> {
        &self.halts
    }

    pub fn trading_allowed(&self) -> bool {
        self.halts.is_empty()
    }

    /// Clips target positions (EUR, signed) to every hard limit. When halted, all
    /// targets become 0 (flatten).
    pub fn clip_targets(
        &self,
        targets: &BTreeMap<InstrumentId, f64>,
        market: &BTreeMap<InstrumentId, MarketInfo>,
        equity: f64,
    ) -> BTreeMap<InstrumentId, f64> {
        if !self.trading_allowed() {
            return targets.keys().map(|k| (*k, 0.0)).collect();
        }
        let deployable = self.deployable(equity);
        let pos_cap = self.cfg.max_position_frac * deployable;
        // per-instrument caps: position and liquidity
        let mut clipped: BTreeMap<InstrumentId, f64> = targets
            .iter()
            .map(|(i, &t)| {
                let Some(m) = market.get(i) else { return (*i, 0.0) }; // unknown market → no position
                let liq_cap = self.cfg.liquidity_k * m.horizon_volume_eur;
                let cap = pos_cap.min(liq_cap);
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
            let cap = self.cfg.leverage_cap.get(&c).copied().unwrap_or(0.0) * deployable;
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
        clipped
    }
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
    }

    #[test]
    fn position_and_liquidity_caps() {
        let e = engine();
        let (a, b) = (InstrumentId::new(), InstrumentId::new());
        let t = BTreeMap::from([(a, 5_000.0), (b, -5_000.0)]);
        let m = BTreeMap::from([
            (a, MarketInfo { asset_class: AssetClass::Equity, horizon_volume_eur: 1e9 }),
            (b, MarketInfo { asset_class: AssetClass::Crypto, horizon_volume_eur: 50_000.0 }),
        ]);
        let c = e.clip_targets(&t, &m, 11_000.0);
        assert_eq!(c[&a], 1_000.0); // 10% of 10k deployable
        assert_eq!(c[&b], -500.0); // 1% of 50k volume
    }

    #[test]
    fn leverage_cap_scales_class() {
        let mut e = engine();
        e.cfg.max_position_frac = 10.0;
        e.cfg.leverage_cap.insert(AssetClass::Fx, 0.5);
        let (a, b) = (InstrumentId::new(), InstrumentId::new());
        let t = BTreeMap::from([(a, 4_000.0), (b, -6_000.0)]);
        let m: BTreeMap<_, _> = [a, b].into_iter().map(|i| (i, MarketInfo { asset_class: AssetClass::Fx, horizon_volume_eur: 1e12 })).collect();
        let c = e.clip_targets(&t, &m, 11_000.0);
        assert!((c[&a].abs() + c[&b].abs() - 5_000.0).abs() < 1e-6);
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
    }
}
