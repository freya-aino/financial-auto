//! Live paper allocator tournament (spec 40 §6.2: "all allocators run in parallel in paper;
//! their portfolio P&L is compared on the same strategy signals", §9 acceptance).
//!
//! Every entry allocates over the same strategy posteriors each decision tick; its virtual
//! portfolio earns `Σ_s w_s(t−1) · r_s(t)`, where `r_s` is strategy `s`'s sleeve return over
//! the tick ([`crate::PerformanceTracker::step_return`]) — identical inputs for every entry,
//! so the reports are directly comparable. Each step also charges `cost_bps` on the weight
//! turnover `Σ_s |w_s(t) − w_s(t−1)|` (a coarse stand-in for the netted execution cost the
//! offline replay `fa_backtest::allocator_replay` models exactly). The active allocator can be
//! fed in as an external entry so its own record sits next to the controls.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Allocation, Allocator, StrategyStats};

/// One entry's running record.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Growth of 1 unit of deployable capital (compounded tick returns).
    pub equity: f64,
    pub peak: f64,
    pub max_drawdown: f64,
    pub steps: u64,
    pub sum_r: f64,
    pub sum_r2: f64,
    pub turnover: f64,
    pub costs: f64,
}

impl Record {
    fn new() -> Self {
        Self { equity: 1.0, peak: 1.0, ..Default::default() }
    }

    fn step(&mut self, r: f64) {
        let r = if r.is_finite() { r } else { 0.0 };
        self.equity *= 1.0 + r;
        self.peak = self.peak.max(self.equity);
        if self.peak > 0.0 {
            self.max_drawdown = self.max_drawdown.max(1.0 - self.equity / self.peak);
        }
        self.steps += 1;
        self.sum_r += r;
        self.sum_r2 += r * r;
    }

    /// Per-step Sharpe of the tick returns (0 with fewer than 2 steps or no variance).
    pub fn sharpe(&self) -> f64 {
        if self.steps < 2 {
            return 0.0;
        }
        let n = self.steps as f64;
        let m = self.sum_r / n;
        let var = (self.sum_r2 / n - m * m) * n / (n - 1.0);
        if var > 0.0 { m / var.sqrt() } else { 0.0 }
    }
}

struct Entry {
    allocator: Option<Box<dyn Allocator>>,
    weights: Allocation,
    record: Record,
}

/// Comparable paper records of several allocators on identical strategy returns.
pub struct Tournament {
    entries: BTreeMap<String, Entry>,
    pub cost_bps: f64,
}

/// Report row of one entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryReport {
    pub name: String,
    pub cumulative_return: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub steps: u64,
    pub turnover: f64,
    pub costs: f64,
    pub weights: Allocation,
}

impl Tournament {
    pub fn new(cost_bps: f64) -> Self {
        Self { entries: BTreeMap::new(), cost_bps: cost_bps.max(0.0) }
    }

    /// The spec 40 §6.2 initial set: equal weight (control), risk parity, fractional Kelly and
    /// the Thompson bandit + risk parity.
    pub fn standard(max_weight: f64, kelly_fraction: f64, cost_bps: f64) -> Self {
        let mut t = Self::new(cost_bps);
        for name in ["equal_weight", "risk_parity", "fractional_kelly", "thompson_rp"] {
            if let Some(a) = crate::by_name(name, max_weight, kelly_fraction) {
                t.add(name, a);
            }
        }
        t
    }

    pub fn add(&mut self, name: &str, allocator: Box<dyn Allocator>) {
        self.entries.insert(
            name.to_owned(),
            Entry { allocator: Some(allocator), weights: Allocation::new(), record: Record::new() },
        );
    }

    /// Realises one tick for every entry: the weights set at the previous tick earn the
    /// strategies' returns over this tick.
    pub fn realise(&mut self, strategy_returns: &BTreeMap<String, f64>) {
        for e in self.entries.values_mut() {
            let r: f64 = e.weights.iter().map(|(s, w)| w * strategy_returns.get(s).copied().unwrap_or(0.0)).sum();
            e.record.step(r);
        }
    }

    /// Re-allocates every internal entry on `stats` and records turnover and its cost.
    pub fn allocate(&mut self, stats: &[StrategyStats]) {
        let names: Vec<String> =
            self.entries.iter().filter(|(_, e)| e.allocator.is_some()).map(|(n, _)| n.clone()).collect();
        for n in names {
            let w = self.entries[&n].allocator.as_ref().expect("filtered").allocate(stats);
            self.set_weights(&n, w);
        }
    }

    /// Sets an entry's weights for the next tick (creates an external entry, e.g. `active`).
    pub fn set_weights(&mut self, name: &str, weights: Allocation) {
        let cost_bps = self.cost_bps;
        let e = self.entries.entry(name.to_owned()).or_insert_with(|| Entry {
            allocator: None,
            weights: Allocation::new(),
            record: Record::new(),
        });
        let weights: Allocation = weights.into_iter().filter(|(_, w)| w.is_finite() && *w > 0.0).collect();
        let mut turn = 0.0;
        for (s, w) in &weights {
            turn += (w - e.weights.get(s).copied().unwrap_or(0.0)).abs();
        }
        for (s, w) in &e.weights {
            if !weights.contains_key(s) {
                turn += w.abs();
            }
        }
        let cost = turn * cost_bps / 10_000.0;
        e.record.turnover += turn;
        e.record.costs += cost;
        e.record.equity *= 1.0 - cost;
        e.weights = weights;
    }

    pub fn report(&self) -> Vec<EntryReport> {
        self.entries
            .iter()
            .map(|(n, e)| EntryReport {
                name: n.clone(),
                cumulative_return: e.record.equity - 1.0,
                sharpe: e.record.sharpe(),
                max_drawdown: e.record.max_drawdown,
                steps: e.record.steps,
                turnover: e.record.turnover,
                costs: e.record.costs,
                weights: e.weights.clone(),
            })
            .collect()
    }

    pub fn record(&self, name: &str) -> Option<&Record> {
        self.entries.get(name).map(|e| &e.record)
    }

    /// Records for persistence (`name → record`); weights restart empty after a restore.
    pub fn records(&self) -> BTreeMap<String, Record> {
        self.entries.iter().map(|(n, e)| (n.clone(), e.record.clone())).collect()
    }

    /// Restores persisted records for the entries that exist (and external ones by name).
    pub fn restore(&mut self, records: BTreeMap<String, Record>) {
        for (n, r) in records {
            let e = self.entries.entry(n).or_insert_with(|| Entry {
                allocator: None,
                weights: Allocation::new(),
                record: Record::new(),
            });
            e.record = r;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(id: &str, m: f64, v: f64) -> StrategyStats {
        StrategyStats { strategy_id: id.into(), mean_return: m, volatility: v, observations: 1_000 }
    }

    #[test]
    fn entries_earn_the_same_strategy_returns_through_their_own_weights() {
        let mut t = Tournament::standard(1.0, 0.25, 0.0);
        let stats = vec![st("win", 0.001, 0.01), st("lose", -0.001, 0.01)];
        let r = BTreeMap::from([("win".to_owned(), 0.01), ("lose".to_owned(), -0.01)]);
        for _ in 0..20 {
            t.realise(&r);
            t.allocate(&stats);
        }
        let rep: BTreeMap<String, EntryReport> = t.report().into_iter().map(|e| (e.name.clone(), e)).collect();
        assert_eq!(rep.len(), 4);
        assert!(rep.values().all(|e| e.steps == 20), "identical tick count");
        // equal weight holds both (net 0); risk parity / Kelly / Thompson hold only the winner
        assert!(rep["equal_weight"].cumulative_return.abs() < 1e-12);
        for n in ["risk_parity", "fractional_kelly", "thompson_rp"] {
            assert!(rep[n].cumulative_return > 0.0, "{n}: {:?}", rep[n]);
        }
        assert!(rep["risk_parity"].cumulative_return > rep["equal_weight"].cumulative_return);
    }

    #[test]
    fn external_entries_turnover_costs_and_restore() {
        let mut t = Tournament::new(10.0);
        t.set_weights("active", Allocation::from([("a".to_owned(), 0.5)]));
        let r = BTreeMap::from([("a".to_owned(), 0.02)]);
        t.realise(&r);
        let rec = t.record("active").unwrap().clone();
        assert!((rec.turnover - 0.5).abs() < 1e-12 && (rec.costs - 0.0005).abs() < 1e-12);
        assert!((rec.equity - (1.0 - 0.0005) * 1.01).abs() < 1e-12);
        t.set_weights("active", Allocation::new());
        assert!((t.record("active").unwrap().turnover - 1.0).abs() < 1e-12, "closing counts too");
        let saved = t.records();
        let mut u = Tournament::standard(0.5, 0.25, 10.0);
        u.restore(saved);
        assert_eq!(u.record("active").unwrap().steps, 1);
        assert_eq!(u.report().len(), 5);
        // non-finite weights and returns never poison the record
        u.set_weights("active", Allocation::from([("a".to_owned(), f64::NAN)]));
        u.realise(&BTreeMap::from([("a".to_owned(), f64::NAN)]));
        assert!(u.record("active").unwrap().equity.is_finite());
    }
}
