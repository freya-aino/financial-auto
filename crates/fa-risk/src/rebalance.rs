//! Cost-aware rebalancing (spec 40 §6.1 step 6, §6.3): the decision whether a netted target
//! is worth trading toward at all. The D-041 no-trade band (0.5 % / 25 EUR) sat far below the
//! venue's 1 EUR fee minimum, so 1 s strategies paid 30–50 bp per ~300 EUR rebalance
//! (2026-11-10 19:00 – 11-11 02:00: ~7 000 fills, ~7 000 EUR of fees). This module replaces
//! "trade whenever the target moved by more than the band" with "trade only when the
//! expected benefit exceeds the full expected cost":
//!
//! ```text
//! cost(Δ)    = max(commission_min, |Δ|·commission_bps) + |Δ|·half_spread_bps + |Δ|·k·σ_day·sqrt(|Δ|/ADV)
//! benefit(Δ) = edge · |Δ_open|  +  risk_aversion · σ_hold · |Δ_reduce|
//! trade iff  benefit ≥ min_benefit_cost_ratio · cost   and   |Δ| ≥ min order notional
//! ```
//!
//! * **edge** — expected return per EUR of target exposure over the holding period, from the
//!   contributing strategies' tracked step returns (`fa_allocator::PerformanceTracker`), per
//!   unit exposure, scaled to `holding_horizons` × their horizon and shrunk toward
//!   `prior_edge_bps` with `prior_obs` pseudo-observations ([`strategy_edge`]). A strategy
//!   that has been losing money has a non-positive edge and opens nothing it has to pay for.
//! * **risk term** — reducing an over-sized position toward its target removes risk the
//!   strategies no longer want: `σ_hold = σ_day·sqrt(holding/1 d)`; a position no strategy
//!   targets any more (orphan) is held without horizon, so its σ is a full day's.
//! * **min order notional** = `max(min_order_eur, commission_min · 1e4 / max_fee_bps)`: below
//!   it the fee minimum alone costs more than `max_fee_bps` of the order.
//! * **urgent** orders (halts, kill switch, strategy flatten, stale-feed flatten) and
//!   **limit-bound** reductions (the cap chain cut the target below the position) bypass the
//!   cost test: risk limits are never traded off against fees.
//! * **batching** — a non-urgent trade must stay worthwhile for `confirm_secs` before it is
//!   sent, so strategies whose targets swing against each other within seconds net out in the
//!   portfolio target (spec 40 §6.3) instead of producing a buy and a sell.
//! * **turnover budget** — every strategy may cause at most `turnover_budget_per_day` × its
//!   sleeve of traded notional per UTC day (orders attributed evenly to the contributors, as
//!   fills are, D-051). A strategy over budget is *frozen*: the caller keeps netting its last
//!   distribution instead of its new signals ([`Rebalancer::over_budget`]).
//!
//! The same [`Rebalancer`] runs in the risk-engine tick and in `fa_backtest::allocator_replay`
//! (`TradeRule::CostAware`), so the replay of recorded signals measures exactly the live rule.

use std::collections::{BTreeMap, BTreeSet};

use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};

const MS_PER_DAY: i64 = 86_400_000;

/// `[rebalance]` in `risk.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RebalanceConfig {
    /// Off: only the legacy no-trade band applies (the D-041 behaviour).
    pub enabled: bool,
    /// Expected benefit must be at least this multiple of the full expected cost.
    pub min_benefit_cost_ratio: f64,
    /// Edge (bps of the target exposure per holding period) of a strategy without evidence.
    pub prior_edge_bps: f64,
    /// Tracker observations that weigh as much as the prior.
    pub prior_obs: f64,
    /// Estimated edges are clamped to ±this (bps per holding period).
    pub max_edge_bps: f64,
    /// Holding period = this many of the contributing strategies' horizons …
    pub holding_horizons: f64,
    /// … but at least this long (a target is rarely revisited faster than the cadence it is
    /// re-decided at; also the floor for the reduction risk term).
    pub min_holding_secs: f64,
    /// Benefit of removing one σ of unwanted risk, per EUR (mean-variance risk aversion in
    /// the linearised form); 0 = reductions need edge like openings.
    pub risk_aversion: f64,
    /// Min order notional: the fee minimum may cost at most this many bps of the order.
    pub max_fee_bps: f64,
    /// Traded notional a strategy may cause per UTC day, in multiples of its sleeve; 0 = no budget.
    pub turnover_budget_per_day: f64,
    /// A non-urgent trade must stay worthwhile this long before it is sent.
    pub confirm_secs: f64,
}

impl Default for RebalanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_benefit_cost_ratio: 1.0,
            prior_edge_bps: 20.0,
            prior_obs: 600.0,
            max_edge_bps: 200.0,
            holding_horizons: 1.0,
            min_holding_secs: 60.0,
            risk_aversion: 1.0,
            max_fee_bps: 10.0,
            turnover_budget_per_day: 20.0,
            confirm_secs: 3.0,
        }
    }
}

/// What one order is expected to cost at the venue: the fee schedule of the paper venue's
/// cost model (`costs.yaml`, possibly recalibrated) plus the instrument's market state.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct CostQuote {
    pub commission_bps: f64,
    pub commission_min_eur: f64,
    /// Half the quoted (or modelled) spread, bps.
    pub half_spread_bps: f64,
    /// Square-root impact coefficient.
    pub impact_k: f64,
    /// Daily volatility (fraction).
    pub sigma_day: f64,
    /// Daily traded notional, EUR.
    pub adv_eur: f64,
}

impl CostQuote {
    /// Full expected cost in EUR of trading `notional` EUR (commission with its minimum, half
    /// spread, square-root impact) — the gateway's `CostModel.Price` in expectation.
    pub fn cost_eur(&self, notional: f64) -> f64 {
        let n = notional.abs();
        if n <= 0.0 {
            return 0.0;
        }
        let fee = self.commission_min_eur.max(n * self.commission_bps * 1e-4);
        let spread = n * self.half_spread_bps * 1e-4;
        let impact =
            if self.adv_eur > 0.0 { n * self.impact_k * self.sigma_day * (n / self.adv_eur).sqrt() } else { 0.0 };
        fee + spread + impact
    }

    /// Smallest order worth sending: the fee minimum costs at most `max_fee_bps` of it.
    pub fn min_order_eur(&self, cfg: &RebalanceConfig, floor_eur: f64) -> f64 {
        let from_fee = if cfg.max_fee_bps > 0.0 { self.commission_min_eur * 1e4 / cfg.max_fee_bps } else { 0.0 };
        from_fee.max(floor_eur)
    }
}

/// Expected return per EUR of exposure over the holding period of one strategy (fraction):
/// `mean_step_return / mean|exposure| · holding/step`, shrunk toward the prior with
/// `prior_obs` pseudo-observations and clamped to `±max_edge_bps`.
pub fn strategy_edge(
    cfg: &RebalanceConfig,
    mean_step_return: f64,
    mean_abs_exposure: f64,
    observations: u64,
    step_secs: f64,
    holding_secs: f64,
) -> f64 {
    let prior = cfg.prior_edge_bps * 1e-4;
    let raw = if observations > 0 && mean_abs_exposure > 1e-6 && step_secs > 0.0 && mean_step_return.is_finite() {
        mean_step_return / mean_abs_exposure.max(0.05) * (holding_secs / step_secs)
    } else {
        prior
    };
    let n = observations as f64;
    let shrunk = (n * raw + cfg.prior_obs.max(0.0) * prior) / (n + cfg.prior_obs.max(0.0)).max(1e-9);
    let cap = cfg.max_edge_bps.abs() * 1e-4;
    shrunk.clamp(-cap, cap)
}

/// Effective sample size of an EWMA mean with `decay` after `n` observations: the tracker
/// forgets, so its mean is never worth more than `(1 + decay) / (1 − decay)` observations
/// (39 at the risk-engine's 0.95) against the prior.
pub fn effective_obs(n: u64, decay: f64) -> u64 {
    let d = decay.clamp(0.0, 0.999_999);
    let cap = ((1.0 + d) / (1.0 - d)).round().max(1.0) as u64;
    n.min(cap)
}

/// One instrument's rebalance as the tick sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct TradeCandidate {
    pub instrument: InstrumentId,
    /// Position incl. pending intents, EUR.
    pub current_eur: f64,
    /// Post-risk target, EUR.
    pub target_eur: f64,
    /// Exposure-weighted edge of the strategies targeting the instrument (fraction per holding).
    pub edge: f64,
    /// Holding period of the contributors, seconds (orphans: ignored, a day is used).
    pub holding_secs: f64,
    /// Halt / kill switch / strategy flatten / stale-feed flatten: trade at any cost.
    pub urgent: bool,
    /// The cap chain cut the target below the position: the reduction is a limit, not a choice.
    pub limit_bound: bool,
    /// Strategies whose signals target the instrument (empty = orphan position).
    pub contributors: Vec<String>,
    pub cost: CostQuote,
}

impl TradeCandidate {
    pub fn delta(&self) -> f64 {
        self.target_eur - self.current_eur
    }

    /// Expected benefit in EUR (module docs): a trade **with** the strategies' view (opening,
    /// adding, or flipping a position that is against it) earns the edge on the whole delta; a
    /// reduction toward a smaller target on the same side, or toward zero, sheds risk worth
    /// `risk_aversion · σ_hold` per EUR minus the edge it gives up.
    pub fn benefit_eur(&self, cfg: &RebalanceConfig) -> f64 {
        let (c, t) = (self.current_eur, self.target_eur);
        let delta = (t - c).abs();
        let edge = self.edge.max(0.0);
        let with_view = t != 0.0 && (c == 0.0 || c.signum() != t.signum() || t.abs() >= c.abs());
        if with_view {
            return delta * edge;
        }
        let orphan = self.contributors.is_empty() && t == 0.0;
        let hold = if orphan { 86_400.0 } else { self.holding_secs.max(cfg.min_holding_secs) };
        let sigma_hold = self.cost.sigma_day.max(0.0) * (hold / 86_400.0).sqrt();
        let given_up = if t == 0.0 { 0.0 } else { edge };
        delta * (cfg.risk_aversion.max(0.0) * sigma_hold - given_up).max(0.0)
    }

    /// The position shrinks toward the target without changing side.
    pub fn reduces(&self) -> bool {
        let (c, t) = (self.current_eur, self.target_eur);
        t.abs() < c.abs() && (t == 0.0 || t.signum() == c.signum())
    }
}

fn sign(x: f64) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// Why a candidate was not traded (heartbeat `rebalance.skip.<reason>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    /// Inside the legacy no-trade band.
    Band,
    /// Below the min order notional derived from the fee minimum.
    MinNotional,
    /// Expected benefit below the expected cost.
    Cost,
    /// Worthwhile, but not yet for `confirm_secs`.
    Confirming,
    /// Every contributing strategy has spent its turnover budget today.
    Budget,
}

impl SkipReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Band => "band",
            Self::MinNotional => "min_notional",
            Self::Cost => "cost",
            Self::Confirming => "confirming",
            Self::Budget => "budget",
        }
    }
}

/// The decision for one candidate.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Trade { benefit_eur: f64, cost_eur: f64, bypass: bool },
    Skip { reason: SkipReason, benefit_eur: f64, cost_eur: f64 },
}

impl Verdict {
    pub fn trades(&self) -> bool {
        matches!(self, Self::Trade { .. })
    }
}

/// Running totals for the heartbeat and replay reports.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RebalanceStats {
    pub trades: u64,
    pub bypass: u64,
    pub skipped: BTreeMap<String, u64>,
    /// Expected cost of the orders sent (EUR).
    pub expected_cost_eur: f64,
    /// Expected cost of the orders NOT sent because cost exceeded benefit (EUR).
    pub avoided_cost_eur: f64,
}

/// Stateful decision maker: confirmation clocks per instrument, turnover per strategy per UTC
/// day, frozen (over-budget) strategies.
#[derive(Debug, Clone, Default)]
pub struct Rebalancer {
    pub cfg: RebalanceConfig,
    worthwhile_since: BTreeMap<InstrumentId, (i64, i8)>,
    day: i64,
    turnover: BTreeMap<String, f64>,
    frozen: BTreeSet<String>,
    pub stats: RebalanceStats,
}

impl Rebalancer {
    pub fn new(cfg: RebalanceConfig) -> Self {
        Self { cfg, ..Default::default() }
    }

    fn roll(&mut self, now_ms: i64) {
        let day = now_ms.div_euclid(MS_PER_DAY);
        if day != self.day {
            self.day = day;
            self.turnover.clear();
            self.frozen.clear();
        }
    }

    /// Strategies that spent their budget today: the caller nets their last distribution
    /// (frozen sub-target) instead of new signals until the UTC day rolls.
    pub fn over_budget(&mut self, now_ms: i64) -> &BTreeSet<String> {
        self.roll(now_ms);
        &self.frozen
    }

    /// Traded notional attributed to `strategy` today (EUR).
    pub fn turnover_eur(&self, strategy: &str) -> f64 {
        self.turnover.get(strategy).copied().unwrap_or(0.0)
    }

    /// Decides one candidate. `band_eur` is the legacy no-trade band (still applied first),
    /// `floor_eur` the configured absolute min order, `sleeves` the strategies' sleeves (EUR)
    /// for the turnover budget.
    pub fn decide(
        &mut self,
        c: &TradeCandidate,
        band_eur: f64,
        floor_eur: f64,
        sleeves: &BTreeMap<String, f64>,
        now_ms: i64,
    ) -> Verdict {
        self.roll(now_ms);
        let delta = c.delta().abs();
        let cost = c.cost.cost_eur(delta);
        let skip = |me: &mut Self, reason: SkipReason, benefit: f64| {
            if reason != SkipReason::Confirming {
                me.worthwhile_since.remove(&c.instrument);
            }
            *me.stats.skipped.entry(reason.as_str().to_owned()).or_insert(0) += 1;
            if reason == SkipReason::Cost {
                me.stats.avoided_cost_eur += cost;
            }
            Verdict::Skip { reason, benefit_eur: benefit, cost_eur: cost }
        };
        if delta < band_eur.max(1e-9) && !(c.urgent && c.target_eur == 0.0 && c.current_eur != 0.0) {
            return skip(self, SkipReason::Band, 0.0);
        }
        // risk first: urgent orders and limit-bound reductions never wait for a fee to pay off
        if c.urgent || (c.limit_bound && c.reduces()) || !self.cfg.enabled {
            self.worthwhile_since.remove(&c.instrument);
            self.stats.trades += 1;
            if c.urgent || c.limit_bound {
                self.stats.bypass += 1;
            }
            self.stats.expected_cost_eur += cost;
            return Verdict::Trade { benefit_eur: 0.0, cost_eur: cost, bypass: c.urgent || c.limit_bound };
        }
        let benefit = c.benefit_eur(&self.cfg);
        let orphan_close = c.contributors.is_empty() && c.target_eur == 0.0;
        if delta < c.cost.min_order_eur(&self.cfg, floor_eur) && !orphan_close {
            return skip(self, SkipReason::MinNotional, benefit);
        }
        if benefit < self.cfg.min_benefit_cost_ratio.max(0.0) * cost {
            return skip(self, SkipReason::Cost, benefit);
        }
        if !c.contributors.is_empty() && c.contributors.iter().all(|s| self.frozen.contains(s)) {
            return skip(self, SkipReason::Budget, benefit);
        }
        let side = sign(c.delta());
        let since = self.worthwhile_since.entry(c.instrument).or_insert((now_ms, side));
        if since.1 != side {
            *since = (now_ms, side);
        }
        if ((now_ms - since.0) as f64) < self.cfg.confirm_secs.max(0.0) * 1000.0 {
            return skip(self, SkipReason::Confirming, benefit);
        }
        self.worthwhile_since.remove(&c.instrument);
        // attribute the turnover evenly (as fills are) and freeze strategies over budget
        if !c.contributors.is_empty() {
            let share = delta / c.contributors.len() as f64;
            for s in &c.contributors {
                let t = self.turnover.entry(s.clone()).or_insert(0.0);
                *t += share;
                let sleeve = sleeves.get(s).copied().unwrap_or(0.0);
                if self.cfg.turnover_budget_per_day > 0.0
                    && sleeve > 0.0
                    && *t >= self.cfg.turnover_budget_per_day * sleeve
                {
                    self.frozen.insert(s.clone());
                }
            }
        }
        self.stats.trades += 1;
        self.stats.expected_cost_eur += cost;
        Verdict::Trade { benefit_eur: benefit, cost_eur: cost, bypass: false }
    }

    /// Drops the confirmation clock of an instrument (e.g. its order was sent another way).
    pub fn forget(&mut self, inst: &InstrumentId) {
        self.worthwhile_since.remove(inst);
    }
}

/// EWMA of squared log returns between trades per second → daily volatility (the paper venue's
/// σ estimate, `paper.go`).
#[derive(Debug, Clone, Default)]
pub struct VolEstimator {
    last: BTreeMap<InstrumentId, (i64, f64)>,
    var_per_sec: BTreeMap<InstrumentId, f64>,
}

impl VolEstimator {
    const ALPHA: f64 = 0.05;

    pub fn observe(&mut self, inst: InstrumentId, at_ms: i64, price: f64) {
        if !(price > 0.0 && price.is_finite()) {
            return;
        }
        if let Some((t0, p0)) = self.last.get(&inst).copied()
            && at_ms > t0
            && p0 > 0.0
        {
            let r = (price / p0).ln();
            let dt = ((at_ms - t0) as f64 / 1000.0).max(1.0);
            let v = r * r / dt;
            let e = self.var_per_sec.entry(inst).or_insert(v);
            *e = (1.0 - Self::ALPHA) * *e + Self::ALPHA * v;
        }
        if self.last.get(&inst).is_none_or(|(t0, _)| at_ms >= *t0) {
            self.last.insert(inst, (at_ms, price));
        }
    }

    /// Daily volatility (fraction), `None` until two trades were seen.
    pub fn sigma_day(&self, inst: &InstrumentId) -> Option<f64> {
        self.var_per_sec.get(inst).map(|v| (v * 86_400.0).sqrt()).filter(|s| s.is_finite() && *s > 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The equity schedule of `costs.yaml` (5 bp, 1 EUR minimum, 5 bp spread) on a liquid stock.
    fn equity() -> CostQuote {
        CostQuote {
            commission_bps: 5.0,
            commission_min_eur: 1.0,
            half_spread_bps: 2.5,
            impact_k: 0.1,
            sigma_day: 0.02,
            adv_eur: 10_000_000.0,
        }
    }

    fn cand(inst: InstrumentId, cur: f64, tgt: f64, edge: f64, contributors: &[&str]) -> TradeCandidate {
        TradeCandidate {
            instrument: inst,
            current_eur: cur,
            target_eur: tgt,
            edge,
            holding_secs: 5.0,
            urgent: false,
            limit_bound: false,
            contributors: contributors.iter().map(|s| (*s).to_owned()).collect(),
            cost: equity(),
        }
    }

    #[test]
    fn costs_include_the_fee_minimum_spread_and_impact() {
        let q = equity();
        // 300 EUR: 1 EUR minimum + 0.075 spread + tiny impact ≈ 36 bp
        let c = q.cost_eur(300.0);
        assert!((c - (1.0 + 0.075 + 300.0 * 0.1 * 0.02 * (300.0f64 / 1e7).sqrt())).abs() < 1e-9);
        assert!(c / 300.0 * 1e4 > 35.0);
        // 10 000 EUR: proportional 5 bp + 2.5 bp + impact
        assert!((q.cost_eur(10_000.0) / 10_000.0 * 1e4 - 7.5).abs() < 1.0);
        assert_eq!(q.min_order_eur(&RebalanceConfig::default(), 25.0), 1_000.0);
        let crypto = CostQuote { commission_min_eur: 0.0, commission_bps: 26.0, ..q };
        assert_eq!(crypto.min_order_eur(&RebalanceConfig::default(), 25.0), 25.0);
    }

    #[test]
    fn ewma_means_count_as_their_effective_sample() {
        assert_eq!(effective_obs(5, 0.95), 5);
        assert_eq!(effective_obs(100_000, 0.95), 39);
        assert_eq!(effective_obs(100_000, 0.999), 1_999);
    }

    #[test]
    fn edge_shrinks_toward_the_prior_and_losers_get_none() {
        let cfg = RebalanceConfig::default();
        let prior = strategy_edge(&cfg, 0.0, 0.0, 0, 1.0, 5.0);
        assert!((prior - 20e-4).abs() < 1e-12);
        // a 1 s strategy losing 0.1 bp per step at full exposure over 3 600 steps: negative edge
        let loser = strategy_edge(&cfg, -1e-5, 1.0, 3_600, 1.0, 60.0);
        assert!(loser < 0.0, "{loser}");
        // a winner well above the prior, clamped at max_edge_bps
        let winner = strategy_edge(&cfg, 1e-4, 0.5, 100_000, 1.0, 600.0);
        assert!((winner - 200e-4).abs() < 1e-12, "{winner}");
    }

    /// SYNTHETIC replay of the recorded churn pattern: five strategies' 1 s targets on four
    /// equity instruments jitter by a few hundred EUR around their level every second (the
    /// 19:00–02:00 book traded ~300 EUR averages at a 1 EUR minimum). The legacy band trades
    /// nearly every tick; the cost-aware rule trades only the level changes.
    #[test]
    fn the_churn_pattern_is_suppressed() {
        let insts: Vec<InstrumentId> = (0..4).map(|_| InstrumentId::new()).collect();
        let sleeves: BTreeMap<String, f64> = ["a", "b"].iter().map(|s| ((*s).to_owned(), 20_000.0)).collect();
        let run = |cfg: RebalanceConfig, edge: f64| {
            let mut rb = Rebalancer::new(cfg);
            let mut pos: BTreeMap<InstrumentId, f64> = BTreeMap::new();
            let (mut fees, mut n) = (0.0, 0usize);
            let mut rng = fa_core::rng::Rng::seed(7);
            for t in 0..3_600i64 {
                for (k, i) in insts.iter().enumerate() {
                    // level 2 000 EUR, jitter ±400 EUR, a regime switch every 15 minutes
                    let level = if (t / 900) % 2 == 0 { 2_000.0 } else { -1_000.0 } * (1.0 + k as f64 * 0.1);
                    let target = level + 400.0 * rng.normal();
                    let cur = pos.get(i).copied().unwrap_or(0.0);
                    let band = 25.0f64.max(0.005 * target.abs());
                    let c = TradeCandidate { target_eur: target, ..cand(*i, cur, target, edge, &["a", "b"]) };
                    if rb.decide(&c, band, 25.0, &sleeves, t * 1000).trades() {
                        fees += c.cost.cost_eur(c.delta());
                        n += 1;
                        pos.insert(*i, target);
                    }
                }
            }
            (n, fees, rb.stats)
        };
        let legacy = RebalanceConfig { enabled: false, ..Default::default() };
        let (n_old, fee_old, _) = run(legacy, 20e-4);
        let (n_new, fee_new, stats) = run(RebalanceConfig::default(), 20e-4);
        eprintln!("legacy {n_old} orders {fee_old:.0} EUR; cost-aware {n_new} orders {fee_new:.0} EUR; {stats:?}");
        assert!(n_old > 13_000, "legacy band trades nearly every tick: {n_old}");
        assert!(n_new < n_old / 50, "{n_new} vs {n_old}");
        assert!(fee_new < fee_old / 20.0, "{fee_new} vs {fee_old}");
        // the regime switches (3 000 EUR moves, worth 6 EUR of edge vs ~3.3 EUR cost) still trade
        assert!(n_new >= 4 * 3, "{n_new}");
        // a losing strategy (negative edge) opens nothing at all
        let (n_loser, ..) = run(RebalanceConfig::default(), -5e-4);
        assert_eq!(n_loser, 0);
    }

    #[test]
    fn urgent_and_limit_bound_reductions_bypass_the_cost_test() {
        let i = InstrumentId::new();
        let sleeves = BTreeMap::new();
        let mut rb = Rebalancer::new(RebalanceConfig::default());
        // a 300 EUR close at zero edge would never pay its fee …
        let c = cand(i, 300.0, 0.0, 0.0, &["a"]);
        assert!(!rb.decide(&c, 25.0, 25.0, &sleeves, 0).trades());
        // … but a halt flattens it, and a cap cut reduces it, at any cost
        assert!(rb.decide(&TradeCandidate { urgent: true, ..c.clone() }, 25.0, 25.0, &sleeves, 0).trades());
        let capped = TradeCandidate { target_eur: 100.0, limit_bound: true, ..c.clone() };
        assert!(rb.decide(&capped, 25.0, 25.0, &sleeves, 0).trades());
        // a limit-bound target that would INCREASE the position gets no bypass
        let up = TradeCandidate { current_eur: 0.0, target_eur: 300.0, limit_bound: true, ..c };
        assert!(!rb.decide(&up, 25.0, 25.0, &sleeves, 0).trades());
        assert_eq!(rb.stats.bypass, 2);
    }

    #[test]
    fn orphans_close_on_risk_and_reductions_weigh_sigma() {
        let i = InstrumentId::new();
        let sleeves = BTreeMap::new();
        let mut rb = Rebalancer::new(RebalanceConfig { confirm_secs: 0.0, ..Default::default() });
        // nobody targets a 500 EUR position any more: a day of 2 % σ is 10 EUR of risk vs ~1.1 EUR cost
        let orphan = cand(i, 500.0, 0.0, 0.0, &[]);
        assert!(rb.decide(&orphan, 25.0, 25.0, &sleeves, 0).trades());
        // the same reduction for a 5 s strategy (σ over a minute ≈ 5 bp) does not pay
        let owned = cand(i, 1_500.0, 1_000.0, 0.0, &["a"]);
        assert!(!rb.decide(&owned, 25.0, 25.0, &sleeves, 0).trades());
    }

    #[test]
    fn confirmation_batches_swings_and_the_budget_freezes_strategies() {
        let i = InstrumentId::new();
        let sleeves: BTreeMap<String, f64> = [("a".to_owned(), 10_000.0)].into();
        let cfg = RebalanceConfig { confirm_secs: 2.0, turnover_budget_per_day: 1.0, ..Default::default() };
        let mut rb = Rebalancer::new(cfg);
        let up = cand(i, 0.0, 6_000.0, 20e-4, &["a"]);
        assert_eq!(
            rb.decide(&up, 25.0, 25.0, &sleeves, 0),
            Verdict::Skip {
                reason: SkipReason::Confirming,
                benefit_eur: up.benefit_eur(&rb.cfg),
                cost_eur: up.cost.cost_eur(6_000.0)
            }
        );
        // the target flips sign within the window: the clock restarts, nothing traded
        let down = cand(i, 0.0, -6_000.0, 20e-4, &["a"]);
        assert!(!rb.decide(&down, 25.0, 25.0, &sleeves, 1_000).trades());
        assert!(!rb.decide(&down, 25.0, 25.0, &sleeves, 2_000).trades());
        assert!(rb.decide(&down, 25.0, 25.0, &sleeves, 3_000).trades());
        assert!(rb.over_budget(3_000).is_empty());
        // 6 000 + 6 000 ≥ 1 × 10 000 sleeve: frozen until the UTC day rolls
        assert!(!rb.decide(&up, 25.0, 25.0, &sleeves, 10_000).trades());
        assert!(rb.decide(&up, 25.0, 25.0, &sleeves, 12_000).trades());
        assert!(rb.over_budget(12_000).contains("a"));
        assert!((rb.turnover_eur("a") - 12_000.0).abs() < 1e-9);
        let again = cand(i, 6_000.0, -6_000.0, 20e-4, &["a"]);
        rb.decide(&again, 25.0, 25.0, &sleeves, 20_000);
        assert!(matches!(
            rb.decide(&again, 25.0, 25.0, &sleeves, 30_000),
            Verdict::Skip { reason: SkipReason::Budget, .. }
        ));
        assert!(rb.over_budget(MS_PER_DAY + 1).is_empty(), "budget resets at the UTC day");
    }

    #[test]
    fn volatility_estimate_scales_to_a_day() {
        let mut v = VolEstimator::default();
        let i = InstrumentId::new();
        let mut p = 100.0;
        for k in 0..2_000i64 {
            p *= if k % 2 == 0 { 1.0001 } else { 1.0 / 1.0001 };
            v.observe(i, k * 1000, p);
        }
        let s = v.sigma_day(&i).unwrap();
        // |r| = 1e-4 per second → σ_day = 1e-4 · sqrt(86 400) ≈ 2.9 %
        assert!((s - 1e-4 * 86_400f64.sqrt()).abs() < 1e-3, "{s}");
    }
}
