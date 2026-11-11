//! Randomised risk-chain tests (spec 40 §9 AC 1: "no order can reach a venue without passing
//! the full risk chain; tested by property tests injecting random targets").
//!
//! Random limits, markets (class, venue, liquidity, provisional, shortable), equity, payout
//! reserve, current positions, halts and targets — including non-finite and absurd ones — go
//! through the same chain the risk-engine runs per tick: `cap_targets` → `gate_for` (halts) →
//! the trade decision (`Rebalancer::decide`) → the order delta. Every limit is asserted on the
//! output, not on the code path that should have enforced it.
use std::collections::{BTreeMap, BTreeSet};

use fa_core::{AssetClass, InstrumentId};
use fa_risk::{
    HaltScope, MarketInfo, RiskConfig, RiskEngine, gate_for,
    rebalance::{CostQuote, RebalanceConfig, Rebalancer, TradeCandidate},
};
use proptest::prelude::*;
use uuid::Uuid;

const CLASSES: [AssetClass; 9] = [
    AssetClass::Equity,
    AssetClass::Etf,
    AssetClass::Fx,
    AssetClass::Crypto,
    AssetClass::Commodity,
    AssetClass::Rates,
    AssetClass::Index,
    AssetClass::Derivative,
    AssetClass::PredictionMarket,
];
const VENUES: [&str; 4] = ["kraken", "yahoo", "sim", ""];
const SCOPES: [HaltScope; 6] =
    [HaltScope::Portfolio, HaltScope::DailyLoss, HaltScope::Source, HaltScope::Model, HaltScope::Broker, HaltScope::Fx];

fn iid(n: usize) -> InstrumentId {
    InstrumentId(Uuid::from_u128(n as u128 + 1))
}

/// Relative tolerance of the proportional scalings (several multiplicative passes).
fn tol(cap: f64) -> f64 {
    1e-9 * cap.abs().max(1.0)
}

/// A target value: mostly ordinary EUR amounts, sometimes zero, huge or not a number.
fn target() -> impl Strategy<Value = f64> {
    prop_oneof![
        8 => -2.0e6..2.0e6f64,
        1 => Just(0.0),
        1 => prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY), Just(1e300), Just(-1e300)],
    ]
}

fn config() -> impl Strategy<Value = RiskConfig> {
    (
        (0.0..0.5f64, 0.0..1.0f64, 0.0..1.5f64, 0.0..4.0f64, 0.0..0.2f64),
        proptest::collection::btree_map(0usize..9, 0.0..3.0f64, 0..9),
        proptest::collection::btree_map(0usize..9, 0.0..1.5f64, 0..9),
        (0.0..2.0f64, proptest::option::of(0.0..0.5f64), 0.0..5_000.0f64),
    )
        .prop_map(|((pos, class, venue, gross, liq), lev, margin, (budget, prov, quota))| {
            RiskConfig {
                max_position_frac: pos,
                max_class_frac: class,
                max_venue_frac: venue,
                portfolio_gross_cap: gross,
                liquidity_k: liq,
                leverage_cap: lev.into_iter().map(|(c, v)| (CLASSES[c], v)).collect(),
                short_margin: margin.into_iter().map(|(c, v)| (CLASSES[c], v)).collect(),
                short_margin_budget_frac: budget,
                provisional_position_frac: prov,
                daily_payout_quota_eur: quota,
                ..RiskConfig::default()
            }
        })
}

#[derive(Debug, Clone)]
struct Inst {
    market: Option<MarketInfo>,
    target: f64,
    current: f64,
}

fn instrument() -> impl Strategy<Value = Inst> {
    (
        proptest::option::weighted(0.9, (0usize..9, 0usize..4, 0.0..5.0e7f64, any::<bool>(), proptest::option::of(any::<bool>()))),
        target(),
        prop_oneof![3 => Just(0.0), 7 => -1.0e5..1.0e5f64],
    )
        .prop_map(|(m, target, current)| Inst {
            market: m.map(|(c, v, vol, prov, short)| MarketInfo {
                asset_class: CLASSES[c],
                venue: VENUES[v].to_owned(),
                horizon_volume_eur: vol,
                provisional: prov,
                shortable: short,
            }),
            target,
            current,
        })
}

#[derive(Debug, Clone)]
struct Case {
    cfg: RiskConfig,
    equity: f64,
    ledger_reserve: f64,
    insts: Vec<Inst>,
    halts: BTreeSet<HaltScope>,
}

fn case() -> impl Strategy<Value = Case> {
    (
        config(),
        prop_oneof![9 => 0.0..1.0e6f64, 1 => Just(0.0)],
        0.0..10_000.0f64,
        proptest::collection::vec(instrument(), 0..12),
        proptest::collection::btree_set(0usize..6, 0..3),
    )
        .prop_map(|(cfg, equity, ledger_reserve, insts, halts)| Case {
            cfg,
            equity,
            ledger_reserve,
            insts,
            halts: halts.into_iter().map(|i| SCOPES[i]).collect(),
        })
}

struct Chain {
    risk: RiskEngine,
    market: BTreeMap<InstrumentId, MarketInfo>,
    targets: BTreeMap<InstrumentId, f64>,
    current: BTreeMap<InstrumentId, f64>,
    capped: BTreeMap<InstrumentId, f64>,
    gated: BTreeMap<InstrumentId, f64>,
}

fn run(c: &Case) -> Chain {
    let mut risk = RiskEngine::new(c.cfg.clone(), c.equity);
    risk.set_reserved_payout(c.ledger_reserve);
    let mut market = BTreeMap::new();
    let mut targets = BTreeMap::new();
    let mut current = BTreeMap::new();
    for (n, i) in c.insts.iter().enumerate() {
        if let Some(m) = &i.market {
            market.insert(iid(n), m.clone());
        }
        targets.insert(iid(n), i.target);
        if i.current != 0.0 {
            current.insert(iid(n), i.current);
        }
    }
    let capped = risk.cap_targets(&targets, &market, c.equity);
    // the engine adds a 0 target for every held instrument nobody targets any more
    let mut full = capped.clone();
    for i in current.keys() {
        full.entry(*i).or_insert(0.0);
    }
    let gated = gate_for(c.halts.iter().copied(), &full, &current);
    Chain { risk, market, targets, current, capped, gated }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2_000, ..ProptestConfig::default() })]

    /// The caps alone: every output finite, never larger or of another sign than the target,
    /// and within every per-instrument, short-margin, class, venue and portfolio cap.
    #[test]
    fn capped_targets_respect_every_limit(c in case()) {
        let ch = run(&c);
        let cfg = &c.cfg;
        let deployable = ch.risk.deployable(c.equity);
        prop_assert!(deployable >= 0.0 && deployable <= c.equity.max(0.0));
        prop_assert_eq!(ch.capped.keys().collect::<Vec<_>>(), ch.targets.keys().collect::<Vec<_>>());
        let mut class_gross: BTreeMap<AssetClass, f64> = BTreeMap::new();
        let mut venue_gross: BTreeMap<String, f64> = BTreeMap::new();
        let (mut gross, mut margin) = (0.0, 0.0);
        for (i, v) in &ch.capped {
            prop_assert!(v.is_finite(), "non-finite capped target {v} for {i}");
            let t = ch.targets[i];
            let t = if t.is_finite() { t } else { 0.0 };
            prop_assert!(v.abs() <= t.abs() + tol(t), "capped {v} exceeds target {t}");
            prop_assert!(*v == 0.0 || v.signum() == t.signum(), "capped {v} flips target {t}");
            let Some(m) = ch.market.get(i) else {
                prop_assert_eq!(*v, 0.0, "unknown market must not be traded");
                continue;
            };
            let pos_cap = if m.provisional { cfg.provisional_position_frac() } else { cfg.max_position_frac } * deployable;
            prop_assert!(v.abs() <= pos_cap + tol(pos_cap), "position cap {pos_cap} broken by {v}");
            let liq = cfg.liquidity_k * m.horizon_volume_eur;
            prop_assert!(v.abs() <= liq + tol(liq), "liquidity cap {liq} broken by {v}");
            if *v < 0.0 {
                let mf = m.short_margin(cfg);
                prop_assert!(mf.is_some(), "short {v} without borrow");
                margin += v.abs() * mf.unwrap_or(0.0);
            }
            *class_gross.entry(m.asset_class).or_default() += v.abs();
            *venue_gross.entry(m.venue.clone()).or_default() += v.abs();
            gross += v.abs();
        }
        let mcap = cfg.short_margin_budget_frac * deployable;
        prop_assert!(margin <= mcap + tol(mcap), "short margin {margin} > {mcap}");
        for (class, g) in class_gross {
            let cap = cfg.leverage_cap.get(&class).copied().unwrap_or(0.0).min(cfg.max_class_frac) * deployable;
            prop_assert!(g <= cap + tol(cap), "class {class:?} gross {g} > {cap}");
        }
        let vcap = cfg.max_venue_frac * deployable;
        for (venue, g) in venue_gross {
            prop_assert!(g <= vcap + tol(vcap), "venue {venue:?} gross {g} > {vcap}");
        }
        let gcap = cfg.portfolio_gross_cap * deployable;
        prop_assert!(gross <= gcap + tol(gcap), "portfolio gross {gross} > {gcap}");
    }

    /// The halt gate: no halt passes the capped targets through; a flattening halt closes
    /// everything (held instruments included); reduce-only halts only ever shrink positions.
    /// The resulting order deltas therefore never add risk while any halt is active.
    #[test]
    fn halts_only_ever_reduce(c in case()) {
        let ch = run(&c);
        let flatten = c.halts.iter().any(|s| s.flattens());
        for i in ch.current.keys() {
            prop_assert!(ch.gated.contains_key(i), "held instrument {i} has no gated target");
        }
        for (i, g) in &ch.gated {
            prop_assert!(g.is_finite());
            let cur = ch.current.get(i).copied().unwrap_or(0.0);
            if c.halts.is_empty() {
                prop_assert_eq!(*g, ch.capped.get(i).copied().unwrap_or(0.0));
            } else if flatten {
                prop_assert_eq!(*g, 0.0);
            } else {
                // reduce-only: the order delta moves toward zero and never through it
                prop_assert!(g.abs() <= cur.abs() + tol(cur), "reduce-only target {g} grows position {cur}");
                prop_assert!(*g == 0.0 || g.signum() == cur.signum(), "reduce-only target {g} flips {cur}");
                let after = cur + (g - cur);
                prop_assert!(after.abs() <= cur.abs() + tol(cur));
            }
            // halted or not, the gate never loosens a cap
            let capped = ch.capped.get(i).copied().unwrap_or(0.0);
            prop_assert!(g.abs() <= capped.abs() + tol(capped), "gated {g} looser than capped {capped}");
        }
    }

    /// The trade decision cannot override the chain: it only decides *whether* to send the
    /// gated delta. Urgent closes always go out, limit-bound reductions are never held back by
    /// fees, and a trade it sends moves the position exactly to the gated target.
    #[test]
    fn rebalance_never_bypasses_limits(
        c in case(),
        edge in -0.05..0.05f64,
        fee_min in 0.0..5.0f64,
        bps in 0.0..50.0f64,
        urgent_flag in any::<bool>(),
    ) {
        let ch = run(&c);
        let mut rb = Rebalancer::new(RebalanceConfig { confirm_secs: 0.0, ..RebalanceConfig::default() });
        let sleeves = BTreeMap::from([("s".to_owned(), 10_000.0)]);
        let urgent = urgent_flag || !c.halts.is_empty();
        for (i, g) in &ch.gated {
            let cur = ch.current.get(i).copied().unwrap_or(0.0);
            let raw = ch.targets.get(i).copied().filter(|t| t.is_finite()).unwrap_or(0.0);
            let cand = TradeCandidate {
                instrument: *i,
                current_eur: cur,
                target_eur: *g,
                edge,
                holding_secs: 60.0,
                urgent,
                limit_bound: g.abs() + 1e-6 < raw.abs(),
                contributors: vec!["s".into()],
                cost: CostQuote { commission_bps: bps, commission_min_eur: fee_min, half_spread_bps: 2.0, impact_k: 0.1, sigma_day: 0.02, adv_eur: 1e7 },
            };
            let band = 25.0;
            let v = rb.decide(&cand, band, 25.0, &sleeves, 0);
            let delta = g - cur;
            if urgent && *g == 0.0 && cur != 0.0 {
                prop_assert!(v.trades(), "urgent close of {cur} held back: {v:?}");
            }
            if cand.limit_bound && cand.reduces() && delta.abs() >= band {
                prop_assert!(v.trades(), "limit-bound reduction {cur} → {g} held back: {v:?}");
            }
            if v.trades() {
                // the order is exactly the gated delta: position after == gated target
                prop_assert!(((cur + delta) - g).abs() <= tol(*g));
                if !c.halts.is_empty() {
                    prop_assert!((cur + delta).abs() <= cur.abs() + tol(cur), "trade adds risk under a halt");
                }
            }
        }
    }
}
