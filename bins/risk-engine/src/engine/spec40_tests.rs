//! Engine-level tests of the spec 40 §9 additions: a randomised run of the whole decision chain
//! (signals → allocation → netting → caps → halts → trade rule → intents), carry booking,
//! causal pro-rata fill attribution, the session daily-loss day, the risk-chain history and the
//! side-effect-free pre-trade check.
use std::collections::BTreeMap;

use chrono::{TimeZone, Utc};
use fa_core::InstrumentId;
use fa_proto::v1::{
    AssetExposure, ExposureDistribution as ProtoDist, Halt, OrderEvent, OrderIntent, Side, order_event,
};
use fa_risk::HaltScope;
use fa_warehouse::calendar::CalendarRule;

use super::{CarryCharge, Engine, Out, PretradeRequest, ts_of};
use crate::settings::Settings;

fn settings() -> Settings {
    let mut s = Settings::parse(include_str!("../../risk.toml")).unwrap();
    s.feed.stale_freeze_bars = 0.0;
    s.feed.jump_max = 0.0;
    s.rebalance.enabled = false;
    s
}

fn iid(n: u128) -> InstrumentId {
    InstrumentId(uuid::Uuid::from_u128(0x0190_0000_0000_7000_8000_0000_0000_0000 + n))
}

fn signal(id: &str, at: i64, exposures: &[(InstrumentId, f64)]) -> ProtoDist {
    ProtoDist {
        strategy_id: id.into(),
        genome_id: "baseline".into(),
        horizon: "5x1s".into(),
        as_of: Some(ts_of(at)),
        exposures: exposures
            .iter()
            .map(|(i, m)| AssetExposure { instrument_id: i.to_string(), mean: *m, stddev: 0.0 })
            .collect(),
    }
}

fn approved(out: &[Out]) -> Vec<OrderIntent> {
    out.iter().filter_map(|o| if let Out::Approved(i) = o { Some(i.clone()) } else { None }).collect()
}

fn fill(e: &mut Engine, intent: &OrderIntent, px: f64, now: i64) {
    let ev = OrderEvent {
        intent_id: intent.intent_id.clone(),
        venue_order_id: format!("paper-{}", intent.intent_id),
        kind: order_event::Kind::Fill as i32,
        fill_price: px,
        fill_quantity: intent.quantity,
        fee: 0.0,
        fee_currency: "EUR".into(),
        event_time: Some(ts_of(now)),
        ..Default::default()
    };
    e.on_order_event(&ev, now);
}

/// Deterministic xorshift (no extra dev-dependency in the binary).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

/// Spec 40 §9 AC 1 at engine level: random signals (incl. absurd means), random price paths and
/// random halts / resumes for 3 strategies × 4 instruments over 600 ticks and 8 seeds. Every
/// order the engine approves must leave the instrument within the position cap and the book
/// within the gross cap of that tick, and under any halt it may only reduce the position.
#[test]
fn random_runs_never_approve_an_order_beyond_the_limits() {
    let t0 = 1_762_560_000_000i64;
    for seed in 1..=8u64 {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed);
        let mut e = Engine::new(settings());
        let insts: Vec<InstrumentId> = (0..4).map(iid).collect();
        let mut px: BTreeMap<InstrumentId, f64> = insts.iter().map(|i| (*i, rng.range(10.0, 500.0))).collect();
        for k in 0..3_600 {
            for i in &insts {
                e.on_trade(*i, t0 - 3_600_000 + k * 1000, px[i], rng.range(1.0, 50.0));
            }
        }
        let (cap, gross_cap) = (e.risk.cfg.max_position_frac, e.risk.cfg.portfolio_gross_cap);
        let mut checked = 0;
        for k in 0..600i64 {
            let now = t0 + k * 1000;
            for i in &insts {
                let p = px.get_mut(i).expect("known");
                *p *= (rng.range(-0.01, 0.01)).exp();
                e.on_trade(*i, now, *p, rng.range(1.0, 50.0));
            }
            for s in ["a", "b", "c"] {
                if rng.next() < 0.5 {
                    let mut ex: Vec<(InstrumentId, f64)> = Vec::new();
                    for i in &insts {
                        if rng.next() < 0.7 {
                            ex.push((*i, if rng.next() < 0.02 { f64::NAN } else { rng.range(-3.0, 3.0) }));
                        }
                    }
                    e.on_signal(&signal(s, now, &ex), now);
                }
            }
            match (rng.next() * 40.0) as u32 {
                0 => e.on_halt(&Halt { scope: "portfolio".into(), reason: "test".into(), since: None, manual: true }, now),
                1 => e.on_halt(&Halt { scope: "daily_loss".into(), reason: "test".into(), since: None, manual: true }, now),
                2 => e.on_resume("portfolio"),
                3 => e.on_resume("daily_loss"),
                _ => {}
            }
            let halted = !e.risk.halts().is_empty();
            let deployable = e.risk.deployable(e.equity());
            let before = e.current_eur();
            let out = e.tick(now);
            let mut after = before.clone();
            for intent in approved(&out) {
                let inst = InstrumentId(uuid::Uuid::parse_str(&intent.instrument_id).unwrap());
                let mark = e.marks[&inst];
                let signed = if intent.side == Side::Sell as i32 { -intent.quantity } else { intent.quantity };
                assert!(signed.is_finite() && intent.quantity > 0.0, "seed {seed}: bad quantity {intent:?}");
                let cur = before.get(&inst).copied().unwrap_or(0.0);
                let post = cur + signed * mark;
                *after.entry(inst).or_insert(0.0) = post;
                if halted {
                    assert!(post.abs() <= cur.abs() + 1e-6, "seed {seed} tick {k}: {cur} → {post} under a halt");
                } else {
                    assert!(post.abs() <= cap * deployable + 1e-6, "seed {seed} tick {k}: {post} > cap {}", cap * deployable);
                }
                checked += 1;
            }
            if !halted {
                let g: f64 = after.values().map(|v| v.abs()).sum();
                let g_before: f64 = before.values().map(|v| v.abs()).sum();
                assert!(g <= (gross_cap * deployable).max(g_before) + 1e-6, "seed {seed} tick {k}: gross {g}");
            }
            for intent in approved(&out) {
                let inst = InstrumentId(uuid::Uuid::parse_str(&intent.instrument_id).unwrap());
                fill(&mut e, &intent, px[&inst], now);
            }
        }
        assert!(checked > 20, "seed {seed}: the run must actually trade ({checked} orders)");
    }
}

#[test]
fn carry_charges_are_booked_once_into_the_paper_book() {
    let mut e = Engine::new(settings());
    let cash = e.ledger().cash_eur;
    let c = CarryCharge {
        account: "paper".into(),
        instrument_id: iid(1).to_string(),
        kind: "borrow".into(),
        amount: -0.25,
        ccy: String::new(),
        bps_day: 2.0,
        notional_days: 1250.0,
        period_end: Utc.with_ymd_and_hms(2026, 11, 12, 13, 0, 0).unwrap(),
        reference: format!("carry:{}:borrow:1794488400", iid(1)),
    };
    let b = e.carry_booking(&c, "paper").expect("booking");
    assert_eq!((b.movement.kind.as_str(), b.movement.amount, b.movement.fx_rate_eur), ("borrow", -0.25, 1.0));
    assert!(e.apply_carry(&b));
    assert!((e.ledger().cash_eur - (cash - 0.25)).abs() < 1e-12);
    // a re-announced charge and another account's are ignored
    assert!(e.carry_booking(&c, "paper").is_none());
    assert!(e.carry_booking(&CarryCharge { account: "paper-test".into(), ..c.clone() }, "paper").is_none());
    assert_eq!(e.counters["carry.booked"], 1.0);
}

/// Spec 40 §10 Q3: the order goes to the strategies whose own targets moved.
#[test]
fn fills_are_attributed_to_the_strategies_that_caused_them() {
    let t0 = 1_762_560_000_000i64;
    let inst = iid(7);
    let mut e = Engine::new(settings());
    for k in 0..3_600 {
        e.on_trade(inst, t0 - 3_600_000 + k * 1000, 100.0, 50_000.0);
    }
    // A alone goes long: its order, its fill
    e.on_signal(&signal("a", t0, &[(inst, 0.05)]), t0);
    let out = e.tick(t0);
    for i in approved(&out) {
        fill(&mut e, &i, 100.0, t0);
    }
    let qa = e.books["a"].positions[&inst].quantity;
    assert!(qa > 0.0);
    // B joins with the same view: the extra quantity goes to B, A keeps what it had
    let t1 = t0 + 1_000;
    e.on_signal(&signal("a", t1, &[(inst, 0.05)]), t1);
    e.on_signal(&signal("b", t1, &[(inst, 0.05)]), t1);
    let out = e.tick(t1);
    let intents = approved(&out);
    assert!(!intents.is_empty(), "B's target needs an order");
    for i in &intents {
        assert_eq!(e.intents[&i.intent_id].weights.len(), i.strategy_ids.len());
        fill(&mut e, i, 100.0, t1);
    }
    let (qa2, qb) = (e.books["a"].positions[&inst].quantity, e.books["b"].positions[&inst].quantity);
    assert!(qb > 0.0, "B gets its order");
    assert!(qa2 <= qa + 1e-9, "A's sub-book does not grow for B's trade (even split would): {qa} → {qa2}");
}

/// Spec 40 §10 Q4: a loss on a US-listed instrument halts until the US exchange day ends (the
/// close, 21:00 UTC in November), not until UTC midnight.
#[test]
fn a_session_loss_halts_until_its_exchange_day_ends() {
    let ms = |h: u32, m: u32| Utc.with_ymd_and_hms(2026, 11, 12, h, m, 0).unwrap().timestamp_millis();
    let inst = iid(9);
    let mut s = settings();
    s.limits.max_position_frac = 0.5; // a position large enough to lose 2 % of equity
    s.limits.max_class_frac = 1.0;
    s.limits.max_venue_frac = 1.0;
    s.limits.leverage_cap.insert(fa_core::AssetClass::Equity, 1.0);
    let mut e = Engine::new(s);
    e.set_calendars(BTreeMap::from([(inst, CalendarRule::Us)]));
    let t = ms(15, 0);
    for k in 0..3_600 {
        e.on_trade(inst, t - 3_600_000 + k * 1000, 100.0, 1_000.0);
    }
    e.tick(t - 1_000); // records the day keys
    e.on_signal(&signal("a", t, &[(inst, 1.0)]), t);
    for i in approved(&e.tick(t)) {
        fill(&mut e, &i, 100.0, t);
    }
    assert!(e.ledger().positions[&inst].quantity > 0.0);
    // the stock drops 20 % in the afternoon: the daily-loss halt is raised
    e.on_trade(inst, ms(20, 0), 80.0, 1.0);
    e.tick(ms(20, 0));
    assert!(e.has_halt(HaltScope::DailyLoss), "loss {:.4}", 1.0 - e.equity() / e.risk.day_start_equity());
    e.on_trade(inst, ms(20, 59), 80.0, 1.0);
    e.tick(ms(20, 59));
    assert!(e.has_halt(HaltScope::DailyLoss), "still the same exchange day");
    // after the close the day is banked: the halt clears hours before UTC midnight
    e.on_trade(inst, ms(21, 1), 80.0, 1.0);
    e.tick(ms(21, 1));
    assert!(!e.has_halt(HaltScope::DailyLoss), "the US day ended");
    assert!(e.counters["daily_loss.rolls"] >= 1.0);
    // the record survives a restart
    let rec = e.risk_record();
    let json = serde_json::to_string(&rec).unwrap();
    assert!(json.contains("session_days"));
}

#[test]
fn ticks_keep_allocation_and_target_history_on_change() {
    let t0 = 1_762_560_000_000i64;
    let inst = iid(3);
    let mut e = Engine::new(settings());
    for k in 0..3_600 {
        e.on_trade(inst, t0 - 3_600_000 + k * 1000, 100.0, 50.0);
    }
    e.on_signal(&signal("a", t0, &[(inst, 0.4)]), t0);
    let out = e.tick(t0);
    for i in approved(&out) {
        fill(&mut e, &i, 100.0, t0);
    }
    // unchanged ticks inside the snapshot interval are not kept
    for k in 1..10 {
        e.on_signal(&signal("a", t0 + k * 1000, &[(inst, 0.4)]), t0 + k * 1000);
        e.tick(t0 + k * 1000);
    }
    let h = e.take_history();
    assert_eq!(h.len(), 1, "only the first (trading) tick: {h:?}");
    let t = &h[0].targets[0];
    assert_eq!(t.instrument, inst);
    assert_eq!(t.decision, "trade");
    assert!(t.intent_id.is_some() && t.raw_eur > 0.0 && t.capped_eur <= t.raw_eur && t.gated_eur == t.capped_eur);
    assert!(t.contributors.contains_key("a"));
    assert!((h[0].weights["a"] - 1.0).abs() < 1e-9 || h[0].weights["a"] > 0.0);
    // after the interval a snapshot is kept again
    e.on_signal(&signal("a", t0 + 61_000, &[(inst, 0.4)]), t0 + 61_000);
    e.tick(t0 + 61_000);
    assert_eq!(e.take_history().len(), 1);
}

#[test]
fn pretrade_check_answers_without_side_effects() {
    let t0 = 1_762_560_000_000i64;
    let inst = iid(5);
    let mut e = Engine::new(settings());
    for k in 0..3_600 {
        e.on_trade(inst, t0 - 3_600_000 + k * 1000, 100.0, 50.0);
    }
    let req = |q: f64, side: &str| PretradeRequest {
        instrument_id: inst.to_string(),
        side: side.into(),
        quantity: q,
        limit_price: None,
        mode: Some("paper".into()),
    };
    let counters = e.counters.clone();
    let intents = e.intents.len();
    let ok = e.pretrade_check(&req(1.0, "buy"), t0);
    assert!(ok.ok && ok.reasons.is_empty(), "{ok:?}");
    // fat finger: far above max_order_frac × deployable
    let big = e.pretrade_check(&req(1e6, "buy"), t0);
    assert!(!big.ok && !big.reasons.is_empty(), "{big:?}");
    let live = e.pretrade_check(&PretradeRequest { mode: Some("live".into()), ..req(1.0, "buy") }, t0);
    assert!(!live.ok);
    let bad = e.pretrade_check(&req(1.0, "hold"), t0);
    assert!(!bad.ok);
    // a halt blocks a buy
    e.on_halt(&Halt { scope: "portfolio".into(), reason: "test".into(), since: None, manual: true }, t0);
    let halted = e.pretrade_check(&req(1.0, "buy"), t0);
    assert_eq!(halted.reasons, vec!["trading halted".to_owned()]);
    // nothing changed: no intent remembered, no counter moved
    assert_eq!(e.intents.len(), intents);
    let c2: BTreeMap<_, _> = e.counters.iter().filter(|(k, _)| !k.starts_with("halts")).map(|(k, v)| (*k, *v)).collect();
    let c1: BTreeMap<_, _> = counters.iter().filter(|(k, _)| !k.starts_with("halts")).map(|(k, v)| (*k, *v)).collect();
    assert_eq!(c1, c2);
}
