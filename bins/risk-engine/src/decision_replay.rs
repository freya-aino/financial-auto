//! Decision-level replay (spec 41 §9 "replaying a recorded live window reproduces the live paper
//! decisions exactly — same orders, same timestamps"): strategy decisions
//! (`signal.exposure.*`) and the tape (`md.trade.*`) go through the risk-engine's own
//! [`Engine`] — allocation, netting, hard limits, the cost-aware rebalance, algorithm choice —
//! and the intents it emits are compared with the ones recorded live.
//!
//! * **Record** (`FA_DECISION_RECORD_SECS=<n> cargo test -p risk-engine -- --ignored
//!   record_live_decision_window`): an engine subscribes read-only to the live bus for `n`
//!   seconds, applies every message on arrival with its receive time (as `main.rs` does) and
//!   ticks on the 1 s wall-clock grid (as the deployed engine does since the grid alignment);
//!   every input, tick and approved intent goes into `testdata/decision_fixture.jsonl`. It
//!   publishes nothing: its intents are filled in-process at the mark with the cost model's
//!   commission (the paper book is never touched, no probe order is sent).
//! * **Replay** (`decision_fixture_reproduces_the_live_intents`, always run): a fresh engine
//!   fed the fixture's inputs in recorded order must emit exactly the recorded intents at the
//!   recorded ticks (instrument, side, quantity, algorithm, contributors; intent ids are random
//!   and excluded).
//! * **From the retained streams** (`FA_DECISION_STREAMS=1 … --ignored
//!   decision_window_from_retained_streams`): the same window rebuilt from the `SIGNAL` and `MD`
//!   JetStream streams alone (knowledge time = stream publish time) instead of the receive
//!   times; reports how many live intents the stream replay reproduces.
use std::path::PathBuf;

use fa_backtest::cost::CostModel;
use fa_bus::replay::RecordedMessage;
use fa_core::InstrumentId;
use fa_proto::{
    Message,
    v1::{OrderEvent, OrderIntent, order_event},
};
use uuid::Uuid;

use crate::{
    engine::{Engine, Out},
    inputs::Input,
    settings::Settings,
};

pub const TICK: &str = "tick";
pub const INTENT: &str = "intent";

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/decision_fixture.jsonl")
}

fn costs() -> CostModel {
    CostModel::from_yaml(include_str!("../../../go/services/broker-gateway/costs.yaml")).expect("costs.yaml")
}

fn settings() -> Settings {
    let mut s = Settings::parse(include_str!("../risk.toml")).expect("risk.toml");
    s.account = "decision-replay".into();
    s
}

/// The intent as compared: the random id cleared.
fn normalized(mut i: OrderIntent) -> OrderIntent {
    i.intent_id.clear();
    i
}

/// FIFO map of deployed intents onto replay intents **without the quantity** (D-105): the join
/// key is (tick, instrument, side, algo, strategies) and ties at one key resolve in emission
/// order. The 2026-11-14 full-day replay joined on the exact quantity too, so one small
/// allocation difference unmapped every later fill of the instrument (5 803 of 6 840 events)
/// and snowballed; quantities are now compared and reported, not joined on.
pub struct FillMap {
    by_key: std::collections::HashMap<String, std::collections::VecDeque<(String, f64)>>,
    pub mapped: usize,
    pub qty_diff_sum: f64,
    pub qty_diff_max: f64,
}

impl FillMap {
    pub fn key(t: i64, i: &OrderIntent) -> String {
        format!("{t} {} side={} algo={} by={:?}", i.instrument_id, i.side, i.exec_algo, i.strategy_ids)
    }

    pub fn new(deployed: impl IntoIterator<Item = (i64, OrderIntent)>) -> Self {
        let mut by_key: std::collections::HashMap<String, std::collections::VecDeque<(String, f64)>> =
            Default::default();
        for (t, i) in deployed {
            by_key.entry(Self::key(t, &i)).or_default().push_back((i.intent_id.clone(), i.quantity));
        }
        Self { by_key, mapped: 0, qty_diff_sum: 0.0, qty_diff_max: 0.0 }
    }

    /// Pops the deployed intent id matching a replay intent at tick `t`, recording the quantity
    /// difference as a diagnostic.
    pub fn map(&mut self, t: i64, replayed: &OrderIntent) -> Option<String> {
        let (id, qty) = self.by_key.get_mut(&Self::key(t, replayed))?.pop_front()?;
        self.mapped += 1;
        let d = (qty - replayed.quantity).abs();
        self.qty_diff_sum += d;
        self.qty_diff_max = self.qty_diff_max.max(d);
        Some(id)
    }

    pub fn mean_qty_diff(&self) -> f64 {
        if self.mapped == 0 { 0.0 } else { self.qty_diff_sum / self.mapped as f64 }
    }
}

/// Applies one recorded input through the live decoder ([`Input::parse`]) and application
/// ([`Input::apply`]) — one code path for live and replay (D-105); on a tick returns the
/// approved intents (normalized) after filling them in-process at the mark with the venue
/// commission.
pub fn apply(e: &mut Engine, r: &RecordedMessage) -> Vec<OrderIntent> {
    let t = r.published_ms;
    if r.subject != TICK {
        if let Some(input) = Input::parse(&r.subject, &r.payload) {
            let _ = input.apply(e, t);
        }
    } else {
        let outs = e.tick(t);
        let approved: Vec<OrderIntent> =
            outs.into_iter().filter_map(|o| if let Out::Approved(i) = o { Some(i) } else { None }).collect();
        for i in &approved {
            let Ok(uuid) = Uuid::parse_str(&i.instrument_id) else { continue };
            let inst = InstrumentId(uuid);
            let Some(mark) = e.mark(&inst) else { continue };
            let q = e.cost_quote(inst, t);
            let notional = i.quantity * mark;
            let fee = q.commission_min_eur.max(notional * q.commission_bps * 1e-4);
            let ev = OrderEvent {
                intent_id: i.intent_id.clone(),
                venue_order_id: format!("replay-{}", i.intent_id),
                kind: order_event::Kind::Fill as i32,
                fill_price: mark,
                fill_quantity: i.quantity,
                fee,
                fee_currency: "EUR".into(),
                event_time: i.created_at,
                ..Default::default()
            };
            e.on_order_event(&ev, t);
        }
        return approved.into_iter().map(normalized).collect();
    }
    Vec::new()
}

/// Replays inputs (signals, trades, ticks; recorded intents are skipped) through a fresh
/// engine; returns `(tick ms, intent)` in emission order.
pub fn replay(records: &[RecordedMessage]) -> Vec<(i64, OrderIntent)> {
    replay_with(records, settings()).0
}

/// [`replay`] under other settings; also returns the commission the in-process fills paid.
pub fn replay_with(records: &[RecordedMessage], settings: Settings) -> (Vec<(i64, OrderIntent)>, f64) {
    let mut e = Engine::new(settings);
    e.set_costs(costs());
    let mut out = Vec::new();
    let mut fees = 0.0;
    for r in records.iter().filter(|r| r.subject != INTENT) {
        for i in apply(&mut e, r) {
            if let Ok(uuid) = Uuid::parse_str(&i.instrument_id) {
                let inst = InstrumentId(uuid);
                let q = e.cost_quote(inst, r.published_ms);
                let n = i.quantity * e.mark(&inst).unwrap_or(0.0);
                fees += q.commission_min_eur.max(n * q.commission_bps * 1e-4);
            }
            out.push((r.published_ms, i));
        }
    }
    (out, fees)
}

/// The live intents of a fixture.
pub fn recorded_intents(records: &[RecordedMessage]) -> Vec<(i64, OrderIntent)> {
    records
        .iter()
        .filter(|r| r.subject == INTENT)
        .map(|r| (r.published_ms, OrderIntent::decode(r.payload.as_slice()).expect("intent")))
        .collect()
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use futures::StreamExt;

    use super::*;

    fn load() -> Vec<RecordedMessage> {
        let text = std::fs::read_to_string(fixture_path()).expect("testdata/decision_fixture.jsonl");
        fa_bus::replay::from_jsonl(&text).expect("fixture parses")
    }

    /// Fingerprint used in failure messages.
    fn key(t: i64, i: &OrderIntent) -> String {
        format!(
            "{t} {} side={} qty={:.9} algo={} by={:?}",
            i.instrument_id, i.side, i.quantity, i.exec_algo, i.strategy_ids
        )
    }

    #[test]
    fn decision_fixture_reproduces_the_live_intents() {
        let records = load();
        let live = recorded_intents(&records);
        let ticks = records.iter().filter(|r| r.subject == TICK).count();
        let signals = records.iter().filter(|r| r.subject.starts_with("signal.")).count();
        assert!(ticks >= 60 && signals >= 100, "fixture too small: {ticks} ticks, {signals} signals");
        assert!(!live.is_empty(), "the recorded window must contain live intents");
        let replayed = replay(&records);
        let a: Vec<String> = live.iter().map(|(t, i)| key(*t, i)).collect();
        let b: Vec<String> = replayed.iter().map(|(t, i)| key(*t, i)).collect();
        assert_eq!(a, b, "same orders at the same ticks");
        assert_eq!(live, replayed, "byte-identical intents (ids excluded)");
        eprintln!("decision replay: {ticks} ticks, {signals} signals, {} intents reproduced", live.len());
        // the same live window under the D-041 band alone
        let (_, fees_now) = replay_with(&records, settings());
        let mut band = settings();
        band.rebalance.enabled = false;
        let (legacy, fees_band) = replay_with(&records, band);
        eprintln!(
            "same window, band only: {} intents, {fees_band:.2} EUR commission; cost-aware: {} intents, {fees_now:.2} EUR",
            legacy.len(),
            live.len()
        );
        assert!(legacy.len() > live.len() && fees_band >= fees_now);
    }

    /// Records a live window (see module docs). Read-only on the bus; nothing is published.
    #[tokio::test]
    #[ignore = "records from the live bus; FA_DECISION_RECORD_SECS=<n>"]
    async fn record_live_decision_window() {
        let secs: i64 = std::env::var("FA_DECISION_RECORD_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        if secs <= 0 {
            return;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let client = fa_bus::async_nats::connect(&url).await.expect("nats");
        let mut msgs = futures::stream::select(
            client.subscribe("signal.exposure.>").await.expect("sub"),
            client.subscribe("md.trade.>").await.expect("sub"),
        );
        let mut e = Engine::new(settings());
        e.set_costs(costs());
        let mut rec: Vec<RecordedMessage> = Vec::new();
        let period = 1000i64;
        let phase = std::time::Duration::from_millis((period - now_ms().rem_euclid(period)) as u64);
        let mut tick =
            tokio::time::interval_at(tokio::time::Instant::now() + phase, std::time::Duration::from_millis(1000));
        let end = now_ms() + secs * 1000;
        let mut seq = 0u64;
        loop {
            tokio::select! {
                Some(m) = msgs.next() => {
                    seq += 1;
                    let r = RecordedMessage { subject: m.subject.to_string(), stream_seq: seq, published_ms: now_ms(), payload: m.payload.to_vec() };
                    apply(&mut e, &r);
                    rec.push(r);
                }
                _ = tick.tick() => {
                    let t = now_ms();
                    let grid = t - t.rem_euclid(period);
                    if grid > end { break; }
                    seq += 1;
                    let r = RecordedMessage { subject: TICK.into(), stream_seq: seq, published_ms: grid, payload: Vec::new() };
                    let intents = apply(&mut e, &r);
                    rec.push(r);
                    for i in intents {
                        seq += 1;
                        rec.push(RecordedMessage { subject: INTENT.into(), stream_seq: seq, published_ms: grid, payload: i.encode_to_vec() });
                    }
                }
            }
        }
        let n = rec.iter().filter(|r| r.subject == INTENT).count();
        eprintln!("recorded {} messages, {n} intents", rec.len());
        std::fs::write(fixture_path(), fa_bus::replay::to_jsonl(&rec)).expect("write fixture");
    }

    /// Rebuilds the fixture's window from the retained `SIGNAL` and `MD` streams (knowledge time
    /// = stream publish time) and compares the replayed intents with the live ones.
    #[tokio::test]
    #[ignore = "reads the live JetStream streams; FA_DECISION_STREAMS=1"]
    async fn decision_window_from_retained_streams() {
        if std::env::var("FA_DECISION_STREAMS").is_err() {
            return;
        }
        let fixture = load();
        let ticks: Vec<i64> = fixture.iter().filter(|r| r.subject == TICK).map(|r| r.published_ms).collect();
        let (t0, t1) = (*ticks.first().expect("ticks"), *ticks.last().expect("ticks"));
        let first_input = fixture
            .iter()
            .filter(|r| r.subject != TICK && r.subject != INTENT)
            .map(|r| r.published_ms)
            .min()
            .unwrap_or(t0);
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let js = fa_bus::async_nats::jetstream::new(fa_bus::async_nats::connect(&url).await.expect("nats"));
        let opts = fa_bus::replay::ReadOptions::default();
        let mut inputs = fa_bus::replay::read_range(&js, "SIGNAL", "signal.exposure.>", first_input, t1, &opts)
            .await
            .expect("SIGNAL");
        inputs.extend(fa_bus::replay::read_range(&js, "MD", "md.trade.>", first_input, t1, &opts).await.expect("MD"));
        inputs.sort_by_key(|r| (r.published_ms, r.subject.clone(), r.stream_seq));
        // inputs published at or before a tick are known to it (knowledge time ≤ decision time)
        let mut merged: Vec<RecordedMessage> = Vec::new();
        let mut k = 0usize;
        for &t in &ticks {
            while k < inputs.len() && inputs[k].published_ms <= t {
                merged.push(inputs[k].clone());
                k += 1;
            }
            merged.push(RecordedMessage { subject: TICK.into(), stream_seq: 0, published_ms: t, payload: Vec::new() });
        }
        let live = recorded_intents(&fixture);
        let replayed = replay(&merged);
        let count = |v: &[(i64, OrderIntent)]| {
            let mut m: BTreeMap<String, usize> = BTreeMap::new();
            for (t, i) in v {
                *m.entry(key(*t, i)).or_default() += 1;
            }
            m
        };
        let (a, b) = (count(&live), count(&replayed));
        let same: usize = a.iter().map(|(k, n)| (*n).min(b.get(k).copied().unwrap_or(0))).sum();
        let in_fixture = fixture.iter().filter(|r| r.subject != TICK && r.subject != INTENT).count();
        let fixture_payloads: std::collections::BTreeSet<&[u8]> =
            fixture.iter().filter(|r| r.subject != TICK && r.subject != INTENT).map(|r| r.payload.as_slice()).collect();
        let retained = inputs.iter().filter(|r| fixture_payloads.contains(r.payload.as_slice())).count();
        eprintln!(
            "retained streams: {} inputs read ({retained} of the fixture's {in_fixture} found byte-identical); live intents {}, stream replay {}, identical {same} ({:.1} %)",
            inputs.len(),
            live.len(),
            replayed.len(),
            100.0 * same as f64 / live.len().max(1) as f64
        );
        assert!(same * 100 >= live.len() * 95, "the retained streams must reproduce the live decisions");
    }

    /// The **deployed** service, from the streams alone (spec 41 §9, full-day replay): a cold
    /// engine with the deployed `risk.toml` (account `FA_REPLAY_ACCOUNT`, default `paper`) is fed
    /// every `signal.exposure.*` and `md.trade.*` retained since `FA_REPLAY_FROM_MS` (the deployed
    /// process start on an empty ledger, e.g. a fresh ephemeral stack) plus the **gateway's real
    /// order events** from `ORDERS` (their intent ids mapped onto the replay's own), ticking on the
    /// 1 s wall-clock grid like the deployed engine. Its approved intents are compared tick by tick
    /// with the `order.approved.<account>` the deployed engine published.
    #[tokio::test]
    #[ignore = "reads the live JetStream streams; FA_REPLAY_FROM_MS=<deployed start ms>"]
    async fn deployed_decisions_from_retained_streams() {
        use std::collections::HashMap;
        let Some(from) = std::env::var("FA_REPLAY_FROM_MS").ok().and_then(|s| s.parse::<i64>().ok()) else {
            return;
        };
        let to = std::env::var("FA_REPLAY_TO_MS").ok().and_then(|s| s.parse().ok()).unwrap_or_else(now_ms);
        let account = std::env::var("FA_REPLAY_ACCOUNT").unwrap_or_else(|_| "paper".into());
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let js = fa_bus::async_nats::jetstream::new(fa_bus::async_nats::connect(&url).await.expect("nats"));
        let opts = fa_bus::replay::ReadOptions { max_messages: 50_000_000, ..Default::default() };
        let read = |stream: &'static str, filter: String| {
            let js = js.clone();
            async move { fa_bus::replay::read_range(&js, stream, &filter, from, to, &opts).await.expect(stream) }
        };
        let mut inputs = read("SIGNAL", "signal.exposure.>".into()).await;
        // the deployed service's market-data mode (D-103; FA_MARKET_DATA, default real)
        let mode = fa_core::market_data::MarketDataMode::from_env().expect("FA_MARKET_DATA");
        inputs.extend(read("MD", "md.trade.>".into()).await.into_iter().filter(|r| mode.admits_subject(&r.subject)));
        // halts the deployed engine heard (its own and the ingestor's risk.halt.source:*),
        // resumes and the model-drift kernel updates (D-105; ingest.health.> is not retained in
        // any stream — the ingestor's risk.halt.source:* mirror covers the downs)
        inputs.extend(read("ORDERS", "risk.halt.>".into()).await);
        inputs.extend(read("ORDERS", "risk.resume.>".into()).await);
        inputs.extend(read("LATENT", "latent.mcmc.>".into()).await);
        let events = read("ORDERS", format!("order.event.{account}")).await;
        let live: Vec<OrderIntent> = read("ORDERS", format!("order.approved.{account}"))
            .await
            .iter()
            .filter_map(|r| OrderIntent::decode(r.payload.as_slice()).ok())
            .collect();
        let live_raw: Vec<OrderIntent> = read("ORDERS", format!("order.intent.{account}"))
            .await
            .iter()
            .filter_map(|r| OrderIntent::decode(r.payload.as_slice()).ok())
            .collect();
        inputs.extend(events.iter().cloned());
        inputs.sort_by_key(|r| (r.published_ms, r.stream_seq));
        let created = |i: &OrderIntent| i.created_at.map_or(0, |t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000);
        let tick_of = |ms: i64| ms - ms.rem_euclid(1000);
        let fp = |t: i64, i: &OrderIntent| {
            format!(
                "{t} {} side={} qty={:.6} algo={} by={:?}",
                i.instrument_id, i.side, i.quantity, i.exec_algo, i.strategy_ids
            )
        };
        // the gateway's events are remapped onto replay intent ids without the quantity (D-105)
        let mut fills = FillMap::new(live_raw.iter().map(|i| (tick_of(created(i)), i.clone())));
        let mut s = settings();
        s.account = account.clone();
        let mut e = Engine::new(s);
        e.set_costs(costs());
        let mut id_map: HashMap<String, String> = HashMap::new(); // deployed → replay
        let (mut replayed, mut unmapped_events) = (Vec::<(i64, String)>::new(), 0usize);
        let mut k = 0usize;
        let mut t = tick_of(from) + 1000;
        while t <= to {
            while k < inputs.len() && inputs[k].published_ms <= t {
                let r = &inputs[k];
                k += 1;
                if r.subject.starts_with("order.event.") {
                    let Ok(mut ev) = OrderEvent::decode(r.payload.as_slice()) else { continue };
                    match id_map.get(&ev.intent_id) {
                        Some(id) => {
                            ev.intent_id = id.clone();
                            e.on_order_event(&ev, r.published_ms);
                        }
                        None => unmapped_events += 1,
                    }
                } else if let Some(input) = Input::parse(&r.subject, &r.payload) {
                    // the live decoder+application (D-105): signals, trades, halts, resumes, kernels
                    let _ = input.apply(&mut e, r.published_ms);
                }
            }
            for o in e.tick(t) {
                match o {
                    Out::Intent(i) => {
                        if let Some(dep) = fills.map(t, &i) {
                            id_map.insert(dep, i.intent_id.clone());
                        }
                    }
                    Out::Approved(i) => replayed.push((t, fp(t, &i))),
                    _ => {}
                }
            }
            t += 1000;
        }
        let live: Vec<(i64, String)> = live.iter().map(|i| (tick_of(created(i)), fp(tick_of(created(i)), i))).collect();
        fn count<'a>(v: impl IntoIterator<Item = &'a (i64, String)>) -> BTreeMap<&'a str, usize> {
            let mut m: BTreeMap<&str, usize> = BTreeMap::new();
            for (_, x) in v {
                *m.entry(x.as_str()).or_default() += 1;
            }
            m
        }
        let (a, b) = (count(&live), count(&replayed));
        let same: usize = a.iter().map(|(k, n)| (*n).min(b.get(k).copied().unwrap_or(0))).sum();
        // identity per hour of the window, so a mid-day divergence (a halt the replay missed, a
        // stalled stretch) is located instead of drowned in the global count
        let hours = ((to - from) / 3_600_000 + 1) as usize;
        for h in 0..hours {
            let in_hour = |(t, _): &&(i64, String)| ((*t - from) / 3_600_000) as usize == h;
            let (lh, rh) = (count(live.iter().filter(in_hour)), count(replayed.iter().filter(in_hour)));
            let live_n: usize = lh.values().sum();
            if live_n == 0 && rh.is_empty() {
                continue;
            }
            let same_h: usize = lh.iter().map(|(k, n)| (*n).min(rh.get(k).copied().unwrap_or(0))).sum();
            eprintln!(
                "  hour {h:>3}: live {live_n:>5}, replayed {:>5}, identical {same_h:>5} ({:.1} %)",
                rh.values().sum::<usize>(),
                100.0 * same_h as f64 / live_n.max(1) as f64
            );
        }
        let first_diff = live.iter().zip(&replayed).position(|(x, y)| x.1 != y.1);
        eprintln!(
            "deployed replay {from}..{to} ({:.1} h): {} inputs, {} order events ({unmapped_events} unmapped, {} mapped; qty diff mean {:.9} max {:.9}); live approved {}, replayed {}, identical {same} ({:.2} %); first divergence at {:?}",
            (to - from) as f64 / 3.6e6,
            inputs.len(),
            events.len(),
            fills.mapped,
            fills.mean_qty_diff(),
            fills.qty_diff_max,
            live.len(),
            replayed.len(),
            100.0 * same as f64 / live.len().max(1) as f64,
            first_diff.map(|p| (&live[p].1, &replayed[p].1))
        );
        assert!(!live.is_empty(), "the deployed engine approved nothing in the window");
        assert!(same * 100 >= live.len() * 95, "the deployed decisions must be reproduced from the streams");
    }

    #[test]
    fn fill_map_joins_without_the_quantity() {
        let intent = |id: &str, qty: f64| OrderIntent {
            intent_id: id.into(),
            instrument_id: "0190".into(),
            side: 1,
            quantity: qty,
            exec_algo: "market".into(),
            strategy_ids: vec!["s1".into()],
            ..Default::default()
        };
        // two deployed intents at one tick with the same key, different quantities
        let mut fm = FillMap::new([(1000, intent("dep-a", 3.037264)), (1000, intent("dep-b", 1.5))]);
        // the replay's quantity differs (the 03:08:16 divergence shape): still mapped, FIFO
        assert_eq!(fm.map(1000, &intent("rep-1", 2.329854)), Some("dep-a".into()));
        assert_eq!(fm.map(1000, &intent("rep-2", 1.5)), Some("dep-b".into()));
        assert_eq!(fm.map(1000, &intent("rep-3", 1.0)), None, "no deployed intent left at the key");
        assert_eq!(fm.mapped, 2);
        assert!((fm.qty_diff_max - 0.70741).abs() < 1e-6);
        assert!((fm.mean_qty_diff() - 0.353705).abs() < 1e-6);
        // a different tick, side or strategy set does not join
        let mut fm = FillMap::new([(1000, intent("dep-a", 1.0))]);
        assert_eq!(fm.map(2000, &intent("x", 1.0)), None);
        let mut sell = intent("x", 1.0);
        sell.side = 2;
        assert_eq!(fm.map(1000, &sell), None);
    }
}
