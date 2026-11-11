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
    v1::{Envelope, ExposureDistribution, OrderEvent, OrderIntent, envelope::Payload, order_event},
};
use uuid::Uuid;

use crate::{
    engine::{Engine, Out},
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

/// Applies one recorded input; on a tick returns the approved intents (normalized) after filling
/// them in-process at the mark with the venue commission.
pub fn apply(e: &mut Engine, r: &RecordedMessage) -> Vec<OrderIntent> {
    let t = r.published_ms;
    if r.subject.starts_with("signal.exposure.") {
        if let Ok(m) = ExposureDistribution::decode(r.payload.as_slice()) {
            e.on_signal(&m, t);
        }
    } else if let Some(rest) = r.subject.strip_prefix("md.trade.") {
        if let Ok(env) = Envelope::decode(r.payload.as_slice())
            && let Some(Payload::Trade(tr)) = &env.payload
            && let Ok(uuid) = Uuid::parse_str(&env.instrument_id)
        {
            let inst = InstrumentId(uuid);
            if let Some(venue) = rest.split('.').next() {
                e.note_md_venue(inst, venue);
            }
            let at = env.event_time.map_or(t, |ts| ts.seconds * 1000 + i64::from(ts.nanos) / 1_000_000);
            let _ = e.on_trade(inst, at, tr.price, tr.size);
        }
    } else if r.subject == TICK {
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
}
