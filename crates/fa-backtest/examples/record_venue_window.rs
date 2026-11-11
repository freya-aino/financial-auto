//! Records a live paper-venue window as `VenueRecord` JSONL (`fa_backtest::venue`) for the
//! replay-vs-live identity test (spec 41 §9):
//!
//! - trades (`fa_trade_price` + `fa_trade_size`) and quotes (`fa_quote_bid_px` / `fa_quote_ask_px`)
//!   at their event time, raw samples from VictoriaMetrics `/api/v1/export`;
//! - `order.approved.<account>` intents and `order.event.<account>` fills from the `ORDERS`
//!   JetStream stream at their publish time.
//!
//! The window should start when the broker-gateway started (its estimators run from its first
//! trade), so the replay sees exactly the history the gateway saw. **Caveat:** stored samples
//! carry the trade's *event* time, not when it became known; feeds that publish late (Kraken
//! trades reach the bus 20–90 s after their event time) then look earlier in the replay than
//! they were live, so `export` windows only reproduce feeds without that lag.
//!
//! `live` captures the bus instead, stamping every message with its receive time (knowledge
//! time) and the trade's event time separately, writes a `reset` line when a new broker-gateway
//! instance starts heart-beating, and every `probe_secs` publishes a small market `OrderIntent`
//! per probe instrument on `order.approved.<account>` (alternating buy / sell; use a throwaway
//! account such as `paper-test` that the risk-engine and ledger ignore).
//!
//! ```sh
//! cargo run -p fa-backtest --example record_venue_window -- \
//!   export <start_ms> <end_ms> [account=paper] [nats=nats://127.0.0.1:4222] [vm=http://127.0.0.1:8428] > window.jsonl
//! cargo run -p fa-backtest --example record_venue_window -- \
//!   live <secs> <account> <probe_secs> <inst>=<qty>[,<inst>=<qty>…] [nats] > venue_fixture.jsonl
//! ```
use std::collections::BTreeMap;

use fa_backtest::venue::{VenueRecord, subject_token};
use fa_bus::replay::{ReadOptions, read_range};
use fa_proto::{
    Message,
    v1::{Envelope, Heartbeat, OrderEvent, OrderIntent, Side, TradingMode, envelope, order_event},
};
use futures::StreamExt;

type Samples = BTreeMap<(String, String, i64), f64>;

/// `(inst token, venue, ts_ms) → value` for one metric over the window.
async fn export(http: &reqwest::Client, vm: &str, metric: &str, start_ms: i64, end_ms: i64) -> Samples {
    let body = http
        .get(format!("{vm}/api/v1/export"))
        .query(&[
            ("match[]", metric),
            ("start", &(start_ms / 1000).to_string()),
            ("end", &(end_ms / 1000 + 1).to_string()),
        ])
        .send()
        .await
        .expect("VictoriaMetrics export")
        .text()
        .await
        .expect("export body");
    let mut out = Samples::new();
    for line in body.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).expect("export line");
        let inst = v["metric"]["inst"].as_str().unwrap_or_default().to_owned();
        let venue = v["metric"]["venue"].as_str().unwrap_or_default().to_owned();
        let (Some(vals), Some(ts)) = (v["values"].as_array(), v["timestamps"].as_array()) else { continue };
        for (x, t) in vals.iter().zip(ts) {
            let (Some(x), Some(t)) = (x.as_f64(), t.as_i64()) else { continue };
            if (start_ms..=end_ms).contains(&t) {
                out.insert((inst.clone(), venue.clone(), t), x);
            }
        }
    }
    out
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("export") => export_mode(&args[2..]).await,
        Some("live") => live_mode(&args[2..]).await,
        _ => panic!(
            "usage: record_venue_window export <start_ms> <end_ms> … | live <secs> <account> <probe_secs> <probes>"
        ),
    }
}

async fn live_mode(args: &[String]) {
    let secs: u64 = args.first().and_then(|s| s.parse().ok()).expect("secs");
    let account = args.get(1).cloned().expect("account");
    let probe_secs: u64 = args.get(2).and_then(|s| s.parse().ok()).expect("probe_secs");
    let probes: Vec<(String, f64)> = args
        .get(3)
        .map(|p| {
            p.split(',')
                .filter_map(|x| x.split_once('=').map(|(i, q)| (i.to_owned(), q.parse().expect("probe qty"))))
                .collect()
        })
        .unwrap_or_default();
    let nats = args.get(4).cloned().unwrap_or_else(|| "nats://127.0.0.1:4222".into());
    let bus = fa_bus::Bus::connect(&nats).await.expect("nats");
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<VenueRecord>();
    let mut sub = futures::stream::select_all([
        bus.subscribe_raw("md.trade.>".into()).await.expect("sub"),
        bus.subscribe_raw("md.quote.>".into()).await.expect("sub"),
        bus.subscribe_raw(format!("order.approved.{account}")).await.expect("sub"),
        bus.subscribe_raw(format!("order.event.{account}")).await.expect("sub"),
        bus.subscribe_raw("svc.heartbeat.broker-gateway".into()).await.expect("sub"),
    ]);
    let txs = tx.clone();
    tokio::spawn(async move {
        let mut instance: Option<String> = None;
        while let Some(m) = sub.next().await {
            let t = now_ms();
            let subject = m.subject.as_str();
            let parts: Vec<&str> = subject.split('.').collect();
            let rec = if subject.starts_with("md.") && parts.len() == 4 {
                let Ok(e) = Envelope::decode(m.payload.as_ref()) else { continue };
                let at = e.event_time.map(|ts| ts.seconds * 1000 + i64::from(ts.nanos) / 1_000_000);
                match e.payload {
                    Some(envelope::Payload::Trade(tr)) => VenueRecord::Trade {
                        t,
                        at,
                        venue: parts[2].to_owned(),
                        token: parts[3].to_owned(),
                        price: tr.price,
                        size: tr.size,
                    },
                    Some(envelope::Payload::Quote(q)) => {
                        VenueRecord::Quote { t, token: parts[3].to_owned(), bid: q.bid_price, ask: q.ask_price }
                    }
                    _ => continue,
                }
            } else if subject.starts_with("order.approved.") {
                let Ok(i) = OrderIntent::decode(m.payload.as_ref()) else { continue };
                VenueRecord::Intent {
                    t,
                    intent_id: i.intent_id,
                    instrument_id: i.instrument_id,
                    sell: i.side == Side::Sell as i32,
                    qty: i.quantity,
                    exec_algo: i.exec_algo,
                }
            } else if subject.starts_with("order.event.") {
                let Ok(e) = OrderEvent::decode(m.payload.as_ref()) else { continue };
                if e.kind != order_event::Kind::Fill as i32 {
                    continue;
                }
                VenueRecord::Fill {
                    t,
                    intent_id: e.intent_id,
                    fill_price: e.fill_price,
                    fill_qty: e.fill_quantity,
                    fee: e.fee,
                    arrival_price: e.arrival_price,
                    slippage_bps: e.slippage_bps,
                }
            } else {
                let Ok(h) = Heartbeat::decode(m.payload.as_ref()) else { continue };
                let new = instance.as_ref().is_some_and(|i| *i != h.instance_id);
                instance = Some(h.instance_id);
                if !new {
                    continue;
                }
                eprintln!("broker-gateway restarted at {t}");
                VenueRecord::Reset { t }
            };
            if txs.send(rec).is_err() {
                break;
            }
        }
    });
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(secs);
    let mut probe = tokio::time::interval(std::time::Duration::from_secs(probe_secs.max(1)));
    probe.tick().await;
    let mut n = 0u64;
    let mut recs = Vec::new();
    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(deadline) => break,
            _ = probe.tick() => {
                for (inst, qty) in &probes {
                    n += 1;
                    let intent = OrderIntent {
                        intent_id: ulid::Ulid::generate().to_string(),
                        account_id: account.clone(),
                        instrument_id: inst.clone(),
                        side: if n.is_multiple_of(2) { Side::Sell } else { Side::Buy } as i32,
                        quantity: *qty,
                        exec_algo: "market".into(),
                        mode: TradingMode::Paper as i32,
                        created_at: Some(fa_proto::now()),
                        ..Default::default()
                    };
                    if let Err(e) = bus.publish(format!("order.approved.{account}"), &intent).await {
                        eprintln!("probe publish failed: {e}");
                    }
                }
            }
            Some(r) = rx.recv() => recs.push(r),
        }
    }
    while let Ok(r) = rx.try_recv() {
        recs.push(r);
    }
    recs.sort_by_key(VenueRecord::t);
    for r in &recs {
        println!("{}", serde_json::to_string(r).expect("record"));
    }
    eprintln!("{} records, {n} probes", recs.len());
}

async fn export_mode(args: &[String]) {
    let start_ms: i64 = args.first().and_then(|s| s.parse().ok()).expect("start_ms");
    let end_ms: i64 = args.get(1).and_then(|s| s.parse().ok()).expect("end_ms");
    let account = args.get(2).cloned().unwrap_or_else(|| "paper".into());
    let nats = args.get(3).cloned().unwrap_or_else(|| "nats://127.0.0.1:4222".into());
    let vm = args.get(4).cloned().unwrap_or_else(|| "http://127.0.0.1:8428".into());

    let http = reqwest::Client::new();
    let mut recs: Vec<VenueRecord> = Vec::new();
    let prices = export(&http, &vm, "fa_trade_price", start_ms, end_ms).await;
    let sizes = export(&http, &vm, "fa_trade_size", start_ms, end_ms).await;
    for ((inst, venue, t), price) in &prices {
        let size = sizes.get(&(inst.clone(), venue.clone(), *t)).copied().unwrap_or(0.0);
        recs.push(VenueRecord::Trade {
            t: *t,
            at: None,
            venue: venue.clone(),
            token: subject_token(inst),
            price: *price,
            size,
        });
    }
    let bids = export(&http, &vm, "fa_quote_bid_px", start_ms, end_ms).await;
    let asks = export(&http, &vm, "fa_quote_ask_px", start_ms, end_ms).await;
    for ((inst, venue, t), bid) in &bids {
        if let Some(ask) = asks.get(&(inst.clone(), venue.clone(), *t)) {
            recs.push(VenueRecord::Quote { t: *t, token: subject_token(inst), bid: *bid, ask: *ask });
        }
    }

    let bus = fa_bus::Bus::connect(&nats).await.expect("nats");
    let js = bus.jetstream();
    let opts = ReadOptions::default();
    for m in
        read_range(&js, "ORDERS", &format!("order.approved.{account}"), start_ms, end_ms, &opts).await.expect("ORDERS")
    {
        let Ok(i) = OrderIntent::decode(m.payload.as_ref()) else { continue };
        recs.push(VenueRecord::Intent {
            t: m.published_ms,
            intent_id: i.intent_id,
            instrument_id: i.instrument_id,
            sell: i.side == Side::Sell as i32,
            qty: i.quantity,
            exec_algo: i.exec_algo,
        });
    }
    for m in
        read_range(&js, "ORDERS", &format!("order.event.{account}"), start_ms, end_ms, &opts).await.expect("ORDERS")
    {
        let Ok(e) = OrderEvent::decode(m.payload.as_ref()) else { continue };
        if e.kind != order_event::Kind::Fill as i32 {
            continue;
        }
        recs.push(VenueRecord::Fill {
            t: m.published_ms,
            intent_id: e.intent_id,
            fill_price: e.fill_price,
            fill_qty: e.fill_quantity,
            fee: e.fee,
            arrival_price: e.arrival_price,
            slippage_bps: e.slippage_bps,
        });
    }
    recs.sort_by_key(VenueRecord::t);
    for r in &recs {
        println!("{}", serde_json::to_string(r).expect("record"));
    }
    eprintln!("{} records ({} trades, {} quotes)", recs.len(), prices.len(), bids.len());
}
