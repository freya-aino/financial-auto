//! Measures the event-time → bus lag per market-data venue (spec 41 §6.2 knowledge time) from the
//! retained `MD` JetStream stream: for every `md.trade.<venue>.<token>` message, `lag = stream
//! publish time − Envelope.event_time`. The result backs `backfill.knowledge_delay_secs` of the
//! source in `config/sources.yaml` (read by `fa_backtest::knowledge`).
//!
//! ```text
//! cargo run -p fa-backtest --example knowledge_delay -- [minutes=90] [nats=nats://127.0.0.1:4222]
//! ```
use std::collections::BTreeMap;

use fa_bus::replay::{ReadOptions, read_range};
use fa_proto::{
    Message,
    v1::{Envelope, envelope},
};

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

fn pct(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v[((v.len() - 1) as f64 * q).round() as usize]
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let minutes: i64 = args.get(1).and_then(|m| m.parse().ok()).unwrap_or(90);
    let nats = args.get(2).cloned().unwrap_or_else(|| "nats://127.0.0.1:4222".into());
    let js = fa_bus::async_nats::jetstream::new(fa_bus::async_nats::connect(&nats).await.expect("nats"));
    let end = now_ms();
    let start = end - minutes * 60_000;
    let opts = ReadOptions { max_messages: 1_000_000, ..Default::default() };
    let recs = read_range(&js, "MD", "md.trade.>", start, end, &opts).await.expect("read MD");
    let mut lags: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut first: BTreeMap<String, (i64, i64)> = BTreeMap::new();
    for r in &recs {
        let venue = r.subject.split('.').nth(2).unwrap_or("?").to_owned();
        let Ok(e) = Envelope::decode(r.payload.as_slice()) else { continue };
        if !matches!(e.payload, Some(envelope::Payload::Trade(_))) {
            continue;
        }
        let Some(at) = e.event_time.map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000) else { continue };
        lags.entry(venue.clone()).or_default().push((r.published_ms - at) as f64 / 1000.0);
        let f = first.entry(venue).or_insert((r.published_ms, r.published_ms));
        f.0 = f.0.min(r.published_ms);
        f.1 = f.1.max(r.published_ms);
    }
    println!("window {start}..{end} ({minutes} min), {} trade messages", recs.len());
    println!(
        "{:<10} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "venue", "n", "min_s", "p10_s", "p50_s", "p90_s", "p99_s", "max_s"
    );
    for (v, mut l) in lags {
        l.sort_by(f64::total_cmp);
        println!(
            "{:<10} {:>7} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
            v,
            l.len(),
            l[0],
            pct(&l, 0.1),
            pct(&l, 0.5),
            pct(&l, 0.9),
            pct(&l, 0.99),
            l[l.len() - 1]
        );
    }
}
