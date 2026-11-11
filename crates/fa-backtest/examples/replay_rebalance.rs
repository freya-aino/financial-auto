//! Replays recorded strategy signals (`SIGNAL` stream) and VictoriaMetrics trade marks through
//! the risk-engine's allocation → netting → rebalance rule and the paper venue's cost model,
//! comparing the D-041 no-trade band with the cost-aware rule (`fa_risk::rebalance`). Used to
//! size the fee churn of 2026-11-10 19:00 – 11-11 02:00 (spec 40 §6.1 step 6).
//!
//! ```text
//! cargo run -p fa-backtest --example replay_rebalance -- <start_rfc3339> <end_rfc3339> \
//!     [costs=go/services/broker-gateway/costs.yaml] [allocator=alloc:name=equal_weight;cap=0.5] \
//!     [nats=nats://127.0.0.1:4222] [vm=http://127.0.0.1:8428]
//! ```
use std::collections::BTreeMap;

use fa_allocator::genome::AllocatorGenome;
use fa_backtest::{
    allocator_replay::{
        AllocatorReplaySpec, SIGNAL_FILTER, SIGNAL_STREAM, TradeRule, decode_signals, replay_allocator,
    },
    cost::CostModel,
    replay::{BarSeries, ReplayControls},
};
use fa_bus::replay::{ReadOptions, read_range};
use fa_risk::rebalance::RebalanceConfig;

fn ms(s: &str) -> i64 {
    chrono_like(s)
}

/// Minimal RFC 3339 (`YYYY-MM-DDTHH:MM:SSZ`) → epoch ms, or a raw integer.
fn chrono_like(s: &str) -> i64 {
    if let Ok(v) = s.parse::<i64>() {
        return v;
    }
    let b = s.as_bytes();
    let n = |a: usize, l: usize| s[a..a + l].parse::<i64>().expect("timestamp");
    assert!(b.len() >= 19, "timestamp {s}");
    let (y, mo, d, h, mi, se) = (n(0, 4), n(5, 2), n(8, 2), n(11, 2), n(14, 2), n(17, 2));
    // days from civil (Howard Hinnant)
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let doy = (153 * (mo + if mo > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    ((days * 86_400) + h * 3_600 + mi * 60 + se) * 1000
}

async fn marks(http: &reqwest::Client, vm: &str, start_ms: i64, end_ms: i64) -> Vec<BarSeries> {
    let body = http
        .get(format!("{vm}/api/v1/export"))
        .query(&[
            ("match[]", "fa_trade_price"),
            ("start", &(start_ms / 1000).to_string()),
            ("end", &(end_ms / 1000 + 1).to_string()),
        ])
        .send()
        .await
        .expect("VictoriaMetrics export")
        .text()
        .await
        .expect("export body");
    let mut pts: BTreeMap<(String, String), Vec<(i64, f64)>> = BTreeMap::new();
    for line in body.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).expect("export line");
        let inst = v["metric"]["inst"].as_str().unwrap_or_default().to_owned();
        let venue = v["metric"]["venue"].as_str().unwrap_or_default().to_owned();
        let (Some(vals), Some(ts)) = (v["values"].as_array(), v["timestamps"].as_array()) else { continue };
        let e = pts.entry((inst, venue)).or_default();
        for (x, t) in vals.iter().zip(ts) {
            if let (Some(x), Some(t)) = (x.as_f64(), t.as_i64()) {
                e.push((t, x));
            }
        }
    }
    pts.into_iter()
        .map(|((inst, venue), mut p)| {
            p.sort_by_key(|x| x.0);
            BarSeries::from_points(&inst, &venue, &p, 1_000)
        })
        .collect()
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let start = ms(args.get(1).expect("start"));
    let end = ms(args.get(2).expect("end"));
    let costs_path = args.get(3).cloned().unwrap_or_else(|| "go/services/broker-gateway/costs.yaml".into());
    let genome_text = args.get(4).cloned().unwrap_or_else(|| "alloc:name=equal_weight;cap=0.5".into());
    let nats = args.get(5).cloned().unwrap_or_else(|| "nats://127.0.0.1:4222".into());
    let vm = args.get(6).cloned().unwrap_or_else(|| "http://127.0.0.1:8428".into());
    let cost = CostModel::from_yaml(&std::fs::read_to_string(&costs_path).expect("costs")).expect("cost model");
    let genome = AllocatorGenome::parse(&genome_text).expect("genome");

    let js = fa_bus::async_nats::jetstream::new(fa_bus::async_nats::connect(&nats).await.expect("nats"));
    let opts = ReadOptions { max_messages: 2_000_000, ..Default::default() };
    let raw = read_range(&js, SIGNAL_STREAM, SIGNAL_FILTER, start - 60_000, end, &opts).await.expect("SIGNAL");
    let (signals, bad) = decode_signals(&raw);
    let http = reqwest::Client::new();
    let bars = marks(&http, &vm, start - 3_600_000, end).await;
    eprintln!(
        "{} signals ({bad} undecodable), {} mark series ({} points), costs {costs_path}",
        signals.len(),
        bars.len(),
        bars.iter().map(BarSeries::len).sum::<usize>()
    );
    let base = AllocatorReplaySpec { signal_ttl_ms: 60_000, ..AllocatorReplaySpec::new(start, end) };
    let rules = [
        ("band (D-041: 0.5 % / 25 EUR)", TradeRule::Band { frac: 0.005, min_eur: 25.0 }),
        (
            "cost-aware (risk.toml [rebalance])",
            TradeRule::CostAware { frac: 0.005, min_eur: 25.0, cfg: RebalanceConfig::default() },
        ),
    ];
    println!(
        "{:<36} {:>8} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9}",
        "rule", "orders", "notional", "fees_eur", "cost_eur", "gross_pnl", "net_pnl", "avg_eur"
    );
    for (name, rule) in rules {
        let spec = AllocatorReplaySpec { trade_rule: rule, ..base.clone() };
        let r = replay_allocator(&genome, &signals, &bars, &spec, &cost, &ReplayControls::default()).expect("replay");
        let notional = r.turnover * r.capital * r.n_steps as f64;
        println!(
            "{:<36} {:>8} {:>10.0} {:>10.1} {:>10.1} {:>10.1} {:>9.1} {:>9.0}",
            name,
            r.n_trades,
            notional,
            r.total_fees,
            r.total_cost,
            r.gross_pnl,
            r.net_pnl,
            if r.n_trades > 0 { notional / r.n_trades as f64 } else { 0.0 }
        );
        if let Some(st) = &r.rebalance {
            println!("    {st:?}");
        }
        eprintln!("    strategies {:?}", r.strategies);
    }
}
