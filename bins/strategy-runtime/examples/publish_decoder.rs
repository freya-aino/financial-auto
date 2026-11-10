//! Publishes synthetic text / visual decoder outputs on the model-runtime contract so the
//! strategy-runtime consumer can be verified end to end without the real producer:
//! `DecoderOutput{family: text.news|text.summary|text.narrative|visual.chart|visual.future,
//! quantiles[0] = score in [-1, 1], output_refs: [inst:<uuid> | inst:*, topic:<t> | kind:<k>, src:<s>],
//! timescale, snapshot_id, as_of}` on `decoder.output.<family>`.
//!
//! Instruments are discovered from `md.trade.>` (first 5 s) unless given with `--inst`.
//! Per instrument and period: `text.news`, `text.narrative`, `visual.chart`; market-wide
//! (`inst:*`): `text.summary`, `visual.future`. Scores are slow sine waves per instrument.
//!
//! ```sh
//! cargo run -p strategy-runtime --example publish_decoder -- --secs 60
//! cargo run -p strategy-runtime --example publish_decoder -- --secs 30 --period-ms 500 --inst <uuid>
//! ```
use std::{collections::BTreeSet, time::Duration};

use fa_bus::{Bus, subjects};
use fa_proto::v1::{DecoderOutput, Envelope};
use futures::StreamExt;

fn arg(name: &str) -> Option<String> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == name {
            return it.next();
        }
    }
    None
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let secs: u64 = arg("--secs").and_then(|s| s.parse().ok()).unwrap_or(60);
    let period_ms: u64 = arg("--period-ms").and_then(|s| s.parse().ok()).unwrap_or(1_000);
    let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
    let bus = Bus::connect(&url).await?;

    let mut insts: BTreeSet<String> = arg("--inst").into_iter().collect();
    if insts.is_empty() {
        let mut trades = bus.subscribe::<Envelope>(subjects::ALL_MD_TRADES.into()).await?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while let Ok(Some((_, env))) = tokio::time::timeout_at(deadline, trades.next()).await {
            if !env.instrument_id.is_empty() {
                insts.insert(env.instrument_id);
            }
        }
    }
    println!("publishing for {} instrument(s) + market-wide, {secs} s every {period_ms} ms", insts.len());

    let frame = |family: &str, inst_ref: String, tag: String, score: f64, now: fa_proto::Timestamp| DecoderOutput {
        decoder_id: format!("synthetic-{family}"),
        genome_id: format!("{family}:synthetic"),
        family: family.into(),
        snapshot_id: "synthetic".into(),
        timescale: "1s".into(),
        as_of: Some(now),
        output_refs: vec![inst_ref, tag, "src:publish_decoder".into()],
        quantiles: vec![score.clamp(-1.0, 1.0)],
        analogs: vec![],
    };
    let start = tokio::time::Instant::now();
    let mut tick = tokio::time::interval(Duration::from_millis(period_ms.max(50)));
    let mut sent = 0u64;
    while start.elapsed() < Duration::from_secs(secs) {
        tick.tick().await;
        let t = start.elapsed().as_secs_f64();
        let now = fa_proto::now();
        let mut out = vec![
            frame("text.summary", "inst:*".into(), "topic:summary".into(), (t / 20.0).sin() * 0.5, now),
            frame("visual.future", "inst:*".into(), "kind:future".into(), (t / 15.0).cos() * 0.8, now),
        ];
        for (n, i) in insts.iter().enumerate() {
            let ph = n as f64;
            out.push(frame("text.news", format!("inst:{i}"), "topic:news".into(), (t / 10.0 + ph).sin(), now));
            out.push(frame(
                "text.narrative",
                format!("inst:{i}"),
                "topic:narrative".into(),
                (t / 25.0 + ph).cos() * 0.6,
                now,
            ));
            out.push(frame("visual.chart", format!("inst:{i}"), "kind:chart".into(), (t / 12.0 + ph).sin() * 0.7, now));
        }
        for o in out {
            bus.publish(subjects::decoder_output(&o.family), &o).await?;
            sent += 1;
        }
    }
    println!("sent {sent} decoder outputs");
    Ok(())
}
