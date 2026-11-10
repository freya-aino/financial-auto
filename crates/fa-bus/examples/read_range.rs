//! Reads a JetStream stream by time range and prints what it holds (`fa_bus::replay`).
//!
//! ```sh
//! cargo run -p fa-bus --example read_range -- LATENT 'latent.state.1s.>' 2026-11-10T06:00:00Z 2026-11-10T06:05:00Z
//! cargo run -p fa-bus --example read_range -- JOB_RESULTS 'jobs.result.evo.eval.strategy' -2h now --json
//! ```
//! Times are RFC 3339, epoch milliseconds, `now`, or `-<n>h|m|s` relative to now. `--json` decodes
//! `JobResult` payloads (`jobs.result.>`) and prints their JSON; `--limit N` caps the output.
//! `FA_NATS_URL` (default `nats://127.0.0.1:4222`).

use fa_bus::replay::{LatentMessage, ReadOptions, read_range};
use fa_proto::{Message, v1::JobResult};

fn now_ms() -> i64 {
    i64::try_from(std::time::UNIX_EPOCH.elapsed().expect("clock").as_millis()).expect("fits")
}

/// `now`, `-15m`, epoch ms or RFC 3339.
fn parse_time(s: &str) -> Result<i64, String> {
    if s == "now" {
        return Ok(now_ms());
    }
    if let Some(rel) = s.strip_prefix('-') {
        let (num, unit) = rel.split_at(rel.len() - 1);
        let n: i64 = num.parse().map_err(|e| format!("{s}: {e}"))?;
        let mult = match unit {
            "h" => 3_600_000,
            "m" => 60_000,
            "s" => 1_000,
            _ => return Err(format!("{s}: unit must be h, m or s")),
        };
        return Ok(now_ms() - n * mult);
    }
    if let Ok(ms) = s.parse::<i64>() {
        return Ok(ms);
    }
    let t = async_nats::datetime::DateTime::parse(s, &time::format_description::well_known::Rfc3339)
        .map_err(|e| format!("{s}: {e}"))?;
    Ok(i64::try_from(t.unix_timestamp_nanos() / 1_000_000).expect("fits"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if positional.len() < 4 {
        eprintln!("usage: read_range <STREAM> <filter subject> <start> <end> [--json] [--limit N]");
        std::process::exit(2);
    }
    let json = args.iter().any(|a| a == "--json");
    let limit = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let (stream, filter) = (positional[0], positional[1]);
    let (start, end) = (parse_time(positional[2])?, parse_time(positional[3])?);
    let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
    let client = async_nats::connect(&url).await?;
    let js = async_nats::jetstream::new(client);
    let records = read_range(&js, stream, filter, start, end, &ReadOptions::default()).await?;
    eprintln!("{} messages on {stream} `{filter}` in [{start}, {end}]", records.len());
    for r in records.iter().take(limit) {
        if json && r.subject.starts_with("jobs.result.") {
            let res = JobResult::decode(r.payload.as_slice())?;
            let body: serde_json::Value = serde_json::from_slice(&res.result).unwrap_or(serde_json::Value::Null);
            println!(
                "{}",
                serde_json::json!({"subject": r.subject, "seq": r.stream_seq, "published_ms": r.published_ms,
                    "job_id": res.job_id, "status": res.status, "error": res.error, "result": body})
            );
        } else {
            match LatentMessage::decode(&r.subject, &r.payload) {
                Ok(m) => println!(
                    "{} seq={} at={} {} snapshot={} as_of={:?}",
                    r.subject,
                    r.stream_seq,
                    r.published_ms,
                    m.kind(),
                    m.snapshot_id(),
                    m.as_of_ms()
                ),
                Err(_) => println!("{r}"),
            }
        }
    }
    Ok(())
}
