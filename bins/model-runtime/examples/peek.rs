//! Prints region ids / lineages of `latent.state.<ts>.>` and `latent.shadow.<ts>.>` side by side
//! (smoke-test helper): `cargo run -p model-runtime --example peek -- 1m 20`.
use fa_proto::{Message, v1::LatentState};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let ts = args.next().unwrap_or_else(|| "1m".into());
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
    let client = async_nats::connect(url).await?;
    let a = client.subscribe(format!("latent.state.{ts}.>")).await?;
    let b = client.subscribe(format!("latent.shadow.{ts}.>")).await?;
    let mut all = futures::stream::select(a, b);
    for _ in 0..n {
        let Some(m) = all.next().await else { break };
        let s = LatentState::decode(m.payload.as_ref())?;
        let regions: Vec<String> =
            s.experts.iter().map(|e| format!("{}~{}", e.region_id, e.region_lineage_id)).collect();
        println!("{} {} {}", m.subject, s.snapshot_id, regions.join(" "));
    }
    Ok(())
}
