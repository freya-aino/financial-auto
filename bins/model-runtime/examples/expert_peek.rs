//! Prints `latent.expert.>` traffic (spec 32 §4.3): `ExpertView` per step and `ExpertHealth`
//! (smoke-test helper): `cargo run -p model-runtime --example expert_peek -- 10`.
use fa_proto::{
    Message,
    v1::{ExpertHealth, ExpertView},
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
    let client = async_nats::connect(url).await?;
    let mut sub = client.subscribe("latent.expert.>").await?;
    for _ in 0..n {
        let Some(m) = sub.next().await else { break };
        if m.subject.as_str() == "latent.expert.health" {
            let h = ExpertHealth::decode(m.payload.as_ref())?;
            println!(
                "{} {} k_eff={:.2} corr_max={:.2} skill={:?} n={}",
                m.subject, h.snapshot_id, h.k_eff, h.resid_corr_max, h.skill, h.n_windows
            );
            continue;
        }
        let v = ExpertView::decode(m.payload.as_ref())?;
        let e: Vec<String> = v
            .experts
            .iter()
            .map(|x| {
                format!(
                    "e{} gate={:.2} dim={} region={} next_step={}",
                    x.expert_id,
                    x.gate,
                    x.mu.len(),
                    x.region.as_ref().and_then(|r| r.weights.first()).map_or("-", |w| w.region_id.as_str()),
                    x.next_step.as_ref().map_or(0, |p| p.mean.len())
                )
            })
            .collect();
        println!(
            "{} {} [{}] prior={:?} | {}",
            m.subject,
            v.snapshot_id,
            v.encoder_family,
            v.has_prior_score.then_some(v.prior_score),
            e.join("; ")
        );
    }
    Ok(())
}
