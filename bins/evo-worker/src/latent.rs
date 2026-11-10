//! Latent conditioning: subscribes to `latent.state.>` and keeps the latest `LatentState`s
//! per instrument so `evo.eval.decoder` can hand the retrieval decoder a point-in-time
//! `DecodeCtx` (spec 32 → spec 34 §6.5). Each state becomes one `LatentPoint` whose vector is
//! the concatenation of the expert means in `expert_index` order (missing experts → empty).

use std::sync::{Arc, Mutex};

use fa_bus::{Bus, subjects};
use fa_decoders::{DecodeCtx, LatentPoint, instrument_token};
use fa_proto::v1::LatentState;
use futures::StreamExt;

/// Latents kept per instrument.
pub const MAX_POINTS_PER_INSTRUMENT: usize = 20_000;

#[derive(Clone, Default)]
pub struct LatentBuffer {
    inner: Arc<Mutex<DecodeCtx>>,
}

/// Concatenated expert means (expert order), or `None` when the state carries no numbers.
pub fn vector_of(state: &LatentState) -> Option<Vec<f32>> {
    let mut experts: Vec<_> = state.experts.iter().collect();
    experts.sort_by_key(|e| e.expert_index);
    let v: Vec<f32> = experts.iter().flat_map(|e| e.mean.iter().copied()).collect();
    (!v.is_empty()).then_some(v)
}

impl LatentBuffer {
    pub fn push(&self, state: &LatentState) {
        let Some(vec) = vector_of(state) else { return };
        if state.instrument_id.is_empty() {
            return;
        }
        let as_of_ms = state.as_of.as_ref().map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000).unwrap_or(0);
        let mut ctx = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let key = instrument_token(&state.instrument_id);
        ctx.push(&key, LatentPoint { as_of_ms, vec });
        if let Some(v) = ctx.latents.get_mut(&key)
            && v.len() > MAX_POINTS_PER_INSTRUMENT
        {
            let drop = v.len() - MAX_POINTS_PER_INSTRUMENT;
            v.drain(..drop);
        }
    }

    /// Snapshot for one job (cheap: cloned once per job).
    pub fn ctx(&self) -> Arc<DecodeCtx> {
        Arc::new(self.inner.lock().unwrap_or_else(|e| e.into_inner()).clone())
    }

    /// `(instruments, points)`.
    pub fn stats(&self) -> (usize, usize) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).stats()
    }

    /// Consumes `latent.state.>` forever.
    pub async fn run(self, bus: Bus) {
        let mut sub = match bus.subscribe::<LatentState>(subjects::ALL_LATENT_STATES.into()).await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "cannot subscribe to {}", subjects::ALL_LATENT_STATES);
                return;
            }
        };
        while let Some((_, state)) = sub.next().await {
            self.push(&state);
        }
        tracing::warn!("latent.state subscription ended");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_proto::v1::ExpertLatent;

    #[test]
    fn concatenates_expert_means_in_order_and_is_point_in_time() {
        let b = LatentBuffer::default();
        let state = |ms: i64, means: Vec<Vec<f32>>| LatentState {
            snapshot_id: "s".into(),
            timescale: "1s".into(),
            instrument_id: "0190-AB".into(),
            as_of: Some(crate::job::ts_ms(ms)),
            experts: means
                .into_iter()
                .enumerate()
                .rev()
                .map(|(i, m)| ExpertLatent { expert_index: i as u32, mean: m, ..Default::default() })
                .collect(),
            router_weights: vec![],
        };
        b.push(&state(2000, vec![vec![1.0, 2.0], vec![3.0]]));
        b.push(&state(1000, vec![vec![0.0], vec![0.5]]));
        b.push(&LatentState { instrument_id: "x".into(), ..Default::default() }); // no numbers → ignored
        assert_eq!(b.stats(), (1, 2));
        let ctx = b.ctx();
        assert_eq!(ctx.latent_as_of("0190ab", 1500).unwrap().vec, vec![0.0, 0.5]);
        assert_eq!(ctx.latent_as_of("0190ab", 2000).unwrap().vec, vec![1.0, 2.0, 3.0]);
        assert!(ctx.latent_as_of("0190ab", 999).is_none());
    }
}
