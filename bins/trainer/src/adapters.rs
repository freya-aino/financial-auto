//! Continuous online adapters (spec 36 §6.5, acceptance 4; decision recorded in the trainer
//! README): adapters are trained as short, frequent jobs — the planner enqueues one per active
//! snapshot (statistical **and** neural) every `FA_TRAIN_ADAPTER_EVERY` (default 15 min, the
//! spec's `adapter_checkpoint_interval`) over the last `adapter_window_hours` (the replay
//! buffer) — and every LoRA job **resumes the factors of the snapshot's newest served
//! (non-frozen) adapter** ([`previous_lora`] → `train_lora_from`), so successive jobs form one
//! continuing optimisation instead of independent hourly fits. Each job is still versioned
//! (`AdapterState`), guarded against the frozen snapshot (`max_delta_norm`,
//! `max_latent_shift`; the guard sees the accumulated ΔW) and applied by model-runtime on the
//! job result. A step-level trainer inside model-runtime was rejected: it would put SGD on the
//! P0 live-inference path, which spec 36 §3 forbids (adapters are P1, preemptible).

use candle_core::Device;
use fa_expert_space::lora::LoraSet;
use serde_json::Value;

use crate::run::Services;

/// The newest non-frozen LoRA of `snapshot_id` in Garage (`adapters/<snapshot>/<ms>.json` with a
/// served `lora.key`), with that key.
pub async fn previous_lora(svc: &Services<'_>, snapshot_id: &str, dev: &Device) -> Option<(LoraSet, String)> {
    let blob = svc.stores.blob.as_ref()?;
    let prefix = format!("adapters/{snapshot_id}/");
    let mut arts: Vec<(i64, String)> = fa_blob::BlobStore::list(blob, fa_blob::buckets::MODELS, &prefix)
        .await
        .ok()?
        .into_iter()
        .filter_map(|o| {
            let ms: i64 = o.key.strip_prefix(&prefix)?.strip_suffix(".json")?.parse().ok()?;
            Some((ms, o.key))
        })
        .collect();
    arts.sort();
    for (_, key) in arts.into_iter().rev() {
        let Ok(b) = fa_blob::BlobStore::get(blob, &fa_blob::ObjectKey { bucket: fa_blob::buckets::MODELS, key }).await
        else {
            continue;
        };
        let Ok(art) = serde_json::from_slice::<Value>(&b) else { continue };
        if let Some(lk) = servable_lora_key(&art) {
            let bytes = fa_blob::BlobStore::get(
                blob,
                &fa_blob::ObjectKey { bucket: fa_blob::buckets::MODELS, key: lk.clone() },
            )
            .await
            .ok()?;
            return LoraSet::from_bytes(&bytes, dev).ok().map(|(s, _)| (s, lk));
        }
    }
    None
}

/// The LoRA key of an adapter artifact when that LoRA was served (neither the artifact nor its
/// LoRA frozen).
pub fn servable_lora_key(art: &Value) -> Option<String> {
    if art["frozen"] == Value::Bool(true) || art["lora"]["frozen"] == Value::Bool(true) {
        return None;
    }
    art["lora"]["key"].as_str().filter(|k| !k.is_empty()).map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn only_served_loras_are_resumed() {
        assert_eq!(
            servable_lora_key(&json!({ "frozen": false, "lora": { "key": "k", "frozen": false } })).as_deref(),
            Some("k")
        );
        assert_eq!(servable_lora_key(&json!({ "frozen": false, "lora": { "key": "k", "frozen": true } })), None);
        assert_eq!(servable_lora_key(&json!({ "frozen": true, "lora": { "key": "k" } })), None);
        assert_eq!(servable_lora_key(&json!({ "frozen": false, "lora": null })), None);
    }
}
