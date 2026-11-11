//! Snapshot lifecycle on the runtime side (spec 36 §6.4, spec 31 §4.3): which encoder serves a
//! timescale, where it comes from, and the `ctl.model.reload` hot swap.
//!
//! On start [`SnapshotLoader::resolve`] asks Helix for the **active** `Snapshot` of the
//! timescale (any family: `statistical` first, then the neural families, [`active_any_family`]).
//! A row with a `garage_key` is a trained snapshot: its `manifest.json` / `weights.bin` /
//! `region_map.json` are read from Garage through `fa_blob` and the encoder is rebuilt by its
//! family ([`encoder_from_manifest`]: `StatisticalEncoder` or `NeuralEncoder`); the region sets
//! start seeded from the region map so matched lineages continue (D-007). No active row, a row
//! without an artifact (the in-process baseline `stat-v1-<ts>` registered by this service as
//! `active` gen 0), Helix down or no Garage credentials → the in-process baseline, as before.
//!
//! `ctl.model.reload` (request/reply, `CtlRequest{verb: "reload", args: timescale, snapshot_id?}`)
//! resolves the named snapshot (default: the active one) and swaps it into the running pipeline
//! ([`crate::pipeline::TimescalePipeline::swap_encoder`], with kernel transfer); control-api's
//! promote/rollback call it after changing the Helix status. The reply says what is now serving.
//! `ctl.model.shadow` (`timescale`, `snapshot_id` | `none`) resolves a candidate the same way and
//! serves it alongside the active one ([`crate::shadow`]); `ctl.model.adapter` (`timescale`)
//! applies the latest online-adapter artifact of the active snapshot ([`SnapshotLoader::adapter`]).

use fa_blob::{BlobStore, S3BlobStore};
use fa_core::{Secrets, Timescale, config::PlatformConfig};
use fa_encoders::{
    EncoderFamily, MANIFEST_FILE, NormStats, REGION_MAP_FILE, STATISTICAL_FAMILY, SnapshotManifest, StatisticalEncoder,
    WEIGHTS_FILE,
    neural::{InputNorm, NeuralEncoder, NeuralFamily, is_neural_family},
};
use fa_hub::HubClient;
use fa_patterns::RegionMap;
use fa_proto::v1::{CtlReply, CtlRequest};
use futures::StreamExt;
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot};

use crate::{hub_sink::SnapshotDesc, region_method::RegionMethod};

/// What serves a timescale after a resolve.
pub struct Loaded {
    pub encoder: Box<dyn EncoderFamily>,
    pub region_map: Option<RegionMap>,
    pub desc: SnapshotDesc,
    /// `garage` (trained artifact) or `baseline` (in-process statistical encoder).
    pub source: &'static str,
    /// Why the baseline was chosen, when it was not the active trained snapshot.
    pub note: Option<String>,
}

pub struct SnapshotLoader {
    hub: Option<HubClient>,
    blob: Option<S3BlobStore>,
}

impl SnapshotLoader {
    /// `hub_url = None` disables Helix lookups; Garage credentials come from `config/secrets.toml`
    /// (`FA_GARAGE_*` overrides), missing credentials only disable trained snapshots.
    pub fn new(cfg: &PlatformConfig, hub_url: Option<String>) -> Self {
        let hub = hub_url.map(|u| HubClient::with_timeout(u, std::time::Duration::from_secs(5)));
        let blob = match Secrets::load()
            .map_err(|e| e.to_string())
            .and_then(|s| S3BlobStore::from_config(cfg, &s).map_err(|e| e.to_string()))
        {
            Ok(b) => Some(b),
            Err(e) => {
                tracing::warn!(error = %e, "no Garage client; only the in-process baseline encoder can serve");
                None
            }
        };
        Self { hub, blob }
    }

    /// The in-process baseline of a timescale, registered as `active` gen 0.
    pub fn baseline(ts: Timescale, note: Option<String>) -> Loaded {
        let enc = StatisticalEncoder::new(ts);
        Loaded {
            desc: SnapshotDesc {
                snapshot_id: enc.snapshot_id().to_owned(),
                timescale: ts,
                n_experts: enc.sizing().max_experts,
                dim: enc.sizing().latent_dim,
                generation: 0,
                garage_key: String::new(),
                status: "active".into(),
                region_method: RegionMethod::resolve(enc.snapshot_id(), None, enc.descriptor().region_discovery, None),
            },
            encoder: Box::new(enc),
            region_map: None,
            source: "baseline",
            note,
        }
    }

    /// Resolves `snapshot_id` (or the active snapshot) of `ts`. Only an explicitly named snapshot
    /// that cannot be served is an error; every implicit failure falls back to the baseline.
    pub async fn resolve(&self, ts: Timescale, snapshot_id: Option<&str>) -> anyhow::Result<Loaded> {
        let Some(hub) = &self.hub else {
            return Ok(Self::baseline(ts, Some("hub disabled".into())));
        };
        let row = match snapshot_id {
            Some(id) => match hub.snapshot_by_id(id).await? {
                Some(r) => r,
                None => anyhow::bail!("snapshot {id} not found in Helix"),
            },
            None => match active_any_family(hub, ts).await {
                Ok(Some(r)) => r,
                Ok(None) => return Ok(Self::baseline(ts, Some(format!("no active snapshot for {ts} in Helix")))),
                Err(e) => return Ok(Self::baseline(ts, Some(format!("helix unavailable: {e}")))),
            },
        };
        if row.timescale != ts.as_str() {
            anyhow::bail!("snapshot {} is for {} not {ts}", row.snapshot_id, row.timescale);
        }
        if row.garage_key.is_empty() {
            // The in-process baseline (or any row without an artifact) under its own id/status.
            let mut l = Self::baseline(ts, Some(format!("{} has no Garage artifact", row.snapshot_id)));
            if row.snapshot_id != l.desc.snapshot_id {
                anyhow::bail!(
                    "snapshot {} has no artifact and is not the baseline {}",
                    row.snapshot_id,
                    l.desc.snapshot_id
                );
            }
            l.desc.generation = row.generation;
            l.desc.status = row.status.clone();
            return Ok(l);
        }
        let Some(blob) = &self.blob else {
            let msg = format!("snapshot {} needs Garage credentials to load", row.snapshot_id);
            if snapshot_id.is_some() {
                anyhow::bail!(msg);
            }
            return Ok(Self::baseline(ts, Some(msg)));
        };
        let id = row.snapshot_id.as_str();
        let manifest = SnapshotManifest::from_json(&blob.get(&fa_blob::snapshot_file(id, MANIFEST_FILE)).await?)?;
        let weights = blob.get(&fa_blob::snapshot_file(id, WEIGHTS_FILE)).await?;
        let region_map = match blob.get(&fa_blob::snapshot_file(id, REGION_MAP_FILE)).await {
            Ok(b) => Some(RegionMap::from_json(&b)?),
            Err(fa_blob::BlobError::NotFound(..)) => None,
            Err(e) => return Err(e.into()),
        };
        if manifest.timescale != ts {
            anyhow::bail!("manifest of {id} is for {} not {ts}", manifest.timescale);
        }
        let enc = encoder_from_manifest(&manifest, &weights)?;
        let region_method =
            RegionMethod::resolve(id, Some(&manifest.params), enc.descriptor().region_discovery, region_map.as_ref());
        Ok(Loaded {
            desc: SnapshotDesc {
                snapshot_id: enc.snapshot_id().to_owned(),
                timescale: ts,
                n_experts: enc.sizing().max_experts,
                dim: enc.sizing().latent_dim,
                generation: manifest.generation,
                garage_key: row.garage_key.clone(),
                status: row.status.clone(),
                region_method,
            },
            encoder: enc,
            region_map,
            source: "garage",
            note: None,
        })
    }

    /// The online adapter of `snapshot_id` (spec 36 §6.5): `artifact_key` when given (from a
    /// `jobs.result.train.adapter`), else the newest `fa-models/adapters/<snapshot>/<ms>.json`.
    /// The artifact's re-fitted normalisation statistics replace the snapshot's (a trained
    /// snapshot keeps its manifest params, e.g. `context_steps`). A `frozen` artifact (drift
    /// beyond `max_drift`) is refused, as is one for another timescale.
    pub async fn adapter(
        &self,
        ts: Timescale,
        snapshot_id: &str,
        artifact_key: Option<&str>,
    ) -> anyhow::Result<LoadedAdapter> {
        let Some(blob) = &self.blob else {
            anyhow::bail!("adapters need Garage credentials");
        };
        let key = match artifact_key.filter(|k| !k.is_empty()) {
            Some(k) => k.to_owned(),
            None => {
                let prefix = format!("adapters/{snapshot_id}/");
                let listed = blob.list(fa_blob::buckets::MODELS, &prefix).await?;
                listed
                    .into_iter()
                    .filter_map(|o| adapter_ms(&o.key).map(|ms| (ms, o.key)))
                    .max()
                    .map(|x| x.1)
                    .ok_or_else(|| anyhow::anyhow!("no adapter artifact under fa-models/{prefix}"))?
            }
        };
        let bytes = blob.get(&fa_blob::ObjectKey { bucket: fa_blob::buckets::MODELS, key: key.clone() }).await?;
        let art = AdapterArtifact::from_json(&bytes)?;
        if art.snapshot_id != snapshot_id {
            anyhow::bail!("adapter {key} is for {} not {snapshot_id}", art.snapshot_id);
        }
        if art.timescale != ts {
            anyhow::bail!("adapter {key} is for {} not {ts}", art.timescale);
        }
        if art.frozen {
            anyhow::bail!("adapter {key} is frozen (drift {:?} > max {:?})", art.drift, art.max_drift);
        }
        let manifest = match blob.get(&fa_blob::snapshot_file(snapshot_id, MANIFEST_FILE)).await {
            Ok(m) => Some(SnapshotManifest::from_json(&m)?),
            Err(fa_blob::BlobError::NotFound(..)) => None,
            Err(e) => return Err(e.into()),
        };
        let encoder: Box<dyn EncoderFamily> = match (&manifest, &art.stats, &art.input_norm) {
            // neural snapshot: the adapter re-fits the input normalisation (spec 36 §6.5 input adapter)
            (Some(m), _, Some(norm)) if is_neural_family(&m.encoder_family) => {
                let weights = blob.get(&fa_blob::snapshot_file(snapshot_id, WEIGHTS_FILE)).await?;
                let (dev, _) = fa_encoders::neural::device_from_env();
                let base = NeuralEncoder::from_manifest_on(m, &weights, &dev)?;
                let mut model = base.model().clone();
                model.norm = norm.clone();
                let mut enc = NeuralEncoder::new(model, ts, snapshot_id.to_owned());
                // weight-level LoRA (spec 36 §6.5), unless its own guard froze it
                if let Some(l) = art.lora.as_ref().filter(|l| !l.frozen)
                    && let Some(lk) = l.key.as_deref().filter(|k| !k.is_empty())
                {
                    let bytes =
                        blob.get(&fa_blob::ObjectKey { bucket: fa_blob::buckets::MODELS, key: lk.to_owned() }).await?;
                    enc = enc.with_lora(&bytes)?;
                    tracing::info!(snapshot = snapshot_id, lora = lk, "LoRA adapter folded into the serving encoder");
                }
                Box::new(enc)
            }
            (Some(m), Some(stats), _) => Box::new(StatisticalEncoder::from_manifest(m, &stats.to_bytes())?),
            (None, Some(stats), _) => {
                Box::new(StatisticalEncoder::with_stats(ts, snapshot_id.to_owned(), stats.clone()))
            }
            _ => anyhow::bail!("adapter {key} carries neither statistics nor an input normalisation for this snapshot"),
        };
        Ok(LoadedAdapter { encoder, key, ts_ms: art.ts_ms, drift: art.drift })
    }
}

impl SnapshotLoader {
    /// The snapshot Helix records as the shadow of `ts` (`status = shadow`; the newest
    /// generation when several), restored after a restart (spec 36 §6.4).
    pub async fn shadow_snapshot(&self, ts: Timescale) -> anyhow::Result<Option<String>> {
        let Some(hub) = &self.hub else { return Ok(None) };
        let rows = hub.list_snapshots().await?;
        Ok(pick_shadow(&rows, ts))
    }
}

/// The newest `shadow` row of `ts`.
pub fn pick_shadow(rows: &[fa_hub::Snapshot], ts: Timescale) -> Option<String> {
    rows.iter()
        .filter(|r| r.status == "shadow" && r.timescale == ts.as_str())
        .max_by(|a, b| a.generation.cmp(&b.generation).then(a.created_at.cmp(&b.created_at)))
        .map(|r| r.snapshot_id.clone())
}

/// The active snapshot of `ts` whatever its encoder family (control-api keeps at most one
/// `active` row per timescale across families: promotion retires the replaced one).
pub async fn active_any_family(hub: &HubClient, ts: Timescale) -> anyhow::Result<Option<fa_hub::Snapshot>> {
    let mut families = vec![STATISTICAL_FAMILY];
    families.extend(NeuralFamily::ALL.iter().map(|f| f.as_str()));
    for f in families {
        if let Some(r) = hub.active_snapshot(ts.as_str(), f).await? {
            return Ok(Some(r));
        }
    }
    Ok(None)
}

/// Builds the serving encoder of a snapshot artifact by its `encoder_family`: the statistical
/// family from `FASTAT01` statistics, the neural families (all four) from
/// safetensors on the configured device (`FA_ML_DEVICE`, default cpu).
pub fn encoder_from_manifest(manifest: &SnapshotManifest, weights: &[u8]) -> anyhow::Result<Box<dyn EncoderFamily>> {
    if is_neural_family(&manifest.encoder_family) {
        let (dev, note) = fa_encoders::neural::device_from_env();
        if let Some(n) = note {
            tracing::warn!(snapshot = %manifest.snapshot_id, "{n}");
        }
        let enc = NeuralEncoder::from_manifest_on(manifest, weights, &dev)?;
        tracing::info!(snapshot = %manifest.snapshot_id, family = %manifest.encoder_family, experts = enc.sizing().max_experts, dim = enc.sizing().latent_dim, context = enc.sizing().context_steps, "neural encoder loaded");
        return Ok(Box::new(enc));
    }
    Ok(Box::new(StatisticalEncoder::from_manifest(manifest, weights)?))
}

/// An adapter ready to replace the serving encoder.
pub struct LoadedAdapter {
    pub encoder: Box<dyn EncoderFamily>,
    pub key: String,
    pub ts_ms: i64,
    pub drift: Option<f64>,
}

/// The part of the trainer's `fa-models/adapters/<snapshot>/<ms>.json` the runtime needs
/// (written by `bins/trainer` `run_adapter`, `protocol: train/v0`).
#[derive(Debug, Clone, Deserialize)]
pub struct AdapterArtifact {
    pub snapshot_id: String,
    pub timescale: Timescale,
    pub ts_ms: i64,
    /// Statistical family: re-fitted normalisation statistics.
    #[serde(default)]
    pub stats: Option<NormStats>,
    /// Neural families: re-fitted input normalisation (`fa_encoders::neural::InputNorm`).
    #[serde(default)]
    pub input_norm: Option<InputNorm>,
    /// Neural families: the LoRA layer (`fa-lora-v1` safetensors next to the JSON).
    #[serde(default)]
    pub lora: Option<LoraRef>,
    #[serde(default)]
    pub drift: Option<f64>,
    #[serde(default)]
    pub max_drift: Option<f64>,
    #[serde(default)]
    pub frozen: bool,
}

/// The `lora` part of a neural adapter artifact.
#[derive(Debug, Clone, Deserialize)]
pub struct LoraRef {
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub frozen: bool,
}

impl AdapterArtifact {
    pub fn from_json(b: &[u8]) -> anyhow::Result<Self> {
        Ok(serde_json::from_slice(b)?)
    }
}

/// `<ms>` of an `adapters/<snapshot>/<ms>.json` key.
pub fn adapter_ms(key: &str) -> Option<i64> {
    key.rsplit('/').next()?.strip_suffix(".json")?.parse().ok()
}

/// A `ctl.model.<verb>` request handed to the pipeline loop.
pub struct CtlCommand {
    pub timescale: Timescale,
    pub verb: CtlVerb,
    pub reply: oneshot::Sender<CtlReply>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CtlVerb {
    /// Swap the active encoder (`snapshot_id`, default: the active Helix snapshot).
    Reload { snapshot_id: Option<String> },
    /// Serve a candidate in shadow (`None` = stop shadowing).
    Shadow { snapshot_id: Option<String> },
    /// Apply the latest online adapter of the active snapshot.
    Adapter,
}

/// Parses verb + args; `Err` is the reply message for a malformed request.
pub fn parse_ctl(verb: &str, req: &CtlRequest) -> Result<(Timescale, CtlVerb), String> {
    let ts = match req.args.get("timescale").map(|s| s.parse::<Timescale>()) {
        Some(Ok(t)) => t,
        Some(Err(e)) => return Err(format!("bad timescale: {e}")),
        None => return Err("missing arg `timescale`".into()),
    };
    let snapshot_id = req.args.get("snapshot_id").map(|s| s.trim()).filter(|s| !s.is_empty()).map(str::to_owned);
    let v = match verb {
        "reload" => CtlVerb::Reload { snapshot_id },
        "shadow" => CtlVerb::Shadow { snapshot_id: snapshot_id.filter(|s| !s.eq_ignore_ascii_case("none")) },
        "adapter" => CtlVerb::Adapter,
        other => return Err(format!("unknown verb `{other}` (reload | shadow | adapter)")),
    };
    Ok((ts, v))
}

/// Subscribes to `ctl.model.>` (`reload`, `shadow`, `adapter`) and forwards decoded requests;
/// answers the NATS reply subject with whatever the loop sends back (or an error for malformed
/// requests).
pub async fn run_ctl_listener(bus: fa_bus::Bus, tx: mpsc::Sender<CtlCommand>) {
    let subject = "ctl.model.>".to_owned();
    let Ok(mut sub) = bus.subscribe_raw(subject.clone()).await else {
        tracing::error!(subject, "cannot subscribe to ctl.model.>");
        return;
    };
    tracing::info!(subject, "listening for snapshot reload / shadow / adapter requests");
    while let Some(msg) = sub.next().await {
        let reply_to = msg.reply.clone();
        let verb = msg.subject.as_str().strip_prefix("ctl.model.").unwrap_or_default().to_owned();
        let answer = match <CtlRequest as fa_proto::Message>::decode(msg.payload.as_ref()) {
            Ok(req) => match parse_ctl(&verb, &req) {
                Ok((timescale, verb)) => {
                    let (rtx, rrx) = oneshot::channel();
                    if tx.send(CtlCommand { timescale, verb, reply: rtx }).await.is_err() {
                        CtlReply { ok: false, message: "pipeline loop gone".into() }
                    } else {
                        rrx.await.unwrap_or(CtlReply { ok: false, message: "request dropped".into() })
                    }
                }
                Err(m) => CtlReply { ok: false, message: m },
            },
            Err(e) => CtlReply { ok: false, message: format!("undecodable CtlRequest: {e}") },
        };
        if let Some(reply_to) = reply_to {
            let bytes = fa_proto::Message::encode_to_vec(&answer);
            if let Err(e) = bus.client().publish(reply_to, bytes.into()).await {
                tracing::warn!(error = %e, "ctl reply failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctl_requests_parse() {
        let req = |args: &[(&str, &str)]| CtlRequest {
            verb: String::new(),
            args: args.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect(),
        };
        assert_eq!(
            parse_ctl("shadow", &req(&[("timescale", "1m"), ("snapshot_id", "stat-v1-1m-g0-1")])).unwrap(),
            (Timescale::Minute, CtlVerb::Shadow { snapshot_id: Some("stat-v1-1m-g0-1".into()) })
        );
        assert_eq!(
            parse_ctl("shadow", &req(&[("timescale", "1m"), ("snapshot_id", "none")])).unwrap().1,
            CtlVerb::Shadow { snapshot_id: None }
        );
        assert_eq!(parse_ctl("shadow", &req(&[("timescale", "1h")])).unwrap().1, CtlVerb::Shadow { snapshot_id: None });
        assert_eq!(parse_ctl("reload", &req(&[("timescale", "1d")])).unwrap().1, CtlVerb::Reload { snapshot_id: None });
        assert_eq!(parse_ctl("adapter", &req(&[("timescale", "1m")])).unwrap().1, CtlVerb::Adapter);
        assert!(parse_ctl("adapter", &req(&[])).unwrap_err().contains("timescale"));
        assert!(parse_ctl("reload", &req(&[("timescale", "2m")])).unwrap_err().contains("bad timescale"));
        assert!(parse_ctl("explode", &req(&[("timescale", "1m")])).unwrap_err().contains("unknown verb"));
    }

    #[test]
    fn trainer_adapter_artifact_parses() {
        // shape written by bins/trainer run_adapter (protocol train/v0)
        let stats = NormStats { n_windows: 3, mean: vec![vec![0.0; 8]; 3], std: vec![vec![1.0; 8]; 3], gain: vec![] };
        let json = serde_json::json!({
            "protocol": "train/v0", "snapshot_id": "stat-v1-1m", "timescale": "1m", "ts_ms": 1_700_000_000_000i64,
            "created_at": "2026-11-10T00:00:00Z",
            "window": {"start_ms": 0, "end_ms": 1, "instrument": "x", "n_bars": 60},
            "n_windows": 3, "stats": stats, "reference": "baseline (no fitted statistics)",
            "drift": null, "max_drift": 1.0, "frozen": false, "last_latent_rms": [0.1, 0.2, 0.3]
        });
        let a = AdapterArtifact::from_json(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(
            (a.snapshot_id.as_str(), a.timescale, a.ts_ms, a.frozen),
            ("stat-v1-1m", Timescale::Minute, 1_700_000_000_000, false)
        );
        assert_eq!(a.stats, Some(stats));
        assert!(a.input_norm.is_none());
        // neural adapter artifacts carry an input normalisation instead
        let neural = serde_json::json!({
            "snapshot_id": "fsq-v1-1m-g0-1", "timescale": "1m", "ts_ms": 1, "input_norm": InputNorm::default(), "frozen": false
        });
        let n = AdapterArtifact::from_json(&serde_json::to_vec(&neural).unwrap()).unwrap();
        assert!(n.stats.is_none() && n.input_norm.is_some() && n.lora.is_none());
        // with the LoRA layer (trainer run_adapter, spec 36 §6.5)
        let mut with_lora = neural.clone();
        with_lora["lora"] = serde_json::json!({
            "key": "adapters/fsq-v1-1m-g0-1/1.lora.safetensors", "rank": 8, "alpha": 8.0, "frozen": false,
            "report": {"delta_norm": 0.01}
        });
        let l = AdapterArtifact::from_json(&serde_json::to_vec(&with_lora).unwrap()).unwrap().lora.unwrap();
        assert_eq!((l.key.as_deref(), l.frozen), (Some("adapters/fsq-v1-1m-g0-1/1.lora.safetensors"), false));
        assert_eq!(adapter_ms("adapters/x/1.lora.safetensors"), None, "the LoRA weights are not an artifact");
        assert_eq!(adapter_ms("adapters/stat-v1-1m/1700000000000.json"), Some(1_700_000_000_000));
        assert_eq!(adapter_ms("adapters/stat-v1-1m/notes.txt"), None);
    }

    #[test]
    fn persisted_shadow_is_the_newest_shadow_row_of_the_timescale() {
        let row = |id: &str, ts: &str, status: &str, generation: u32| fa_hub::Snapshot {
            id: 1,
            snapshot_id: id.into(),
            timescale: ts.into(),
            encoder_family: "jepa_vae".into(),
            generation,
            garage_key: String::new(),
            status: status.into(),
            created_at: None,
            carried_mass: None,
            manifest: None,
            previous_active: None,
        };
        let rows = vec![
            row("a", "1m", "active", 3),
            row("b", "1m", "shadow", 1),
            row("c", "1m", "shadow", 2),
            row("d", "1s", "shadow", 9),
            row("e", "1m", "candidate", 7),
        ];
        assert_eq!(pick_shadow(&rows, Timescale::Minute).as_deref(), Some("c"));
        assert_eq!(pick_shadow(&rows, Timescale::Second).as_deref(), Some("d"));
        assert_eq!(pick_shadow(&rows, Timescale::Hour), None);
    }
}
