//! Snapshot lifecycle on the runtime side (spec 36 §6.4, spec 31 §4.3): which encoder serves a
//! timescale, where it comes from, and the `ctl.model.reload` hot swap.
//!
//! On start [`SnapshotLoader::resolve`] asks Helix for the **active** `Snapshot` of the
//! timescale (`encoder_family = statistical`). A row with a `garage_key` is a trained snapshot:
//! its `manifest.json` / `weights.bin` / `region_map.json` are read from Garage through
//! `fa_blob` and the encoder is rebuilt with `StatisticalEncoder::from_manifest`; the region sets
//! start seeded from the region map so matched lineages continue (D-007). No active row, a row
//! without an artifact (the in-process baseline `stat-v1-<ts>` registered by this service as
//! `active` gen 0), Helix down or no Garage credentials → the in-process baseline, as before.
//!
//! `ctl.model.reload` (request/reply, `CtlRequest{verb: "reload", args: timescale, snapshot_id?}`)
//! resolves the named snapshot (default: the active one) and swaps it into the running pipeline
//! ([`crate::pipeline::TimescalePipeline::swap_encoder`]); control-api's promote/rollback call
//! it after changing the Helix status. The reply says what is now serving.

use fa_blob::{BlobStore, S3BlobStore};
use fa_core::{Secrets, Timescale, config::PlatformConfig};
use fa_encoders::{
    EncoderFamily, MANIFEST_FILE, REGION_MAP_FILE, STATISTICAL_FAMILY, SnapshotManifest, StatisticalEncoder,
    WEIGHTS_FILE,
};
use fa_hub::HubClient;
use fa_patterns::RegionMap;
use fa_proto::v1::{CtlReply, CtlRequest};
use futures::StreamExt;
use tokio::sync::{mpsc, oneshot};

use crate::hub_sink::SnapshotDesc;

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
            None => match hub.active_snapshot(ts.as_str(), STATISTICAL_FAMILY).await {
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
        let enc = StatisticalEncoder::from_manifest(&manifest, &weights)?;
        Ok(Loaded {
            desc: SnapshotDesc {
                snapshot_id: enc.snapshot_id().to_owned(),
                timescale: ts,
                n_experts: enc.sizing().max_experts,
                dim: enc.sizing().latent_dim,
                generation: manifest.generation,
                garage_key: row.garage_key.clone(),
                status: row.status.clone(),
            },
            encoder: Box::new(enc),
            region_map,
            source: "garage",
            note: None,
        })
    }
}

/// A `ctl.model.reload` request handed to the pipeline loop.
pub struct ReloadRequest {
    pub timescale: Timescale,
    pub snapshot_id: Option<String>,
    pub reply: oneshot::Sender<CtlReply>,
}

/// Subscribes to `ctl.model.reload` and forwards decoded requests; answers the NATS reply
/// subject with whatever the loop sends back (or an error for malformed requests).
pub async fn run_reload_listener(bus: fa_bus::Bus, tx: mpsc::Sender<ReloadRequest>) {
    let subject = fa_bus::subjects::ctl("model", "reload");
    let Ok(mut sub) = bus.subscribe_raw(subject.clone()).await else {
        tracing::error!(subject, "cannot subscribe to ctl.model.reload");
        return;
    };
    tracing::info!(subject, "listening for snapshot reloads");
    while let Some(msg) = sub.next().await {
        let reply_to = msg.reply.clone();
        let answer = match <CtlRequest as fa_proto::Message>::decode(msg.payload.as_ref()) {
            Ok(req) => {
                let ts = req.args.get("timescale").map(|s| s.parse::<Timescale>());
                match ts {
                    Some(Ok(timescale)) => {
                        let (rtx, rrx) = oneshot::channel();
                        let snapshot_id = req.args.get("snapshot_id").filter(|s| !s.is_empty()).cloned();
                        if tx.send(ReloadRequest { timescale, snapshot_id, reply: rtx }).await.is_err() {
                            CtlReply { ok: false, message: "pipeline loop gone".into() }
                        } else {
                            rrx.await.unwrap_or(CtlReply { ok: false, message: "reload dropped".into() })
                        }
                    }
                    Some(Err(e)) => CtlReply { ok: false, message: format!("bad timescale: {e}") },
                    None => CtlReply { ok: false, message: "missing arg `timescale`".into() },
                }
            }
            Err(e) => CtlReply { ok: false, message: format!("undecodable CtlRequest: {e}") },
        };
        if let Some(reply_to) = reply_to {
            let bytes = fa_proto::Message::encode_to_vec(&answer);
            if let Err(e) = bus.client().publish(reply_to, bytes.into()).await {
                tracing::warn!(error = %e, "reload reply failed");
            }
        }
    }
}
