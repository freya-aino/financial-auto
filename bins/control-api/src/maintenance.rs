//! Scheduled storage maintenance (spec 20 §6.5), run by control-api because it is off the trading
//! path and already holds the VictoriaMetrics URL, the Garage credentials, NATS and Postgres:
//!
//! - **VictoriaMetrics retention** every `FA_VM_RETENTION_SECS` (default 86 400 = daily; `0`
//!   disables): `fa_tsdb::retention::enforce` with the per-class policy (overrides in
//!   `FA_VM_RETENTION_POLICY`, e.g. `book=7d`), tightened one priority step while
//!   `vm_data_size_bytes` is above `FA_VM_DISK_BUDGET_GIB` (default 50, §8).
//! - **Genome-weight GC** every `FA_GENOME_GC_SECS` (default 604 800 = weekly; `0` disables):
//!   `fa_blob::gc` over `fa-models/decoders/` with the default policy (newest 5 generations, 8
//!   card elites per family, nothing younger than 7 days), protecting every bundle a snapshot
//!   manifest or model-runtime's serving set (`ctl.decoders.status`) references. A run whose
//!   serving set cannot be read deletes nothing (same rule as `examples/genome_gc.rs --apply`).
//! - **Media downsampling** on the GC cadence: `fa_blob::media::downsample` drops a media item's
//!   full `video/` once its sampled `frames/` exist and it is older than
//!   `garage.media_downsample_after_d` (default 30); audio and transcripts are kept forever.
//!
//! Each run is logged and appended to `audit.log` (`ops.vm_retention`, `ops.genome_gc`, `ops.media_downsample`) with its
//! report; `FA_MAINT_DRY_RUN=1` plans without deleting.
use std::{collections::BTreeSet, time::Duration};

use bytes::Bytes;
use fa_blob::{BlobStore, S3BlobStore, buckets, gc};
use fa_proto::{
    Message,
    v1::{CtlReply, CtlRequest},
};
use fa_tsdb::retention::{EnforceOptions, RetentionPolicy, enforce, tighten};
use serde_json::{Value, json};

use crate::state::AppState;

const ACTOR: &str = "system:control-api";
const FIRST_RUN_DELAY: Duration = Duration::from_secs(300);

fn env_secs(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn dry_run() -> bool {
    std::env::var("FA_MAINT_DRY_RUN").is_ok_and(|v| v == "1")
}

pub fn spawn(state: &AppState) {
    let vm_every = env_secs("FA_VM_RETENTION_SECS", 86_400);
    let gc_every = env_secs("FA_GENOME_GC_SECS", 7 * 86_400);
    if vm_every > 0 {
        let s = state.clone();
        tokio::spawn(async move {
            tokio::time::sleep(FIRST_RUN_DELAY).await;
            loop {
                let report = vm_retention(&s).await;
                record(&s, "ops.vm_retention", report).await;
                tokio::time::sleep(Duration::from_secs(vm_every)).await;
            }
        });
    } else {
        tracing::info!("VictoriaMetrics retention disabled (FA_VM_RETENTION_SECS=0)");
    }
    if gc_every > 0 {
        let s = state.clone();
        tokio::spawn(async move {
            tokio::time::sleep(FIRST_RUN_DELAY).await;
            loop {
                let report = genome_gc(&s).await;
                record(&s, "ops.genome_gc", report).await;
                let report = media_downsample(&s).await;
                record(&s, "ops.media_downsample", report).await;
                tokio::time::sleep(Duration::from_secs(gc_every)).await;
            }
        });
    } else {
        tracing::info!("genome GC disabled (FA_GENOME_GC_SECS=0)");
    }
}

async fn record(s: &AppState, action: &str, report: Result<Value, String>) {
    let payload = match &report {
        Ok(v) => {
            tracing::info!(action, report = %v, "maintenance run");
            json!({ "ok": true, "report": v })
        }
        Err(e) => {
            tracing::warn!(action, error = %e, "maintenance run failed (retried next interval)");
            json!({ "ok": false, "error": e })
        }
    };
    if let Some(pool) = s.pg.get()
        && let Err(e) = crate::audit::append(pool, ACTOR, action, None, payload).await
    {
        tracing::warn!(action, error = %e, "audit append failed");
    }
}

async fn vm_retention(s: &AppState) -> Result<Value, String> {
    let vm = fa_tsdb::Tsdb::new(s.cfg.victoriametrics.url.clone());
    let mut policy = RetentionPolicy::default();
    if let Ok(o) = std::env::var("FA_VM_RETENTION_POLICY") {
        policy = policy.with_overrides(&o).map_err(|e| format!("FA_VM_RETENTION_POLICY: {e}"))?;
    }
    let data_bytes = vm.data_size_bytes().await.map_err(|e| e.to_string())?;
    let budget_gib = std::env::var("FA_VM_DISK_BUDGET_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(50.0);
    let (policy, tightened) = tighten(&policy, data_bytes, (budget_gib * (1u64 << 30) as f64) as u64);
    let opts = EnforceOptions { dry_run: dry_run(), ..Default::default() };
    let classes =
        enforce(&vm, &policy, chrono::Utc::now().timestamp_millis(), &opts).await.map_err(|e| e.to_string())?;
    Ok(
        json!({ "dry_run": opts.dry_run, "data_bytes": data_bytes, "budget_gib": budget_gib, "tightened": tightened, "classes": classes }),
    )
}

/// Weight bundles model-runtime serves or shadows (`ctl.decoders.status`); `Err` when it does
/// not answer, so the caller deletes nothing.
async fn serving_keys(s: &AppState) -> Result<BTreeSet<String>, String> {
    let client = s.nats.as_ref().filter(|_| s.nats_connected()).ok_or("NATS not connected")?;
    let req = CtlRequest { verb: "status".into(), args: [("actor".to_owned(), ACTOR.to_owned())].into() };
    let msg = tokio::time::timeout(
        Duration::from_secs(10),
        client.request(fa_bus::subjects::ctl("decoders", "status"), Bytes::from(req.encode_to_vec())),
    )
    .await
    .map_err(|_| "ctl.decoders.status: no reply within 10 s".to_owned())?
    .map_err(|e| format!("ctl.decoders.status: {e}"))?;
    let reply = CtlReply::decode(msg.payload.as_ref()).map_err(|e| format!("ctl.decoders.status reply: {e}"))?;
    if !reply.ok {
        return Err(format!("ctl.decoders.status: {}", reply.message));
    }
    let v: Value = serde_json::from_str(&reply.message).map_err(|e| format!("ctl.decoders.status: {e}"))?;
    if v.get("slots").is_none() {
        return Err("ctl.decoders.status: no `slots` in reply".into());
    }
    let mut out = BTreeSet::new();
    gc::collect_bundle_refs(&v, &mut out);
    Ok(out)
}

async fn genome_gc(s: &AppState) -> Result<Value, String> {
    let secrets = fa_core::Secrets::load().map_err(|e| format!("garage secrets: {e}"))?;
    let store = S3BlobStore::from_config(&s.cfg, &secrets).map_err(|e| format!("garage: {e}"))?;
    let serving = serving_keys(s).await.map_err(|e| format!("serving set unknown, nothing deleted: {e}"))?;
    let listing = store.list(buckets::MODELS, gc::DECODER_PREFIX).await.map_err(|e| e.to_string())?;
    let mut prot = gc::Protection::default();
    prot.extend(gc::snapshot_refs(&store).await.map_err(|e| e.to_string())?);
    prot.extend(serving);
    let policy = gc::GcPolicy::default();
    let (cards, bad_cards) = gc::load_cards(&store, &listing).await;
    let elites = gc::card_elites(&cards, policy.elite_per_family);
    let plan = gc::plan(&listing, &policy, &prot, &elites, chrono::Utc::now());
    let report = gc::execute(&store, &plan, dry_run()).await.map_err(|e| e.to_string())?;
    Ok(json!({
        "dry_run": report.dry_run, "objects": listing.len(), "cards": cards.len(), "bad_cards": bad_cards,
        "kept": plan.keep.len(), "deleted": report.deleted, "deleted_bytes": report.deleted_bytes,
        "failed": report.failed.len(),
    }))
}

async fn media_downsample(s: &AppState) -> Result<Value, String> {
    let secrets = fa_core::Secrets::load().map_err(|e| format!("garage secrets: {e}"))?;
    let store = S3BlobStore::from_config(&s.cfg, &secrets).map_err(|e| format!("garage: {e}"))?;
    let after = chrono::Duration::days(i64::from(s.cfg.garage.media_downsample_after_d));
    let report =
        fa_blob::media::downsample(&store, chrono::Utc::now(), after, dry_run()).await.map_err(|e| e.to_string())?;
    serde_json::to_value(report).map_err(|e| e.to_string())
}
