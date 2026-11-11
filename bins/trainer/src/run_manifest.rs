//! Run manifests (spec 36 §5 "Run manifest", §6.6, acceptance 6).
//!
//! Every successful fit (`pretrain`, `generation`; statistical and neural) writes
//! `fa-models/runs/<snapshot_id>/manifest.json` (and `<out>/runs/<snapshot_id>.json` locally)
//! holding everything needed to re-run it: the build (git commit + dirty flag, `Cargo.lock` and
//! `go.sum` SHA-256 embedded at compile time by `build.rs`), the job kind and the full
//! [`FitSpec`] with the window end pinned, its config hash, the seeds, the data window and
//! instrument pins, the data version (SHA-256 of the bars read, holdouts excluded:
//! [`crate::holdout::DataAudit`]), the device / backend and the host, and the SHA-256 of the
//! written `weights.bin`. The snapshot manifest carries the build block as `params.build` and
//! the job result `run_manifest_key`.
//!
//! [`reproduce`] re-runs a manifest (same spec, same window, same instruments; offline against
//! frozen bars or against VictoriaMetrics) and compares the data hash and the weights hash:
//! on the CPU backend a parent-free run reproduces bit for bit (`reproduction_is_bit_exact`).
//! Runs with a parent (`generation`) additionally need the parent snapshot and the elite
//! decoders they read, so they reproduce against the live stores only.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    data::Instruments,
    fit::FitSpec,
    holdout::DataAudit,
    run::{Control, Kind, Outcome, Services},
};

pub const PROTOCOL: &str = "fa-run/v1";

/// Compile-time provenance (`build.rs`).
pub fn build_info() -> Value {
    json!({
        "git_commit": env!("FA_BUILD_GIT_COMMIT"),
        "git_dirty": env!("FA_BUILD_GIT_DIRTY"),
        "cargo_lock_sha256": env!("FA_BUILD_CARGO_LOCK_SHA256"),
        "go_sum_sha256": env!("FA_BUILD_GO_SUM_SHA256"),
        "binary": "trainer",
        "version": env!("CARGO_PKG_VERSION"),
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Digest of `weights.bin`: for safetensors (neural) the SHA-256 over the parameter tensors in
/// name order (the header's metadata map is serialised in arbitrary order, so raw bytes are not
/// stable), else the SHA-256 of the bytes (`FASTAT01`).
pub fn weights_digest(bytes: &[u8]) -> String {
    if let Ok((vals, _)) = fa_expert_space::nn::from_safetensors(bytes, &candle_core::Device::Cpu) {
        let mut names: Vec<&String> = vals.keys().collect();
        names.sort();
        let mut h = Sha256::new();
        for n in names {
            h.update(n.as_bytes());
            if let Ok(v) =
                vals[n].flatten_all().and_then(|t| t.to_dtype(candle_core::DType::F32)).and_then(|t| t.to_vec1::<f32>())
            {
                for x in v {
                    h.update(x.to_le_bytes());
                }
            }
        }
        return format!("{:x}", h.finalize());
    }
    sha256_hex(bytes)
}

/// SHA-256 of a value's canonical JSON (serde_json orders object keys).
pub fn config_hash(v: &Value) -> String {
    sha256_hex(&serde_json::to_vec(v).unwrap_or_default())
}

fn host() -> Value {
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        "hostname": std::env::var("HOSTNAME").unwrap_or_default(),
    })
}

/// The spec as it must be re-run: window end pinned, instruments pinned to what was read.
pub fn pinned_spec(spec: &FitSpec, audit: Option<&DataAudit>, result: &Value) -> FitSpec {
    let mut s = spec.clone();
    if let Some(a) = audit {
        s.end_ms = Some(a.end_ms);
    }
    let insts: Vec<String> = result["data"]["instruments"]
        .as_array()
        .map(|a| a.iter().filter_map(|i| i["instrument"].as_str().map(str::to_owned)).collect())
        .unwrap_or_default();
    if !insts.is_empty() {
        let mut uniq = insts;
        uniq.dedup();
        s.instruments = Some(Instruments::List(uniq));
        s.instrument = None;
    }
    s
}

/// Builds the run manifest of a finished fit.
pub fn build(
    kind: Kind,
    spec: &FitSpec,
    audit: Option<&DataAudit>,
    result: &Value,
    job_id: &str,
    weights: Option<&[u8]>,
) -> Value {
    let pinned = pinned_spec(spec, audit, result);
    let spec_json = serde_json::to_value(&pinned).unwrap_or(Value::Null);
    let neural = spec.family.as_deref().is_some_and(fa_encoders::neural::is_neural_family);
    json!({
        "protocol": PROTOCOL,
        "run_id": result["snapshot_id"],
        "snapshot_id": result["snapshot_id"],
        "job_id": job_id,
        "kind": kind.job_kind(),
        "created_at": chrono::Utc::now().to_rfc3339(),
        "build": build_info(),
        "spec": spec_json,
        "config_sha256": config_hash(&spec_json),
        "seeds": { "fit": spec.seed, "minibatch": "Rng::seed(seed ^ step)", "router_noise": "per step from the minibatch rng" },
        "data": result["data"],
        "data_audit": audit,
        "parent_snapshot_id": result["parent_snapshot_id"],
        "generation": result["generation"],
        "feedback_source": result["feedback"]["source"],
        "device": result["device"].as_str().unwrap_or("cpu"),
        "backend": if neural { json!({ "library": "candle", "version": "0.10", "dtype": "f32" }) } else { json!({ "library": "fa_encoders::StatisticalEncoder" }) },
        "nondeterminism": "none on CPU (bit-exact); CUDA kernels unverified (no GPU on the dev host)",
        "host": host(),
        "outputs": {
            "manifest_key": result["manifest_key"],
            "weights_sha256": weights.map(weights_digest),
        },
    })
}

async fn written_weights(svc: &Services<'_>, result: &Value) -> Option<Vec<u8>> {
    let id = result["snapshot_id"].as_str()?;
    if let Some(dir) = result["persisted"]["local_dir"].as_str()
        && let Ok(b) = std::fs::read(std::path::Path::new(dir).join(fa_encoders::WEIGHTS_FILE))
    {
        return Some(b);
    }
    let blob = svc.stores.blob.as_ref()?;
    fa_blob::BlobStore::get(blob, &fa_blob::snapshot_file(id, fa_encoders::WEIGHTS_FILE)).await.ok()
}

pub fn run_key(snapshot_id: &str) -> fa_blob::ObjectKey {
    fa_blob::ObjectKey { bucket: fa_blob::buckets::MODELS, key: format!("runs/{snapshot_id}/manifest.json") }
}

/// Writes the run manifest of an OK fit and links it from the job result.
pub async fn attach(
    out: &mut Outcome,
    kind: Kind,
    spec: &FitSpec,
    audit: Option<DataAudit>,
    svc: &Services<'_>,
    ctl: &Control,
) {
    let Some(id) = out.result["snapshot_id"].as_str().map(str::to_owned) else { return };
    let weights = written_weights(svc, &out.result).await;
    let m = build(kind, spec, audit.as_ref(), &out.result, &ctl.job_id, weights.as_deref());
    let bytes = serde_json::to_vec_pretty(&m).unwrap_or_default();
    let mut where_ = json!({});
    if let Some(dir) = &svc.stores.out_dir {
        let p = dir.join("runs");
        if std::fs::create_dir_all(&p).is_ok() && std::fs::write(p.join(format!("{id}.json")), &bytes).is_ok() {
            where_["local"] = json!(p.join(format!("{id}.json")).display().to_string());
        }
    }
    if let Some(blob) = &svc.stores.blob {
        let key = run_key(&id);
        match fa_blob::BlobStore::put(blob, &key, bytes, fa_blob::JSON).await {
            Ok(_) => {
                where_["garage"] = json!(key.key);
                out.output_refs.push(key.key);
            }
            Err(e) => tracing::warn!(error = %e, "run manifest not written to Garage"),
        }
    }
    out.result["run_manifest"] = json!({
        "location": where_, "config_sha256": m["config_sha256"], "weights_sha256": m["outputs"]["weights_sha256"],
        "bars_sha256": m["data_audit"]["bars_sha256"], "git_commit": m["build"]["git_commit"],
    });
    if let Some(k) = where_.get("garage") {
        out.result["run_manifest_key"] = k.clone();
    }
}

/// Re-runs a run manifest and compares the data and weights hashes. `svc` decides where the
/// bars come from (frozen bars or VictoriaMetrics) and where the re-run snapshot is written
/// (use local-only stores: the re-run gets a new snapshot id).
pub async fn reproduce(manifest: &Value, svc: &Services<'_>) -> anyhow::Result<Value> {
    let kind = manifest["kind"]
        .as_str()
        .and_then(Kind::parse_job_kind)
        .ok_or_else(|| anyhow::anyhow!("run manifest: no kind"))?;
    let spec: FitSpec = serde_json::from_value(manifest["spec"].clone())?;
    if spec.end_ms.is_none() {
        anyhow::bail!("run manifest has no pinned window end");
    }
    let out = crate::run::run_fit(kind, spec, svc, &Control::local("reproduce")).await;
    if out.status != fa_proto::v1::job_result::Status::Ok {
        anyhow::bail!("re-run failed: {}", out.error);
    }
    let want_w = manifest["outputs"]["weights_sha256"].as_str().unwrap_or_default();
    let want_d = manifest["data_audit"]["bars_sha256"].as_str().unwrap_or_default();
    let got_w = out.result["run_manifest"]["weights_sha256"].as_str().unwrap_or_default().to_owned();
    let got_d = out.result["run_manifest"]["bars_sha256"].as_str().unwrap_or_default().to_owned();
    let same_build = manifest["build"]["git_commit"] == build_info()["git_commit"]
        && manifest["build"]["cargo_lock_sha256"] == build_info()["cargo_lock_sha256"];
    Ok(json!({
        "original": manifest["snapshot_id"],
        "rerun": out.result["snapshot_id"],
        "data_identical": !want_d.is_empty() && want_d == got_d,
        "weights_identical": !want_w.is_empty() && want_w == got_w,
        "same_build": same_build,
        "bars_sha256": { "original": want_d, "rerun": got_d },
        "weights_sha256": { "original": want_w, "rerun": got_w },
        "verified": !want_w.is_empty() && want_w == got_w && want_d == got_d,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{artifact::Stores, holdout::tests::history};

    fn small_neural_spec() -> FitSpec {
        FitSpec {
            family: Some("fsq_hybrid".into()),
            neural: Some(json!({ "steps": 20, "context_steps": 16, "d_model": 8, "layers": 2, "batch": 16,
                                  "moe": { "max_experts": 2, "expert_dim": 4, "fsq_levels": [5, 5] } })),
            window_hours: 24.0,
            instruments: Some(Instruments::All(crate::data::AllTag::All)),
            probe_max: 60,
            seed: 7,
            end_ms: Some(400 * 60_000),
            ..FitSpec::new(fa_core::Timescale::Minute)
        }
    }

    /// Acceptance 6: the run manifest re-runs the job bit for bit (CPU, frozen bars).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reproduction_is_bit_exact() {
        let bars = vec![
            history("aaa", fa_core::Timescale::Minute, 400, 0),
            history("bbbb", fa_core::Timescale::Minute, 380, 20 * 60_000),
        ];
        let dir = std::env::temp_dir().join(format!("fa-run-manifest-{}", std::process::id()));
        let stores = Stores { blob: None, hub: None, out_dir: Some(dir.clone()) };
        let tsdb = fa_tsdb::Tsdb::new("http://127.0.0.1:9");
        let svc = Services { frozen: Some(&bars), ..Services::new(&stores, &tsdb, None) };
        let out = crate::run::run_fit(Kind::Pretrain, small_neural_spec(), &svc, &Control::local("t")).await;
        assert_eq!(out.status, fa_proto::v1::job_result::Status::Ok, "{}", out.error);
        let id = out.result["snapshot_id"].as_str().unwrap().to_owned();
        let m: Value =
            serde_json::from_slice(&std::fs::read(dir.join("runs").join(format!("{id}.json"))).unwrap()).unwrap();
        assert_eq!(m["protocol"], PROTOCOL);
        assert_eq!(m["spec"]["end_ms"], json!(400 * 60_000));
        assert!(m["build"]["cargo_lock_sha256"].as_str().unwrap().len() == 64);
        assert!(m["outputs"]["weights_sha256"].as_str().unwrap().len() == 64);
        assert_eq!(m["data_audit"]["source"], "frozen");
        let r = reproduce(&m, &svc).await.unwrap();
        assert_eq!(r["verified"], json!(true), "{r}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
