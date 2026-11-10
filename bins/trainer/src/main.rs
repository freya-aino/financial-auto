//! trainer: encoder training lifecycle (spec 36) with the statistical family standing in for a
//! neural model (D-004): `pretrain` fits a gen-0 snapshot from VictoriaMetrics history,
//! `generation` fits gen N+1 from a parent snapshot (recording the latest decoder-feedback
//! summary), `adapter` re-fits statistics on the last hour, and `serve` runs the `JOBS_TRAIN`
//! work-queue worker. Every fit writes `manifest.json`, `weights.bin`, `regions.json` and
//! `region_map.json` to `fa-models/snapshots/<snapshot_id>/` (and/or `--out`), mirrors the
//! snapshot in Helix as a `candidate` with `SnapshotParent` / `RegionMatch` edges, and the
//! region lineage is carried over by probe-set alignment against the parent (D-007).
//! Promotion is control-api's `POST /api/latent/snapshots/{id}/promote`, which reloads
//! model-runtime (`ctl.model.reload`).

mod artifact;
mod data;
mod feedback;
mod fit;
mod gpu;
mod run;
mod worker;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use fa_blob::S3BlobStore;
use fa_core::{Secrets, Timescale, config::PlatformConfig};
use fa_proto::v1::job_result::Status;

use crate::{
    artifact::Stores,
    fit::FitSpec,
    run::{AdapterSpec, Control, Kind, Services},
};

#[derive(Parser)]
#[command(about = "Encoder training: gen-0 pretraining, generations, online adapters, job worker")]
struct Args {
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: String,
    /// Write the snapshot folder here as well (`<out>/<snapshot_id>/…`, adapters under `<out>/adapters/`).
    #[arg(long, global = true)]
    out: Option<PathBuf>,
    /// Skip Garage and Helix (local folder only).
    #[arg(long, global = true, default_value_t = false)]
    local_only: bool,
    /// VictoriaMetrics URL (default `FA_VM_URL`, else `victoriametrics.url`).
    #[arg(long, global = true)]
    vm_url: Option<String>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct FitArgs {
    #[arg(long)]
    timescale: Timescale,
    /// History length ending now.
    #[arg(long, default_value_t = 6.0)]
    window_hours: f64,
    /// Instrument (VM token or UUID); default: the one with the most trades.
    #[arg(long)]
    instrument: Option<String>,
    /// Snapshot to align with (default: the active snapshot of the timescale in Helix).
    #[arg(long)]
    parent: Option<String>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 1)]
    stride: usize,
    #[arg(long, default_value_t = 1.0)]
    region_radius: f32,
    #[arg(long, default_value_t = 0.5)]
    carry_threshold: f64,
    #[arg(long, default_value_t = 512)]
    probe_max: usize,
}

impl FitArgs {
    fn spec(&self) -> FitSpec {
        FitSpec {
            window_hours: self.window_hours,
            instrument: self.instrument.clone(),
            parent_snapshot_id: self.parent.clone(),
            seed: self.seed,
            fit: fa_encoders::FitConfig { stride: self.stride.max(1), ..Default::default() },
            region_radius: self.region_radius,
            carry_threshold: self.carry_threshold,
            probe_max: self.probe_max,
            ..FitSpec::new(self.timescale)
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Gen 0: fit the statistical snapshot of a timescale from history (candidate).
    Pretrain(FitArgs),
    /// Gen N+1 from the parent (active) snapshot, recording the latest decoder feedback.
    Generation(FitArgs),
    /// Online adapter: re-fit statistics on the last hour of a snapshot.
    Adapter {
        /// Snapshot to adapt (default: the active snapshot of `--timescale`).
        #[arg(long)]
        snapshot: Option<String>,
        #[arg(long)]
        timescale: Option<Timescale>,
        #[arg(long, default_value_t = 1.0)]
        window_hours: f64,
        #[arg(long)]
        instrument: Option<String>,
        #[arg(long, default_value_t = 1.0)]
        max_drift: f64,
    },
    /// Work-queue worker for `jobs.train.>`.
    Serve {
        /// GPU memory requested from the arbiter per job (0 disables leasing).
        #[arg(long, env = "FA_TRAIN_GPU_LEASE_MB", default_value_t = 256)]
        gpu_lease_mb: u32,
        /// Checkpoint the fit to `fa-ckpt/<job_id>/state.json` this often (0 = only on preemption).
        #[arg(long, env = "FA_TRAIN_CHECKPOINT_SECS", default_value_t = 30)]
        checkpoint_secs: u64,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();
    let args = Args::parse();
    let cfg = PlatformConfig::load(&args.config)?;
    let vm_url = args
        .vm_url
        .clone()
        .or_else(|| std::env::var("FA_VM_URL").ok().filter(|u| !u.is_empty()))
        .unwrap_or_else(|| cfg.victoriametrics.url.clone());
    let tsdb = fa_tsdb::Tsdb::new(vm_url);
    if let Err(e) = tsdb.health().await {
        tracing::warn!(url = tsdb.base_url(), error = %e, "VictoriaMetrics unreachable; fits will fail until it answers");
    }
    let stores = if args.local_only {
        Stores { blob: None, hub: None, out_dir: args.out.clone() }
    } else {
        let secrets = Secrets::load()?;
        let blob = match S3BlobStore::from_config(&cfg, &secrets) {
            Ok(s) => Some(s),
            Err(fa_blob::BlobError::NotConfigured) => {
                tracing::warn!("no Garage credentials in config/secrets.toml; snapshots are not written to Garage");
                None
            }
            Err(e) => {
                tracing::warn!(error = %e, "Garage client unavailable");
                None
            }
        };
        let hub = fa_hub::HubClient::with_timeout(
            fa_hub::HubClient::url_from_env_or(&cfg.helix.url),
            std::time::Duration::from_secs(10),
        );
        let hub = match hub.ensure_schema_ready(std::time::Duration::from_secs(30)).await {
            Ok(()) => Some(hub),
            Err(e) => {
                tracing::warn!(url = hub.base_url(), error = %e, "Helix unreachable; snapshots are not mirrored");
                None
            }
        };
        Stores { blob, hub, out_dir: args.out.clone() }
    };
    let svc = Services { stores: &stores, tsdb: &tsdb, pg_url: Some(cfg.postgres.url.as_str()) };

    let outcome = match args.cmd {
        Cmd::Pretrain(f) => run::run_fit(Kind::Pretrain, f.spec(), &svc, &Control::local("")).await,
        Cmd::Generation(f) => run::run_fit(Kind::Generation, f.spec(), &svc, &Control::local("")).await,
        Cmd::Adapter { snapshot, timescale, window_hours, instrument, max_drift } => {
            let spec = AdapterSpec {
                snapshot_id: snapshot,
                timescale,
                window_hours,
                instrument,
                max_drift,
                ..Default::default()
            };
            run::run_adapter(spec, &svc, &Control::local("")).await
        }
        Cmd::Serve { gpu_lease_mb, checkpoint_secs } => {
            let bus = fa_bus::Bus::connect(&cfg.nats.url).await?;
            return worker::serve(bus, svc, worker::WorkerConfig { gpu_lease_mb, checkpoint_secs }).await;
        }
    };
    println!("{}", serde_json::to_string_pretty(&outcome.result)?);
    if outcome.status != Status::Ok {
        anyhow::bail!("{:?}: {}", outcome.status, outcome.error);
    }
    Ok(())
}
