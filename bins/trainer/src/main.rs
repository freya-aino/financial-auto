//! trainer: encoder training lifecycle (spec 36) for the statistical family and the four neural
//! families of spec 31 (candle, [`neural_fit`]; D-058, D-065, D-070, D-073): `pretrain` fits a
//! gen-0 snapshot from VictoriaMetrics history, `generation` fits gen N+1 from a parent snapshot
//! (statistical: feedback search; neural: warm start, D-007 anchor, D-009 elite-decoder feedback),
//! `adapter` re-fits statistics (neural: input normalisation + LoRA) on the last hour, `sweep`
//! compares neural genomes without writing, and `serve` runs the `JOBS_TRAIN` work-queue worker. Every fit writes `manifest.json`, `weights.bin`, `regions.json` and
//! `region_map.json` to `fa-models/snapshots/<snapshot_id>/` (and/or `--out`), mirrors the
//! snapshot in Helix as a `candidate` with `SnapshotParent` / `RegionMatch` edges, and the
//! region lineage is carried over by probe-set alignment against the parent (D-007).
//! Promotion is control-api's `POST /api/latent/snapshots/{id}/promote`, which checks the
//! spec 36 §6.4 checklist (carried mass, shadow serving, neural health metrics) and reloads model-runtime
//! (`ctl.model.reload`). Fits read connector bars (`fa_bar_*`) and fall back to trade prices
//! per instrument; a statistical `generation` runs the feedback search of [`search`]; `adapter` writes an
//! `AdapterState` node; `serve` can plan adapter / generation jobs periodically.

mod adapters;
mod artifact;
mod data;
mod expert_health;
mod feedback;
mod fit;
mod generation;
mod gpu;
mod holdout;
mod neural_fit;
mod plan;
mod promotion;
mod run;
mod run_manifest;
mod search;
mod watch;
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
    /// Instrument (VM token or UUID); default: the one with the most bars.
    #[arg(long)]
    instrument: Option<String>,
    /// Multi-instrument fit: comma-separated tokens/UUIDs or `all`.
    #[arg(long)]
    instruments: Option<String>,
    /// Pool normalisation statistics with equal weight per instrument (`params.per_instrument_norm`).
    #[arg(long, default_value_t = false)]
    per_instrument_norm: bool,
    /// Feedback-search candidates of a `generation` (1 disables the search).
    #[arg(long, default_value_t = 12)]
    search_candidates: usize,
    /// Adapter drift guard recorded in the manifest (`params.adapter_guard.max_drift`).
    #[arg(long, default_value_t = 1.0)]
    adapter_max_drift: f64,
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
    /// Encoder family: `statistical` (default) or a neural family (candle): `fsq_hybrid`,
    /// `hier_vae_flow`, `jepa_vae`, `latent_diffusion` (day/week only, D-010).
    #[arg(long, default_value = "statistical")]
    family: String,
    /// Neural genome overrides as JSON, e.g. `{"steps":300,"backbone":"gru","moe":{"max_experts":6}}`.
    #[arg(long)]
    neural: Option<String>,
    /// SGD steps of a neural fit (shorthand for `--neural '{"steps":N}'`).
    #[arg(long)]
    steps: Option<usize>,
    /// Compute device of a neural fit (`cpu`, `cuda[:n]`; CUDA needs the `cuda` feature).
    #[arg(long, env = "FA_ML_DEVICE")]
    device: Option<String>,
}

impl FitArgs {
    fn spec(&self) -> anyhow::Result<FitSpec> {
        let mut neural: Option<serde_json::Value> = match &self.neural {
            Some(s) => Some(serde_json::from_str(s).map_err(|e| anyhow::anyhow!("--neural: {e}"))?),
            None => None,
        };
        if let Some(n) = self.steps {
            neural.get_or_insert_with(|| serde_json::json!({}))["steps"] = serde_json::json!(n);
        }
        Ok(FitSpec {
            family: Some(self.family.clone()),
            neural,
            device: self.device.clone(),
            window_hours: self.window_hours,
            instrument: self.instrument.clone(),
            instruments: self.instruments.as_deref().map(data::Instruments::parse_cli),
            params: fit::FitParams { per_instrument_norm: self.per_instrument_norm },
            search: search::SearchSpec { candidates: self.search_candidates.max(1), ..Default::default() },
            adapter_guard: Some(fa_encoders::AdapterGuard { max_drift: self.adapter_max_drift, ..Default::default() }),
            parent_snapshot_id: self.parent.clone(),
            seed: self.seed,
            fit: fa_encoders::FitConfig { stride: self.stride.max(1), ..Default::default() },
            region_radius: self.region_radius,
            carry_threshold: self.carry_threshold,
            probe_max: self.probe_max,
            ..FitSpec::new(self.timescale)
        })
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Gen 0: fit a snapshot of a timescale (statistical or `--family`) from history (candidate).
    Pretrain(FitArgs),
    /// Gen N+1 from the parent (active) snapshot, recording the latest decoder feedback.
    Generation(FitArgs),
    /// Online adapter: re-fit statistics (neural: input normalisation + LoRA) on the last hour of a snapshot.
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
        /// Override the snapshot's `params.adapter_guard.max_drift`.
        #[arg(long)]
        max_drift: Option<f64>,
        /// Neural snapshots: input-normalisation adapter only (no LoRA).
        #[arg(long, default_value_t = false)]
        no_lora: bool,
        /// Neural snapshots: LoRA SGD steps.
        #[arg(long)]
        lora_steps: Option<usize>,
    },
    /// List the `AdapterState` versions of a snapshot (Helix), newest first.
    Adapters {
        #[arg(long)]
        snapshot: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Train a list of neural genomes on the same history and print their expert-space health
    /// and checklist verdict, one JSON line each (no snapshot is written).
    Sweep {
        #[command(flatten)]
        fit: FitArgs,
        /// JSON array of genome overrides (or `@file.json`).
        #[arg(long)]
        grid: String,
        /// Cache the loaded bars here (read if it exists, so every sweep sees the same data).
        #[arg(long)]
        bars_cache: Option<PathBuf>,
        /// Locked holdout `start_ms:end_ms` to exclude when `validation.holdout` cannot be read
        /// (repeatable; the recorded holdouts are always excluded when readable).
        #[arg(long)]
        holdout_ms: Vec<String>,
    },
    /// One encoder generation through the spec 36 §4.1 state machine (data freeze → family
    /// candidates → Pareto selection → re-embedding → regions → MCMC init → decoder / strategy
    /// seeds → validation); never promotes.
    Cycle(CycleArgs),
    /// Gen-0 bootstrap end to end (acceptance 1): `cycle` with `pretrain` fits of every family;
    /// `--bars-file` (recorded bars, e.g. a sweep `--bars-cache`) and/or `--synthetic N` run it
    /// offline.
    Bootstrap(CycleArgs),
    /// Re-runs a run manifest (`fa-models/runs/<snapshot>/manifest.json` or a local file) and
    /// compares the data and weights hashes (spec 36 §6.6). Use with `--local-only --out <dir>`.
    Reproduce {
        /// Local path or Garage key (`runs/<snapshot>/manifest.json`).
        #[arg(long)]
        run: String,
        /// Re-run against these recorded bars instead of VictoriaMetrics.
        #[arg(long)]
        bars_file: Option<PathBuf>,
    },
    /// Print the planner schedule and the jobs its next tick would enqueue.
    Plan(PlanArgs),
    /// Cancel a training job (`ctl.train.cancel {job_id}`).
    Cancel { job_id: String },
    /// Work-queue worker for `jobs.train.>` (+ the periodic planner).
    Serve {
        /// GPU memory requested from the arbiter per job (0 disables leasing).
        #[arg(long, env = "FA_TRAIN_GPU_LEASE_MB", default_value_t = 256)]
        gpu_lease_mb: u32,
        /// Checkpoint the fit to `fa-ckpt/<job_id>/state.json` this often (0 = only on preemption).
        #[arg(long, env = "FA_TRAIN_CHECKPOINT_SECS", default_value_t = 30)]
        checkpoint_secs: u64,
        #[command(flatten)]
        plan: PlanArgs,
    },
}

#[derive(clap::Args, Clone)]
struct CycleArgs {
    #[arg(long)]
    timescale: Timescale,
    /// Comma-separated neural families (default: every family allowed at the timescale).
    #[arg(long)]
    families: Option<String>,
    #[arg(long, default_value_t = 12.0)]
    window_hours: f64,
    #[arg(long)]
    instruments: Option<String>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Genome overrides shared by every candidate.
    #[arg(long)]
    neural: Option<String>,
    #[arg(long)]
    steps: Option<usize>,
    #[arg(long)]
    parent: Option<String>,
    /// Recorded bars (`Vec<BarHistory>` JSON, e.g. a `sweep --bars-cache` file).
    #[arg(long)]
    bars_file: Option<PathBuf>,
    /// Add N synthetic instruments (regime-switching, not market data) of `--synthetic-bars` bars.
    #[arg(long, default_value_t = 0)]
    synthetic: usize,
    #[arg(long, default_value_t = 600)]
    synthetic_bars: usize,
    #[arg(long, default_value_t = 8)]
    decoder_seeds: usize,
    #[arg(long, default_value_t = 16)]
    strategy_seeds: usize,
    #[arg(long, default_value_t = 512)]
    probe_max: usize,
}

impl CycleArgs {
    fn spec(&self, kind: &str) -> anyhow::Result<generation::CycleSpec> {
        let mut neural: Option<serde_json::Value> = match &self.neural {
            Some(s) => Some(serde_json::from_str(s).map_err(|e| anyhow::anyhow!("--neural: {e}"))?),
            None => None,
        };
        if let Some(n) = self.steps {
            neural.get_or_insert_with(|| serde_json::json!({}))["steps"] = serde_json::json!(n);
        }
        Ok(generation::CycleSpec {
            families: self
                .families
                .as_deref()
                .map(|f| f.split(',').map(|x| x.trim().to_owned()).collect())
                .unwrap_or_default(),
            window_hours: self.window_hours,
            instruments: self.instruments.as_deref().map(data::Instruments::parse_cli),
            seed: self.seed,
            neural,
            parent_snapshot_id: self.parent.clone(),
            decoder_seeds: self.decoder_seeds,
            strategy_seeds: self.strategy_seeds,
            probe_max: self.probe_max,
            ..generation::CycleSpec::new(self.timescale, kind)
        })
    }

    /// Recorded + synthetic bars (`None` = read VictoriaMetrics).
    fn frozen(&self) -> anyhow::Result<Option<Vec<data::BarHistory>>> {
        let mut bars: Vec<data::BarHistory> = match &self.bars_file {
            Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
            None => Vec::new(),
        };
        if self.synthetic > 0 {
            let end = bars
                .iter()
                .filter(|h| h.timescale == self.timescale)
                .map(|h| h.end_ms)
                .max()
                .unwrap_or_else(data::now_ms);
            let end = end - end.rem_euclid(data::step_ms(self.timescale));
            bars.extend(generation::synthetic_bars(
                self.timescale,
                self.synthetic,
                self.synthetic_bars,
                end,
                self.seed,
            ));
        }
        Ok((!bars.is_empty()).then_some(bars))
    }
}

#[derive(clap::Args, Clone)]
struct PlanArgs {
    /// Enqueue an adapter job per active snapshot (any family) this often (`off` disables); each
    /// LoRA job resumes the previous factors (continuous adapters, spec 36 §6.5).
    #[arg(long, env = "FA_TRAIN_ADAPTER_EVERY", default_value = "15m")]
    adapter_every: String,
    /// Enqueue a generation job per active snapshot this often (default `off`).
    #[arg(long, env = "FA_TRAIN_GENERATION_EVERY", default_value = "off")]
    generation_every: String,
    /// `instruments` of planned generation jobs.
    #[arg(long, default_value = "all")]
    generation_instruments: String,
    #[arg(long, default_value_t = 6.0)]
    generation_window_hours: f64,
}

impl PlanArgs {
    fn config(&self) -> anyhow::Result<plan::PlanConfig> {
        Ok(plan::PlanConfig {
            adapter_every: plan::parse_every(&self.adapter_every).map_err(anyhow::Error::msg)?,
            generation_every: plan::parse_every(&self.generation_every).map_err(anyhow::Error::msg)?,
            generation_window_hours: self.generation_window_hours,
            generation_instruments: self.generation_instruments.clone(),
            ..Default::default()
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fa_core::logging::init("trainer"); // FA_LOG_FORMAT=json|text (spec 60 §4)
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
    let svc = Services::new(&stores, &tsdb, Some(cfg.postgres.url.as_str()));

    let outcome = match args.cmd {
        Cmd::Pretrain(f) => run::run_fit(Kind::Pretrain, f.spec()?, &svc, &Control::local("")).await,
        Cmd::Generation(f) => run::run_fit(Kind::Generation, f.spec()?, &svc, &Control::local("")).await,
        Cmd::Adapter { snapshot, timescale, window_hours, instrument, max_drift, no_lora, lora_steps } => {
            let spec = AdapterSpec {
                snapshot_id: snapshot,
                timescale,
                window_hours,
                instrument,
                max_drift,
                no_lora,
                lora: lora_steps.map(|steps| fa_encoders::neural::adapter::LoraConfig { steps, ..Default::default() }),
                ..Default::default()
            };
            run::run_adapter(spec, &svc, &Control::local("")).await
        }
        Cmd::Adapters { snapshot, limit } => {
            let hub = stores.hub.as_ref().ok_or_else(|| anyhow::anyhow!("Helix unavailable"))?;
            let rows = hub.snapshot_adapters(&snapshot, limit).await?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({ "snapshot_id": snapshot, "adapters": rows }))?
            );
            return Ok(());
        }
        Cmd::Sweep { fit, grid, bars_cache, holdout_ms } => {
            let spec = fit.spec()?;
            let family = neural_fit::family_of(&spec)?
                .ok_or_else(|| anyhow::anyhow!("sweep needs a neural --family (fsq_hybrid | jepa_vae | ...)"))?;
            let grid_text = match grid.strip_prefix('@') {
                Some(path) => std::fs::read_to_string(path)?,
                None => grid,
            };
            let grid: Vec<serde_json::Value> = serde_json::from_str(&grid_text)?;
            let history: Vec<data::BarHistory> = match bars_cache.as_ref().filter(|p| p.exists()) {
                Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
                None => {
                    let (start_ms, end_ms) = data::window(spec.timescale, spec.window_hours, data::now_ms());
                    let h = data::load_set(
                        &tsdb,
                        spec.timescale,
                        spec.instruments.as_ref(),
                        spec.instrument.as_deref(),
                        start_ms,
                        end_ms,
                        None,
                    )
                    .await?;
                    if let Some(p) = &bars_cache {
                        std::fs::write(p, serde_json::to_vec(&h)?)?;
                    }
                    h
                }
            };
            // spec 36 §6.1 / acceptance 7: a sweep trains too, so cached or loaded bars lose every
            // locked holdout (+ embargo) like a fit's
            let history = sweep_exclude_holdouts(history, svc.pg_url, spec.timescale, &holdout_ms).await?;
            let (dev, _) = fa_expert_space::nn::device_from_str(spec.device.as_deref().unwrap_or("cpu"));
            neural_fit::sweep(family, spec.timescale, &history, &grid, spec.seed, &dev, |row| {
                println!("{row}");
            })?;
            return Ok(());
        }
        Cmd::Cycle(c) => return run_cycle_cmd(&c, "cycle", &svc, &cfg).await,
        Cmd::Bootstrap(c) => return run_cycle_cmd(&c, "bootstrap", &svc, &cfg).await,
        Cmd::Reproduce { run, bars_file } => {
            let bytes = match std::fs::read(&run) {
                Ok(b) => b,
                Err(_) => {
                    let blob =
                        stores.blob.as_ref().ok_or_else(|| anyhow::anyhow!("{run}: not a file and no Garage"))?;
                    fa_blob::BlobStore::get(
                        blob,
                        &fa_blob::ObjectKey { bucket: fa_blob::buckets::MODELS, key: run.clone() },
                    )
                    .await?
                }
            };
            let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
            let frozen: Option<Vec<data::BarHistory>> = match bars_file {
                Some(p) => Some(serde_json::from_slice(&std::fs::read(p)?)?),
                None => None,
            };
            let rsvc = Services { frozen: frozen.as_deref(), ..svc };
            let r = run_manifest::reproduce(&manifest, &rsvc).await?;
            println!("{}", serde_json::to_string_pretty(&r)?);
            if r["verified"] != serde_json::json!(true) {
                anyhow::bail!("not reproduced");
            }
            return Ok(());
        }
        Cmd::Plan(p) => {
            let pc = p.config()?;
            let active = match stores.hub.as_ref() {
                Some(h) => plan::active_snapshots(h).await?,
                None => Vec::new(),
            };
            println!("{}", serde_json::to_string_pretty(&plan::describe(&pc, &active))?);
            return Ok(());
        }
        Cmd::Cancel { job_id } => {
            let bus = fa_bus::Bus::connect(&cfg.nats.url).await?;
            let req = fa_proto::v1::CtlRequest {
                verb: "cancel".into(),
                args: [("job_id".to_owned(), job_id)].into_iter().collect(),
            };
            let msg = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                bus.client()
                    .request(fa_bus::subjects::ctl("train", "cancel"), fa_proto::Message::encode_to_vec(&req).into()),
            )
            .await
            .map_err(|_| anyhow::anyhow!("no trainer answered ctl.train.cancel within 5 s"))??;
            let reply = <fa_proto::v1::CtlReply as fa_proto::Message>::decode(msg.payload.as_ref())?;
            println!("{}: {}", if reply.ok { "ok" } else { "refused" }, reply.message);
            return Ok(());
        }
        Cmd::Serve { gpu_lease_mb, checkpoint_secs, plan } => {
            let bus = fa_bus::Bus::connect(&cfg.nats.url).await?;
            let plan = plan.config()?;
            return worker::serve(bus, svc, worker::WorkerConfig { gpu_lease_mb, checkpoint_secs, plan }).await;
        }
    };
    println!("{}", serde_json::to_string_pretty(&outcome.result)?);
    if outcome.status != Status::Ok {
        anyhow::bail!("{:?}: {}", outcome.status, outcome.error);
    }
    Ok(())
}

/// `trainer cycle|bootstrap`: runs the generation state machine locally and prints its result.
async fn run_cycle_cmd(c: &CycleArgs, kind: &str, svc: &Services<'_>, cfg: &PlatformConfig) -> anyhow::Result<()> {
    let mut spec = c.spec(kind)?;
    let frozen = c.frozen()?;
    if let Some(f) = &frozen {
        spec.end_ms = f.iter().filter(|h| h.timescale == c.timescale).map(|h| h.end_ms).max();
    }
    let bus = if svc.stores.hub.is_some() { fa_bus::Bus::connect(&cfg.nats.url).await.ok() } else { None };
    let pg = if svc.stores.hub.is_some() { svc.pg_url } else { None };
    let store = generation::GenStore::connect(pg, bus).await;
    let csvc = Services { frozen: frozen.as_deref(), ..*svc };
    let out = generation::run_cycle(spec, &csvc, &run::Control::local("cli"), &store).await;
    println!("{}", serde_json::to_string_pretty(&out.result)?);
    if out.status != Status::Ok {
        anyhow::bail!("{:?}: {}", out.status, out.error);
    }
    Ok(())
}

/// Locked holdouts of a sweep: `validation.holdout` when readable, plus `--holdout-ms` ranges;
/// unreadable without explicit ranges fails closed (unless `FA_TRAIN_HOLDOUT_OPTIONAL=1`).
async fn sweep_exclude_holdouts(
    history: Vec<data::BarHistory>,
    pg_url: Option<&str>,
    ts: Timescale,
    explicit: &[String],
) -> anyhow::Result<Vec<data::BarHistory>> {
    let mut hs: Vec<fa_backtest::replay::Holdout> = explicit
        .iter()
        .map(|r| {
            let (a, b) = r.split_once(':').ok_or_else(|| anyhow::anyhow!("--holdout-ms wants start_ms:end_ms"))?;
            Ok(fa_backtest::replay::Holdout {
                timescale: ts,
                start_ms: a.trim().parse()?,
                end_ms: b.trim().parse()?,
                locked_at_ms: 0,
                embargo_ms: None,
            })
        })
        .collect::<anyhow::Result<_>>()?;
    let read = match pg_url {
        Some(u) => holdout::holdouts(u, ts).await,
        None => Err(anyhow::anyhow!("no Postgres URL")),
    };
    match read {
        Ok(h) => hs.extend(h),
        Err(e) if !explicit.is_empty() || holdout::optional() => {
            eprintln!("validation.holdout unreadable ({e}); excluding the {} explicit range(s) only", explicit.len())
        }
        Err(e) => anyhow::bail!("locked holdouts unreadable ({e}); pass --holdout-ms or FA_TRAIN_HOLDOUT_OPTIONAL=1"),
    }
    let mut audit = holdout::DataAudit::default();
    // the widest horizon a sweep genome uses (JEPA / diffusion look 4–8 steps ahead)
    let out = holdout::exclude(history, &hs, data::step_ms(ts) * 8, &mut audit);
    eprintln!(
        "sweep data: {} bars excluded by {} holdout(s) {:?}",
        audit.excluded_bars,
        hs.len(),
        audit.excluded_ranges
    );
    Ok(out)
}
