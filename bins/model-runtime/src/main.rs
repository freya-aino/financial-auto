//! model-runtime: live latent pipeline per timescale (spec 01 §6, D-005, D-018).
//!
//! Consumes `md.trade.>` and `md.bar.>`, builds rolling bars per timescale and instrument
//! ([`bars`]: from bus bars where a venue publishes them, resampled upward, else from trades)
//! and runs the baseline pipeline ([`pipeline`]): statistical encoder → evidence-softmax router
//! → online leader-clustering regions → Dirichlet transition posteriors. It publishes
//! `LatentState` on `latent.state.<ts>.<instrument>`, `RegionEvent` on `latent.region.<ts>`
//! and `TransitionUpdate` on `latent.transition.<ts>`, plus a heartbeat with counters. The
//! baseline text (`news.item.>`) and visual (bar windows) decoders publish on
//! `decoder.output.<family>` ([`decoders`]); a candidate snapshot can run in shadow
//! ([`shadow`]); online adapters are applied on `ctl.model.adapter` / adapter job results.
//!
//! Neural encoders are off (D-004): the "snapshot" is the versioned statistical baseline
//! (`stat-v1-<ts>`) or a **trained statistical snapshot** from Garage (spec 36, [`snapshot`]):
//! the active Helix `Snapshot` of each timescale is loaded at start and hot-swapped on
//! `ctl.model.reload` (promote / rollback). Every stage sits behind a trait so trained
//! models can replace the baselines without touching the bus contract.

mod bars;
mod decoders;
mod hub_sink;
mod pipeline;
mod shadow;
mod snapshot;

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::{Duration, Instant},
};

use bars::{BarRow, BarSeries, BarUpdate};
use clap::Parser;
use fa_bus::{Bus, subjects};
use fa_core::{Timescale, config::PlatformConfig};
use fa_decoders::text_baseline::InstrumentIndex;
use fa_encoders::MarketBar;
use fa_patterns::OnlineRegionCfg;
use fa_proto::v1::{CtlReply, DecoderOutput, Envelope, Heartbeat, JobResult, envelope::Payload};
use futures::StreamExt;
use pipeline::{PipelineConfig, StepInput, TimescalePipeline};
use serde_json::json;
use snapshot::{CtlCommand, CtlVerb};
use tokio::sync::{Mutex, mpsc, watch};

#[derive(Parser)]
struct Args {
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: String,
    #[arg(long, default_value_t = 2)]
    heartbeat_secs: u64,
    /// Between bar closes, re-encode the forming bar and publish a preview state at most this
    /// often per (timescale, instrument). Keeps slow timescales (1h, 1d) visible in the UI.
    #[arg(long, default_value_t = 5)]
    preview_secs: u64,
    /// Leader-clustering radius in latent units (features are bounded to (-1, 1) per dim).
    #[arg(long, default_value_t = 1.0)]
    region_radius: f32,
    /// Hard cap on live regions per (timescale, expert).
    #[arg(long, default_value_t = 64)]
    max_regions: usize,
    /// HelixDB base URL (default: `FA_HELIX_URL`, else `helix.url` of the platform config).
    #[arg(long)]
    hub_url: Option<String>,
    /// Coalescing window for Transition edge writes to Helix: at most one write per edge per
    /// this many seconds (spec 33 `helix_flush_interval`).
    #[arg(long, default_value_t = 10)]
    hub_flush_secs: u64,
    /// Timescales whose closed-bar latents are stored as `LatentVec`s in Helix (comma separated;
    /// empty disables latent storage).
    #[arg(long, default_value = "1m,1h,1d,1w", value_delimiter = ',')]
    hub_latent_timescales: Vec<String>,
    /// Bounded queue between the pipeline and the Helix writer; overflow drops writes.
    #[arg(long, default_value_t = 4096)]
    hub_queue: usize,
    /// Do not persist anything to HelixDB.
    #[arg(long)]
    no_hub: bool,
    /// Timescales whose bar closes publish `visual.chart` / `visual.future` (empty disables).
    #[arg(long, default_value = "1m,1h,1d,1w", value_delimiter = ',')]
    visual_timescales: Vec<String>,
    /// Timescales of the `text.news` / `text.summary` outputs (empty disables the text decoder).
    #[arg(long, default_value = "1m,1h,1d,1w", value_delimiter = ',')]
    text_timescales: Vec<String>,
    /// Re-publish the text scores of every key with items in range this often (decay is visible
    /// between news arrivals).
    #[arg(long, default_value_t = 60)]
    text_refresh_secs: u64,
    /// PostgreSQL for the text decoder's reference snapshot (default: `FA_POSTGRES_URL`, else
    /// `postgres.url` of the config).
    #[arg(long)]
    pg_url: Option<String>,
}

type Counters = Arc<Mutex<BTreeMap<String, f64>>>;

/// Where the bars of one (timescale, instrument) come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BarSource {
    Trades,
    Bus,
}

impl BarSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Trades => "trades",
            Self::Bus => "bus",
        }
    }
}

struct TimescaleState {
    pipeline: TimescalePipeline,
    series: HashMap<String, BarSeries>,
    source: HashMap<String, BarSource>,
    last_preview: HashMap<String, Instant>,
    shadow: Option<shadow::Shadow>,
    /// `ts_ms` of the applied online adapter (reset by a reload).
    adapter_ms: Option<i64>,
}

fn env_ms(env: &Envelope) -> i64 {
    env.event_time
        .as_ref()
        .or(env.ingest_time.as_ref())
        .map_or_else(|| fa_proto::now().seconds * 1000, |ts| ts.seconds * 1000 + i64::from(ts.nanos) / 1_000_000)
}

fn trade_of(env: &Envelope) -> Option<(i64, f64, f64)> {
    let Some(Payload::Trade(t)) = &env.payload else {
        return None;
    };
    if env.instrument_id.is_empty() || t.price <= 0.0 || !t.price.is_finite() {
        return None;
    }
    Some((env_ms(env), t.price, t.size.max(0.0)))
}

/// `(timeframe token, timeframe ms, bar start ms, row)` of an `md.bar.<tf>.<venue>.<inst>` envelope.
/// Bars are stamped at their **close** (`event_time = open + tf`, the Yahoo connector's convention).
fn bar_of(subject: &str, env: &Envelope) -> Option<(String, i64, i64, BarRow)> {
    let Some(Payload::Bar(b)) = &env.payload else {
        return None;
    };
    if env.instrument_id.is_empty() || !(b.close > 0.0 && b.close.is_finite()) {
        return None;
    }
    let tf = if b.timeframe.is_empty() {
        subject.split('.').nth(2).unwrap_or_default().to_owned()
    } else {
        b.timeframe.clone()
    };
    let tf_ms = bars::parse_tf_ms(&tf)?;
    let start_ms = env_ms(env) - tf_ms;
    let row = BarRow::from_bus(b.open, b.high, b.low, b.close, b.volume.max(0.0), b.trade_count, b.vwap);
    Some((tf, tf_ms, start_ms, row))
}

fn now_ms() -> i64 {
    let t = fa_proto::now();
    t.seconds * 1000 + i64::from(t.nanos) / 1_000_000
}

/// Everything the main loop owns.
struct Runtime {
    bus: Bus,
    counters: Counters,
    states: Vec<TimescaleState>,
    loader: snapshot::SnapshotLoader,
    pcfg: PipelineConfig,
    sink: Option<hub_sink::HubSink>,
    region_radius: f64,
    preview: Duration,
    /// Finest bus-bar timeframe (ms) seen per instrument.
    bus_tf: HashMap<String, i64>,
    visual: Vec<Timescale>,
    text: decoders::TextDecoder,
    refs: watch::Receiver<Arc<InstrumentIndex>>,
}

impl Runtime {
    fn snapshot_of(&self, ts: Timescale) -> String {
        self.states
            .iter()
            .find(|s| s.pipeline.timescale() == ts)
            .map_or_else(|| fa_encoders::StatisticalEncoder::baseline_id(ts), |s| s.pipeline.snapshot_id().to_owned())
    }

    async fn on_trade(&mut self, subject: &str, env: &Envelope) {
        {
            let mut c = self.counters.lock().await;
            *c.entry("trades_total".into()).or_default() += 1.0;
            *c.entry(format!("trades.{subject}")).or_default() += 1.0;
        }
        let Some((t_ms, price, size)) = trade_of(env) else {
            return;
        };
        let inst = env.instrument_id.clone();
        let bus_tf = self.bus_tf.get(&inst).copied();
        for i in 0..self.states.len() {
            let st = &mut self.states[i];
            let ts = st.pipeline.timescale();
            let step_ms = ts.duration().as_millis() as i64;
            if bus_tf.is_some_and(|tf| tf <= step_ms) {
                continue; // bars of this instrument come from the bus at this timescale
            }
            let context = st.pipeline.context_steps();
            st.source.entry(inst.clone()).or_insert(BarSource::Trades);
            let series = st.series.entry(inst.clone()).or_insert_with(|| BarSeries::new(ts, context));
            let upd = series.on_trade(t_ms, price, size);
            self.after_update(i, &inst, upd, t_ms).await;
        }
    }

    async fn on_bar(&mut self, subject: &str, env: &Envelope) {
        let Some((tf, tf_ms, start_ms, row)) = bar_of(subject, env) else {
            return;
        };
        *self.counters.lock().await.entry(format!("bars_in.{tf}")).or_default() += 1.0;
        let inst = env.instrument_id.clone();
        let finest = self.bus_tf.entry(inst.clone()).or_insert(tf_ms);
        *finest = (*finest).min(tf_ms);
        let finest = *finest;
        if tf_ms != finest {
            return; // a coarser feed of an instrument that also has finer bars: resample the finer ones
        }
        for i in 0..self.states.len() {
            let st = &mut self.states[i];
            let ts = st.pipeline.timescale();
            if tf_ms > ts.duration().as_millis() as i64 {
                continue; // too coarse for this timescale: trades keep feeding it
            }
            let context = st.pipeline.context_steps();
            if st.source.insert(inst.clone(), BarSource::Bus) != Some(BarSource::Bus) {
                // switch from trade-built bars: start over so volume is not double counted
                st.series.insert(inst.clone(), BarSeries::new(ts, context));
                tracing::info!(%ts, instrument = %inst, tf, "bars now from the bus");
            }
            let series = st.series.entry(inst.clone()).or_insert_with(|| BarSeries::new(ts, context));
            let upd = series.on_bar(start_ms, tf_ms, row);
            self.after_update(i, &inst, upd, start_ms + tf_ms).await;
        }
    }

    /// Runs the pipeline (and the shadow, and the visual decoder) after a bar update.
    async fn after_update(&mut self, i: usize, inst: &str, upd: BarUpdate, t_ms: i64) {
        let now = Instant::now();
        let st = &mut self.states[i];
        let ts = st.pipeline.timescale();
        let Some(series) = st.series.get(inst) else { return };
        let (bars, as_of_ms, closed): (Vec<MarketBar>, i64, bool) = match upd {
            BarUpdate::Closed { bar_end_ms } => (series.closed(), bar_end_ms, true),
            BarUpdate::Extended => {
                let due = st.last_preview.get(inst).is_none_or(|t| now.duration_since(*t) >= self.preview);
                if !due {
                    return;
                }
                (series.with_forming(), t_ms, false)
            }
            BarUpdate::Ignored => return,
        };
        st.last_preview.insert(inst.to_owned(), now);
        let input = StepInput { instrument_id: inst, as_of_ms, bars: &bars, bar_closed: closed };
        let out = match st.pipeline.step(&input) {
            Ok(o) => o,
            Err(e) => {
                tracing::warn!(%ts, instrument = inst, error = %e, "encode failed");
                return;
            }
        };
        let token = inst.replace('-', "");
        let shadow_state = st.shadow.as_mut().and_then(|sh| sh.step(&input, out.state.as_ref()));
        let shadow_gauges = st.shadow.as_ref().map(|sh| sh.gauges(ts));
        let source = st.source.get(inst).copied().unwrap_or(BarSource::Trades);
        let snapshot_id = st.pipeline.snapshot_id().to_owned();
        if let Some(sink) = &self.sink {
            sink.observe(ts, closed, &out, self.region_radius);
        }
        publish_step(&self.bus, &self.counters, ts, inst, closed, out, &self.states[i].pipeline).await;
        if let Some(s) = shadow_state
            && let Err(e) = self.bus.publish(shadow::shadow_subject(ts, &token), &s).await
        {
            tracing::warn!(error = %e, "shadow state publish failed");
        }
        if let Some(g) = shadow_gauges {
            let mut c = self.counters.lock().await;
            c.extend(g);
        }
        if closed
            && self.visual.contains(&ts)
            && let Some(outs) = decoders::visual_outputs(inst, ts, &snapshot_id, as_of_ms, &bars, source.as_str())
        {
            self.publish_outputs(&outs).await;
        }
    }

    async fn publish_outputs(&self, outs: &[DecoderOutput]) {
        let mut c = self.counters.lock().await;
        for o in outs {
            if let Err(e) = self.bus.publish(subjects::decoder_output(&o.family), o).await {
                tracing::warn!(error = %e, family = %o.family, "decoder output publish failed");
                continue;
            }
            *c.entry(format!("decoder_outputs.{}", o.family)).or_default() += 1.0;
        }
    }

    /// One `news.item.<source>` envelope; `publish = false` while replaying the stream backlog.
    async fn on_news(&mut self, subject: &str, env: &Envelope, publish: bool) {
        let Some((dedup, doc)) = decoders::news_doc(subject, env) else {
            return;
        };
        let matched = self.refs.borrow().match_doc(&doc);
        let now = now_ms();
        let Some(keys) = self.text.ingest(dedup, &doc, &matched, now) else {
            *self.counters.lock().await.entry("news.duplicates".into()).or_default() += 1.0;
            return;
        };
        {
            let mut c = self.counters.lock().await;
            *c.entry("news.items".into()).or_default() += 1.0;
            *c.entry(if matched.is_empty() { "news.market".into() } else { "news.matched".to_owned() }).or_default() +=
                1.0;
        }
        if publish {
            self.publish_text(&keys, now).await;
        }
    }

    async fn publish_text(&self, keys: &[String], now: i64) {
        let mut outs = Vec::new();
        for key in keys {
            for &ts in self.text.timescales() {
                if let Some(s) = self.text.score(key, ts, now) {
                    outs.extend(decoders::text_outputs(key, ts, &self.snapshot_of(ts), now, &s));
                }
            }
        }
        self.publish_outputs(&outs).await;
    }

    async fn text_tick(&mut self) {
        let now = now_ms();
        self.text.prune(now);
        let keys: Vec<String> = self.text.keys().into_iter().collect();
        self.publish_text(&keys, now).await;
        let mut c = self.counters.lock().await;
        c.insert("news.keys".into(), keys.len() as f64);
        c.insert("ref_instruments".into(), self.refs.borrow().len() as f64);
    }

    /// Gauges that change without steps (shadow minutes, bar sources).
    async fn gauge_tick(&self) {
        let mut c = self.counters.lock().await;
        for st in &self.states {
            let ts = st.pipeline.timescale();
            if let Some(sh) = &st.shadow {
                c.extend(sh.gauges(ts));
            }
            for src in [BarSource::Bus, BarSource::Trades] {
                let n = st.source.values().filter(|s| **s == src).count();
                c.insert(format!("bars_source.{ts}.{}", src.as_str()), n as f64);
            }
        }
    }

    async fn on_ctl(&mut self, cmd: CtlCommand) {
        let ts = cmd.timescale;
        let reply = match cmd.verb.clone() {
            CtlVerb::Reload { snapshot_id } => self.reload(ts, snapshot_id).await,
            CtlVerb::Shadow { snapshot_id } => self.shadow(ts, snapshot_id).await,
            CtlVerb::Adapter => self.adapter(ts, None).await,
        };
        let _ = cmd.reply.send(reply);
    }

    /// `ctl.model.reload`: resolve the snapshot, swap the encoder with kernel transfer, re-announce
    /// the snapshot and its seeded regions (bus + Helix), answer with what is now serving.
    async fn reload(&mut self, ts: Timescale, snapshot_id: Option<String>) -> CtlReply {
        let Some(i) = self.states.iter().position(|s| s.pipeline.timescale() == ts) else {
            return CtlReply { ok: false, message: format!("timescale {ts} not served") };
        };
        let loaded = match self.loader.resolve(ts, snapshot_id.as_deref()).await {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(%ts, snapshot = ?snapshot_id, error = %e, "reload refused");
                return CtlReply { ok: false, message: e.to_string() };
            }
        };
        let st = &mut self.states[i];
        let previous = st.pipeline.snapshot_id().to_owned();
        if previous == loaded.desc.snapshot_id {
            return CtlReply { ok: true, message: format!("unchanged {previous}") };
        }
        let seeded = loaded.region_map.as_ref().map_or(0, fa_patterns::RegionMap::n_regions);
        let carried = loaded.region_map.as_ref().map_or(0.0, fa_patterns::RegionMap::carried_mass);
        let report = st.pipeline.swap_encoder(&self.pcfg, loaded.encoder, loaded.region_map.as_ref());
        st.adapter_ms = None;
        tracing::info!(%ts, from = %previous, to = %loaded.desc.snapshot_id, generation = loaded.desc.generation, source = loaded.source, seeded_regions = seeded, carried_mass = carried, kernel_carried = report.carried, kernel_dropped = report.dropped, kernel_rows = report.rows_carried, "encoder swapped");
        if let Some(sink) = &self.sink {
            sink.announce_snapshot(loaded.desc.clone());
        }
        let out = pipeline::StepOutput { region_events: report.events, ..Default::default() };
        if let Some(sink) = &self.sink {
            sink.observe(ts, false, &out, self.region_radius);
        }
        publish_step(&self.bus, &self.counters, ts, "", false, out, &self.states[i].pipeline).await;
        {
            let mut c = self.counters.lock().await;
            *c.entry("reloads".into()).or_default() += 1.0;
            *c.entry(format!("reloads.{ts}")).or_default() += 1.0;
            *c.entry(format!("kernel_transfer.{ts}.carried")).or_default() += report.carried;
            c.insert(format!("kernel_transfer.{ts}.rows"), report.rows_carried as f64);
            c.insert(format!("kernel_transfer.{ts}.dropped"), report.dropped);
            c.remove(&format!("adapter.{ts}.ts_ms"));
        }
        CtlReply {
            ok: true,
            message: format!(
                "{ts}: {previous} -> {} (gen {}, {}, {seeded} seeded regions, carried mass {carried:.3}, kernel carried {:.3} over {} rows)",
                loaded.desc.snapshot_id, loaded.desc.generation, loaded.source, report.carried, report.rows_carried
            ),
        }
    }

    /// `ctl.model.shadow`: start (or replace, or stop) the shadow of a timescale. Reply JSON.
    async fn shadow(&mut self, ts: Timescale, snapshot_id: Option<String>) -> CtlReply {
        let Some(i) = self.states.iter().position(|s| s.pipeline.timescale() == ts) else {
            return CtlReply {
                ok: false,
                message: json!({"ok": false, "error": format!("timescale {ts} not served")}).to_string(),
            };
        };
        let active = self.states[i].pipeline.snapshot_id().to_owned();
        let previous = self.states[i].shadow.as_ref().map(|s| s.snapshot_id.clone());
        self.counters.lock().await.retain(|k, _| !k.starts_with(&shadow::gauge_prefix(ts)));
        let Some(id) = snapshot_id else {
            self.states[i].shadow = None;
            tracing::info!(%ts, stopped = ?previous, "shadow stopped");
            return CtlReply {
                ok: true,
                message: json!({"ok": true, "timescale": ts, "active": active, "shadow": null, "stopped": previous})
                    .to_string(),
            };
        };
        let loaded = match self.loader.resolve(ts, Some(&id)).await {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(%ts, snapshot = %id, error = %e, "shadow refused");
                return CtlReply { ok: false, message: json!({"ok": false, "error": e.to_string()}).to_string() };
            }
        };
        let seeded = loaded.region_map.as_ref().map_or(0, fa_patterns::RegionMap::n_regions);
        let pl = TimescalePipeline::with_encoder(ts, &self.pcfg, loaded.encoder, loaded.region_map.as_ref());
        let sh = shadow::Shadow::new(pl, loaded.desc.generation, loaded.region_map.as_ref());
        let reply = json!({
            "ok": true, "timescale": ts, "active": active, "shadow": sh.snapshot_id, "generation": sh.generation,
            "source": loaded.source, "seeded_regions": seeded, "carried_mass": sh.carried_mass, "replaced": previous,
            "subject": format!("latent.shadow.{ts}.<instrument>"),
        });
        tracing::info!(%ts, active = %active, shadow = %sh.snapshot_id, generation = sh.generation, seeded_regions = seeded, "shadow started");
        self.counters.lock().await.extend(sh.gauges(ts));
        self.states[i].shadow = Some(sh);
        CtlReply { ok: true, message: reply.to_string() }
    }

    /// Applies the online adapter of the active snapshot (`artifact_key` from a job result, else
    /// the newest artifact).
    async fn adapter(&mut self, ts: Timescale, artifact_key: Option<&str>) -> CtlReply {
        let Some(i) = self.states.iter().position(|s| s.pipeline.timescale() == ts) else {
            return CtlReply { ok: false, message: format!("timescale {ts} not served") };
        };
        let snapshot_id = self.states[i].pipeline.snapshot_id().to_owned();
        let a = match self.loader.adapter(ts, &snapshot_id, artifact_key).await {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(%ts, snapshot = %snapshot_id, error = %e, "adapter refused");
                return CtlReply { ok: false, message: e.to_string() };
            }
        };
        let st = &mut self.states[i];
        if st.adapter_ms == Some(a.ts_ms) {
            return CtlReply { ok: true, message: format!("{ts}: adapter {} already applied", a.key) };
        }
        if let Err(e) = st.pipeline.set_encoder(a.encoder) {
            return CtlReply { ok: false, message: e };
        }
        st.adapter_ms = Some(a.ts_ms);
        tracing::info!(%ts, snapshot = %snapshot_id, key = %a.key, drift = ?a.drift, "online adapter applied");
        {
            let mut c = self.counters.lock().await;
            *c.entry(format!("adapters_applied.{ts}")).or_default() += 1.0;
            c.insert(format!("adapter.{ts}.ts_ms"), a.ts_ms as f64);
        }
        CtlReply { ok: true, message: format!("{ts}: {snapshot_id} + adapter {} (drift {:?})", a.key, a.drift) }
    }

    /// `jobs.result.train.>`: an OK `train.adapter` result for a snapshot that is serving applies it.
    async fn on_job_result(&mut self, r: &JobResult) {
        if r.status != fa_proto::v1::job_result::Status::Ok as i32 {
            return;
        }
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&r.result) else { return };
        if v.get("kind").and_then(|k| k.as_str()) != Some("train.adapter") {
            return;
        }
        let Some(snapshot_id) = v.get("snapshot_id").and_then(|s| s.as_str()) else { return };
        let Some(ts) =
            self.states.iter().find(|s| s.pipeline.snapshot_id() == snapshot_id).map(|s| s.pipeline.timescale())
        else {
            tracing::debug!(snapshot = snapshot_id, "adapter result for a snapshot that is not serving");
            return;
        };
        let key = v.get("artifact_key").and_then(|k| k.as_str()).map(str::to_owned);
        let reply = self.adapter(ts, key.as_deref()).await;
        tracing::info!(%ts, job = %r.job_id, ok = reply.ok, message = %reply.message, "adapter job result handled");
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();
    let args = Args::parse();
    fa_bus::metrics::serve_from_env("model-runtime", env!("CARGO_PKG_VERSION")); // GET /metrics on FA_METRICS_BIND (spec 60 §4, D-056)
    let cfg = PlatformConfig::load(&args.config)?;
    let bus = Bus::connect(&cfg.nats.url).await?;

    let pcfg = PipelineConfig {
        regions: OnlineRegionCfg {
            radius: args.region_radius,
            merge_radius: args.region_radius * 0.35,
            max_regions: args.max_regions,
            ..OnlineRegionCfg::default()
        },
        ..PipelineConfig::default()
    };
    // Active snapshot per timescale from Helix + Garage, else the in-process baseline (spec 36).
    let hub_url = args.hub_url.clone().unwrap_or_else(|| fa_hub::HubClient::url_from_env_or(&cfg.helix.url));
    let loader = snapshot::SnapshotLoader::new(&cfg, (!args.no_hub).then(|| hub_url.clone()));
    let mut states: Vec<TimescaleState> = Vec::with_capacity(cfg.timescales.len());
    let mut descs = Vec::with_capacity(cfg.timescales.len());
    for &ts in &cfg.timescales {
        let loaded = match loader.resolve(ts, None).await {
            Ok(l) => l,
            Err(e) => snapshot::SnapshotLoader::baseline(ts, Some(format!("active snapshot unusable: {e}"))),
        };
        tracing::info!(%ts, snapshot = %loaded.desc.snapshot_id, generation = loaded.desc.generation, source = loaded.source, note = ?loaded.note, seeded_regions = loaded.region_map.as_ref().map_or(0, fa_patterns::RegionMap::n_regions), "encoder resolved");
        states.push(TimescaleState {
            pipeline: TimescalePipeline::with_encoder(ts, &pcfg, loaded.encoder, loaded.region_map.as_ref()),
            series: HashMap::new(),
            source: HashMap::new(),
            last_preview: HashMap::new(),
            shadow: None,
            adapter_ms: None,
        });
        descs.push(loaded.desc);
    }
    let parse_ts = |v: &[String]| -> Vec<Timescale> { v.iter().filter_map(|t| t.trim().parse().ok()).collect() };
    tracing::info!(
        timescales = ?cfg.timescales,
        snapshots = ?states.iter().map(|s| s.pipeline.snapshot_id().to_owned()).collect::<Vec<_>>(),
        region_radius = args.region_radius,
        visual = ?parse_ts(&args.visual_timescales),
        text = ?parse_ts(&args.text_timescales),
        "model-runtime starting (statistical encoders, baseline text/visual decoders, no neural models)"
    );

    let counters: Counters = Arc::default();
    let instance_id = ulid::Ulid::generate().to_string();
    spawn_heartbeat(bus.clone(), counters.clone(), instance_id, args.heartbeat_secs);
    let sink = (!args.no_hub).then(|| {
        let sink = hub_sink::HubSink::spawn(
            hub_sink::SinkConfig {
                url: hub_url.clone(),
                flush: Duration::from_secs(args.hub_flush_secs.max(1)),
                latent_timescales: args.hub_latent_timescales.iter().filter_map(|t| t.trim().parse().ok()).collect(),
                queue: args.hub_queue,
                region_method: "leader".into(),
            },
            counters.clone(),
        );
        for d in &descs {
            sink.announce_snapshot(d.clone());
        }
        sink
    });
    let region_radius = f64::from(args.region_radius);
    // Seeded regions of loaded snapshots are announced like births so the bus and Helix know them.
    for st in &states {
        let events = st.pipeline.seeded_region_events();
        if !events.is_empty() {
            let out = pipeline::StepOutput { region_events: events, ..Default::default() };
            if let Some(sink) = &sink {
                sink.observe(st.pipeline.timescale(), false, &out, region_radius);
            }
            publish_step(&bus, &counters, st.pipeline.timescale(), "", false, out, &st.pipeline).await;
        }
    }

    let (ctl_tx, mut ctl_rx) = mpsc::channel::<CtlCommand>(8);
    tokio::spawn(snapshot::run_ctl_listener(bus.clone(), ctl_tx));

    // Text decoder: reference snapshot (Postgres) + news (JetStream backlog of `TEXT`, then live).
    let text_ts = parse_ts(&args.text_timescales);
    let (refs_tx, refs_rx) = watch::channel(Arc::new(InstrumentIndex::default()));
    let (news_tx, mut news_rx) = mpsc::channel::<(String, Envelope, bool)>(1024);
    if !text_ts.is_empty() {
        let pg_url = args
            .pg_url
            .clone()
            .or_else(|| std::env::var("FA_POSTGRES_URL").ok())
            .unwrap_or_else(|| cfg.postgres.url.clone());
        decoders::spawn_ref_refresh(pg_url, refs_tx);
        tokio::spawn(run_news(bus.clone(), news_tx, refs_rx.clone()));
    }

    let mut rt = Runtime {
        bus: bus.clone(),
        counters: counters.clone(),
        states,
        loader,
        pcfg,
        sink,
        region_radius,
        preview: Duration::from_secs(args.preview_secs),
        bus_tf: HashMap::new(),
        visual: parse_ts(&args.visual_timescales),
        text: decoders::TextDecoder::new(text_ts),
        refs: refs_rx,
    };

    let mut trades = bus.subscribe::<Envelope>(subjects::ALL_MD_TRADES.into()).await?;
    let mut bars_sub = bus.subscribe::<Envelope>("md.bar.>".into()).await?;
    let mut jobs = bus.subscribe::<JobResult>("jobs.result.train.>".into()).await?;
    let mut text_tick = tokio::time::interval(Duration::from_secs(args.text_refresh_secs.max(5)));
    let mut gauge_tick = tokio::time::interval(Duration::from_secs(args.heartbeat_secs.max(1)));
    loop {
        tokio::select! {
            msg = trades.next() => match msg {
                Some((subject, env)) => rt.on_trade(&subject, &env).await,
                None => break,
            },
            Some((subject, env)) = bars_sub.next() => rt.on_bar(&subject, &env).await,
            Some((subject, env, publish)) = news_rx.recv() => {
                rt.on_news(&subject, &env, publish).await;
                if subject.is_empty() {
                    // end of the backlog replay: publish every key once
                    rt.text_tick().await;
                }
            }
            Some((_, r)) = jobs.next() => rt.on_job_result(&r).await,
            Some(cmd) = ctl_rx.recv() => rt.on_ctl(cmd).await,
            _ = text_tick.tick() => if !rt.text.timescales().is_empty() { rt.text_tick().await },
            _ = gauge_tick.tick() => rt.gauge_tick().await,
        }
    }
    Ok(())
}

/// Feeds `news.item.>` into the loop: first the `TEXT` stream's retained items (ordered
/// consumer, deliver all; published = false so the backlog does not flood the bus), then live.
/// A marker with an empty subject ends the backlog. Without JetStream: core subscription. Waits
/// (at most 15 s) for the first reference snapshot so the backlog is matched to instruments.
async fn run_news(
    bus: Bus,
    tx: mpsc::Sender<(String, Envelope, bool)>,
    mut refs: watch::Receiver<Arc<InstrumentIndex>>,
) {
    use async_nats::jetstream::consumer::{DeliverPolicy, pull::OrderedConfig};
    // Map the backlog with the reference snapshot, not the empty index (Postgres down → go on).
    if tokio::time::timeout(Duration::from_secs(15), refs.changed()).await.is_err() {
        tracing::warn!("no reference snapshot after 15 s; replaying news without instrument matching");
    }
    let filter = "news.item.>".to_owned();
    let js = bus.jetstream();
    let ordered = async {
        let stream = js.get_stream("TEXT").await?;
        let consumer = stream
            .create_consumer(OrderedConfig {
                filter_subject: filter.clone(),
                deliver_policy: DeliverPolicy::All,
                ..Default::default()
            })
            .await?;
        anyhow::Ok(consumer.messages().await?)
    }
    .await;
    match ordered {
        Ok(mut msgs) => {
            let mut backlog = true;
            let mut replayed = 0u64;
            while let Some(m) = msgs.next().await {
                let Ok(m) = m else { continue };
                let pending = m.info().map(|i| i.pending).unwrap_or(0);
                if let Ok(env) = <Envelope as fa_proto::Message>::decode(m.payload.as_ref()) {
                    if tx.send((m.subject.to_string(), env, !backlog)).await.is_err() {
                        return;
                    }
                    replayed += u64::from(backlog);
                }
                if backlog && pending == 0 {
                    backlog = false;
                    tracing::info!(replayed, "news backlog replayed from the TEXT stream; live from here");
                    let _ = tx.send((String::new(), Envelope::default(), false)).await;
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "TEXT stream unavailable; news from the live subscription only");
            let Ok(mut sub) = bus.subscribe::<Envelope>(filter).await else {
                tracing::error!("cannot subscribe to news.item.>");
                return;
            };
            while let Some((subject, env)) = sub.next().await {
                if tx.send((subject, env, true)).await.is_err() {
                    return;
                }
            }
        }
    }
}

async fn publish_step(
    bus: &Bus,
    counters: &Counters,
    ts: Timescale,
    inst: &str,
    closed: bool,
    out: pipeline::StepOutput,
    pl: &TimescalePipeline,
) {
    let mut published = BTreeMap::<&str, f64>::new();
    for ev in &out.region_events {
        if let Err(e) = bus.publish(subjects::latent_region(ts), ev).await {
            tracing::warn!(error = %e, "region event publish failed");
        }
        *published.entry("region_events").or_default() += 1.0;
    }
    for tr in &out.transitions {
        if let Err(e) = bus.publish(subjects::latent_transition(ts), tr).await {
            tracing::warn!(error = %e, "transition publish failed");
        }
        *published.entry("transitions_published").or_default() += 1.0;
    }
    if let Some(state) = &out.state {
        // Subject per instrument: `latent.state.<ts>.<instrument token>` (D-017 token).
        let token = inst.replace('-', "");
        if let Err(e) = bus.publish(subjects::latent_state(ts, &token), state).await {
            tracing::warn!(error = %e, "latent state publish failed");
        }
        *published.entry("states_published").or_default() += 1.0;
    }
    let mut c = counters.lock().await;
    for (k, v) in published {
        *c.entry(k.to_owned()).or_default() += v;
        *c.entry(format!("{k}.{ts}")).or_default() += v;
    }
    if closed {
        *c.entry(format!("bars_closed.{ts}")).or_default() += 1.0;
    }
    c.insert(format!("regions_live.{ts}"), pl.live_regions() as f64);
    c.insert(format!("transition_rows.{ts}"), pl.transition_rows() as f64);
}

fn spawn_heartbeat(bus: Bus, counters: Counters, instance_id: String, every_secs: u64) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(every_secs));
        loop {
            tick.tick().await;
            let hb = Heartbeat {
                service: "model-runtime".into(),
                instance_id: instance_id.clone(),
                at: Some(fa_proto::now()),
                version: env!("CARGO_PKG_VERSION").into(),
                counters: counters.lock().await.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            };
            if let Err(e) = bus.publish(subjects::heartbeat("model-runtime"), &hb).await {
                tracing::warn!(error = %e, "heartbeat publish failed");
            }
        }
    });
}
