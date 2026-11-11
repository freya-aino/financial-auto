//! Persisting the live pipeline into HelixDB (spec 30 §4, spec 33 §6 `helix_flush_interval`,
//! spec 20 §5.2, D-048).
//!
//! The pipeline never talks to Helix: [`HubSink::observe`] turns a [`StepOutput`] into
//! [`HubEvent`]s and `try_send`s them into a **bounded** channel; a single background task
//! drains it and writes through `fa_hub`. Bus publishing is therefore never blocked or slowed by
//! Helix. When Helix is unreachable the task logs once, drops writes (counted in
//! `hub.dropped`) and retries connecting every [`RECONNECT_EVERY`].
//!
//! What is written
//! - the `Snapshot` (`stat-v1-<ts>`) and its `ExpertSpace`s once per process (get-or-create);
//! - a `LatentRegion` per region birth (upsert: the baseline's ids are process-local, D-039,
//!   so a restart revives rows of the same id), `status = merged | retired` on those events;
//! - `Transition` edges with posterior mean / interval / observation count, **coalesced** to at
//!   most one write per edge per `flush` ([`Coalescer`]), and only for successors that this
//!   process has observed at least once (prior-only mass stays in the in-memory view, D-032);
//! - one `LatentVec` (`kind = observation`, `ref_id = region_id`) per expert per **closed bar**
//!   for the configured timescales (default 1m and slower), after `ensure_latent_index_ready`;
//! - region lifecycle (spec 30 §5): `DescendsFrom` edges child → parent for online splits and
//!   merges and for a loaded snapshot's alignment, death marks (`died_at`, `retire_reason`), the
//!   centroid `LatentVec` (`kind = centroid`) of every born / split / carried region;
//! - every `--hub-state-secs`: the `RegionSet` (serialised region set + kernel, restored on
//!   restart), region statistics and moved centroids;
//! - `KernelModel` per expert space on every `KernelUpdate`, `Hier` edges on every hierarchy
//!   update, `CrossPrior` rows on every cross-asset update (spec 33 §5, §6.5).

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    hash::Hash,
    sync::Arc,
    time::{Duration, Instant},
};

use fa_core::Timescale;
use fa_hub::{
    HubClient, HubError, NewCrossPrior, NewDescent, NewExpertSpace, NewHierEdge, NewKernelModel, NewLatent, NewRegion,
    NewRegionSet, NewSnapshot, NewTransition, RegionStatsUpdate, TransitionUpsert,
};
use fa_proto::v1::{
    CrossAssetPriors, ExpertLatent, ExpertSpaceRef, KernelUpdate, RegionEvent, RegionHierarchy, RegionLifecycleEvent,
    TransitionUpdate, region_event, region_lifecycle_event,
};
use tokio::sync::{Mutex, mpsc};

use crate::{
    pipeline::{ExpertPersist, StepOutput},
    region_method::RegionMethod,
};

pub type Counters = Arc<Mutex<BTreeMap<String, f64>>>;

/// How long to wait before probing an unreachable server again.
pub const RECONNECT_EVERY: Duration = Duration::from_secs(30);
/// Per-request timeout of the sink's client; a stalled Helix must not stall the drain task.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const INDEX_WAIT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone)]
pub struct SinkConfig {
    pub url: String,
    /// Coalescing window for `Transition` edge writes (`--hub-flush-secs`).
    pub flush: Duration,
    /// Timescales whose closed-bar latents are stored as `LatentVec`s.
    pub latent_timescales: BTreeSet<Timescale>,
    /// Channel capacity; when full, events are dropped (`hub.dropped`).
    pub queue: usize,
}

/// A pipeline's snapshot as the sink needs it to (re)create `Snapshot` + `ExpertSpace` rows.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotDesc {
    pub snapshot_id: String,
    pub timescale: Timescale,
    pub n_experts: u32,
    pub dim: u32,
    /// Spec 36 lineage fields (`ensure_snapshot` is get-or-create, so an existing row keeps its own).
    pub generation: u32,
    pub garage_key: String,
    pub status: String,
    /// Region discovery of the snapshot's family and its online form: the `method` labels of its
    /// `RegionSet`s and `Region`s (D-080; follows reloads because it travels with the snapshot).
    pub region_method: RegionMethod,
}

#[derive(Debug, Clone)]
pub enum HubEvent {
    Snapshot(SnapshotDesc),
    Region {
        event: RegionEvent,
        radius: f64,
    },
    /// A published posterior row plus the successor actually observed at this step.
    Transition {
        update: TransitionUpdate,
        observed_to: Option<String>,
    },
    /// Closed-bar latents of one instrument (one `LatentVec` per expert).
    Latent {
        space: ExpertSpaceRef,
        instrument_id: String,
        ts_ms: i64,
        experts: Vec<ExpertLatent>,
    },
    Lifecycle(RegionLifecycleEvent),
    Kernel(KernelUpdate),
    /// Region sets, kernels and statistics of one timescale's experts.
    Persist(Vec<ExpertPersist>),
    Hierarchy(RegionHierarchy),
    CrossPriors(CrossAssetPriors),
}

/// Expert-space business id: `<snapshot_id>-e<k>` (what `LatentVec.expert_space_id` partitions on).
pub fn expert_space_id(snapshot_id: &str, expert_index: u32) -> String {
    format!("{snapshot_id}-e{expert_index}")
}

/// Per-key write throttle: the first value for a key goes out immediately, later values within
/// `interval` are held (latest wins) and released by [`Coalescer::due`]. Pure; no clock inside.
#[derive(Debug)]
pub struct Coalescer<K, V> {
    interval: Duration,
    last_write: HashMap<K, Instant>,
    pending: HashMap<K, V>,
}

impl<K: Eq + Hash + Clone, V> Coalescer<K, V> {
    pub fn new(interval: Duration) -> Self {
        Self { interval, last_write: HashMap::new(), pending: HashMap::new() }
    }

    /// `Some(value)` when it should be written now (and the key's window starts at `now`);
    /// `None` when it was parked for the next [`Self::due`].
    pub fn offer(&mut self, key: K, value: V, now: Instant) -> Option<V> {
        let open = self.last_write.get(&key).is_none_or(|t| now.duration_since(*t) >= self.interval);
        if open {
            self.last_write.insert(key.clone(), now);
            self.pending.remove(&key);
            Some(value)
        } else {
            self.pending.insert(key, value);
            None
        }
    }

    /// Parked values whose window has elapsed; their windows restart at `now`.
    pub fn due(&mut self, now: Instant) -> Vec<(K, V)> {
        let keys: Vec<K> = self
            .pending
            .keys()
            .filter(|k| self.last_write.get(*k).is_none_or(|t| now.duration_since(*t) >= self.interval))
            .cloned()
            .collect();
        keys.into_iter()
            .filter_map(|k| {
                let v = self.pending.remove(&k)?;
                self.last_write.insert(k.clone(), now);
                Some((k, v))
            })
            .collect()
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Forget keys not written for a long time (retired regions) so the maps stay bounded.
    pub fn prune(&mut self, now: Instant, older_than: Duration) {
        self.last_write.retain(|k, t| self.pending.contains_key(k) || now.duration_since(*t) < older_than);
    }
}

/// Handle held by the pipeline side.
#[derive(Clone)]
pub struct HubSink {
    tx: mpsc::Sender<HubEvent>,
    counters: Counters,
    latent_timescales: BTreeSet<Timescale>,
}

impl HubSink {
    /// Start the drain task. `counters` receives `hub.writes`, `hub.errors`, `hub.dropped`,
    /// `hub.coalesced`, `hub.pending` and `hub.skipped_zero_vec`.
    pub fn spawn(cfg: SinkConfig, counters: Counters) -> Self {
        let (tx, rx) = mpsc::channel(cfg.queue.max(1));
        let latent_timescales = cfg.latent_timescales.clone();
        tokio::spawn(Drain::new(cfg, counters.clone()).run(rx));
        Self { tx, counters, latent_timescales }
    }

    /// Enqueue without waiting; a full queue drops the event (`hub.dropped`).
    pub fn send(&self, ev: HubEvent) {
        if self.tx.try_send(ev).is_err() {
            let c = self.counters.clone();
            tokio::spawn(async move { bump(&c, "hub.dropped", 1.0).await });
        }
    }

    pub fn announce_snapshot(&self, desc: SnapshotDesc) {
        self.send(HubEvent::Snapshot(desc));
    }

    /// Turn one pipeline step into hub events. Called after the bus publishes.
    pub fn observe(&self, ts: Timescale, bar_closed: bool, out: &StepOutput, region_radius: f64) {
        for ev in &out.region_events {
            self.send(HubEvent::Region { event: ev.clone(), radius: region_radius });
        }
        // after the region rows exist: lineage edges, death marks, centroids
        for ev in &out.lifecycle {
            self.send(HubEvent::Lifecycle(ev.clone()));
        }
        for k in &out.kernels {
            self.send(HubEvent::Kernel(k.clone()));
        }
        if !bar_closed {
            return;
        }
        let state = out.state.as_ref();
        for tr in &out.transitions {
            // The observed successor of expert k is the region the state landed in at this step.
            let observed_to = tr.space.as_ref().and_then(|sp| {
                state?.experts.iter().find(|e| e.expert_index == sp.expert_index).map(|e| e.region_id.clone())
            });
            self.send(HubEvent::Transition { update: tr.clone(), observed_to });
        }
        if let Some(st) = state.filter(|_| self.latent_timescales.contains(&ts)) {
            let ts_ms = st.as_of.as_ref().map_or(0, |t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000);
            self.send(HubEvent::Latent {
                space: ExpertSpaceRef {
                    snapshot_id: st.snapshot_id.clone(),
                    timescale: st.timescale.clone(),
                    expert_index: 0,
                },
                instrument_id: st.instrument_id.clone(),
                ts_ms,
                experts: st.experts.clone(),
            });
        }
    }
}

async fn bump(counters: &Counters, key: &str, by: f64) {
    *counters.lock().await.entry(key.to_owned()).or_default() += by;
}

type EdgeKey = (String, String, String); // from, to, sampler

struct Drain {
    hub: HubClient,
    cfg: SinkConfig,
    counters: Counters,
    /// Schema ensured and the last request succeeded.
    ready: bool,
    warned: bool,
    next_probe: Instant,
    snapshots: BTreeMap<String, SnapshotDesc>,
    /// `expert_space_id` → Helix node id.
    spaces: HashMap<String, u64>,
    latent_dims: HashSet<usize>,
    coalescer: Coalescer<EdgeKey, TransitionUpsert>,
    /// Observations per edge since process start (what `Transition.count` holds).
    counts: HashMap<EdgeKey, i64>,
    /// Last centroid written per region (only moved centroids are rewritten).
    centroids: HashMap<String, Vec<f32>>,
}

/// A centroid is rewritten when it moved farther than this (latent units).
const CENTROID_MOVE: f32 = 0.05;

impl Drain {
    fn new(cfg: SinkConfig, counters: Counters) -> Self {
        Self {
            hub: HubClient::with_timeout(cfg.url.clone(), REQUEST_TIMEOUT),
            coalescer: Coalescer::new(cfg.flush),
            cfg,
            counters,
            ready: false,
            warned: false,
            next_probe: Instant::now(),
            snapshots: BTreeMap::new(),
            spaces: HashMap::new(),
            latent_dims: HashSet::new(),
            counts: HashMap::new(),
            centroids: HashMap::new(),
        }
    }

    async fn run(mut self, mut rx: mpsc::Receiver<HubEvent>) {
        tracing::info!(url = %self.cfg.url, flush_secs = self.cfg.flush.as_secs(), latent_timescales = ?self.cfg.latent_timescales, "hub sink started");
        let mut tick = tokio::time::interval(Duration::from_secs(1).min(self.cfg.flush));
        loop {
            tokio::select! {
                ev = rx.recv() => match ev {
                    Some(ev) => self.handle(ev).await,
                    None => break,
                },
                _ = tick.tick() => self.flush_due().await,
            }
        }
        tracing::info!("hub sink stopped");
    }

    /// Make sure the server is reachable and the schema exists; drops writes while it is not.
    async fn connected(&mut self) -> bool {
        if self.ready {
            return true;
        }
        let now = Instant::now();
        if now < self.next_probe {
            return false;
        }
        self.next_probe = now + RECONNECT_EVERY;
        match self.hub.ensure_schema_ready(INDEX_WAIT).await {
            Ok(()) => {
                self.ready = true;
                self.warned = false;
                self.latent_dims.clear();
                self.spaces.clear();
                tracing::info!(url = %self.cfg.url, "helix reachable; persisting snapshots, regions, transitions and latents");
                let descs: Vec<SnapshotDesc> = self.snapshots.values().cloned().collect();
                for d in descs {
                    self.ensure_snapshot(&d).await;
                }
                self.ready
            }
            Err(e) => {
                self.fail(&e, "helix unreachable at start; dropping hub writes until it answers").await;
                false
            }
        }
    }

    async fn fail(&mut self, e: &HubError, what: &str) {
        bump(&self.counters, "hub.errors", 1.0).await;
        if e.is_unreachable() {
            self.ready = false;
            self.next_probe = Instant::now() + RECONNECT_EVERY;
            if !self.warned {
                self.warned = true;
                tracing::warn!(url = %self.cfg.url, error = %e, "{what}");
            }
        } else {
            // Server answered with an error: the connection is fine, the write is not.
            tracing::debug!(error = %e, "{what}");
        }
    }

    async fn handle(&mut self, ev: HubEvent) {
        match ev {
            HubEvent::Snapshot(desc) => {
                self.snapshots.insert(desc.snapshot_id.clone(), desc.clone());
                if self.connected().await {
                    self.ensure_snapshot(&desc).await;
                }
            }
            HubEvent::Region { event, radius } => {
                if !self.connected().await {
                    bump(&self.counters, "hub.dropped", 1.0).await;
                    return;
                }
                self.region(event, radius).await;
            }
            HubEvent::Transition { update, observed_to } => self.transition(update, observed_to).await,
            HubEvent::Latent { space, instrument_id, ts_ms, experts } => {
                if !self.connected().await {
                    bump(&self.counters, "hub.dropped", 1.0).await;
                    return;
                }
                self.latents(&space.snapshot_id, &instrument_id, ts_ms, &experts).await;
            }
            other => {
                if !self.connected().await {
                    bump(&self.counters, "hub.dropped", 1.0).await;
                    return;
                }
                let res = match other {
                    HubEvent::Lifecycle(ev) => self.lifecycle(ev).await,
                    HubEvent::Kernel(k) => self.kernel(k).await,
                    HubEvent::Persist(rows) => self.persist(rows).await,
                    HubEvent::Hierarchy(h) => self.hierarchy(h).await,
                    HubEvent::CrossPriors(c) => self.cross_priors(c).await,
                    _ => Ok(()),
                };
                match res {
                    Ok(()) => bump(&self.counters, "hub.writes", 1.0).await,
                    Err(e) => self.fail(&e, "region/graph write failed").await,
                }
            }
        }
    }

    async fn centroid(&mut self, space: &ExpertSpaceRef, region_id: &str, c: &[f32]) -> Result<(), HubError> {
        if c.is_empty() {
            return Ok(());
        }
        let moved = self.centroids.get(region_id).is_none_or(|old| {
            old.len() != c.len()
                || old.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt() > CENTROID_MOVE
        });
        if !moved {
            return Ok(());
        }
        if !self.latent_dims.contains(&c.len()) {
            self.hub.ensure_latent_index_ready(c.len(), INDEX_WAIT).await?;
            self.latent_dims.insert(c.len());
        }
        let space_id = expert_space_id(&space.snapshot_id, space.expert_index);
        if self.hub.upsert_centroid(&space_id, region_id, c).await? {
            self.centroids.insert(region_id.to_owned(), c.to_vec());
        }
        Ok(())
    }

    async fn lifecycle(&mut self, ev: RegionLifecycleEvent) -> Result<(), HubError> {
        use region_lifecycle_event::Kind;
        let Some(space) = ev.space.clone() else { return Ok(()) };
        match Kind::try_from(ev.kind).unwrap_or(Kind::Unspecified) {
            Kind::Died => {
                self.centroids.remove(&ev.region_id);
                match self.hub.mark_region_died(&ev.region_id, &ev.reason, &ev.related_region_id).await {
                    Err(HubError::NotFound { .. }) => Ok(()),
                    r => r,
                }
            }
            Kind::Unspecified => Ok(()),
            _ => {
                let descents: Vec<NewDescent> = ev
                    .ancestors
                    .iter()
                    .filter(|a| !a.region_id.is_empty())
                    .map(|a| NewDescent {
                        child_region_id: ev.region_id.clone(),
                        parent_region_id: a.region_id.clone(),
                        kind: a.kind.clone(),
                        overlap: a.overlap,
                    })
                    .collect();
                self.hub.upsert_descents(&descents).await?;
                self.centroid(&space, &ev.region_id, &ev.centroid).await
            }
        }
    }

    async fn kernel(&mut self, k: KernelUpdate) -> Result<(), HubError> {
        let Some(space) = k.space.as_ref() else { return Ok(()) };
        let scores: Vec<serde_json::Value> = k
            .samplers
            .iter()
            .map(|s| {
                serde_json::json!({
                    "sampler_id": s.sampler_id, "log_score": s.log_score, "brier": s.brier, "ece": s.ece,
                    "pit_ks": s.pit_ks, "cost_us": s.cost_us, "evals": s.evals, "pareto": s.pareto, "serving": s.serving,
                })
            })
            .collect();
        let stationary: Vec<serde_json::Value> = k
            .stationary
            .iter()
            .map(|d| serde_json::json!({"region_id": d.region_id, "mean": d.mean, "lower": d.lower, "upper": d.upper, "sd": d.sd}))
            .collect();
        self.hub
            .upsert_kernel_model(&NewKernelModel {
                expert_space_id: expert_space_id(&space.snapshot_id, space.expert_index),
                snapshot_id: space.snapshot_id.clone(),
                timescale: space.timescale.clone(),
                expert_index: space.expert_index,
                kernel_type: "dirichlet".into(),
                serving_sampler: k.serving_sampler_id.clone(),
                scores: serde_json::Value::from(scores).to_string(),
                stationary: serde_json::Value::from(stationary).to_string(),
                relaxation_steps: k.relaxation_steps,
                drift_kl: k.drift_kl,
                n_rows: k.n_rows as u32,
                observation_mass: k.observation_mass,
                live_regions: k.live_regions,
            })
            .await
    }

    async fn persist(&mut self, rows: Vec<ExpertPersist>) -> Result<(), HubError> {
        for r in rows {
            let Some(state) = r.region_state.as_ref() else { continue };
            // make sure the expert space exists (a reload may have introduced it)
            if self.space_node(&r.space).await.is_none() {
                continue;
            }
            let cfg = state.get("cfg").map(ToString::to_string).unwrap_or_default();
            self.hub
                .upsert_region_set(&NewRegionSet {
                    expert_space_id: expert_space_id(&r.space.snapshot_id, r.space.expert_index),
                    snapshot_id: r.space.snapshot_id.clone(),
                    timescale: r.space.timescale.clone(),
                    expert_index: r.space.expert_index,
                    family: r.family.clone(),
                    method: self.set_method(&r.space.snapshot_id),
                    n_regions: r.stats.len() as u32,
                    cfg_hash: format!("{:016x}", fnv1a(cfg.as_bytes())),
                    state: state.to_string(),
                    kernel_state: r.kernel_state.clone().unwrap_or_default(),
                })
                .await?;
            let stats: Vec<RegionStatsUpdate> = r
                .stats
                .iter()
                .map(|s| RegionStatsUpdate {
                    region_id: s.region_id.clone(),
                    mass: s.mass,
                    occupancy: s.occupancy,
                    spread: s.spread,
                    age: s.age,
                })
                .collect();
            self.hub.update_region_stats(&stats).await?;
            for s in &r.stats {
                self.centroid(&r.space, &s.region_id, &s.centroid).await?;
            }
        }
        Ok(())
    }

    async fn hierarchy(&mut self, h: RegionHierarchy) -> Result<(), HubError> {
        let rows: Vec<NewHierEdge> = h
            .edges
            .iter()
            .map(|e| NewHierEdge {
                fine_region_id: e.fine_region_id.clone(),
                coarse_region_id: e.coarse_region_id.clone(),
                p_coarse_given_fine: e.p_coarse_given_fine,
                p_fine_given_coarse: e.p_fine_given_coarse,
                support: e.support,
                pmi_bits: e.pmi_bits,
                mi_bits: h.mutual_information_bits,
            })
            .collect();
        self.hub.upsert_hier_edges(&rows).await
    }

    async fn cross_priors(&mut self, c: CrossAssetPriors) -> Result<(), HubError> {
        let rows: Vec<NewCrossPrior> = c
            .proposals
            .iter()
            .enumerate()
            .map(|(i, p)| NewCrossPrior {
                timescale: c.timescale.clone(),
                source_instrument_id: p.source_instrument_id.clone(),
                target_instrument_id: p.target_instrument_id.clone(),
                n: p.n,
                correlation: p.correlation,
                shrinkage: p.shrinkage,
                lead_lag: p.lead_lag,
                lead_corr: p.lead_corr,
                predictive_gain: p.predictive_gain,
                uncertainty: p.uncertainty,
                score: p.score,
                rank: i as u32 + 1,
            })
            .collect();
        self.hub.upsert_cross_priors(&rows).await
    }

    async fn ensure_snapshot(&mut self, d: &SnapshotDesc) {
        let snap = NewSnapshot {
            snapshot_id: d.snapshot_id.clone(),
            timescale: d.timescale.to_string(),
            encoder_family: "statistical".into(),
            generation: d.generation,
            garage_key: d.garage_key.clone(),
            status: d.status.clone(),
        };
        let snap = match self.hub.ensure_snapshot(&snap).await {
            Ok(s) => s,
            Err(e) => return self.fail(&e, "ensure_snapshot failed").await,
        };
        bump(&self.counters, "hub.writes", 1.0).await;
        for k in 0..d.n_experts {
            let id = expert_space_id(&d.snapshot_id, k);
            let space = NewExpertSpace {
                expert_space_id: id.clone(),
                snapshot_id: d.snapshot_id.clone(),
                index: k,
                dim: d.dim,
            };
            match self.hub.ensure_expert_space(snap.id, &space).await {
                Ok(s) => {
                    self.spaces.insert(id, s.id);
                    bump(&self.counters, "hub.writes", 1.0).await;
                }
                Err(e) => return self.fail(&e, "ensure_expert_space failed").await,
            }
        }
    }

    /// `RegionSet.method` of a snapshot's sets (`leader` for an unannounced snapshot).
    fn set_method(&self, snapshot_id: &str) -> String {
        self.snapshots.get(snapshot_id).map_or("leader", |d| d.region_method.set_label()).to_owned()
    }

    /// `Region.method` of one region of a snapshot (`leader` for an unannounced snapshot).
    fn region_method(&self, snapshot_id: &str, region_id: &str) -> String {
        self.snapshots.get(snapshot_id).map_or("leader", |d| d.region_method.region_label(region_id)).to_owned()
    }

    async fn space_node(&mut self, space: &ExpertSpaceRef) -> Option<u64> {
        let id = expert_space_id(&space.snapshot_id, space.expert_index);
        if let Some(n) = self.spaces.get(&id) {
            return Some(*n);
        }
        let desc = self.snapshots.get(&space.snapshot_id).cloned()?;
        self.ensure_snapshot(&desc).await;
        self.spaces.get(&id).copied()
    }

    async fn region(&mut self, ev: RegionEvent, radius: f64) {
        let Some(space) = ev.space.clone() else { return };
        let kind = region_event::Kind::try_from(ev.kind).unwrap_or(region_event::Kind::Unspecified);
        let related = ev.related_region_ids.first().cloned().unwrap_or_default();
        let res = match kind {
            // a split child is a new region like a birth (its lineage edge comes with the lifecycle)
            region_event::Kind::Created | region_event::Kind::Split => {
                let Some(space_node) = self.space_node(&space).await else {
                    bump(&self.counters, "hub.dropped", 1.0).await;
                    return;
                };
                let region = NewRegion {
                    region_id: ev.region_id.clone(),
                    lineage_id: ev.lineage_id.clone(),
                    expert_space_id: expert_space_id(&space.snapshot_id, space.expert_index),
                    method: self.region_method(&space.snapshot_id, &ev.region_id),
                    radius,
                    density: 0.0,
                };
                self.hub.upsert_region(space_node, &region).await.map(drop)
            }
            // Merged{survivor = region_id, absorbed = related}: the absorbed region carries the mark.
            region_event::Kind::Merged if !related.is_empty() => {
                self.hub.mark_region_status(&related, "merged", &ev.region_id).await.map(drop)
            }
            region_event::Kind::Retired => {
                self.hub.mark_region_status(&ev.region_id, "retired", &related).await.map(drop)
            }
            _ => return,
        };
        match res {
            Ok(()) => bump(&self.counters, "hub.writes", 1.0).await,
            Err(e) => self.fail(&e, "region write failed").await,
        }
    }

    async fn transition(&mut self, update: TransitionUpdate, observed_to: Option<String>) {
        let Some(space) = update.space.as_ref() else { return };
        if let Some(to) = observed_to.as_deref() {
            *self
                .counts
                .entry((update.from_region_id.clone(), to.to_owned(), update.sampler_id.clone()))
                .or_default() += 1;
        }
        let now = Instant::now();
        let kernel = format!("{{\"timescale\":\"{}\",\"expert\":{}}}", space.timescale, space.expert_index);
        let mut immediate = Vec::new();
        for p in &update.to {
            let key = (update.from_region_id.clone(), p.region_id.clone(), update.sampler_id.clone());
            let Some(&count) = self.counts.get(&key) else { continue };
            let row = TransitionUpsert {
                from_region_id: key.0.clone(),
                to_region_id: key.1.clone(),
                transition: NewTransition {
                    sampler_id: update.sampler_id.clone(),
                    count,
                    mean: p.mean,
                    lower: p.lower,
                    upper: p.upper,
                    kernel_params: kernel.clone(),
                },
            };
            match self.coalescer.offer(key, row, now) {
                Some(row) => immediate.push(row),
                None => bump(&self.counters, "hub.coalesced", 1.0).await,
            }
        }
        self.write_transitions(immediate).await;
    }

    async fn flush_due(&mut self) {
        let now = Instant::now();
        let rows: Vec<TransitionUpsert> = self.coalescer.due(now).into_iter().map(|(_, v)| v).collect();
        self.coalescer.prune(now, self.cfg.flush * 60);
        self.counters.lock().await.insert("hub.pending".into(), self.coalescer.pending_len() as f64);
        self.write_transitions(rows).await;
    }

    async fn write_transitions(&mut self, rows: Vec<TransitionUpsert>) {
        if rows.is_empty() {
            return;
        }
        if !self.connected().await {
            bump(&self.counters, "hub.dropped", rows.len() as f64).await;
            return;
        }
        match self.hub.upsert_transitions(&rows).await {
            Ok(_) => bump(&self.counters, "hub.writes", 1.0).await,
            Err(e) => self.fail(&e, "transition upsert failed").await,
        }
    }

    async fn latents(&mut self, snapshot_id: &str, instrument_id: &str, ts_ms: i64, experts: &[ExpertLatent]) {
        let mut rows = Vec::with_capacity(experts.len());
        for e in experts {
            let dim = e.mean.len();
            // The cosine index rejects the zero vector (400 `zero_norm_cosine_vector`); the first
            // closed bar of an instrument has all-zero z-scores, so skip it rather than fail the batch.
            if dim == 0 || !e.mean.iter().any(|x| *x != 0.0 && x.is_finite()) {
                bump(&self.counters, "hub.skipped_zero_vec", 1.0).await;
                continue;
            }
            if !self.latent_dims.contains(&dim) {
                if let Err(err) = self.hub.ensure_latent_index_ready(dim, INDEX_WAIT).await {
                    return self.fail(&err, "ensure_latent_index failed").await;
                }
                self.latent_dims.insert(dim);
            }
            let var = if e.log_var.is_empty() {
                0.0
            } else {
                e.log_var.iter().map(|v| f64::from(*v).exp()).sum::<f64>() / e.log_var.len() as f64
            };
            rows.push(NewLatent {
                vector: e.mean.clone(),
                expert_space_id: expert_space_id(snapshot_id, e.expert_index),
                kind: "observation".into(),
                ref_id: e.region_id.clone(),
                instrument_id: instrument_id.to_owned(),
                ts_ms,
                uncertainty: var.sqrt(),
            });
        }
        if rows.is_empty() {
            return;
        }
        match self.hub.add_latents(&rows).await {
            Ok(_) => bump(&self.counters, "hub.writes", 1.0).await,
            Err(e) => self.fail(&e, "latent write failed").await,
        }
    }
}

/// FNV-1a, for the region set's `cfg_hash`.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalescer_writes_at_most_once_per_key_per_interval() {
        let iv = Duration::from_secs(10);
        let mut c = Coalescer::new(iv);
        let t0 = Instant::now();
        // First value for a key goes straight out; later ones inside the window are parked.
        assert_eq!(c.offer("a", 1, t0), Some(1));
        assert_eq!(c.offer("a", 2, t0 + Duration::from_secs(3)), None);
        assert_eq!(c.offer("a", 3, t0 + Duration::from_secs(6)), None, "latest wins");
        assert_eq!(c.offer("b", 10, t0 + Duration::from_secs(6)), Some(10), "other keys are independent");
        assert_eq!(c.pending_len(), 1);
        // Nothing is due before the window closes.
        assert!(c.due(t0 + Duration::from_secs(9)).is_empty());
        // Then exactly the latest parked value is released and the window restarts.
        assert_eq!(c.due(t0 + iv), vec![("a", 3)]);
        assert_eq!(c.pending_len(), 0);
        assert_eq!(c.offer("a", 4, t0 + iv + Duration::from_secs(1)), None);
        assert_eq!(c.offer("a", 5, t0 + 2 * iv), Some(5), "window elapsed: immediate again, parked value superseded");
        assert_eq!(c.pending_len(), 0);
        assert!(c.due(t0 + 3 * iv).is_empty());
        // Pruning forgets idle keys but never pending ones.
        assert_eq!(c.offer("b", 11, t0 + 2 * iv + Duration::from_secs(1)), Some(11), "b idle since t0+6s");
        assert_eq!(c.offer("b", 12, t0 + 2 * iv + Duration::from_secs(2)), None);
        c.prune(t0 + 3 * iv, Duration::from_secs(5));
        assert_eq!(c.last_write.len(), 1);
        assert!(c.last_write.contains_key("b"));
    }

    #[tokio::test]
    async fn region_methods_follow_the_announced_snapshot() {
        let cfg = SinkConfig {
            url: "http://127.0.0.1:9".into(),
            flush: Duration::from_secs(1),
            latent_timescales: BTreeSet::new(),
            queue: 1,
        };
        let mut drain = Drain::new(cfg, Counters::default());
        let desc = |id: &str, rm: RegionMethod| SnapshotDesc {
            snapshot_id: id.into(),
            timescale: Timescale::Minute,
            n_experts: 1,
            dim: 2,
            generation: 1,
            garage_key: String::new(),
            status: "active".into(),
            region_method: rm,
        };
        let neural = RegionMethod {
            declared: fa_patterns::DiscoveryMethod::Mapper,
            online: fa_patterns::DiscoveryMethod::Leader,
            seeded: ["1m-e0-r000".to_owned()].into(),
        };
        drain.snapshots.insert("stat".into(), desc("stat", RegionMethod::default()));
        drain.snapshots.insert("jepa".into(), desc("jepa", neural));
        assert_eq!(drain.set_method("stat"), "leader");
        assert_eq!(drain.region_method("stat", "1m-e0-r000"), "leader");
        assert_eq!(drain.set_method("jepa"), "mapper");
        assert_eq!(drain.region_method("jepa", "1m-e0-r000"), "mapper");
        assert_eq!(drain.region_method("jepa", "1m-e0-r009"), "leader", "born live → online form");
        assert_eq!(drain.set_method("unknown"), "leader");
        // a reload re-announces: the newer desc of the same id replaces the labels
        drain.snapshots.insert("jepa".into(), desc("jepa", RegionMethod::default()));
        assert_eq!(drain.region_method("jepa", "1m-e0-r000"), "leader");
    }

    #[test]
    fn expert_space_ids_follow_snapshot_and_index() {
        assert_eq!(expert_space_id("stat-v1-1m", 2), "stat-v1-1m-e2");
    }
}
