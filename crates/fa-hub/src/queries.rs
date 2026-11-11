//! Typed queries: one function per `QUERY` in `helix/queries.hx` (same name, snake_case),
//! translated into `/v2/query` operation trees, plus the graph reads spec 50 §4.4/§4.5 need.
//!
//! Conventions
//! - `ID` parameters of the `.hx` catalog are Helix node ids (`u64`, the `$id` of a row).
//! - Writes return the created row(s) read back in the same batch (`value_map` /
//!   `edge_properties`), so callers get both the node id and the stored properties.
//! - `Date` properties are written as `{"date_time": unix_ms}` and read back as RFC 3339 strings.
//! - Uniqueness is enforced by the indexes of [`crate::schema`]; a duplicate business id fails
//!   with [`HubError::is_unique_violation`](crate::HubError::is_unique_violation). The `upsert_*`
//!   functions use a conditional batch (lookup → `set_property` or `add_n`) instead.

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::ast::{Batch, Cond, EdgeRef, Input, NodeRef, Op, Pred, Val};
use crate::schema::{LATENT_TENANT, edge, latent_vec_property, node};
use crate::{HubClient, HubError, Result};

// ── row types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Instrument {
    #[serde(rename = "$id")]
    pub id: u64,
    pub instrument_id: String,
    pub asset_class: String,
    pub kind: String,
    pub name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Snapshot {
    #[serde(rename = "$id")]
    pub id: u64,
    pub snapshot_id: String,
    pub timescale: String,
    pub encoder_family: String,
    pub generation: u32,
    pub garage_key: String,
    /// `candidate | active | retired`
    pub status: String,
    #[serde(default)]
    pub created_at: Option<String>,
    /// `manifest.metrics.carried_mass` of a trained snapshot ([`HubClient::set_snapshot_manifest`]).
    #[serde(default)]
    pub carried_mass: Option<f64>,
    /// The snapshot's `manifest.json` as written by the trainer, when mirrored.
    #[serde(default)]
    pub manifest: Option<String>,
    /// `snapshot_id` of the snapshot this one replaced when it was promoted; `rollback` restores it.
    #[serde(default)]
    pub previous_active: Option<String>,
}

/// One online-adapter version of a snapshot (`AdapterState`, spec 36 §5).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct AdapterState {
    #[serde(rename = "$id")]
    pub id: u64,
    pub adapter_id: String,
    pub snapshot_id: String,
    pub timescale: String,
    pub version: u64,
    pub garage_key: String,
    #[serde(default)]
    pub drift: Option<f64>,
    pub max_drift: f64,
    pub frozen: bool,
    #[serde(default)]
    pub n_windows: u64,
    #[serde(default)]
    pub window_start: Option<String>,
    #[serde(default)]
    pub window_end: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ExpertSpace {
    #[serde(rename = "$id")]
    pub id: u64,
    pub expert_space_id: String,
    pub snapshot_id: String,
    pub index: u32,
    pub dim: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LatentRegion {
    #[serde(rename = "$id")]
    pub id: u64,
    pub region_id: String,
    /// Stable across generations (D-007).
    pub lineage_id: String,
    pub expert_space_id: String,
    pub method: String,
    pub radius: f64,
    pub density: f64,
    /// `live | merged | retired` (absent on rows written before the lifecycle marks existed).
    #[serde(default)]
    pub status: Option<String>,
    /// merged: the absorbed region; retired: the survivor this region merged into.
    #[serde(default)]
    pub related_region_id: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

/// A `Transition` edge row (`edge_properties`).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Transition {
    #[serde(rename = "$id")]
    pub id: u64,
    #[serde(rename = "$from")]
    pub from: u64,
    #[serde(rename = "$to")]
    pub to: u64,
    /// Business ids, denormalised onto the edge (absent on rows written by `add_transition`).
    #[serde(default)]
    pub from_region_id: Option<String>,
    #[serde(default)]
    pub to_region_id: Option<String>,
    pub sampler_id: String,
    pub count: i64,
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
    /// JSON.
    pub kernel_params: String,
    #[serde(default)]
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Genome {
    #[serde(rename = "$id")]
    pub id: u64,
    pub genome_id: String,
    /// `encoder | decoder | strategy | sampler | allocator`
    pub kind: String,
    pub family: String,
    /// JSON.
    pub params: String,
    #[serde(default)]
    pub created_at: Option<String>,
    /// Lifecycle status mirrored from `ledger.strategy_state` (spec 35 §5; strategies only):
    /// `candidate | holdout_passed | shadow | active | paused | flattening | retired`.
    #[serde(default)]
    pub status: Option<String>,
    /// Live strategy id the status belongs to (evolved: the genome id itself).
    #[serde(default)]
    pub strategy_id: Option<String>,
    #[serde(default)]
    pub status_reason: Option<String>,
    /// When the status was entered (ms since the epoch, from `ledger.strategy_state.since`).
    #[serde(default)]
    pub status_since_ms: Option<i64>,
    #[serde(default)]
    pub status_updated_at: Option<String>,
}

/// A lifecycle status to mirror onto a `Genome` node (`setGenomeStatus`).
#[derive(Debug, Clone, PartialEq)]
pub struct GenomeStatus {
    pub genome_id: String,
    /// Used only when the genome node does not exist yet (created like `upsertGenome`).
    pub kind: String,
    pub family: String,
    pub params: String,
    pub strategy_id: String,
    pub status: String,
    pub reason: String,
    pub since_ms: i64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FitnessRecord {
    #[serde(rename = "$id")]
    pub id: u64,
    pub genome_id: String,
    pub generation: u32,
    /// JSON: reconstruction, forecast, trading, cost.
    pub objectives: String,
    pub pareto_rank: u32,
}

/// A vector-search hit on `LatentVec` (`$distance`: smaller is closer; cosine).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LatentHit {
    #[serde(rename = "$id")]
    pub id: u64,
    #[serde(rename = "$distance")]
    pub distance: f64,
    pub expert_space_id: String,
    /// `centroid | observation | window`
    pub kind: String,
    pub ref_id: String,
    pub instrument_id: String,
    #[serde(default)]
    pub ts: Option<String>,
    pub uncertainty: f64,
}

/// A stored `LatentVec` row read back with `value_map` (spec 50 `/api/latent/{snapshot}/projection`).
/// The vector sits in the `vec<dim>` property; [`LatentRow::vector`] finds it whatever the dimension.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LatentRow {
    #[serde(rename = "$id")]
    pub id: u64,
    pub expert_space_id: String,
    /// `centroid | observation | window`
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub ref_id: String,
    #[serde(default)]
    pub instrument_id: String,
    #[serde(default)]
    pub ts: Option<String>,
    #[serde(default)]
    pub uncertainty: f64,
    /// Every other property, among them `vec<dim>`.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Json>,
}

impl LatentRow {
    /// The stored vector (`vec<dim>` property as `f32`s), if the row carries one.
    pub fn vector(&self) -> Option<Vec<f32>> {
        self.extra.iter().find_map(|(k, v)| {
            if !k.starts_with("vec") || !k[3..].chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            let arr = v.as_array()?;
            arr.iter().map(|x| x.as_f64().map(|f| f as f32)).collect()
        })
    }
}

/// A genome with its fitness records (spec 50 `/api/evolution/genomes/{id}`).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct GenomeDetail {
    pub genome: Genome,
    pub fitness: Vec<FitnessRecord>,
    pub parents: Vec<Genome>,
}

/// Any edge row as returned by `edge_properties`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct EdgeRow {
    #[serde(rename = "$id")]
    pub id: u64,
    #[serde(rename = "$label")]
    pub label: String,
    #[serde(rename = "$from")]
    pub from: u64,
    #[serde(rename = "$to")]
    pub to: u64,
    #[serde(flatten)]
    pub properties: serde_json::Map<String, Json>,
}

/// Any node row as returned by `value_map`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct NodeRow {
    #[serde(rename = "$id")]
    pub id: u64,
    #[serde(rename = "$label")]
    pub label: String,
    #[serde(flatten)]
    pub properties: serde_json::Map<String, Json>,
}

/// Regions of one expert space and the transitions among them (spec 50 `/api/mcmc/{snapshot}/graph`).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct RegionGraph {
    pub regions: Vec<LatentRegion>,
    pub transitions: Vec<Transition>,
}

/// Everything persisted for one snapshot: its expert spaces, their regions and the transitions
/// among them (spec 50 `/api/latent/{snapshot}/regions`, `/api/mcmc/{snapshot}/graph`).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SnapshotGraph {
    pub snapshot: Snapshot,
    pub spaces: Vec<ExpertSpace>,
    pub regions: Vec<LatentRegion>,
    pub transitions: Vec<Transition>,
}

/// Subgraph around a node (spec 50 `/api/graph/neighborhood`).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct Neighborhood {
    pub nodes: Vec<NodeRow>,
    pub edges: Vec<EdgeRow>,
}

// ── write payloads ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct NewInstrument {
    pub instrument_id: String,
    pub asset_class: String,
    pub kind: String,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewSnapshot {
    pub snapshot_id: String,
    pub timescale: String,
    pub encoder_family: String,
    pub generation: u32,
    pub garage_key: String,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewAdapterState {
    pub snapshot_id: String,
    pub timescale: String,
    /// Unix ms of the adapter fit; `adapter_id` becomes `<snapshot_id>@<version>`.
    pub version: u64,
    pub garage_key: String,
    /// `None` when the snapshot had no fitted reference statistics.
    pub drift: Option<f64>,
    pub max_drift: f64,
    pub frozen: bool,
    pub n_windows: u64,
    pub window_start_ms: i64,
    pub window_end_ms: i64,
}

impl NewAdapterState {
    pub fn adapter_id(&self) -> String {
        adapter_id(&self.snapshot_id, self.version)
    }
}

/// Business id of an adapter version: `<snapshot_id>@<version>`.
pub fn adapter_id(snapshot_id: &str, version: u64) -> String {
    format!("{snapshot_id}@{version}")
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewExpertSpace {
    pub expert_space_id: String,
    pub snapshot_id: String,
    pub index: u32,
    pub dim: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewRegion {
    pub region_id: String,
    pub lineage_id: String,
    pub expert_space_id: String,
    pub method: String,
    pub radius: f64,
    pub density: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewTransition {
    pub sampler_id: String,
    pub count: i64,
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
    pub kernel_params: String,
}

/// One `upsertTransition` row: the edge `from_region_id → to_region_id` for `sampler_id`.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionUpsert {
    pub from_region_id: String,
    pub to_region_id: String,
    pub transition: NewTransition,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewLatent {
    pub vector: Vec<f32>,
    pub expert_space_id: String,
    pub kind: String,
    pub ref_id: String,
    pub instrument_id: String,
    /// Unix milliseconds.
    pub ts_ms: i64,
    pub uncertainty: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewGenome {
    pub genome_id: String,
    pub kind: String,
    pub family: String,
    pub params: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewFitness {
    pub genome_id: String,
    pub generation: u32,
    pub objectives: String,
    pub pareto_rank: u32,
}

fn s(v: &str) -> Input {
    Input::Lit(Val::Str(v.to_owned()))
}

/// `add_n(label, props)` as entry `n`, read back as `row`; returns the batch to extend.
fn add_node(batch: Batch, label: &str, props: Vec<(&str, Input)>) -> Batch {
    batch.var("n", Op::add_n(label, props)).ret("row", Op::nodes(NodeRef::var("n")).value_map(None))
}

/// Lookup by label + unique key, read back with `value_map`.
fn by_key(label: &str, key: &str, value: &str) -> Op {
    Op::nodes_with_label_where(label, Pred::eq(key, value)).value_map(None)
}

impl HubClient {
    // ── reference ───────────────────────────────────────────────────────────

    /// `upsertInstrument`: a real upsert (lookup → update properties, else insert).
    pub async fn upsert_instrument(&self, inst: &NewInstrument) -> Result<Instrument> {
        let existing =
            Op::nodes_with_label_where(node::INSTRUMENT, Pred::eq("instrument_id", inst.instrument_id.as_str()));
        let updated = Op::nodes(NodeRef::var("existing"))
            .set_property("asset_class", inst.asset_class.as_str())
            .set_property("kind", inst.kind.as_str())
            .set_property("name", inst.name.as_str())
            .value_map(None);
        let created = Op::add_n(
            node::INSTRUMENT,
            vec![
                ("instrument_id", s(&inst.instrument_id)),
                ("asset_class", s(&inst.asset_class)),
                ("kind", s(&inst.kind)),
                ("name", s(&inst.name)),
            ],
        );
        let batch = Batch::write()
            .named("upsert_instrument")
            .var("existing", existing)
            .var_if("updated", Cond::VarNotEmpty("existing".into()), updated)
            .var_if("created", Cond::VarEmpty("existing".into()), created)
            .var_if(
                "created_row",
                Cond::VarNotEmpty("created".into()),
                Op::nodes(NodeRef::var("created")).value_map(None),
            )
            .returning(["updated", "created_row"]);
        let mut resp = self.batch(&batch).await?;
        if let Some(row) = resp.take_opt::<Instrument>("updated")? {
            return Ok(row);
        }
        resp.take_one("created_row")
    }

    /// `getInstrument`
    pub async fn get_instrument(&self, instrument_id: &str) -> Result<Option<Instrument>> {
        let mut resp =
            self.batch(&Batch::read().ret("row", by_key(node::INSTRUMENT, "instrument_id", instrument_id))).await?;
        resp.take_opt("row")
    }

    // ── snapshots and expert spaces ─────────────────────────────────────────

    /// `addSnapshot` (fails with a unique violation if `snapshot_id` exists).
    pub async fn add_snapshot(&self, snap: &NewSnapshot) -> Result<Snapshot> {
        let props = vec![
            ("snapshot_id", s(&snap.snapshot_id)),
            ("timescale", s(&snap.timescale)),
            ("encoder_family", s(&snap.encoder_family)),
            ("generation", snap.generation.into()),
            ("garage_key", s(&snap.garage_key)),
            ("status", s(&snap.status)),
            ("created_at", Input::Now),
        ];
        let mut resp = self.batch(&add_node(Batch::write().named("add_snapshot"), node::SNAPSHOT, props)).await?;
        resp.take_one("row")
    }

    /// `addExpertSpace`: inserts the space and the `HasSpace` edge from `snapshot`.
    pub async fn add_expert_space(&self, snapshot: u64, space: &NewExpertSpace) -> Result<ExpertSpace> {
        let props = vec![
            ("expert_space_id", s(&space.expert_space_id)),
            ("snapshot_id", s(&space.snapshot_id)),
            ("index", space.index.into()),
            ("dim", space.dim.into()),
        ];
        let batch = add_node(Batch::write().named("add_expert_space"), node::EXPERT_SPACE, props)
            .var("edge", Op::nodes(snapshot).add_e(edge::HAS_SPACE, NodeRef::var("n"), vec![]));
        let mut resp = self.batch(&batch).await?;
        resp.take_one("row")
    }

    /// `ensureSnapshot`: get-or-create by `snapshot_id` (idempotent across service restarts;
    /// an existing row is returned unchanged so operator status changes survive).
    pub async fn ensure_snapshot(&self, snap: &NewSnapshot) -> Result<Snapshot> {
        let props = vec![
            ("snapshot_id", s(&snap.snapshot_id)),
            ("timescale", s(&snap.timescale)),
            ("encoder_family", s(&snap.encoder_family)),
            ("generation", snap.generation.into()),
            ("garage_key", s(&snap.garage_key)),
            ("status", s(&snap.status)),
            ("created_at", Input::Now),
        ];
        let batch = get_or_create(
            Batch::write().named("ensure_snapshot"),
            node::SNAPSHOT,
            "snapshot_id",
            &snap.snapshot_id,
            props,
        );
        take_existing_or_created(self.batch(&batch).await?)
    }

    /// `ensureExpertSpace`: get-or-create by `expert_space_id`; a new space gets its `HasSpace` edge.
    pub async fn ensure_expert_space(&self, snapshot: u64, space: &NewExpertSpace) -> Result<ExpertSpace> {
        let props = vec![
            ("expert_space_id", s(&space.expert_space_id)),
            ("snapshot_id", s(&space.snapshot_id)),
            ("index", space.index.into()),
            ("dim", space.dim.into()),
        ];
        let batch = get_or_create(
            Batch::write().named("ensure_expert_space"),
            node::EXPERT_SPACE,
            "expert_space_id",
            &space.expert_space_id,
            props,
        )
        .var_if(
            "edge",
            Cond::VarNotEmpty("created".into()),
            Op::nodes(snapshot).add_e(edge::HAS_SPACE, NodeRef::var("created"), vec![]),
        );
        take_existing_or_created(self.batch(&batch).await?)
    }

    /// `listSnapshots`
    pub async fn list_snapshots(&self) -> Result<Vec<Snapshot>> {
        let op = Op::nodes_with_label(node::SNAPSHOT).order_by("snapshot_id", false).value_map(None);
        let mut resp = self.batch(&Batch::read().named("list_snapshots").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// Lookup by business id (what control-api routes receive).
    pub async fn snapshot_by_id(&self, snapshot_id: &str) -> Result<Option<Snapshot>> {
        let mut resp =
            self.batch(&Batch::read().ret("row", by_key(node::SNAPSHOT, "snapshot_id", snapshot_id))).await?;
        resp.take_opt("row")
    }

    /// Lookup an expert space by business id.
    pub async fn space_by_id(&self, expert_space_id: &str) -> Result<Option<ExpertSpace>> {
        let mut resp = self
            .batch(&Batch::read().ret("row", by_key(node::EXPERT_SPACE, "expert_space_id", expert_space_id)))
            .await?;
        resp.take_opt("row")
    }

    /// `snapshotSpaces`: `N<Snapshot>(snapshot)::Out<HasSpace>`
    pub async fn snapshot_spaces(&self, snapshot: u64) -> Result<Vec<ExpertSpace>> {
        let op = Op::nodes(snapshot).out(Some(edge::HAS_SPACE)).value_map(None);
        let mut resp = self.batch(&Batch::read().named("snapshot_spaces").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// Snapshots of one timescale and encoder family with `status` (`candidate | active | retired`),
    /// newest `created_at` first. The trainer and model-runtime resolve the active snapshot with it.
    pub async fn snapshots_by_status(&self, timescale: &str, family: &str, status: &str) -> Result<Vec<Snapshot>> {
        let pred = Pred::and([
            Pred::eq("timescale", timescale),
            Pred::eq("encoder_family", family),
            Pred::eq("status", status),
        ]);
        let op = Op::nodes_with_label_where(node::SNAPSHOT, pred).value_map(None);
        let mut resp = self.batch(&Batch::read().named("snapshots_by_status").ret("rows", op)).await?;
        let mut rows: Vec<Snapshot> = resp.take("rows")?;
        rows.sort_by(|a, b| b.created_at.cmp(&a.created_at).then_with(|| b.snapshot_id.cmp(&a.snapshot_id)));
        Ok(rows)
    }

    /// The active snapshot of a timescale for one encoder family, if any (newest when several).
    pub async fn active_snapshot(&self, timescale: &str, family: &str) -> Result<Option<Snapshot>> {
        Ok(self.snapshots_by_status(timescale, family, "active").await?.into_iter().next())
    }

    /// `SnapshotParent{op}` edge parent → child (spec 36 lineage). Returns the edge id.
    pub async fn link_snapshot_parent(&self, parent: u64, child: u64, op: &str) -> Result<u64> {
        let edge_op = Op::nodes(parent).add_e(edge::SNAPSHOT_PARENT, child, vec![("op", s(op))]);
        let mut resp = self.batch(&Batch::write().named("link_snapshot_parent").ret("e", edge_op)).await?;
        Ok(resp.take_one::<IdRow>("e")?.id)
    }

    /// Parent snapshot of `child` (`In<SnapshotParent>`), if linked.
    pub async fn snapshot_parent(&self, child: u64) -> Result<Option<Snapshot>> {
        let op = Op::nodes(child).r#in(Some(edge::SNAPSHOT_PARENT)).value_map(None);
        let mut resp = self.batch(&Batch::read().named("snapshot_parent").ret("rows", op)).await?;
        let mut rows: Vec<Snapshot> = resp.take("rows")?;
        rows.truncate(1);
        Ok(rows.pop())
    }

    /// Snapshots derived from `parent` (`Out<SnapshotParent>`).
    pub async fn snapshot_children(&self, parent: u64) -> Result<Vec<Snapshot>> {
        let op = Op::nodes(parent).out(Some(edge::SNAPSHOT_PARENT)).value_map(None);
        let mut resp = self.batch(&Batch::read().named("snapshot_children").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// Set `status` on a snapshot (`promote`/`rollback` routes). Returns the updated row.
    pub async fn set_snapshot_status(&self, snapshot: u64, status: &str) -> Result<Snapshot> {
        let op = Op::nodes(snapshot).set_property("status", status).value_map(None);
        let mut resp = self.batch(&Batch::write().named("set_snapshot_status").ret("row", op)).await?;
        resp.take_one("row")
    }

    /// Record which snapshot `snapshot` replaced on promotion, so `rollback` can restore exactly
    /// that one instead of guessing from creation times. Returns the updated row.
    pub async fn set_snapshot_previous_active(&self, snapshot: u64, previous: &str) -> Result<Snapshot> {
        let op = Op::nodes(snapshot).set_property("previous_active", previous).value_map(None);
        let mut resp = self.batch(&Batch::write().named("set_snapshot_previous_active").ret("row", op)).await?;
        resp.take_one("row")
    }

    /// Mirrors a trained snapshot's manifest on its node (`manifest` JSON + `carried_mass`) so
    /// control-api's promotion checklist can answer without Garage. Returns the updated row.
    pub async fn set_snapshot_manifest(
        &self,
        snapshot: u64,
        manifest_json: &str,
        carried_mass: f64,
    ) -> Result<Snapshot> {
        let op = Op::nodes(snapshot)
            .set_property("manifest", manifest_json)
            .set_property("carried_mass", carried_mass)
            .value_map(None);
        let mut resp = self.batch(&Batch::write().named("set_snapshot_manifest").ret("row", op)).await?;
        resp.take_one("row")
    }

    // ── adapters (spec 36 §5 `AdapterState`) ────────────────────────────────

    /// `addAdapterState`: inserts the adapter version and its `HasAdapter` edge from `snapshot`
    /// (the Helix id of the `Snapshot` row). Fails with a unique violation when
    /// `<snapshot_id>@<version>` exists.
    pub async fn add_adapter_state(&self, snapshot: u64, a: &NewAdapterState) -> Result<AdapterState> {
        let mut props = vec![
            ("adapter_id", s(&a.adapter_id())),
            ("snapshot_id", s(&a.snapshot_id)),
            ("timescale", s(&a.timescale)),
            ("version", Input::Lit(Val::I64(i64::try_from(a.version).unwrap_or(i64::MAX)))),
            ("garage_key", s(&a.garage_key)),
            ("max_drift", a.max_drift.into()),
            ("frozen", a.frozen.into()),
            ("n_windows", Input::Lit(Val::I64(i64::try_from(a.n_windows).unwrap_or(i64::MAX)))),
            ("window_start", Input::Lit(Val::DateTime(a.window_start_ms))),
            ("window_end", Input::Lit(Val::DateTime(a.window_end_ms))),
            ("updated_at", Input::Now),
        ];
        if let Some(d) = a.drift.filter(|d| d.is_finite()) {
            props.push(("drift", d.into()));
        }
        let batch = add_node(Batch::write().named("add_adapter_state"), node::ADAPTER_STATE, props)
            .var("edge", Op::nodes(snapshot).add_e(edge::HAS_ADAPTER, NodeRef::var("n"), vec![]));
        let mut resp = self.batch(&batch).await?;
        resp.take_one("row")
    }

    /// Adapter versions of a snapshot by business id, newest `version` first, at most `limit`.
    pub async fn snapshot_adapters(&self, snapshot_id: &str, limit: usize) -> Result<Vec<AdapterState>> {
        let op = Op::nodes_with_label_where(node::ADAPTER_STATE, Pred::eq("snapshot_id", snapshot_id))
            .order_by("version", true)
            .limit(limit.max(1))
            .value_map(None);
        let mut resp = self.batch(&Batch::read().named("snapshot_adapters").ret("rows", op)).await?;
        let mut rows: Vec<AdapterState> = resp.take("rows")?;
        rows.sort_by_key(|a| std::cmp::Reverse(a.version));
        Ok(rows)
    }

    /// `latestAdapter`: the newest adapter version of a snapshot, if any.
    pub async fn latest_adapter(&self, snapshot_id: &str) -> Result<Option<AdapterState>> {
        Ok(self.snapshot_adapters(snapshot_id, 1).await?.into_iter().next())
    }

    // ── regions and transitions ─────────────────────────────────────────────

    /// `addRegion`: inserts the region and its `InSpace` edge to `space`.
    pub async fn add_region(&self, space: u64, region: &NewRegion) -> Result<LatentRegion> {
        let props = vec![
            ("region_id", s(&region.region_id)),
            ("lineage_id", s(&region.lineage_id)),
            ("expert_space_id", s(&region.expert_space_id)),
            ("method", s(&region.method)),
            ("radius", region.radius.into()),
            ("density", region.density.into()),
            ("created_at", Input::Now),
        ];
        let batch = add_node(Batch::write().named("add_region"), node::LATENT_REGION, props)
            .var("edge", Op::nodes(NodeRef::var("n")).add_e(edge::IN_SPACE, space, vec![]));
        let mut resp = self.batch(&batch).await?;
        resp.take_one("row")
    }

    /// `upsertRegion`: by `region_id`. An existing row (same id after a model-runtime restart,
    /// D-039 ids are process-local) is revived: `status = live`, radius/density refreshed.
    /// A new row is inserted with its `InSpace` edge to `space`.
    pub async fn upsert_region(&self, space: u64, region: &NewRegion) -> Result<LatentRegion> {
        let existing = by_key_nodes(node::LATENT_REGION, "region_id", &region.region_id);
        let updated = Op::nodes(NodeRef::var("existing"))
            .set_property("status", "live")
            .set_property("related_region_id", "")
            .set_property("radius", region.radius)
            .set_property("density", region.density)
            .set_property("updated_at", Input::Now)
            .value_map(None);
        let created = Op::add_n(
            node::LATENT_REGION,
            vec![
                ("region_id", s(&region.region_id)),
                ("lineage_id", s(&region.lineage_id)),
                ("expert_space_id", s(&region.expert_space_id)),
                ("method", s(&region.method)),
                ("radius", region.radius.into()),
                ("density", region.density.into()),
                ("status", s("live")),
                ("related_region_id", s("")),
                ("created_at", Input::Now),
                ("updated_at", Input::Now),
            ],
        );
        let batch = Batch::write()
            .named("upsert_region")
            .var("existing", existing)
            .var_if("updated", Cond::VarNotEmpty("existing".into()), updated)
            .var_if("created", Cond::VarEmpty("existing".into()), created)
            .var_if(
                "edge",
                Cond::VarNotEmpty("created".into()),
                Op::nodes(NodeRef::var("created")).add_e(edge::IN_SPACE, space, vec![]),
            )
            .var_if(
                "created_row",
                Cond::VarNotEmpty("created".into()),
                Op::nodes(NodeRef::var("created")).value_map(None),
            )
            .returning(["updated", "created_row"]);
        let mut resp = self.batch(&batch).await?;
        if let Some(row) = resp.take_opt::<LatentRegion>("updated")? {
            return Ok(row);
        }
        resp.take_one("created_row")
    }

    /// `getRegion` under its catalog name; same as [`Self::region_by_id`].
    pub async fn get_region(&self, region_id: &str) -> Result<Option<LatentRegion>> {
        self.region_by_id(region_id).await
    }

    /// `getRegion`: lookup by business id.
    pub async fn region_by_id(&self, region_id: &str) -> Result<Option<LatentRegion>> {
        let mut resp =
            self.batch(&Batch::read().ret("row", by_key(node::LATENT_REGION, "region_id", region_id))).await?;
        resp.take_opt("row")
    }

    /// `markRegionStatus`: lifecycle mark (`merged` | `retired`, or back to `live`) with the
    /// related region (absorbed region / survivor). Returns the updated row, `NotFound` if none.
    pub async fn mark_region_status(
        &self,
        region_id: &str,
        status: &str,
        related_region_id: &str,
    ) -> Result<LatentRegion> {
        let op = by_key_nodes(node::LATENT_REGION, "region_id", region_id)
            .set_property("status", status)
            .set_property("related_region_id", related_region_id)
            .set_property("updated_at", Input::Now)
            .value_map(None);
        let mut resp = self.batch(&Batch::write().named("mark_region_status").ret("row", op)).await?;
        resp.take_opt("row")?
            .ok_or_else(|| HubError::NotFound { label: node::LATENT_REGION.into(), id: region_id.into() })
    }

    /// `spaceRegions`: `N<ExpertSpace>(space)::In<InSpace>`
    pub async fn space_regions(&self, space: u64) -> Result<Vec<LatentRegion>> {
        let op = Op::nodes(space).r#in(Some(edge::IN_SPACE)).value_map(None);
        let mut resp = self.batch(&Batch::read().named("space_regions").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// `addTransition`: `AddE<Transition>::From(from)::To(to)`, returned as an edge row.
    pub async fn add_transition(&self, from: u64, to: u64, t: &NewTransition) -> Result<Transition> {
        let props = vec![
            ("sampler_id", s(&t.sampler_id)),
            ("count", t.count.into()),
            ("mean", t.mean.into()),
            ("lower", t.lower.into()),
            ("upper", t.upper.into()),
            ("kernel_params", s(&t.kernel_params)),
            ("updated_at", Input::Now),
        ];
        let batch = Batch::write()
            .named("add_transition")
            .var("e", Op::nodes(from).add_e(edge::TRANSITION, to, props))
            .ret("row", Op::edges(EdgeRef::Var("e".into())).edge_properties());
        let mut resp = self.batch(&batch).await?;
        resp.take_one("row")
    }

    /// `upsertTransition` for one edge; see [`Self::upsert_transitions`].
    pub async fn upsert_transition(&self, row: &TransitionUpsert) -> Result<Transition> {
        let mut rows = self.upsert_transitions(std::slice::from_ref(row)).await?;
        Ok(rows.pop().expect("one row in, one row out"))
    }

    /// `upsertTransition` for several edges in one write batch (what model-runtime flushes
    /// every `--hub-flush-secs`): for each row the edge `from → to` with the same `sampler_id`
    /// is updated in place (count, mean, interval, kernel params, `updated_at`), else inserted.
    /// Fails with [`HubError::NotFound`] when either region is unknown; the whole batch is then
    /// rolled back by the server (one transaction).
    pub async fn upsert_transitions(&self, rows: &[TransitionUpsert]) -> Result<Vec<Transition>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut batch = Batch::write().named("upsert_transitions");
        for (i, r) in rows.iter().enumerate() {
            let t = &r.transition;
            let (from, to, existing, updated, probe, created, created_row) = (
                format!("from{i}"),
                format!("to{i}"),
                format!("existing{i}"),
                format!("updated{i}"),
                format!("probe{i}"),
                format!("created{i}"),
                format!("created_row{i}"),
            );
            let mut set = Op::edges(EdgeRef::Var(existing.clone()));
            for (k, v) in transition_mutable_props(t) {
                set = set.set_property(k, v);
            }
            batch = batch
                .var(&from, by_key_nodes(node::LATENT_REGION, "region_id", &r.from_region_id))
                .var(&to, by_key_nodes(node::LATENT_REGION, "region_id", &r.to_region_id))
                .var(
                    &existing,
                    Op::nodes(NodeRef::var(&from)).out_e(Some(edge::TRANSITION)).where_(Pred::and([
                        Pred::eq("to_region_id", r.to_region_id.as_str()),
                        Pred::eq("sampler_id", t.sampler_id.as_str()),
                    ])),
                )
                .var_if(&updated, Cond::VarNotEmpty(existing.clone()), set.edge_properties())
                // `add_e` with an empty target is a server error (500 `invalid_query`), not an empty
                // result: `probe` is non-empty only when there is no edge yet AND `to` exists.
                .var_if(&probe, Cond::VarEmpty(existing.clone()), Op::nodes(NodeRef::var(&to)))
                .var_if(
                    &created,
                    Cond::VarNotEmpty(probe.clone()),
                    Op::nodes(NodeRef::var(&from)).add_e(
                        edge::TRANSITION,
                        NodeRef::var(&to),
                        [
                            vec![
                                ("from_region_id", s(&r.from_region_id)),
                                ("to_region_id", s(&r.to_region_id)),
                                ("sampler_id", s(&t.sampler_id)),
                            ],
                            transition_mutable_props(t),
                        ]
                        .concat(),
                    ),
                )
                .var_if(
                    &created_row,
                    Cond::VarNotEmpty(created.clone()),
                    Op::edges(EdgeRef::Var(created.clone())).edge_properties(),
                )
                .returning([updated.clone(), created_row.clone()]);
        }
        let mut resp = self.batch(&batch).await?;
        let mut out = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            let row = match resp.take_opt::<Transition>(&format!("updated{i}"))? {
                Some(row) => row,
                None => match resp.take_opt::<Transition>(&format!("created_row{i}"))? {
                    Some(row) => row,
                    None => {
                        return Err(HubError::NotFound {
                            label: node::LATENT_REGION.into(),
                            id: format!("{} -> {}", r.from_region_id, r.to_region_id),
                        });
                    }
                },
            };
            out.push(row);
        }
        Ok(out)
    }

    /// `regionTransitions`: outgoing `Transition` edges of a region.
    pub async fn region_transitions(&self, region: u64) -> Result<Vec<Transition>> {
        let op = Op::nodes(region).out_e(Some(edge::TRANSITION)).edge_properties();
        let mut resp = self.batch(&Batch::read().named("region_transitions").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// `regionNeighbors`: regions reachable over one `Transition`.
    pub async fn region_neighbors(&self, region: u64) -> Result<Vec<LatentRegion>> {
        let op = Op::nodes(region).out(Some(edge::TRANSITION)).dedup().value_map(None);
        let mut resp = self.batch(&Batch::read().named("region_neighbors").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// `matchRegions`: `RegionMatch{score}` edge prev → next; returns the edge id.
    pub async fn match_regions(&self, prev: u64, next: u64, score: f64) -> Result<u64> {
        let op = Op::nodes(prev).add_e(edge::REGION_MATCH, next, vec![("score", score.into())]);
        let mut resp = self.batch(&Batch::write().named("match_regions").ret("e", op)).await?;
        Ok(resp.take_one::<IdRow>("e")?.id)
    }

    /// Regions of an expert space with all transitions among them (one round trip).
    pub async fn region_graph(&self, space: u64) -> Result<RegionGraph> {
        let batch = Batch::read()
            .named("region_graph")
            .var("regions", Op::nodes(space).r#in(Some(edge::IN_SPACE)))
            .ret("region_rows", Op::nodes(NodeRef::var("regions")).value_map(None))
            .ret("transitions", Op::nodes(NodeRef::var("regions")).out_e(Some(edge::TRANSITION)).edge_properties());
        let mut resp = self.batch(&batch).await?;
        Ok(RegionGraph { regions: resp.take("region_rows")?, transitions: resp.take("transitions")? })
    }

    /// `snapshotGraph`: the snapshot, its spaces, their regions and all transitions among them,
    /// by business id, in one round trip. `None` when the snapshot does not exist.
    pub async fn snapshot_graph(&self, snapshot_id: &str) -> Result<Option<SnapshotGraph>> {
        let batch = Batch::read()
            .named("snapshot_graph")
            .var("snap", by_key_nodes(node::SNAPSHOT, "snapshot_id", snapshot_id))
            .ret("snap_row", Op::nodes(NodeRef::var("snap")).value_map(None))
            .var("spaces", Op::nodes(NodeRef::var("snap")).out(Some(edge::HAS_SPACE)))
            .ret("space_rows", Op::nodes(NodeRef::var("spaces")).value_map(None))
            .var("regions", Op::nodes(NodeRef::var("spaces")).r#in(Some(edge::IN_SPACE)))
            .ret("region_rows", Op::nodes(NodeRef::var("regions")).value_map(None))
            .ret("transitions", Op::nodes(NodeRef::var("regions")).out_e(Some(edge::TRANSITION)).edge_properties());
        let mut resp = self.batch(&batch).await?;
        let Some(snapshot) = resp.take_opt::<Snapshot>("snap_row")? else {
            return Ok(None);
        };
        Ok(Some(SnapshotGraph {
            snapshot,
            spaces: resp.take("space_rows")?,
            regions: resp.take("region_rows")?,
            transitions: resp.take("transitions")?,
        }))
    }

    // ── latent vectors ──────────────────────────────────────────────────────

    /// `addLatent`: a `LatentVec` node whose vector sits in `vec<dim>`. Call
    /// [`HubClient::ensure_latent_index`](crate::HubClient::ensure_latent_index) for that
    /// dimension once per server, or searches report `index_not_found`
    /// ([`HubClient::add_latents_checked`] does both and validates the space's `dim`).
    /// Empty, zero-norm or non-finite vectors fail with [`HubError::InvalidVector`].
    pub async fn add_latent(&self, latent: &NewLatent) -> Result<u64> {
        let mut ids = self.add_latents(std::slice::from_ref(latent)).await?;
        Ok(ids.pop().expect("one row in, one id out"))
    }

    /// Several latents (same or mixed dimensions) in one write batch; returns their node ids.
    pub async fn add_latents(&self, latents: &[NewLatent]) -> Result<Vec<u64>> {
        if latents.is_empty() {
            return Ok(Vec::new());
        }
        let batch = crate::vectors::add_latents_batch(latents)?;
        let mut resp = self.batch(&batch).await?;
        (0..latents.len()).map(|i| Ok(resp.take_one::<IdRow>(&format!("n{i}"))?.id)).collect()
    }

    /// `searchLatent`, scoped to one expert space (the index tenant): k nearest `LatentVec`s
    /// to `vector` by cosine distance. Caller filters by `kind` / time distance (spec 34).
    pub async fn search_latent(&self, expert_space_id: &str, vector: &[f32], k: usize) -> Result<Vec<LatentHit>> {
        let prop = latent_vec_property(vector.len());
        let op =
            Op::vector_search_nodes(node::LATENT_VEC, &prop, vector.to_vec(), k, Some(s(expert_space_id))).value_map(
                Some(&["$id", "$distance", LATENT_TENANT, "kind", "ref_id", "instrument_id", "ts", "uncertainty"]),
            );
        let mut resp = self.batch(&Batch::read().named("search_latent").ret("hits", op)).await?;
        resp.take("hits")
    }

    /// Stored `LatentVec` rows of one expert space, newest first, at most `limit`
    /// (spec 50 `/api/latent/{snapshot}/projection`). `kind` filters on `centroid | observation |
    /// window`; `since_ms` keeps rows with `ts >= since` (unix milliseconds).
    pub async fn space_latents(
        &self,
        expert_space_id: &str,
        kind: Option<&str>,
        since_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<LatentRow>> {
        let mut preds = vec![Pred::eq(LATENT_TENANT, expert_space_id)];
        if let Some(k) = kind.filter(|k| !k.is_empty()) {
            preds.push(Pred::eq("kind", k));
        }
        if let Some(ms) = since_ms {
            preds.push(Pred::gte("ts", Input::Lit(Val::DateTime(ms))));
        }
        let op = Op::nodes_with_label_where(node::LATENT_VEC, Pred::and(preds))
            .order_by("ts", true)
            .limit(limit.max(1))
            .value_map(None);
        let mut resp = self.batch(&Batch::read().named("space_latents").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// Number of `LatentVec` rows in one expert space.
    pub async fn count_space_latents(&self, expert_space_id: &str) -> Result<u64> {
        let op = Op::nodes_with_label_where(node::LATENT_VEC, Pred::eq(LATENT_TENANT, expert_space_id)).count();
        let v = self.query(op).await?;
        v.as_u64().ok_or_else(|| HubError::Shape { name: "count".into(), detail: v.to_string() })
    }

    // ── evolution ───────────────────────────────────────────────────────────

    /// `addGenome`
    pub async fn add_genome(&self, g: &NewGenome) -> Result<Genome> {
        let props = vec![
            ("genome_id", s(&g.genome_id)),
            ("kind", s(&g.kind)),
            ("family", s(&g.family)),
            ("params", s(&g.params)),
            ("created_at", Input::Now),
        ];
        let mut resp = self.batch(&add_node(Batch::write().named("add_genome"), node::GENOME, props)).await?;
        resp.take_one("row")
    }

    /// `upsertGenome`: get-or-create by `genome_id` (evo-worker re-evaluating a known genome).
    pub async fn upsert_genome(&self, g: &NewGenome) -> Result<Genome> {
        let props = vec![
            ("genome_id", s(&g.genome_id)),
            ("kind", s(&g.kind)),
            ("family", s(&g.family)),
            ("params", s(&g.params)),
            ("created_at", Input::Now),
        ];
        let batch =
            get_or_create(Batch::write().named("upsert_genome"), node::GENOME, "genome_id", &g.genome_id, props);
        take_existing_or_created(self.batch(&batch).await?)
    }

    /// `setGenomeStatus`: mirrors a strategy's lifecycle status onto its `Genome` (spec 35 §5),
    /// creating the genome like `upsertGenome` when the evo-worker never wrote it. Idempotent.
    pub async fn set_genome_status(&self, g: &GenomeStatus) -> Result<Genome> {
        let status_props = |op: Op| {
            op.set_property("status", g.status.as_str())
                .set_property("strategy_id", g.strategy_id.as_str())
                .set_property("status_reason", g.reason.as_str())
                .set_property("status_since_ms", g.since_ms)
                .set_property("status_updated_at", Input::Now)
                .value_map(None)
        };
        let created = Op::add_n(
            node::GENOME,
            vec![
                ("genome_id", s(&g.genome_id)),
                ("kind", s(&g.kind)),
                ("family", s(&g.family)),
                ("params", s(&g.params)),
                ("created_at", Input::Now),
            ],
        );
        let batch = Batch::write()
            .named("set_genome_status")
            .var("existing", by_key_nodes(node::GENOME, "genome_id", &g.genome_id))
            .var_if("updated", Cond::VarNotEmpty("existing".into()), status_props(Op::nodes(NodeRef::var("existing"))))
            .var_if("created", Cond::VarEmpty("existing".into()), created)
            .var_if("created_row", Cond::VarNotEmpty("created".into()), status_props(Op::nodes(NodeRef::var("created"))))
            .returning(["updated", "created_row"]);
        let mut resp = self.batch(&batch).await?;
        if let Some(row) = resp.take_opt::<Genome>("updated")? {
            return Ok(row);
        }
        resp.take_one("created_row")
    }

    /// `linkParent`: `ParentOf{op}` by business ids; returns the edge id, `NotFound` when a
    /// genome is missing.
    pub async fn link_parent(&self, parent_id: &str, child_id: &str, op: &str) -> Result<u64> {
        let batch = Batch::write()
            .named("link_parent")
            .var("parent", by_key_nodes(node::GENOME, "genome_id", parent_id))
            .var("child", by_key_nodes(node::GENOME, "genome_id", child_id))
            // An empty `add_e` target is a server error; only add when the child exists.
            .var_if(
                "e",
                Cond::VarNotEmpty("child".into()),
                Op::nodes(NodeRef::var("parent")).add_e(
                    edge::PARENT_OF,
                    NodeRef::var("child"),
                    vec![("op", op.into())],
                ),
            )
            .returning(["e"]);
        let mut resp = self.batch(&batch).await?;
        resp.take_opt::<IdRow>("e")?
            .map(|r| r.id)
            .ok_or_else(|| HubError::NotFound { label: node::GENOME.into(), id: format!("{parent_id} -> {child_id}") })
    }

    /// `addFitnessRecord`: like [`Self::add_fitness`] but keyed by `genome_id`.
    pub async fn add_fitness_record(&self, f: &NewFitness) -> Result<FitnessRecord> {
        let props = vec![
            ("genome_id", s(&f.genome_id)),
            ("generation", f.generation.into()),
            ("objectives", s(&f.objectives)),
            ("pareto_rank", f.pareto_rank.into()),
        ];
        let batch = Batch::write()
            .named("add_fitness_record")
            .var("genome", by_key_nodes(node::GENOME, "genome_id", &f.genome_id))
            .var_if("n", Cond::VarNotEmpty("genome".into()), Op::add_n(node::FITNESS_RECORD, props))
            .var_if(
                "edge",
                Cond::VarNotEmpty("genome".into()),
                Op::nodes(NodeRef::var("genome")).add_e(edge::SCORED, NodeRef::var("n"), vec![]),
            )
            .var_if("row", Cond::VarNotEmpty("genome".into()), Op::nodes(NodeRef::var("n")).value_map(None))
            .returning(["row"]);
        let mut resp = self.batch(&batch).await?;
        resp.take_opt("row")?.ok_or_else(|| HubError::NotFound { label: node::GENOME.into(), id: f.genome_id.clone() })
    }

    pub async fn genome_by_id(&self, genome_id: &str) -> Result<Option<Genome>> {
        let mut resp = self.batch(&Batch::read().ret("row", by_key(node::GENOME, "genome_id", genome_id))).await?;
        resp.take_opt("row")
    }

    /// Genome, its `FitnessRecord`s and direct parents by business id in one round trip
    /// (spec 50 `/api/evolution/genomes/{id}`). `None` when the genome does not exist.
    pub async fn genome_detail(&self, genome_id: &str) -> Result<Option<GenomeDetail>> {
        let batch = Batch::read()
            .named("genome_detail")
            .var("g", by_key_nodes(node::GENOME, "genome_id", genome_id))
            .ret("genome", Op::nodes(NodeRef::var("g")).value_map(None))
            .ret("fitness", Op::nodes(NodeRef::var("g")).out(Some(edge::SCORED)).value_map(None))
            .ret("parents", Op::nodes(NodeRef::var("g")).r#in(Some(edge::PARENT_OF)).value_map(None));
        let mut resp = self.batch(&batch).await?;
        let Some(genome) = resp.take_opt::<Genome>("genome")? else {
            return Ok(None);
        };
        Ok(Some(GenomeDetail { genome, fitness: resp.take("fitness")?, parents: resp.take("parents")? }))
    }

    /// `addLineage`: `ParentOf{op}` parent → child; returns the edge id.
    pub async fn add_lineage(&self, parent: u64, child: u64, op: &str) -> Result<u64> {
        let e = Op::nodes(parent).add_e(edge::PARENT_OF, child, vec![("op", op.into())]);
        let mut resp = self.batch(&Batch::write().named("add_lineage").ret("e", e)).await?;
        Ok(resp.take_one::<IdRow>("e")?.id)
    }

    /// `addFitness`: inserts the record and the `Scored` edge from `genome`.
    pub async fn add_fitness(&self, genome: u64, f: &NewFitness) -> Result<FitnessRecord> {
        let props = vec![
            ("genome_id", s(&f.genome_id)),
            ("generation", f.generation.into()),
            ("objectives", s(&f.objectives)),
            ("pareto_rank", f.pareto_rank.into()),
        ];
        let batch = add_node(Batch::write().named("add_fitness"), node::FITNESS_RECORD, props)
            .var("edge", Op::nodes(genome).add_e(edge::SCORED, NodeRef::var("n"), vec![]));
        let mut resp = self.batch(&batch).await?;
        resp.take_one("row")
    }

    /// `genomeParents`: `N<Genome>(genome)::In<ParentOf>`
    pub async fn genome_parents(&self, genome: u64) -> Result<Vec<Genome>> {
        let op = Op::nodes(genome).r#in(Some(edge::PARENT_OF)).value_map(None);
        let mut resp = self.batch(&Batch::read().named("genome_parents").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// `genomeFitness`: `N<Genome>(genome)::Out<Scored>`
    pub async fn genome_fitness(&self, genome: u64) -> Result<Vec<FitnessRecord>> {
        let op = Op::nodes(genome).out(Some(edge::SCORED)).value_map(None);
        let mut resp = self.batch(&Batch::read().named("genome_fitness").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// Ancestry levels of a genome (spec 50 `/api/graph/lineage/{genome_id}`): element `i` holds
    /// the ancestors `i + 1` `ParentOf` hops away, up to `max_depth`. Trailing empty levels are dropped.
    pub async fn genome_ancestry(&self, genome: u64, max_depth: usize) -> Result<Vec<Vec<Genome>>> {
        let mut batch = Batch::read().named("genome_ancestry").var("f0", Op::nodes(genome));
        for d in 1..=max_depth {
            let prev = format!("f{}", d - 1);
            let name = format!("f{d}");
            batch = batch.var(&name, Op::nodes(NodeRef::var(&prev)).r#in(Some(edge::PARENT_OF)).dedup());
            batch = batch.ret(&format!("r{d}"), Op::nodes(NodeRef::var(&name)).value_map(None));
        }
        let mut resp = self.batch(&batch).await?;
        let mut levels: Vec<Vec<Genome>> =
            (1..=max_depth).map(|d| resp.take(&format!("r{d}"))).collect::<Result<_>>()?;
        while levels.last().is_some_and(Vec::is_empty) {
            levels.pop();
        }
        Ok(levels)
    }

    // ── generic graph reads ─────────────────────────────────────────────────

    /// `resolveNode`: the node carrying `business_id` as any unique business key
    /// (`region_id`, `snapshot_id`, `expert_space_id`, `genome_id`, `instrument_id`), or a node
    /// addressed directly by its numeric `$id`.
    pub async fn resolve_node(&self, id: &str) -> Result<Option<NodeRow>> {
        if let Ok(n) = id.parse::<u64>() {
            let mut resp =
                self.batch(&Batch::read().named("resolve_node").ret("row", Op::nodes(n).value_map(None))).await?;
            return resp.take_opt("row");
        }
        let pred = Pred::or(BUSINESS_KEYS.iter().map(|k| Pred::eq(k, id)));
        let mut resp =
            self.batch(&Batch::read().named("resolve_node").ret("row", Op::nodes_where(pred).value_map(None))).await?;
        let mut rows: Vec<NodeRow> = resp.take("row")?;
        rows.truncate(1);
        Ok(rows.pop())
    }

    /// Nodes within `depth` hops (any edge label, both directions) of `node`, and every edge
    /// touching the inner frontier (spec 50 `/api/graph/neighborhood`). Includes `node` itself.
    pub async fn neighborhood(&self, node_id: u64, depth: usize) -> Result<Neighborhood> {
        let mut batch = Batch::read()
            .named("neighborhood")
            .var("f0", Op::nodes(node_id))
            .ret("n0", Op::nodes(node_id).value_map(None));
        for d in 1..=depth {
            let prev = format!("f{}", d - 1);
            batch = batch
                .var(&format!("f{d}"), Op::nodes(NodeRef::var(&prev)).both(None).dedup())
                .ret(&format!("n{d}"), Op::nodes(NodeRef::var(format!("f{d}"))).value_map(None))
                .ret(&format!("e{d}"), Op::nodes(NodeRef::var(&prev)).both_e(None).edge_properties());
        }
        let mut resp = self.batch(&batch).await?;
        let mut out = Neighborhood::default();
        for d in 0..=depth {
            let rows: Vec<NodeRow> = resp.take(&format!("n{d}"))?;
            for r in rows {
                if !out.nodes.iter().any(|n| n.id == r.id) {
                    out.nodes.push(r);
                }
            }
            if d > 0 {
                let rows: Vec<EdgeRow> = resp.take(&format!("e{d}"))?;
                for r in rows {
                    if !out.edges.iter().any(|e| e.id == r.id) {
                        out.edges.push(r);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Count nodes with a label.
    pub async fn count_nodes(&self, label: &str) -> Result<u64> {
        let v = self.query(Op::nodes_with_label(label).count()).await?;
        v.as_u64().ok_or_else(|| HubError::Shape { name: "count".into(), detail: v.to_string() })
    }

    /// Delete nodes with `label` matching `pred`, together with their incident edges.
    pub async fn drop_nodes(&self, label: &str, pred: Pred) -> Result<()> {
        self.query(Op::nodes_with_label_where(label, pred).drop()).await?;
        Ok(())
    }
}

/// Unique business keys a node can be addressed by ([`HubClient::resolve_node`]).
pub const BUSINESS_KEYS: [&str; 5] = ["region_id", "snapshot_id", "expert_space_id", "genome_id", "instrument_id"];

#[derive(Deserialize)]
struct IdRow {
    #[serde(rename = "$id")]
    id: u64,
}

/// Lookup by label + unique key as a bare node stream (for traversals / mutations).
fn by_key_nodes(label: &str, key: &str, value: &str) -> Op {
    Op::nodes_with_label_where(label, Pred::eq(key, value))
}

/// Get-or-create skeleton: `existing` (lookup, read back as `existing_row`), `created` (only when
/// empty, read back as `created_row`). Callers may hang more conditional entries off `created`.
fn get_or_create(batch: Batch, label: &str, key: &str, value: &str, props: Vec<(&str, Input)>) -> Batch {
    batch
        .var("existing", by_key_nodes(label, key, value))
        .var_if(
            "existing_row",
            Cond::VarNotEmpty("existing".into()),
            Op::nodes(NodeRef::var("existing")).value_map(None),
        )
        .var_if("created", Cond::VarEmpty("existing".into()), Op::add_n(label, props))
        .var_if("created_row", Cond::VarNotEmpty("created".into()), Op::nodes(NodeRef::var("created")).value_map(None))
        .returning(["existing_row", "created_row"])
}

fn take_existing_or_created<T: serde::de::DeserializeOwned>(mut resp: crate::Response) -> Result<T> {
    if let Some(row) = resp.take_opt::<T>("existing_row")? {
        return Ok(row);
    }
    resp.take_one("created_row")
}

/// Transition properties that change on every posterior update (shared by insert and update).
fn transition_mutable_props(t: &NewTransition) -> Vec<(&'static str, Input)> {
    vec![
        ("count", t.count.into()),
        ("mean", t.mean.into()),
        ("lower", t.lower.into()),
        ("upper", t.upper.into()),
        ("kernel_params", s(&t.kernel_params)),
        ("updated_at", Input::Now),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rows_deserialize_from_server_shapes() {
        let t: Transition = serde_json::from_value(json!({
            "$from": 2, "$id": 0, "$label": "Transition", "$to": 3, "sampler_id": "s", "count": 5,
            "mean": 0.25, "lower": 0.1, "upper": 0.4, "kernel_params": "{}", "updated_at": "2026-11-08T17:51:51.867Z"
        }))
        .unwrap();
        assert_eq!((t.from, t.to, t.count), (2, 3, 5));
        let h: LatentHit = serde_json::from_value(json!({
            "$distance": 0.003, "$id": 3, "expert_space_id": "sp1", "kind": "window", "ref_id": "w2",
            "instrument_id": "BTC", "ts": "2025-11-08T16:40:00.000Z", "uncertainty": 0.1
        }))
        .unwrap();
        assert_eq!(h.ref_id, "w2");
        let n: NodeRow = serde_json::from_value(json!({ "$id": 1, "$label": "Snapshot", "status": "active" })).unwrap();
        assert_eq!(n.properties["status"], "active");
    }

    #[test]
    fn get_or_create_batch_is_conditional() {
        let b = get_or_create(Batch::write(), node::GENOME, "genome_id", "g1", vec![("genome_id", s("g1"))]).to_json();
        let entries = b["query"]["write"]["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[1]["query"]["condition"], json!({ "var_not_empty": "existing" }));
        assert_eq!(entries[2]["query"]["condition"], json!({ "var_empty": "existing" }));
        assert_eq!(entries[2]["query"]["root"]["add_n"]["label"], "Genome");
        assert_eq!(b["query"]["write"]["returns"], json!(["existing_row", "created_row"]));
    }

    #[test]
    fn upsert_transition_rows_deserialize_with_denormalised_ids() {
        let t: Transition = serde_json::from_value(json!({
            "$from": 2, "$id": 0, "$label": "Transition", "$to": 3, "from_region_id": "a", "to_region_id": "b",
            "sampler_id": "conjugate", "count": 1, "mean": 0.5, "lower": 0.1, "upper": 0.9, "kernel_params": "{}"
        }))
        .unwrap();
        assert_eq!(t.to_region_id.as_deref(), Some("b"));
        let r: LatentRegion = serde_json::from_value(json!({
            "$id": 1, "region_id": "r", "lineage_id": "l", "expert_space_id": "sp", "method": "codebook",
            "radius": 1.0, "density": 2.0
        }))
        .unwrap();
        assert_eq!(r.status, None);
        let props = transition_mutable_props(&NewTransition {
            sampler_id: "x".into(),
            count: 3,
            mean: 0.1,
            lower: 0.0,
            upper: 0.2,
            kernel_params: "{}".into(),
        });
        assert_eq!(props.len(), 6);
        assert!(matches!(props[5], ("updated_at", Input::Now)));
    }

    #[test]
    fn latent_row_finds_its_vector_property() {
        let r: LatentRow = serde_json::from_value(json!({
            "$id": 7, "$label": "LatentVec", "expert_space_id": "sp", "kind": "observation", "ref_id": "r1",
            "instrument_id": "i", "ts": "2026-11-10T00:00:00.000Z", "uncertainty": 0.5, "vec3": [1.0, 2.0, 3.0],
            "vector_note": "ignored"
        }))
        .unwrap();
        assert_eq!(r.vector(), Some(vec![1.0f32, 2.0, 3.0]));
        let none: LatentRow =
            serde_json::from_value(json!({ "$id": 8, "expert_space_id": "sp", "vecx": [1.0] })).unwrap();
        assert_eq!(none.vector(), None);
    }

    #[test]
    fn add_node_batch_reads_row_back() {
        let b = add_node(Batch::write(), node::SNAPSHOT, vec![("snapshot_id", s("s1"))]).to_json();
        let entries = &b["query"]["write"]["entries"];
        assert_eq!(entries[0]["query"]["name"], "n");
        assert_eq!(
            entries[1]["query"]["root"],
            json!({ "value_map": { "input": { "nodes": { "reference": { "var": "n" } } } } })
        );
        assert_eq!(b["query"]["write"]["returns"], json!(["row"]));
    }
}
