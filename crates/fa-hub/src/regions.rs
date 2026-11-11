//! Region lifecycle, lineage and graph-level inference in Helix (spec 30 §5, spec 33 §5).
//!
//! What model-runtime persists beyond the D-048 rows (`LatentRegion`, `Transition`, `LatentVec`):
//!
//! - region statistics on `LatentRegion` (`mass`, `occupancy`, `spread`, `age`), death marks
//!   (`status = retired`, `died_at`, `retire_reason`) and the centroid as a `LatentVec`
//!   (`kind = centroid`, `ref_id = region_id`, replaced on every write);
//! - `DescendsFrom` edges child → parent (`kind = carry | split | merge | partial`, `overlap`),
//!   upserted by `(child, parent, kind)`; the trainer's `RegionMatch` edges count as carries;
//! - `Hier` edges fine → coarse region (`REFINES`, spec 30 §6.5) with `P(coarse | fine)`,
//!   `P(fine | coarse)`, support and pointwise MI;
//! - a `RegionSet` node per expert space holding the serialised online region set and the
//!   conjugate kernel (`state`, `kernel_state`), so a restarted runtime continues the same
//!   regions and posteriors (spec 30 §8 "never only in memory", spec 33 §8 Helix outage);
//! - a `KernelModel` node per expert space with the serving sampler, the calibration standings,
//!   the stationary distribution and relaxation time (spec 33 §5);
//! - `CrossPrior` nodes per directed instrument pair and timescale (spec 33 §6.5).

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::{
    Batch, Cond, EdgeRef, EdgeRow, HubClient, HubError, Input, NewLatent, NodeRef, NodeRow, Op, Pred, Result, Val,
    schema::{edge, node},
};

/// Labels introduced by this module (mirrored in `helix/schema.hx`).
pub mod labels {
    pub const REGION_SET: &str = "RegionSet";
    pub const KERNEL_MODEL: &str = "KernelModel";
    pub const CROSS_PRIOR: &str = "CrossPrior";
    pub const DESCENDS_FROM: &str = "DescendsFrom";
    pub const HAS_REGION_SET: &str = "HasRegionSet";
    pub const HAS_KERNEL: &str = "HasKernel";
}

fn s(v: &str) -> Input {
    Input::Lit(Val::Str(v.to_owned()))
}

fn by_key(label: &str, key: &str, value: &str) -> Op {
    Op::nodes_with_label_where(label, Pred::eq(key, value))
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegionStatsUpdate {
    pub region_id: String,
    pub mass: f64,
    pub occupancy: f64,
    pub spread: f64,
    pub age: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewDescent {
    pub child_region_id: String,
    pub parent_region_id: String,
    /// `carry | split | merge | partial`
    pub kind: String,
    pub overlap: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewHierEdge {
    pub fine_region_id: String,
    pub coarse_region_id: String,
    pub p_coarse_given_fine: f64,
    pub p_fine_given_coarse: f64,
    pub support: f64,
    pub pmi_bits: f64,
    /// Mutual information of the whole (fine set, coarse set) pair, bits.
    pub mi_bits: f64,
}

/// `RegionSet` row (spec 30 §5): one per expert space.
#[derive(Debug, Clone, PartialEq)]
pub struct NewRegionSet {
    pub expert_space_id: String,
    pub snapshot_id: String,
    pub timescale: String,
    pub expert_index: u32,
    pub family: String,
    pub method: String,
    pub n_regions: u32,
    pub cfg_hash: String,
    /// Serialised online region set (JSON).
    pub state: String,
    /// Serialised conjugate kernel (`fa_mcmc::MarkovGraph`, JSON).
    pub kernel_state: String,
}

/// `KernelModel` row (spec 33 §5): one per expert space.
#[derive(Debug, Clone, PartialEq)]
pub struct NewKernelModel {
    pub expert_space_id: String,
    pub snapshot_id: String,
    pub timescale: String,
    pub expert_index: u32,
    pub kernel_type: String,
    pub serving_sampler: String,
    /// JSON array of sampler standings (calibration suite).
    pub scores: String,
    /// JSON array of `{region_id, mean, lower, upper, sd}`.
    pub stationary: String,
    pub relaxation_steps: f64,
    pub drift_kl: f64,
    pub n_rows: u32,
    pub observation_mass: f64,
    pub live_regions: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewCrossPrior {
    pub timescale: String,
    pub source_instrument_id: String,
    pub target_instrument_id: String,
    pub n: u32,
    pub correlation: f64,
    pub shrinkage: f64,
    pub lead_lag: u32,
    pub lead_corr: f64,
    pub predictive_gain: f64,
    pub uncertainty: f64,
    pub score: f64,
    pub rank: u32,
}

impl NewCrossPrior {
    pub fn pair_id(&self) -> String {
        format!("{}:{}:{}", self.timescale, self.source_instrument_id, self.target_instrument_id)
    }
}

/// A `RegionSet` read back (state included).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RegionSetRow {
    #[serde(rename = "$id")]
    pub id: u64,
    pub region_set_id: String,
    pub snapshot_id: String,
    #[serde(default)]
    pub timescale: String,
    #[serde(default)]
    pub expert_index: u32,
    #[serde(default)]
    pub n_regions: u32,
    #[serde(default)]
    pub cfg_hash: String,
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub kernel_state: String,
    #[serde(default)]
    pub updated_at: Option<String>,
}

/// One step of a lineage walk.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LineageLink {
    pub region: NodeRow,
    /// `carry | split | merge | partial` (`RegionMatch` edges report `carry`).
    pub kind: String,
    pub overlap: f64,
    /// 1 = direct parent / child.
    pub depth: usize,
    /// The region this link was reached from.
    pub via: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RegionLineage {
    pub region: NodeRow,
    /// Every region of the same `lineage_id` (across snapshots).
    pub lineage: Vec<NodeRow>,
    pub ancestors: Vec<LineageLink>,
    pub descendants: Vec<LineageLink>,
}

/// Appends an edge upsert to `batch` (entries suffixed `i`): the edge `label` from the node
/// found by `from` to the node found by `to`, matched among `from`'s out-edges by `matching`;
/// updated with `mutable` when present, else created with `fixed ++ mutable`.
#[allow(clippy::too_many_arguments)]
fn edge_upsert(
    batch: Batch,
    i: usize,
    label: &str,
    from: Op,
    to: Op,
    matching: Pred,
    fixed: Vec<(&str, Input)>,
    mutable: Vec<(&str, Input)>,
) -> Batch {
    let (f, t, ex, up, pr, cr) =
        (format!("f{i}"), format!("t{i}"), format!("ex{i}"), format!("up{i}"), format!("pr{i}"), format!("cr{i}"));
    let mut set = Op::edges(EdgeRef::Var(ex.clone()));
    for (k, v) in mutable.clone() {
        set = set.set_property(k, v);
    }
    batch
        .var(&f, from)
        .var(&t, to)
        .var(&ex, Op::nodes(NodeRef::var(&f)).out_e(Some(label)).where_(matching))
        .var_if(&up, Cond::VarNotEmpty(ex.clone()), set)
        // `add_e` with an empty source or target is a server error: probe both first.
        .var_if(&pr, Cond::VarEmpty(ex.clone()), Op::nodes(NodeRef::var(&t)))
        .var_if(
            &cr,
            Cond::VarNotEmpty(pr.clone()),
            Op::nodes(NodeRef::var(&f)).add_e(label, NodeRef::var(&t), [fixed, mutable].concat()),
        )
}

/// Node upsert by unique key, linking a newly created node from `parent` with `link`.
fn node_upsert(
    batch: Batch,
    label: &str,
    key: &str,
    value: &str,
    props: Vec<(&str, Input)>,
    parent: Option<(Op, &str)>,
) -> Batch {
    let mut set = Op::nodes(NodeRef::var("existing"));
    for (k, v) in props.clone() {
        set = set.set_property(k, v);
    }
    let mut create = vec![(key, s(value))];
    create.extend(props);
    let mut b = batch
        .var("existing", by_key(label, key, value))
        .var_if("updated", Cond::VarNotEmpty("existing".into()), set)
        .var_if("created", Cond::VarEmpty("existing".into()), Op::add_n(label, create));
    if let Some((parent_op, link)) = parent {
        b = b
            .var("parent", parent_op)
            .var_if("probe", Cond::VarNotEmpty("created".into()), Op::nodes(NodeRef::var("parent")))
            .var_if(
                "link",
                Cond::VarNotEmpty("probe".into()),
                Op::nodes(NodeRef::var("parent")).add_e(link, NodeRef::var("created"), vec![]),
            );
    }
    b
}

impl HubClient {
    /// Region statistics (`mass`, `occupancy`, `spread`, `age`, `updated_at`) by `region_id`, one
    /// write batch. Unknown ids are skipped.
    pub async fn update_region_stats(&self, rows: &[RegionStatsUpdate]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut b = Batch::write().named("update_region_stats");
        for (i, r) in rows.iter().enumerate() {
            b = b.var(
                &format!("r{i}"),
                by_key(node::LATENT_REGION, "region_id", &r.region_id)
                    .set_property("mass", r.mass)
                    .set_property("occupancy", r.occupancy)
                    .set_property("spread", r.spread)
                    .set_property("age", Input::Lit(Val::I64(r.age.min(i64::MAX as u64) as i64)))
                    .set_property("updated_at", Input::Now),
            );
        }
        self.batch(&b.returning(Vec::<String>::new())).await.map(drop)
    }

    /// Death mark: `status = retired`, `died_at = now`, `retire_reason` (`merged | support |
    /// unmatched`), `related_region_id` (the survivor of a merge, else empty); versioned as a
    /// `RegionStatusVersion` in the same batch ([`crate::versions`]).
    pub async fn mark_region_died(&self, region_id: &str, reason: &str, related_region_id: &str) -> Result<()> {
        let op = Op::nodes(NodeRef::var("region"))
            .set_property("status", "retired")
            .set_property("retire_reason", reason)
            .set_property("related_region_id", related_region_id)
            .set_property("died_at", Input::Now)
            .set_property("updated_at", Input::Now)
            .value_map(Some(&["$id"]));
        let batch = Batch::write()
            .named("mark_region_died")
            .var("region", Op::nodes_with_label_where(node::LATENT_REGION, Pred::eq("region_id", region_id)))
            .var_if(
                "version",
                Cond::VarNotEmpty("region".into()),
                crate::versions::status_version_node(region_id, "retired", related_region_id, reason),
            )
            .ret("row", op);
        let mut resp = self.batch(&batch).await?;
        let rows: Vec<Json> = resp.take("row")?;
        if rows.is_empty() {
            return Err(HubError::NotFound { label: node::LATENT_REGION.into(), id: region_id.into() });
        }
        Ok(())
    }

    /// Replaces the centroid `LatentVec` of a region (`kind = centroid`, `ref_id = region_id`).
    /// Zero or non-finite centroids are skipped (the cosine index rejects them).
    pub async fn upsert_centroid(&self, expert_space_id: &str, region_id: &str, centroid: &[f32]) -> Result<bool> {
        if centroid.is_empty() || !centroid.iter().all(|x| x.is_finite()) || centroid.iter().all(|x| *x == 0.0) {
            return Ok(false);
        }
        let old = Op::nodes_with_label_where(
            node::LATENT_VEC,
            Pred::and([Pred::eq("ref_id", region_id), Pred::eq("kind", "centroid")]),
        )
        .drop();
        self.batch(&Batch::write().named("drop_centroid").var("old", old).returning(Vec::<String>::new())).await?;
        let now =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64);
        self.add_latents(&[NewLatent {
            vector: centroid.to_vec(),
            expert_space_id: expert_space_id.to_owned(),
            kind: "centroid".into(),
            ref_id: region_id.to_owned(),
            instrument_id: String::new(),
            ts_ms: now,
            uncertainty: 0.0,
        }])
        .await?;
        Ok(true)
    }

    /// `DescendsFrom` edges child → parent, upserted by `(child, parent, kind)`; one batch.
    /// Pairs whose regions do not exist (yet) are skipped by the server-side probe.
    pub async fn upsert_descents(&self, rows: &[NewDescent]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut b = Batch::write().named("upsert_descents");
        for (i, d) in rows.iter().enumerate() {
            b = edge_upsert(
                b,
                i,
                labels::DESCENDS_FROM,
                by_key(node::LATENT_REGION, "region_id", &d.child_region_id),
                by_key(node::LATENT_REGION, "region_id", &d.parent_region_id),
                Pred::and([
                    Pred::eq("parent_region_id", d.parent_region_id.as_str()),
                    Pred::eq("kind", d.kind.as_str()),
                ]),
                vec![
                    ("child_region_id", s(&d.child_region_id)),
                    ("parent_region_id", s(&d.parent_region_id)),
                    ("kind", s(&d.kind)),
                ],
                vec![("overlap", d.overlap.into()), ("updated_at", Input::Now)],
            );
        }
        self.batch(&b.returning(Vec::<String>::new())).await.map(drop)
    }

    /// `Hier` edges fine → coarse, upserted by the pair; one batch.
    pub async fn upsert_hier_edges(&self, rows: &[NewHierEdge]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut b = Batch::write().named("upsert_hier_edges");
        for (i, h) in rows.iter().enumerate() {
            b = edge_upsert(
                b,
                i,
                edge::HIER,
                by_key(node::LATENT_REGION, "region_id", &h.fine_region_id),
                by_key(node::LATENT_REGION, "region_id", &h.coarse_region_id),
                Pred::eq("coarse_region_id", h.coarse_region_id.as_str()),
                vec![("fine_region_id", s(&h.fine_region_id)), ("coarse_region_id", s(&h.coarse_region_id))],
                vec![
                    ("weight", h.p_coarse_given_fine.into()),
                    ("p_coarse_given_fine", h.p_coarse_given_fine.into()),
                    ("p_fine_given_coarse", h.p_fine_given_coarse.into()),
                    ("support", h.support.into()),
                    ("pmi_bits", h.pmi_bits.into()),
                    ("mi_bits", h.mi_bits.into()),
                    ("updated_at", Input::Now),
                ],
            );
        }
        self.batch(&b.returning(Vec::<String>::new())).await.map(drop)
    }

    /// `RegionSet` upsert by `region_set_id = expert_space_id` (linked from its `ExpertSpace`).
    pub async fn upsert_region_set(&self, r: &NewRegionSet) -> Result<()> {
        let props = vec![
            ("snapshot_id", s(&r.snapshot_id)),
            ("timescale", s(&r.timescale)),
            ("expert_index", r.expert_index.into()),
            ("family", s(&r.family)),
            ("method", s(&r.method)),
            ("n_regions", r.n_regions.into()),
            ("cfg_hash", s(&r.cfg_hash)),
            ("state", s(&r.state)),
            ("kernel_state", s(&r.kernel_state)),
            ("fit_time", Input::Now),
            ("updated_at", Input::Now),
        ];
        let b = node_upsert(
            Batch::write().named("upsert_region_set"),
            labels::REGION_SET,
            "region_set_id",
            &r.expert_space_id,
            props,
            Some((by_key(node::EXPERT_SPACE, "expert_space_id", &r.expert_space_id), labels::HAS_REGION_SET)),
        );
        self.batch(&b.returning(Vec::<String>::new())).await.map(drop)
    }

    /// The stored `RegionSet` of an expert space.
    pub async fn region_set(&self, expert_space_id: &str) -> Result<Option<RegionSetRow>> {
        let op = by_key(labels::REGION_SET, "region_set_id", expert_space_id).value_map(None);
        let mut resp = self.batch(&Batch::read().named("region_set").ret("row", op)).await?;
        resp.take_opt("row")
    }

    /// `KernelModel` upsert by `kernel_id = expert_space_id` (linked from its `ExpertSpace`).
    pub async fn upsert_kernel_model(&self, k: &NewKernelModel) -> Result<()> {
        let props = vec![
            ("expert_space_id", s(&k.expert_space_id)),
            ("snapshot_id", s(&k.snapshot_id)),
            ("timescale", s(&k.timescale)),
            ("expert_index", k.expert_index.into()),
            ("kernel_type", s(&k.kernel_type)),
            ("serving_sampler", s(&k.serving_sampler)),
            ("scores", s(&k.scores)),
            ("stationary", s(&k.stationary)),
            ("relaxation_steps", k.relaxation_steps.into()),
            ("drift_kl", k.drift_kl.into()),
            ("n_rows", k.n_rows.into()),
            ("observation_mass", k.observation_mass.into()),
            ("live_regions", k.live_regions.into()),
            ("updated_at", Input::Now),
        ];
        let b = node_upsert(
            Batch::write().named("upsert_kernel_model"),
            labels::KERNEL_MODEL,
            "kernel_id",
            &k.expert_space_id,
            props,
            Some((by_key(node::EXPERT_SPACE, "expert_space_id", &k.expert_space_id), labels::HAS_KERNEL)),
        );
        self.batch(&b.returning(Vec::<String>::new())).await.map(drop)
    }

    /// Every `KernelModel` of a snapshot.
    pub async fn snapshot_kernels(&self, snapshot_id: &str) -> Result<Vec<NodeRow>> {
        let op = by_key(labels::KERNEL_MODEL, "snapshot_id", snapshot_id).value_map(None);
        let mut resp = self.batch(&Batch::read().named("snapshot_kernels").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// `CrossPrior` upserts by `pair_id = <ts>:<source>:<target>`; one batch per row.
    pub async fn upsert_cross_priors(&self, rows: &[NewCrossPrior]) -> Result<()> {
        for c in rows {
            let props = vec![
                ("timescale", s(&c.timescale)),
                ("source_instrument_id", s(&c.source_instrument_id)),
                ("target_instrument_id", s(&c.target_instrument_id)),
                ("n", c.n.into()),
                ("correlation", c.correlation.into()),
                ("shrinkage", c.shrinkage.into()),
                ("lead_lag", c.lead_lag.into()),
                ("lead_corr", c.lead_corr.into()),
                ("predictive_gain", c.predictive_gain.into()),
                ("uncertainty", finite(c.uncertainty).into()),
                ("score", c.score.into()),
                ("rank", c.rank.into()),
                ("updated_at", Input::Now),
            ];
            let b = node_upsert(
                Batch::write().named("upsert_cross_prior"),
                labels::CROSS_PRIOR,
                "pair_id",
                &c.pair_id(),
                props,
                None,
            );
            self.batch(&b.returning(Vec::<String>::new())).await?;
        }
        Ok(())
    }

    /// `CrossPrior` rows of a timescale, best score first.
    pub async fn cross_priors(&self, timescale: &str, limit: usize) -> Result<Vec<NodeRow>> {
        let op = by_key(labels::CROSS_PRIOR, "timescale", timescale)
            .order_by("score", true)
            .limit(limit.max(1))
            .value_map(None);
        let mut resp = self.batch(&Batch::read().named("cross_priors").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// `Hier` edges out of every region of a snapshot.
    pub async fn snapshot_hierarchy(&self, snapshot_id: &str) -> Result<Vec<EdgeRow>> {
        let b = Batch::read()
            .named("snapshot_hierarchy")
            .var("snap", by_key(node::SNAPSHOT, "snapshot_id", snapshot_id))
            .var("spaces", Op::nodes(NodeRef::var("snap")).out(Some(edge::HAS_SPACE)))
            .var("regions", Op::nodes(NodeRef::var("spaces")).r#in(Some(edge::IN_SPACE)))
            .ret("edges", Op::nodes(NodeRef::var("regions")).out_e(Some(edge::HIER)).edge_properties());
        let mut resp = self.batch(&b).await?;
        resp.take("edges")
    }

    /// Every region carrying `lineage_id`, across snapshots.
    pub async fn lineage_members(&self, lineage_id: &str) -> Result<Vec<NodeRow>> {
        let op = by_key(node::LATENT_REGION, "lineage_id", lineage_id).value_map(None);
        let mut resp = self.batch(&Batch::read().named("lineage_members").ret("rows", op)).await?;
        resp.take("rows")
    }

    async fn edges_of(&self, ids: &[u64], out: bool) -> Result<Vec<EdgeRow>> {
        let base = Op::nodes(NodeRef::Ids(ids.to_vec()));
        let (d, m) = if out {
            (base.clone().out_e(Some(labels::DESCENDS_FROM)), base.in_e(Some(edge::REGION_MATCH)))
        } else {
            (base.clone().in_e(Some(labels::DESCENDS_FROM)), base.out_e(Some(edge::REGION_MATCH)))
        };
        let b = Batch::read().named("lineage_edges").ret("d", d.edge_properties()).ret("m", m.edge_properties());
        let mut resp = self.batch(&b).await?;
        let mut v: Vec<EdgeRow> = resp.take("d")?;
        v.extend(resp.take::<Vec<EdgeRow>>("m")?);
        Ok(v)
    }

    async fn nodes_by_ids(&self, ids: &[u64]) -> Result<Vec<NodeRow>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let op = Op::nodes(NodeRef::Ids(ids.to_vec())).value_map(None);
        let mut resp = self.batch(&Batch::read().named("nodes_by_ids").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// Ancestors (`DescendsFrom` out, `RegionMatch` in) and descendants of a region up to
    /// `depth` levels, plus every region of its lineage. `None` when the region is unknown.
    pub async fn region_lineage(&self, region_id: &str, depth: usize) -> Result<Option<RegionLineage>> {
        let mut resp = self
            .batch(
                &Batch::read()
                    .named("region")
                    .ret("row", by_key(node::LATENT_REGION, "region_id", region_id).value_map(None)),
            )
            .await?;
        let Some(region) = resp.take_opt::<NodeRow>("row")? else { return Ok(None) };
        let lineage_id = region.properties.get("lineage_id").and_then(Json::as_str).unwrap_or_default().to_owned();
        let lineage = if lineage_id.is_empty() { Vec::new() } else { self.lineage_members(&lineage_id).await? };
        let mut out = [Vec::new(), Vec::new()];
        for (slot, up) in [(0usize, true), (1usize, false)] {
            let mut seen = vec![region.id];
            let mut frontier = vec![region.id];
            let mut names = std::collections::HashMap::from([(region.id, region_id.to_owned())]);
            for level in 1..=depth.max(1) {
                if frontier.is_empty() {
                    break;
                }
                let edges = self.edges_of(&frontier, up).await?;
                let next: Vec<(u64, &EdgeRow)> = edges
                    .iter()
                    .map(|e| (if (e.label == labels::DESCENDS_FROM) == up { e.to } else { e.from }, e))
                    .filter(|(n, _)| !seen.contains(n))
                    .collect();
                let ids: Vec<u64> = next.iter().map(|x| x.0).collect();
                let rows = self.nodes_by_ids(&ids).await?;
                frontier.clear();
                for (n, e) in next {
                    let Some(row) = rows.iter().find(|r| r.id == n) else { continue };
                    if seen.contains(&n) {
                        continue;
                    }
                    seen.push(n);
                    frontier.push(n);
                    let via_id = if n == e.to { e.from } else { e.to };
                    let rid = row.properties.get("region_id").and_then(Json::as_str).unwrap_or_default().to_owned();
                    names.insert(n, rid);
                    let kind = e.properties.get("kind").and_then(Json::as_str).unwrap_or("carry").to_owned();
                    let overlap = e
                        .properties
                        .get("overlap")
                        .or_else(|| e.properties.get("score"))
                        .and_then(Json::as_f64)
                        .unwrap_or(0.0);
                    out[slot].push(LineageLink {
                        region: row.clone(),
                        kind,
                        overlap,
                        depth: level,
                        via: names.get(&via_id).cloned().unwrap_or_default(),
                    });
                }
            }
        }
        let [ancestors, descendants] = out;
        Ok(Some(RegionLineage { region, lineage, ancestors, descendants }))
    }
}

fn finite(x: f64) -> f64 {
    if x.is_finite() { x } else { f64::MAX }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_upsert_probes_before_creating() {
        let b = edge_upsert(
            Batch::write(),
            0,
            labels::DESCENDS_FROM,
            by_key(node::LATENT_REGION, "region_id", "c"),
            by_key(node::LATENT_REGION, "region_id", "p"),
            Pred::eq("kind", "split"),
            vec![("kind", s("split"))],
            vec![("overlap", 0.4.into())],
        );
        let j = b.to_json().to_string();
        for v in ["\"f0\"", "\"t0\"", "\"ex0\"", "\"up0\"", "\"pr0\"", "\"cr0\"", "DescendsFrom", "overlap"] {
            assert!(j.contains(v), "{v} missing in {j}");
        }
        let n = node_upsert(
            Batch::write(),
            labels::KERNEL_MODEL,
            "kernel_id",
            "s-e0",
            vec![("drift_kl", 0.1.into())],
            Some((by_key(node::EXPERT_SPACE, "expert_space_id", "s-e0"), labels::HAS_KERNEL)),
        )
        .to_json()
        .to_string();
        assert!(n.contains("HasKernel") && n.contains("\"probe\"") && n.contains("kernel_id"));
        let c = NewCrossPrior {
            timescale: "1m".into(),
            source_instrument_id: "a".into(),
            target_instrument_id: "b".into(),
            n: 1,
            correlation: 0.0,
            shrinkage: 0.0,
            lead_lag: 1,
            lead_corr: 0.0,
            predictive_gain: 0.0,
            uncertainty: f64::INFINITY,
            score: 0.0,
            rank: 1,
        };
        assert_eq!(c.pair_id(), "1m:a:b");
        assert_eq!(finite(c.uncertainty), f64::MAX);
    }
}
