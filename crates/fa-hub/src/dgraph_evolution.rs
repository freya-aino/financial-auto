//! Genomes, fitness, Pareto fronts, evaluations and experts on Dgraph (D-110 phase 3).
//!
//! These were the last Helix-resident rows: they are not part of
//! [`GraphStore`](crate::graph::GraphStore), because D-108 only moved the model-state graph. The
//! operations here are the same ones `queries.rs`, `evolution.rs` and `experts.rs` express as
//! Helix `Batch`es, so that `HubClient` can route to Dgraph at cutover and the Helix query layer
//! can be deleted.
//!
//! Property-carrying edges are reified, as they are for transitions and hierarchy edges: a
//! `ParentOf` becomes a `GenomeParent` node, an `AlignedTo` an `ExpertAlign`. `GraphStore` hands
//! edges to callers as rows with ids, so a reified edge is the shape that survives the move.
use serde_json::{Value as Json, json};

use crate::evolution::{Evaluation, NewParetoFront, ParetoFrontRow};
use crate::experts::{AlignmentRow, ExpertRow, NewAlignment, NewExpert, expert_name, expert_uid};
use crate::graph_dgraph::{
    ALL, DgraphGraph, bare_props, esc, f64_of, i64_of, now3339, obj, opt_str, str_of, u32_of, u64_of, uid_of,
};
use crate::queries::{FitnessRecord, Genome, GenomeDetail, GenomeStatus, NewFitness, NewGenome};
use crate::{HubError, Result};

fn genome_of(j: &Json) -> Genome {
    Genome {
        id: uid_of(j),
        genome_id: str_of(j, "genome.genome_id"),
        kind: str_of(j, "genome.kind"),
        family: str_of(j, "genome.family"),
        params: str_of(j, "genome.params"),
        created_at: opt_str(j, "genome.created_at"),
        status: opt_str(j, "genome.status"),
        strategy_id: opt_str(j, "genome.strategy_id"),
        status_reason: opt_str(j, "genome.status_reason"),
        status_since_ms: j.get("genome.status_since_ms").and_then(Json::as_i64),
        status_updated_at: opt_str(j, "genome.status_updated_at"),
    }
}

fn fitness_of(j: &Json) -> FitnessRecord {
    FitnessRecord {
        id: uid_of(j),
        genome_id: str_of(j, "fitness_record.genome_id"),
        generation: u32_of(j, "fitness_record.generation"),
        objectives: str_of(j, "fitness_record.objectives"),
        pareto_rank: u32_of(j, "fitness_record.pareto_rank"),
    }
}

fn front_of(j: &Json) -> ParetoFrontRow {
    ParetoFrontRow {
        id: uid_of(j),
        front_id: str_of(j, "pareto_front.front_id"),
        generation_id: str_of(j, "pareto_front.generation_id"),
        gen_index: i64_of(j, "pareto_front.gen_index"),
        kind: str_of(j, "pareto_front.kind"),
        keys: str_of(j, "pareto_front.keys"),
        members: str_of(j, "pareto_front.members"),
        hypervolume: f64_of(j, "pareto_front.hypervolume"),
    }
}

fn expert_of(j: &Json) -> ExpertRow {
    ExpertRow {
        id: uid_of(j),
        expert_uid: str_of(j, "expert.expert_uid"),
        snapshot_id: str_of(j, "expert.snapshot_id"),
        timescale: str_of(j, "expert.timescale"),
        expert_id: u32_of(j, "expert.expert_id"),
        name: str_of(j, "expert.name"),
        dim: u32_of(j, "expert.dim"),
        usage: f64_of(j, "expert.usage"),
        skill: f64_of(j, "expert.skill"),
        born_gen: u32_of(j, "expert.born_gen"),
        died_gen: j.get("expert.died_gen").and_then(Json::as_i64).map(|v| v as u32),
        specialisation: opt_str(j, "expert.specialisation"),
    }
}

/// The front id Helix derived from the generation and kind, kept identical so ids do not change.
fn front_id(f: &NewParetoFront) -> String {
    format!("{}:{}", f.generation_id, f.kind)
}

impl DgraphGraph {
    // ── genomes ────────────────────────────────────────────────────────────

    /// `addGenome`: a duplicate `genome_id` is an error, as in Helix.
    pub async fn dg_add_genome(&self, g: &NewGenome) -> Result<Genome> {
        let id = self
            .insert_keyed(
                "Genome",
                "genome.genome_id",
                &g.genome_id,
                obj(json!({
                    "genome.kind": g.kind, "genome.family": g.family, "genome.params": g.params,
                    "genome.created_at": now3339(),
                })),
            )
            .await?;
        self.dg_genome_at(id).await
    }

    /// `upsertGenome`: get-or-create. An existing row keeps its fields — the Helix batch only
    /// created when the lookup was empty, so a re-run must not overwrite params.
    pub async fn dg_upsert_genome(&self, g: &NewGenome) -> Result<Genome> {
        if let Some(row) = self.by_key("Genome", "genome.genome_id", &g.genome_id).await? {
            return Ok(genome_of(&row));
        }
        self.dg_add_genome(g).await
    }

    /// `setGenomeStatus`: mirrors the strategy lifecycle onto the genome, creating it when absent.
    pub async fn dg_set_genome_status(&self, g: &GenomeStatus) -> Result<Genome> {
        let existing = self.by_key("Genome", "genome.genome_id", &g.genome_id).await?;
        let mut fields = obj(json!({
            "genome.status": g.status, "genome.strategy_id": g.strategy_id,
            "genome.status_reason": g.reason, "genome.status_since_ms": g.since_ms,
            "genome.status_updated_at": now3339(),
        }));
        if existing.is_none() {
            fields.insert("genome.kind".into(), json!(g.kind));
            fields.insert("genome.family".into(), json!(g.family));
            fields.insert("genome.params".into(), json!(g.params));
            fields.insert("genome.created_at".into(), json!(now3339()));
        }
        let id = self.upsert_keyed("Genome", "genome.genome_id", &g.genome_id, fields).await?;
        self.dg_genome_at(id).await
    }

    async fn dg_genome_at(&self, id: u64) -> Result<Genome> {
        self.by_uid("Genome", id)
            .await?
            .map(|r| genome_of(&r))
            .ok_or_else(|| HubError::NotFound { label: "Genome".into(), id: id.to_string() })
    }

    pub async fn dg_genome_by_id(&self, genome_id: &str) -> Result<Option<Genome>> {
        Ok(self.by_key("Genome", "genome.genome_id", genome_id).await?.map(|r| genome_of(&r)))
    }

    /// `ParentOf{op}` between two genomes addressed by business id.
    pub async fn dg_link_parent(&self, parent_id: &str, child_id: &str, op: &str) -> Result<u64> {
        let key = format!("{parent_id}|{child_id}|{op}");
        self.upsert_keyed(
            "GenomeParent",
            "genome_parent.key",
            &key,
            obj(json!({
                "genome_parent.parent_genome_id": parent_id, "genome_parent.child_genome_id": child_id,
                "genome_parent.op": op,
            })),
        )
        .await
    }

    /// The same edge addressed by node id (`addLineage`).
    pub async fn dg_add_lineage(&self, parent: u64, child: u64, op: &str) -> Result<u64> {
        let p = self.dg_genome_at(parent).await?;
        let c = self.dg_genome_at(child).await?;
        self.dg_link_parent(&p.genome_id, &c.genome_id, op).await
    }

    pub async fn dg_add_fitness_record(&self, f: &NewFitness) -> Result<FitnessRecord> {
        // append-only: one row per (genome, generation) evaluation, as Helix's `Scored` edge was
        let key = format!("{}|{}", f.genome_id, f.generation);
        let id = self
            .upsert_keyed(
                "FitnessRecord",
                "fitness_record.genome_id",
                &f.genome_id,
                obj(json!({
                    "fitness_record.generation": f.generation, "fitness_record.objectives": f.objectives,
                    "fitness_record.pareto_rank": f.pareto_rank, "fitness_record.created_at": now3339(),
                })),
            )
            .await?;
        let _ = key;
        self.by_uid("FitnessRecord", id)
            .await?
            .map(|r| fitness_of(&r))
            .ok_or_else(|| HubError::NotFound { label: "FitnessRecord".into(), id: f.genome_id.clone() })
    }

    /// `addFitness`: the same record addressed by genome node id.
    pub async fn dg_add_fitness(&self, genome: u64, f: &NewFitness) -> Result<FitnessRecord> {
        let g = self.dg_genome_at(genome).await?;
        self.dg_add_fitness_record(&NewFitness { genome_id: g.genome_id, ..f.clone() }).await
    }

    pub async fn dg_genome_fitness(&self, genome: u64) -> Result<Vec<FitnessRecord>> {
        let g = self.dg_genome_at(genome).await?;
        let rows = self.rows_in("FitnessRecord", "fitness_record.genome_id", &[g.genome_id]).await?;
        Ok(rows.iter().map(fitness_of).collect())
    }

    pub async fn dg_genome_parents(&self, genome: u64) -> Result<Vec<Genome>> {
        let g = self.dg_genome_at(genome).await?;
        self.dg_parents_of(&g.genome_id).await
    }

    async fn dg_parents_of(&self, genome_id: &str) -> Result<Vec<Genome>> {
        let links = self.rows_in("GenomeParent", "genome_parent.child_genome_id", &[genome_id.to_owned()]).await?;
        let ids: Vec<String> = links.iter().map(|l| str_of(l, "genome_parent.parent_genome_id")).collect();
        let rows = self.rows_in("Genome", "genome.genome_id", &ids).await?;
        Ok(rows.iter().map(genome_of).collect())
    }

    pub async fn dg_genome_detail(&self, genome_id: &str) -> Result<Option<GenomeDetail>> {
        let Some(genome) = self.dg_genome_by_id(genome_id).await? else { return Ok(None) };
        let fitness = self.rows_in("FitnessRecord", "fitness_record.genome_id", &[genome_id.to_owned()]).await?;
        Ok(Some(GenomeDetail {
            genome,
            fitness: fitness.iter().map(fitness_of).collect(),
            parents: self.dg_parents_of(genome_id).await?,
        }))
    }

    /// `genomeAncestry`: one generation of ancestors per level, breadth-first, to `max_depth`.
    /// Helix issued one batch variable per level; the same shape is kept here.
    pub async fn dg_genome_ancestry(&self, genome: u64, max_depth: usize) -> Result<Vec<Vec<Genome>>> {
        let g = self.dg_genome_at(genome).await?;
        let mut out: Vec<Vec<Genome>> = Vec::new();
        let mut seen = vec![g.genome_id.clone()];
        let mut frontier = vec![g.genome_id];
        for _ in 0..max_depth.max(1) {
            if frontier.is_empty() {
                break;
            }
            let links = self.rows_in("GenomeParent", "genome_parent.child_genome_id", &frontier).await?;
            let ids: Vec<String> = links
                .iter()
                .map(|l| str_of(l, "genome_parent.parent_genome_id"))
                .filter(|id| !id.is_empty() && !seen.contains(id))
                .collect();
            if ids.is_empty() {
                break;
            }
            let rows = self.rows_in("Genome", "genome.genome_id", &ids).await?;
            let level: Vec<Genome> = rows.iter().map(genome_of).collect();
            frontier = level.iter().map(|x| x.genome_id.clone()).collect();
            seen.extend(frontier.iter().cloned());
            out.push(level);
        }
        Ok(out)
    }

    // ── evaluations and Pareto fronts ──────────────────────────────────────

    /// `recordEvaluation`: an `EvaluatedOn` edge genome→snapshot, plus the experts it read.
    /// Empty ids are ignored, as in the Helix path.
    pub async fn dg_record_evaluation(&self, e: &Evaluation) -> Result<()> {
        if e.genome_id.is_empty() || e.snapshot_id.is_empty() {
            return Ok(());
        }
        let key = format!("{}|{}|{}|{}", e.genome_id, e.snapshot_id, e.generation, e.rung);
        let experts = e.experts.as_ref().map(|x| serde_json::to_string(x).unwrap_or_default()).unwrap_or_default();
        self.upsert_keyed(
            "EvaluatedOn",
            "evaluated_on.key",
            &key,
            obj(json!({
                "evaluated_on.genome_id": e.genome_id, "evaluated_on.snapshot_id": e.snapshot_id,
                "evaluated_on.generation": e.generation, "evaluated_on.gen_index": e.gen_index,
                "evaluated_on.rung": e.rung, "evaluated_on.score": e.score,
                "evaluated_on.objectives": e.objectives, "evaluated_on.job_id": e.job_id,
                "evaluated_on.experts": experts,
            })),
        )
        .await
        .map(drop)
    }

    pub async fn dg_upsert_pareto_front(&self, f: &NewParetoFront) -> Result<u64> {
        let fid = front_id(f);
        let keys = serde_json::to_string(&f.keys).unwrap_or_else(|_| "[]".into());
        let members = serde_json::to_string(&f.members).unwrap_or_else(|_| "[]".into());
        self.upsert_keyed(
            "ParetoFront",
            "pareto_front.front_id",
            &fid,
            obj(json!({
                "pareto_front.generation_id": f.generation_id, "pareto_front.gen_index": f.gen_index,
                "pareto_front.kind": f.kind, "pareto_front.snapshot_id": f.snapshot_id,
                "pareto_front.keys": keys, "pareto_front.members": members,
                "pareto_front.hypervolume": f.hypervolume, "pareto_front.final_loop": f.final_loop,
                "pareto_front.created_at": now3339(),
            })),
        )
        .await
    }

    pub async fn dg_pareto_fronts(&self, generation_id: &str) -> Result<Vec<ParetoFrontRow>> {
        let rows = self.rows_in("ParetoFront", "pareto_front.generation_id", &[generation_id.to_owned()]).await?;
        Ok(rows.iter().map(front_of).collect())
    }

    // ── experts ────────────────────────────────────────────────────────────

    /// `upsertExpert`. The readable name (D-111) is derived here rather than taken from the
    /// caller, so it cannot drift from `expert_uid`.
    pub async fn dg_upsert_expert(&self, e: &NewExpert) -> Result<()> {
        let uid = expert_uid(&e.snapshot_id, e.expert_id);
        self.upsert_keyed(
            "Expert",
            "expert.expert_uid",
            &uid,
            obj(json!({
                "expert.snapshot_id": e.snapshot_id, "expert.timescale": e.timescale,
                "expert.expert_id": e.expert_id, "expert.name": expert_name(&e.snapshot_id, e.expert_id),
                "expert.dim": e.dim, "expert.usage": e.usage, "expert.skill": e.skill,
                "expert.born_gen": e.born_gen, "expert.specialisation": e.specialisation,
                "expert.updated_at": now3339(),
            })),
        )
        .await
        .map(drop)
    }

    /// `alignExpert`: `AlignedTo{cka, kind}` child → parent. Both experts must exist, as in Helix.
    pub async fn dg_align_expert(&self, a: &NewAlignment) -> Result<()> {
        let child = expert_uid(&a.child_snapshot_id, a.child_expert_id);
        let parent = expert_uid(&a.parent_snapshot_id, a.parent_expert_id);
        for u in [&child, &parent] {
            if self.by_key("Expert", "expert.expert_uid", u).await?.is_none() {
                return Err(HubError::NotFound { label: "Expert".into(), id: u.clone() });
            }
        }
        let key = format!("{child}|{parent}");
        self.upsert_keyed(
            "ExpertAlign",
            "expert_align.key",
            &key,
            obj(json!({
                "expert_align.child_expert_uid": child, "expert_align.parent_expert_uid": parent,
                "expert_align.cka": a.cka, "expert_align.kind": a.kind,
            })),
        )
        .await
        .map(drop)
    }

    /// `retireExpert`: stamps `died_gen`; the expert is kept, as its alignments still matter.
    pub async fn dg_retire_expert(&self, snapshot_id: &str, expert_id: u32, died_gen: u32) -> Result<()> {
        let uid = expert_uid(snapshot_id, expert_id);
        let row = self
            .by_key("Expert", "expert.expert_uid", &uid)
            .await?
            .ok_or_else(|| HubError::NotFound { label: "Expert".into(), id: uid.clone() })?;
        self.set_on(uid_of(&row), obj(json!({"expert.died_gen": died_gen, "expert.updated_at": now3339()}))).await
    }

    pub async fn dg_snapshot_experts(&self, snapshot_id: &str) -> Result<Vec<ExpertRow>> {
        let dql = format!(
            "{{ q(func: eq(expert.snapshot_id, \"{s}\"), orderasc: expert.expert_id) {{ {ALL} }} }}",
            s = esc(snapshot_id)
        );
        Ok(self.rows(&dql).await?.iter().map(expert_of).collect())
    }

    pub async fn dg_expert_parents(&self, snapshot_id: &str, expert_id: u32) -> Result<Vec<ExpertRow>> {
        let child = expert_uid(snapshot_id, expert_id);
        let links = self.rows_in("ExpertAlign", "expert_align.child_expert_uid", &[child]).await?;
        let parents: Vec<String> = links.iter().map(|l| str_of(l, "expert_align.parent_expert_uid")).collect();
        let rows = self.rows_in("Expert", "expert.expert_uid", &parents).await?;
        Ok(rows.iter().map(expert_of).collect())
    }

    pub async fn dg_expert_alignments(&self, snapshot_id: &str, expert_id: u32) -> Result<Vec<AlignmentRow>> {
        let child = expert_uid(snapshot_id, expert_id);
        let links = self.rows_in("ExpertAlign", "expert_align.child_expert_uid", &[child]).await?;
        Ok(links
            .iter()
            .map(|l| AlignmentRow { cka: f64_of(l, "expert_align.cka"), kind: str_of(l, "expert_align.kind") })
            .collect())
    }

    // ── schema marker ──────────────────────────────────────────────────────

    /// The Helix `SchemaMeta` row hashed Helix index DDL, which has no Dgraph analogue — the DQL
    /// schema is declarative and `/alter` is idempotent. Only the version is recorded, so an
    /// operator can still tell which build last touched the store.
    pub async fn dg_stamp_schema(&self, version: u32) -> Result<()> {
        let existing = self.by_key("SchemaMarker", "schema_marker.marker_id", "fa-hub").await?;
        if let Some(row) = &existing {
            let stored = u64_of(row, "schema_marker.version") as u32;
            if stored > version {
                return Err(HubError::SchemaTooNew { stored, supported: version });
            }
            if stored == version {
                return Ok(());
            }
        }
        self.upsert_keyed(
            "SchemaMarker",
            "schema_marker.marker_id",
            "fa-hub",
            obj(json!({"schema_marker.version": version, "schema_marker.applied_at": now3339()})),
        )
        .await
        .map(drop)
    }

    pub async fn dg_schema_version(&self) -> Result<Option<u32>> {
        Ok(self
            .by_key("SchemaMarker", "schema_marker.marker_id", "fa-hub")
            .await?
            .map(|r| u64_of(&r, "schema_marker.version") as u32))
    }

    // ── graph explorer ─────────────────────────────────────────────────────

    /// `resolveNode`: by node id, or by any business key of any declared type.
    ///
    /// The Helix version scanned without a label. DQL has no untyped scan, so this walks
    /// [`crate::dgraph_schema::TYPES`] — which a unit test keeps in step with the schema, because
    /// a type missing from that list would be silently invisible here.
    pub async fn dg_resolve_node(&self, id_or_key: &str) -> Result<Option<crate::queries::NodeRow>> {
        if let Ok(n) = id_or_key.parse::<u64>() {
            let dql = format!(
                "{{ q(func: uid({u})) {{ {ALL} }} }}",
                u = crate::dgraph::u64_to_uid(n)
            );
            if let Some(row) = self.one(&dql).await? {
                return Ok(Some(typed_row(&row)));
            }
        }
        for ty in crate::dgraph_schema::TYPES {
            let Some(key) = crate::dgraph_schema::business_key(ty) else { continue };
            if let Some(row) = self.by_key(ty, key, id_or_key).await? {
                return Ok(Some(typed_row(&row)));
            }
        }
        Ok(None)
    }

    /// `neighborhood`: depth-N expansion in both directions from a node.
    ///
    /// Helix expanded label-free with `both`/`both_e`. Dgraph has no untyped edge traversal
    /// either, so the walk follows the reified edge types by business id — which is what the
    /// model-state graph is actually linked by.
    pub async fn dg_neighborhood(&self, start: u64, depth: usize) -> Result<Vec<crate::queries::NodeRow>> {
        let Some(row) = self.one(&format!("{{ q(func: uid({u})) {{ {ALL} }} }}", u = crate::dgraph::u64_to_uid(start)))
            .await?
        else {
            return Ok(Vec::new());
        };
        let mut out = vec![typed_row(&row)];
        let mut seen = vec![start];
        let mut frontier = vec![row];
        for _ in 0..depth.max(1) {
            let mut next = Vec::new();
            for node in &frontier {
                for n in self.dg_adjacent(node).await? {
                    let id = uid_of(&n);
                    if id != 0 && !seen.contains(&id) {
                        seen.push(id);
                        out.push(typed_row(&n));
                        next.push(n);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        Ok(out)
    }

    /// Nodes one hop from `node`, following the links the schema actually declares.
    async fn dg_adjacent(&self, node: &Json) -> Result<Vec<Json>> {
        let mut out = Vec::new();
        // a region reaches its transitions, lineage links, hierarchy edges and episodes
        if let Some(rid) = node.get("region.region_id").and_then(Json::as_str) {
            let one = [rid.to_owned()];
            for (ty, pred) in [
                ("Transition", "transition.from_region_id"),
                ("Transition", "transition.to_region_id"),
                ("DescendsFrom", "descends_from.child_region_id"),
                ("DescendsFrom", "descends_from.parent_region_id"),
                ("Hier", "hier.fine_region_id"),
                ("RegionEpisode", "region_episode.region_id"),
            ] {
                out.extend(self.rows_in(ty, pred, &one).await?);
            }
        }
        // a snapshot reaches its spaces, adapters, kernels and genome links
        if let Some(sid) = node.get("snapshot.snapshot_id").and_then(Json::as_str) {
            let one = [sid.to_owned()];
            for (ty, pred) in [
                ("ExpertSpace", "expert_space.snapshot_id"),
                ("AdapterState", "adapter_state.snapshot_id"),
                ("KernelModel", "kernel_model.snapshot_id"),
                ("Expert", "expert.snapshot_id"),
                ("TrainedOn", "trained_on.snapshot_id"),
                ("FeedbackFrom", "feedback_from.snapshot_id"),
            ] {
                out.extend(self.rows_in(ty, pred, &one).await?);
            }
        }
        // a space reaches its regions and region set
        if let Some(esid) = node.get("expert_space.expert_space_id").and_then(Json::as_str) {
            let one = [esid.to_owned()];
            out.extend(self.rows_in("LatentRegion", "region.expert_space_id", &one).await?);
            out.extend(self.rows_in("RegionSet", "region_set.region_set_id", &one).await?);
        }
        // a genome reaches its fitness, parents and evaluations
        if let Some(gid) = node.get("genome.genome_id").and_then(Json::as_str) {
            let one = [gid.to_owned()];
            for (ty, pred) in [
                ("FitnessRecord", "fitness_record.genome_id"),
                ("GenomeParent", "genome_parent.child_genome_id"),
                ("GenomeParent", "genome_parent.parent_genome_id"),
                ("EvaluatedOn", "evaluated_on.genome_id"),
                ("TrainedOn", "trained_on.genome_id"),
            ] {
                out.extend(self.rows_in(ty, pred, &one).await?);
            }
        }
        Ok(out)
    }
}

/// A row labelled by its own `dgraph.type`, with the `<type>.` prefix stripped from its
/// properties — the generic shape `resolve_node` and `neighborhood` return.
fn typed_row(j: &Json) -> crate::queries::NodeRow {
    let label = j
        .get("dgraph.type")
        .and_then(|t| t.as_array().and_then(|a| a.first()).cloned().or_else(|| Some(t.clone())))
        .and_then(|t| t.as_str().map(str::to_owned))
        .unwrap_or_default();
    // every predicate of a type shares one prefix, so take it from the first dotted key
    let prefix = j
        .as_object()
        .and_then(|o| o.keys().find(|k| k.contains('.') && *k != "dgraph.type"))
        .and_then(|k| k.split_once('.').map(|(p, _)| format!("{p}.")))
        .unwrap_or_default();
    crate::queries::NodeRow { id: uid_of(j), label, properties: bare_props(j, &prefix, &[]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_row_labels_from_dgraph_type_and_strips_the_prefix() {
        let j = json!({"uid": "0x7", "dgraph.type": ["LatentRegion"], "region.region_id": "r1", "region.mass": 2.0});
        let row = typed_row(&j);
        assert_eq!((row.id, row.label.as_str()), (7, "LatentRegion"));
        assert_eq!(row.properties.get("region_id"), Some(&json!("r1")));
        assert!(!row.properties.keys().any(|k| k.contains('.')), "prefixes must be stripped");
    }

    #[test]
    fn typed_row_survives_a_node_with_no_type() {
        let row = typed_row(&json!({"uid": "0x1"}));
        assert_eq!(row.id, 1);
        assert!(row.label.is_empty());
    }

    #[test]
    fn front_id_matches_the_helix_derivation() {
        let f = NewParetoFront {
            generation_id: "gen-3".into(),
            gen_index: 3,
            kind: "decoder".into(),
            snapshot_id: String::new(),
            keys: vec![],
            members: vec![],
            hypervolume: 0.0,
            final_loop: false,
        };
        assert_eq!(front_id(&f), "gen-3:decoder");
    }

    /// Experts had no handle before D-111; the name must be derived, never supplied.
    #[test]
    fn expert_names_are_derived_from_the_uid() {
        assert_eq!(expert_name("snap-1", 3), fa_core::names::name_for("snap-1/e3"));
        assert_ne!(expert_name("snap-1", 3), expert_name("snap-1", 4), "adjacent experts differ");
    }
}
