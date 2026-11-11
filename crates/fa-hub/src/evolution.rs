//! Decoder-population rows of spec 34 §5 beyond `Genome` / `ParentOf` / `Scored` / `TrainedOn`:
//!
//! - `READS{expert_id}`: decoder `Genome` → each `Expert` of the snapshot its expert mask selects
//!   (mask 0 = every expert of the snapshot);
//! - `EVALUATED_ON{generation, loop, rung, score, objectives, job_id}`: `Genome` → the encoder
//!   `Snapshot` whose latents an evaluation read, upserted per `(genome, snapshot)` (latest
//!   evaluation wins);
//! - `ParetoFront{front_id, generation_id, loop, kind, keys, members, hypervolume}` per loop of a
//!   generation, with `InFront{rank, hv, crowding, score}` → each member `Genome`.
//!
//! Written best effort by evo-worker (`evo.eval.decoder` → `EVALUATED_ON` + `READS`,
//! `evo.front.decoder` → `ParetoFront`); missing genomes / snapshots / experts are skipped by the
//! server-side probes, never an error.

use serde::{Deserialize, Serialize};

use crate::{
    HubClient, Result,
    ast::{Batch, Cond, Input, NodeRef, Op, Pred, Val},
    schema::node,
};

pub mod labels {
    pub const PARETO_FRONT: &str = "ParetoFront";
    pub const IN_FRONT: &str = "InFront";
    pub const READS: &str = "Reads";
    pub const EVALUATED_ON: &str = "EvaluatedOn";
}

fn s(v: &str) -> Input {
    Input::Lit(Val::Str(v.to_owned()))
}

fn by_key(label: &str, key: &str, value: &str) -> Op {
    Op::nodes_with_label_where(label, Pred::eq(key, value))
}

/// One evaluation of a genome on a snapshot (`EVALUATED_ON`) and the experts it read (`READS`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Evaluation {
    pub genome_id: String,
    pub snapshot_id: String,
    pub generation: String,
    pub gen_index: i32,
    pub rung: i32,
    pub score: f64,
    /// Objectives as JSON text.
    pub objectives: String,
    pub job_id: String,
    /// Expert indices read (`None` = every expert of the snapshot, mask 0).
    pub experts: Option<Vec<u32>>,
}

/// One member of a stored front.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FrontMember {
    pub genome_id: String,
    pub rank: u32,
    pub hv: f64,
    pub crowding: f64,
    pub score: f64,
}

/// A loop's Pareto front.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NewParetoFront {
    pub generation_id: String,
    pub gen_index: i32,
    pub kind: String,
    pub snapshot_id: String,
    /// Ranked objective names.
    pub keys: Vec<String>,
    /// Every ranked candidate (front 0 has `rank` 0).
    pub members: Vec<FrontMember>,
    /// Σ hypervolume contribution of the rank-0 members (normalised objectives).
    pub hypervolume: f64,
    pub final_loop: bool,
}

impl NewParetoFront {
    /// `<generation_id>/<loop>`.
    pub fn front_id(&self) -> String {
        format!("{}/{}", self.generation_id, self.gen_index)
    }
}

/// A `ParetoFront` row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoFrontRow {
    #[serde(rename = "$id")]
    pub id: u64,
    pub front_id: String,
    pub generation_id: String,
    #[serde(default)]
    pub gen_index: i64,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub keys: String,
    #[serde(default)]
    pub members: String,
    #[serde(default)]
    pub hypervolume: f64,
}

impl HubClient {
    /// `EVALUATED_ON` (upsert per genome × snapshot) and `READS` edges of one evaluation.
    pub async fn record_evaluation(&self, e: &Evaluation) -> Result<()> {
        if e.genome_id.is_empty() || e.snapshot_id.is_empty() {
            return Ok(());
        }
        let mut b = Batch::write().named("record_evaluation");
        b = crate::regions::edge_upsert(
            b,
            0,
            labels::EVALUATED_ON,
            by_key(node::GENOME, "genome_id", &e.genome_id),
            by_key(node::SNAPSHOT, "snapshot_id", &e.snapshot_id),
            Pred::eq("snapshot_id", e.snapshot_id.as_str()),
            vec![("genome_id", s(&e.genome_id)), ("snapshot_id", s(&e.snapshot_id))],
            vec![
                ("generation", s(&e.generation)),
                ("gen_index", i64::from(e.gen_index).into()),
                ("rung", i64::from(e.rung).into()),
                ("score", e.score.into()),
                ("objectives", s(&e.objectives)),
                ("job_id", s(&e.job_id)),
                ("updated_at", Input::Now),
            ],
        );
        let experts: Vec<(u32, Op)> = match &e.experts {
            Some(ids) => ids
                .iter()
                .map(|i| (*i, by_key(crate::experts::labels::EXPERT, "expert_uid", &crate::expert_uid(&e.snapshot_id, *i))))
                .collect(),
            None => Vec::new(),
        };
        for (n, (i, op)) in experts.into_iter().enumerate() {
            b = crate::regions::edge_upsert(
                b,
                n + 1,
                labels::READS,
                by_key(node::GENOME, "genome_id", &e.genome_id),
                op,
                Pred::eq("snapshot_id", e.snapshot_id.as_str()),
                vec![("snapshot_id", s(&e.snapshot_id)), ("expert_id", i.into())],
                vec![("updated_at", Input::Now)],
            );
        }
        if e.experts.is_none() {
            // mask 0: every expert of the snapshot (one edge each, skipped when already there)
            b = b
                .var("g_all", by_key(node::GENOME, "genome_id", &e.genome_id))
                .var("x_all", by_key(crate::experts::labels::EXPERT, "snapshot_id", &e.snapshot_id))
                .var(
                    "have_all",
                    Op::nodes(NodeRef::var("g_all"))
                        .out_e(Some(labels::READS))
                        .where_(Pred::eq("snapshot_id", e.snapshot_id.as_str())),
                )
                .var_if("probe_all", Cond::VarEmpty("have_all".into()), Op::nodes(NodeRef::var("x_all")))
                .var_if(
                    "link_all",
                    Cond::VarNotEmpty("probe_all".into()),
                    Op::nodes(NodeRef::var("g_all")).add_e(
                        labels::READS,
                        NodeRef::var("x_all"),
                        vec![("snapshot_id", s(&e.snapshot_id)), ("updated_at", Input::Now)],
                    ),
                );
        }
        self.batch(&b.returning(Vec::<String>::new())).await.map(drop)
    }

    /// Upserts the loop's `ParetoFront` node by `front_id` and links each member genome that
    /// exists with `InFront{rank, hv, crowding, score}` (edges of a re-written front are added
    /// only for members not linked yet).
    pub async fn upsert_pareto_front(&self, f: &NewParetoFront) -> Result<()> {
        let id = f.front_id();
        let props = vec![
            ("generation_id", s(&f.generation_id)),
            ("gen_index", i64::from(f.gen_index).into()),
            ("kind", s(&f.kind)),
            ("snapshot_id", s(&f.snapshot_id)),
            ("keys", s(&serde_json::to_string(&f.keys).unwrap_or_default())),
            ("members", s(&serde_json::to_string(&f.members).unwrap_or_default())),
            ("front_size", (f.members.iter().filter(|m| m.rank == 0).count() as i64).into()),
            ("hypervolume", f.hypervolume.into()),
            ("final", f.final_loop.into()),
            ("updated_at", Input::Now),
        ];
        let mut set = Op::nodes(NodeRef::var("existing"));
        for (k, v) in props.clone() {
            set = set.set_property(k, v);
        }
        let mut create = vec![("front_id", s(&id))];
        create.extend(props);
        let mut b = Batch::write()
            .named("upsert_pareto_front")
            .var("existing", by_key(labels::PARETO_FRONT, "front_id", &id))
            .var_if("updated", Cond::VarNotEmpty("existing".into()), set)
            .var_if("created", Cond::VarEmpty("existing".into()), Op::add_n(labels::PARETO_FRONT, create));
        self.batch(&b.returning(Vec::<String>::new())).await?;
        // member edges (a second batch: the front node exists now)
        b = Batch::write().named("link_pareto_front");
        for (i, m) in f.members.iter().enumerate() {
            b = crate::regions::edge_upsert(
                b,
                i,
                labels::IN_FRONT,
                by_key(labels::PARETO_FRONT, "front_id", &id),
                by_key(node::GENOME, "genome_id", &m.genome_id),
                Pred::eq("genome_id", m.genome_id.as_str()),
                vec![("genome_id", s(&m.genome_id))],
                vec![
                    ("rank", i64::from(m.rank).into()),
                    ("hv", m.hv.into()),
                    ("crowding", if m.crowding.is_finite() { m.crowding } else { f64::MAX }.into()),
                    ("score", m.score.into()),
                ],
            );
        }
        if f.members.is_empty() {
            return Ok(());
        }
        self.batch(&b.returning(Vec::<String>::new())).await.map(drop)
    }

    /// `ParetoFront` rows of a generation (every loop).
    pub async fn pareto_fronts(&self, generation_id: &str) -> Result<Vec<ParetoFrontRow>> {
        let op = by_key(labels::PARETO_FRONT, "generation_id", generation_id).value_map(None);
        let mut resp = self.batch(&Batch::read().named("pareto_fronts").ret("rows", op)).await?;
        let mut rows: Vec<ParetoFrontRow> = resp.take("rows")?;
        rows.sort_by_key(|r| r.gen_index);
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_id_and_batches_build() {
        let f = NewParetoFront { generation_id: "01G".into(), gen_index: 2, ..Default::default() };
        assert_eq!(f.front_id(), "01G/2");
        let e = Evaluation { genome_id: "g".into(), snapshot_id: "s".into(), experts: Some(vec![0, 2]), ..Default::default() };
        assert_eq!(e.experts.as_ref().map(Vec::len), Some(2));
    }
}
