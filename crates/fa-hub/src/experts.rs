//! Expert identity in the graph (spec 32 §5, §6.5): one `Expert` node per surviving expert of a
//! learned (neural) snapshot, `ExpertOf` → its `Snapshot`, and `AlignedTo{cka, kind}` → the
//! parent-generation expert it continues (CKA matching of the trainer). `expert_uid` is
//! `<snapshot_id>/e<expert_id>`; `expert_id` is the stable expert index (the parent's index for
//! matched experts), so an expert's history is the `AlignedTo` chain.

use serde::{Deserialize, Serialize};

use crate::{
    HubClient, Result,
    ast::{Batch, Cond, Input, NodeRef, Op, Pred, Val},
    schema::node,
};

pub mod labels {
    pub const EXPERT: &str = "Expert";
    pub const EXPERT_OF: &str = "ExpertOf";
    pub const ALIGNED_TO: &str = "AlignedTo";
}

/// `<snapshot_id>/e<expert_id>`.
pub fn expert_uid(snapshot_id: &str, expert_id: u32) -> String {
    format!("{snapshot_id}/e{expert_id}")
}

/// One expert of a snapshot as the trainer writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewExpert {
    pub snapshot_id: String,
    pub timescale: String,
    /// Stable expert index (output position after the §6.5 reordering).
    pub expert_id: u32,
    pub dim: u32,
    pub usage: f64,
    pub skill: f64,
    pub born_gen: u32,
    /// JSON specialisation summary (top instruments / regimes by gate mass).
    pub specialisation: String,
}

/// An `AlignedTo` edge to write: child expert → parent-generation expert.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewAlignment {
    pub child_snapshot_id: String,
    pub child_expert_id: u32,
    pub parent_snapshot_id: String,
    pub parent_expert_id: u32,
    pub cka: f64,
    /// `carry` (matched ≥ threshold) | `partial`.
    pub kind: String,
}

/// An `Expert` row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertRow {
    #[serde(rename = "$id")]
    pub id: u64,
    pub expert_uid: String,
    pub snapshot_id: String,
    #[serde(default)]
    pub timescale: String,
    pub expert_id: u32,
    #[serde(default)]
    pub dim: u32,
    #[serde(default)]
    pub usage: f64,
    #[serde(default)]
    pub skill: f64,
    #[serde(default)]
    pub born_gen: u32,
    #[serde(default)]
    pub died_gen: Option<u32>,
    #[serde(default)]
    pub specialisation: Option<String>,
}

/// An `AlignedTo` edge as read back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentRow {
    #[serde(default)]
    pub cka: f64,
    #[serde(default)]
    pub kind: String,
}

fn s(v: &str) -> Input {
    Input::Lit(Val::Str(v.to_owned()))
}

fn by_key(label: &str, key: &str, value: &str) -> Op {
    Op::nodes_with_label_where(label, Pred::eq(key, value))
}

impl HubClient {
    /// `upsertExpert`: insert or update the expert by `expert_uid`; a new node gets its `ExpertOf`
    /// edge to the snapshot.
    pub async fn upsert_expert(&self, e: &NewExpert) -> Result<()> {
        let uid = expert_uid(&e.snapshot_id, e.expert_id);
        let props = vec![
            ("snapshot_id", s(&e.snapshot_id)),
            ("timescale", s(&e.timescale)),
            ("expert_id", e.expert_id.into()),
            ("dim", e.dim.into()),
            ("usage", e.usage.into()),
            ("skill", e.skill.into()),
            ("born_gen", e.born_gen.into()),
            ("specialisation", s(&e.specialisation)),
            ("updated_at", Input::Now),
        ];
        let mut set = Op::nodes(NodeRef::var("existing"));
        for (k, v) in props.clone() {
            set = set.set_property(k, v);
        }
        let mut create = vec![("expert_uid", s(&uid))];
        create.extend(props);
        let b = Batch::write()
            .named("upsert_expert")
            .var("existing", by_key(labels::EXPERT, "expert_uid", &uid))
            .var_if("updated", Cond::VarNotEmpty("existing".into()), set)
            .var_if("created", Cond::VarEmpty("existing".into()), Op::add_n(labels::EXPERT, create))
            .var("snap", by_key(node::SNAPSHOT, "snapshot_id", &e.snapshot_id))
            .var_if("probe", Cond::VarNotEmpty("created".into()), Op::nodes(NodeRef::var("snap")))
            .var_if(
                "link",
                Cond::VarNotEmpty("probe".into()),
                Op::nodes(NodeRef::var("created")).add_e(labels::EXPERT_OF, NodeRef::var("snap"), vec![]),
            );
        self.batch(&b.returning(Vec::<String>::new())).await.map(drop)
    }

    /// `alignExpert`: `AlignedTo{cka, kind}` child → parent (both must exist); the parent's
    /// `died_gen` stays unset (it lives on through the child).
    pub async fn align_expert(&self, a: &NewAlignment) -> Result<()> {
        let b = Batch::write()
            .named("align_expert")
            .var("child", by_key(labels::EXPERT, "expert_uid", &expert_uid(&a.child_snapshot_id, a.child_expert_id)))
            .var("parent", by_key(labels::EXPERT, "expert_uid", &expert_uid(&a.parent_snapshot_id, a.parent_expert_id)))
            .var_if("probe", Cond::VarNotEmpty("child".into()), Op::nodes(NodeRef::var("parent")))
            .var_if(
                "edge",
                Cond::VarNotEmpty("probe".into()),
                Op::nodes(NodeRef::var("child")).add_e(
                    labels::ALIGNED_TO,
                    NodeRef::var("parent"),
                    vec![("cka", a.cka.into()), ("kind", s(&a.kind))],
                ),
            );
        self.batch(&b.returning(Vec::<String>::new())).await.map(drop)
    }

    /// Marks a parent-generation expert that no child continues (`died_gen`).
    pub async fn retire_expert(&self, snapshot_id: &str, expert_id: u32, died_gen: u32) -> Result<()> {
        let b = Batch::write().named("retire_expert").var(
            "row",
            by_key(labels::EXPERT, "expert_uid", &expert_uid(snapshot_id, expert_id))
                .set_property("died_gen", died_gen),
        );
        self.batch(&b.returning(Vec::<String>::new())).await.map(drop)
    }

    /// `snapshotExperts`: experts of a snapshot by `expert_id`.
    pub async fn snapshot_experts(&self, snapshot_id: &str) -> Result<Vec<ExpertRow>> {
        let op = by_key(labels::EXPERT, "snapshot_id", snapshot_id).value_map(None);
        let mut resp = self.batch(&Batch::read().named("snapshot_experts").ret("rows", op)).await?;
        let mut rows: Vec<ExpertRow> = resp.take("rows")?;
        rows.sort_by_key(|r| r.expert_id);
        Ok(rows)
    }

    /// `expertLineage`: the experts an expert is `AlignedTo` (its parent-generation expert(s)).
    pub async fn expert_parents(&self, snapshot_id: &str, expert_id: u32) -> Result<Vec<ExpertRow>> {
        let op = by_key(labels::EXPERT, "expert_uid", &expert_uid(snapshot_id, expert_id))
            .out(Some(labels::ALIGNED_TO))
            .value_map(None);
        let mut resp = self.batch(&Batch::read().named("expert_parents").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// `AlignedTo` edge properties out of an expert.
    pub async fn expert_alignments(&self, snapshot_id: &str, expert_id: u32) -> Result<Vec<AlignmentRow>> {
        let op = by_key(labels::EXPERT, "expert_uid", &expert_uid(snapshot_id, expert_id))
            .out_e(Some(labels::ALIGNED_TO))
            .edge_properties();
        let mut resp = self.batch(&Batch::read().named("expert_alignments").ret("rows", op)).await?;
        resp.take("rows")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uid_and_batches_are_well_formed() {
        assert_eq!(expert_uid("jepa-v1-1m-g1-5", 2), "jepa-v1-1m-g1-5/e2");
        // the batch builders do not panic and name their queries (no server needed)
        let b = Batch::write().named("upsert_expert").var("existing", by_key(labels::EXPERT, "expert_uid", "x"));
        assert!(b.returning(Vec::<String>::new()).to_json().to_string().contains("Expert"));
    }

    /// `FA_HELIX_URL=http://127.0.0.1:6969 cargo test -p fa-hub experts -- --nocapture`: two
    /// generations, experts upserted (idempotent), `AlignedTo` written and read back.
    #[tokio::test]
    async fn live_expert_nodes_and_alignment() {
        let Ok(url) = std::env::var(crate::URL_ENV) else { return };
        let c = HubClient::new(url);
        c.ensure_schema_ready(std::time::Duration::from_secs(30)).await.expect("schema");
        let p = format!("test-experts-{}", std::process::id());
        let (a, b) = (format!("{p}-g0"), format!("{p}-g1"));
        for (id, g) in [(&a, 0), (&b, 1)] {
            c.add_snapshot(&crate::NewSnapshot {
                snapshot_id: id.clone(),
                timescale: "1m".into(),
                encoder_family: "jepa_vae".into(),
                generation: g,
                garage_key: String::new(),
                status: "candidate".into(),
            })
            .await
            .expect("snapshot");
            for e in 0..2u32 {
                let ne = NewExpert {
                    snapshot_id: id.clone(),
                    timescale: "1m".into(),
                    expert_id: e,
                    dim: 8,
                    usage: 0.5,
                    skill: 0.1 * f64::from(e + 1),
                    born_gen: g,
                    specialisation: "{}".into(),
                };
                c.upsert_expert(&ne).await.expect("upsert");
                c.upsert_expert(&NewExpert { usage: 0.6, ..ne }).await.expect("upsert again");
            }
        }
        c.align_expert(&NewAlignment {
            child_snapshot_id: b.clone(),
            child_expert_id: 1,
            parent_snapshot_id: a.clone(),
            parent_expert_id: 1,
            cka: 0.83,
            kind: "carry".into(),
        })
        .await
        .expect("align");
        c.retire_expert(&a, 0, 1).await.expect("retire");
        let rows = c.snapshot_experts(&b).await.expect("experts");
        assert_eq!(rows.iter().map(|r| (r.expert_id, r.usage)).collect::<Vec<_>>(), vec![(0, 0.6), (1, 0.6)]);
        let parents = c.expert_parents(&b, 1).await.expect("parents");
        assert_eq!(parents.len(), 1);
        assert_eq!(parents[0].expert_uid, expert_uid(&a, 1));
        let al = c.expert_alignments(&b, 1).await.expect("alignments");
        assert_eq!((al[0].cka, al[0].kind.as_str()), (0.83, "carry"));
        assert_eq!(c.snapshot_experts(&a).await.unwrap()[0].died_gen, Some(1));
        for (label, key) in [(labels::EXPERT, "expert_uid"), (node::SNAPSHOT, "snapshot_id")] {
            c.drop_nodes(label, Pred::starts_with(key, &p)).await.expect("cleanup");
        }
    }
}
