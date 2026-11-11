//! Helix rows of the decoder population beyond `evo.vary.decoder`'s lineage (spec 34 §5):
//!
//! - every `evo.eval.decoder` with a `Job.snapshot_id` upserts its `Genome` and writes
//!   `EvaluatedOn{generation, gen_index, rung, score, objectives, job_id}` → the `Snapshot` and
//!   `Reads{expert_id}` → each `Expert` of that snapshot its expert mask selects
//!   ([`Worker::record_evaluation`]);
//! - `evo.front.decoder` (sent by the orchestrator at every loop end, fire and forget) stores the
//!   loop's ranked candidates as a `ParetoFront` node with `InFront{rank, hv, crowding, score}`
//!   edges ([`Worker::record_front`]).
//!
//! All best effort: Helix being down never fails an evaluation.

use std::time::{Duration, Instant};

use fa_decoders::genome::DecoderGenome;
use fa_hub::{
    NewGenome,
    evolution::{Evaluation, FrontMember, NewParetoFront},
};
use fa_proto::v1::{Job, JobResult, job_result::Status};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::job::{Outcome, Worker};

pub const FRONT_KIND: &str = "evo.front.decoder";
pub const FRONT_PROTOCOL: &str = "evo.front.decoder/v0";
const HUB_TIMEOUT: Duration = Duration::from_secs(20);

/// Expert indices of a mask (`None` = mask 0, every expert).
pub fn mask_experts(mask: u32) -> Option<Vec<u32>> {
    (mask != 0).then(|| (0..32).filter(|i| mask & (1u32 << i) != 0).collect())
}

/// `Genome` row of a genome id as the orchestrator sent it for a generation of `kind`: decoder
/// genomes in their canonical form (as evo.vary.decoder writes them), others verbatim.
pub fn genome_row(id: &str, kind: &str) -> NewGenome {
    let decoder = kind.is_empty() || kind == crate::job::KIND;
    match DecoderGenome::parse(id) {
        Ok(g) if decoder => NewGenome {
            genome_id: g.to_string(),
            kind: "decoder".into(),
            family: g.family.as_str().into(),
            params: serde_json::to_string(&g.params()).unwrap_or_default(),
        },
        _ => NewGenome {
            genome_id: id.to_owned(),
            kind: if id.starts_with("alloc:") { "allocator" } else { "strategy" }.into(),
            family: id.split(':').next().unwrap_or("").chars().take(64).collect(),
            params: String::new(),
        },
    }
}

#[derive(Debug, Deserialize)]
pub struct FrontEntry {
    pub genome: String,
    #[serde(default)]
    pub rank: u32,
    #[serde(default)]
    pub crowding: f64,
    #[serde(default)]
    pub hv: f64,
    #[serde(default)]
    pub score: f64,
}

/// `evo.front.decoder` spec (the orchestrator's final `generation.summary` of a loop, trimmed).
#[derive(Debug, Deserialize)]
pub struct FrontSpec {
    pub generation_id: String,
    #[serde(default)]
    pub generation: i32,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub snapshot_id: String,
    #[serde(default)]
    pub keys: Vec<String>,
    #[serde(default)]
    pub fronts: Vec<Vec<FrontEntry>>,
    #[serde(default, rename = "final")]
    pub final_loop: bool,
}

impl FrontSpec {
    pub fn to_front(&self) -> NewParetoFront {
        let members: Vec<FrontMember> = self
            .fronts
            .iter()
            .flatten()
            .map(|m| FrontMember {
                genome_id: genome_row(&m.genome, &self.kind).genome_id,
                rank: m.rank,
                hv: m.hv,
                crowding: m.crowding,
                score: m.score,
            })
            .collect();
        let hypervolume = members.iter().filter(|m| m.rank == 0).map(|m| m.hv).sum();
        NewParetoFront {
            generation_id: self.generation_id.clone(),
            gen_index: self.generation,
            kind: self.kind.clone(),
            snapshot_id: self.snapshot_id.clone(),
            keys: self.keys.clone(),
            members,
            hypervolume,
            final_loop: self.final_loop,
        }
    }
}

impl Worker {
    /// `EvaluatedOn` + `Reads` of one evaluation (skipped without Helix or `snapshot_id`).
    pub(crate) async fn record_evaluation(&self, job: &Job, genome: &DecoderGenome, score: f64, objectives: &Value) -> Value {
        let Some(hub) = &self.hub else { return json!({"persisted": false, "reason": "no helix client"}) };
        if job.snapshot_id.is_empty() {
            return json!({"persisted": false, "reason": "job has no snapshot_id"});
        }
        let (gen_index, rung) = Self::parse_position(job);
        let row = NewGenome {
            genome_id: genome.to_string(),
            kind: "decoder".into(),
            family: genome.family.as_str().into(),
            params: serde_json::to_string(&genome.params()).unwrap_or_default(),
        };
        let e = Evaluation {
            genome_id: row.genome_id.clone(),
            snapshot_id: job.snapshot_id.clone(),
            generation: job.parent_job_id.clone(),
            gen_index,
            rung,
            score: if score.is_finite() { score } else { 0.0 },
            objectives: objectives.to_string(),
            job_id: job.job_id.clone(),
            experts: mask_experts(genome.expert_mask),
        };
        let res = tokio::time::timeout(HUB_TIMEOUT, async {
            hub.upsert_genome(&row).await?;
            hub.record_evaluation(&e).await
        })
        .await;
        match res {
            Ok(Ok(())) => json!({"persisted": true, "snapshot_id": e.snapshot_id, "reads": e.experts}),
            Ok(Err(err)) => json!({"persisted": false, "error": err.to_string()}),
            Err(_) => json!({"persisted": false, "error": "helix timeout"}),
        }
    }

    /// `evo.front.decoder`: stores the loop's ranked candidates as a `ParetoFront`.
    pub(crate) async fn record_front(&self, job: &Job) -> Outcome {
        let started = Instant::now();
        let spec: FrontSpec = match serde_json::from_slice(&job.spec) {
            Ok(s) => s,
            Err(e) => return self.failed(job, format!("front spec: {e}"), 0),
        };
        let front = spec.to_front();
        let helix = match &self.hub {
            None => json!({"persisted": false, "reason": "no helix client"}),
            Some(hub) => {
                let res = tokio::time::timeout(HUB_TIMEOUT, async {
                    for m in spec.fronts.iter().flatten() {
                        hub.upsert_genome(&genome_row(&m.genome, &spec.kind)).await?;
                    }
                    hub.upsert_pareto_front(&front).await
                })
                .await;
                match res {
                    Ok(Ok(())) => json!({"persisted": true, "front_id": front.front_id()}),
                    Ok(Err(e)) => json!({"persisted": false, "error": e.to_string()}),
                    Err(_) => json!({"persisted": false, "error": "helix timeout"}),
                }
            }
        };
        let payload = json!({"protocol": FRONT_PROTOCOL, "front_id": front.front_id(), "members": front.members.len(),
                             "front_size": front.members.iter().filter(|m| m.rank == 0).count(),
                             "hypervolume": front.hypervolume, "helix": helix});
        Outcome {
            result: JobResult {
                job_id: job.job_id.clone(),
                status: Status::Ok as i32,
                result: serde_json::to_vec(&payload).unwrap_or_default(),
                used_gpu_ms: started.elapsed().as_millis() as u64,
                worker_id: self.worker_id.clone(),
                ..Default::default()
            },
            outputs: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_rows_and_fronts() {
        assert_eq!(mask_experts(0), None);
        assert_eq!(mask_experts(0b101), Some(vec![0, 2]));
        let r = genome_row("nn_chart:lb=16;res=8", "evo.eval.decoder");
        assert_eq!((r.kind.as_str(), r.family.as_str()), ("decoder", "nn_chart"));
        assert!(r.genome_id.starts_with("nn_chart:ts=1s;"), "canonical: {}", r.genome_id);
        assert_eq!(genome_row("alloc:erc;w=1", "evo.eval.allocator").kind, "allocator");
        assert_eq!(genome_row("{ expr = \"x\" }", "evo.eval.strategy").kind, "strategy");
        let spec: FrontSpec = serde_json::from_value(json!({
            "generation_id": "01G", "generation": 1, "kind": "evo.eval.decoder", "keys": ["crps"], "final": true,
            "fronts": [[{"genome": "forecast:h=3", "rank": 0, "hv": 0.2, "crowding": 1.0e308, "score": 0.1},
                        {"genome": "retrieval:k=5", "rank": 0, "hv": 0.1}],
                       [{"genome": "nn_news:tw=5", "rank": 1}]]
        }))
        .unwrap();
        let f = spec.to_front();
        assert_eq!((f.front_id().as_str(), f.members.len(), f.final_loop), ("01G/1", 3, true));
        assert!((f.hypervolume - 0.3).abs() < 1e-12);
    }
}
