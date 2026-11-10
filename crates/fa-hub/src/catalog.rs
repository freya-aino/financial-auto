//! Named-query catalog for `POST /api/graph/query` (spec 50 §4.4): a **whitelist** of read-only
//! queries from `helix/queries.hx`, addressed by business ids, with JSON parameters. Clients never
//! send operation trees; control-api forwards `{name, params}` to [`HubClient::run_named`].
//!
//! Every entry maps to one typed function of [`crate::queries`], so the catalog cannot express
//! anything the typed layer cannot. Mutations are deliberately absent.

use serde::Serialize;
use serde_json::{Value as Json, json};

use crate::{HubClient, HubError, Result};

/// One catalog entry: name as in `helix/queries.hx`, and its JSON parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct NamedQuery {
    pub name: &'static str,
    /// `name: type` pairs; a trailing `?` marks an optional parameter.
    pub params: &'static [&'static str],
    pub doc: &'static str,
}

/// The whitelist. Order is the order `/api/graph/query` lists them in error replies.
pub const NAMED_QUERIES: &[NamedQuery] = &[
    NamedQuery { name: "getInstrument", params: &["instrument_id: string"], doc: "one Instrument row" },
    NamedQuery { name: "listSnapshots", params: &[], doc: "all Snapshot rows" },
    NamedQuery { name: "snapshotSpaces", params: &["snapshot_id: string"], doc: "ExpertSpaces of a snapshot" },
    NamedQuery {
        name: "snapshotGraph",
        params: &["snapshot_id: string"],
        doc: "snapshot, spaces, regions and transitions",
    },
    NamedQuery { name: "spaceRegions", params: &["expert_space_id: string"], doc: "LatentRegions of a space" },
    NamedQuery { name: "getRegion", params: &["region_id: string"], doc: "one LatentRegion row" },
    NamedQuery { name: "regionTransitions", params: &["region_id: string"], doc: "outgoing Transition edges" },
    NamedQuery { name: "regionNeighbors", params: &["region_id: string"], doc: "successor regions" },
    NamedQuery {
        name: "searchLatent",
        params: &["expert_space_id: string", "vector: [number]", "k: integer?"],
        doc: "k nearest LatentVecs in one expert space (default k = 20)",
    },
    NamedQuery { name: "genomeParents", params: &["genome_id: string"], doc: "direct parents" },
    NamedQuery { name: "genomeFitness", params: &["genome_id: string"], doc: "FitnessRecords of a genome" },
    NamedQuery {
        name: "genomeAncestry",
        params: &["genome_id: string", "max_depth: integer?"],
        doc: "ancestors per ParentOf hop (default 10)",
    },
    NamedQuery { name: "resolveNode", params: &["id: string"], doc: "node by numeric $id or any unique business id" },
    NamedQuery {
        name: "neighborhood",
        params: &["node: string", "depth: integer?"],
        doc: "nodes and edges within depth hops (default 1, max 4)",
    },
];

/// Hard cap on `neighborhood` depth through the catalog (every hop is one more batch entry).
pub const MAX_NEIGHBORHOOD_DEPTH: usize = 4;
/// Hard cap on `genomeAncestry` depth through the catalog.
pub const MAX_ANCESTRY_DEPTH: usize = 32;
/// Hard cap on `searchLatent` k through the catalog.
pub const MAX_SEARCH_K: usize = 500;

pub fn lookup(name: &str) -> Option<&'static NamedQuery> {
    NAMED_QUERIES.iter().find(|q| q.name == name)
}

pub fn names() -> Vec<&'static str> {
    NAMED_QUERIES.iter().map(|q| q.name).collect()
}

fn str_param<'a>(params: &'a Json, key: &str) -> Result<&'a str> {
    params
        .get(key)
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| HubError::BadParam(format!("`{key}` (string) is required")))
}

fn opt_usize(params: &Json, key: &str, default: usize, max: usize) -> Result<usize> {
    match params.get(key) {
        None | Some(Json::Null) => Ok(default),
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| HubError::BadParam(format!("`{key}` must be a non-negative integer")))?;
            Ok((n as usize).min(max))
        }
    }
}

fn vector_param(params: &Json, key: &str) -> Result<Vec<f32>> {
    let arr = params
        .get(key)
        .and_then(Json::as_array)
        .ok_or_else(|| HubError::BadParam(format!("`{key}` ([number]) is required")))?;
    if arr.is_empty() {
        return Err(HubError::BadParam(format!("`{key}` must not be empty")));
    }
    arr.iter()
        .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| HubError::BadParam(format!("`{key}` must hold numbers"))))
        .collect()
}

fn not_found(label: &str, id: &str) -> HubError {
    HubError::NotFound { label: label.into(), id: id.into() }
}

impl HubClient {
    /// Run a whitelisted named query with JSON parameters and return its result as JSON.
    /// Errors: [`HubError::UnknownQuery`], [`HubError::BadParam`], [`HubError::NotFound`] (an id
    /// parameter that matches nothing), plus transport/server errors.
    pub async fn run_named(&self, name: &str, params: &Json) -> Result<Json> {
        use crate::schema::node;
        if lookup(name).is_none() {
            return Err(HubError::UnknownQuery(name.to_owned()));
        }
        if !(params.is_object() || params.is_null()) {
            return Err(HubError::BadParam("`params` must be an object".into()));
        }
        let p = params;
        let out = match name {
            "getInstrument" => json!(self.get_instrument(str_param(p, "instrument_id")?).await?),
            "listSnapshots" => json!(self.list_snapshots().await?),
            "snapshotSpaces" => {
                let id = str_param(p, "snapshot_id")?;
                let snap = self.snapshot_by_id(id).await?.ok_or_else(|| not_found(node::SNAPSHOT, id))?;
                json!(self.snapshot_spaces(snap.id).await?)
            }
            "snapshotGraph" => {
                let id = str_param(p, "snapshot_id")?;
                json!(self.snapshot_graph(id).await?.ok_or_else(|| not_found(node::SNAPSHOT, id))?)
            }
            "spaceRegions" => {
                let id = str_param(p, "expert_space_id")?;
                let space = self.space_by_id(id).await?.ok_or_else(|| not_found(node::EXPERT_SPACE, id))?;
                json!(self.space_regions(space.id).await?)
            }
            "getRegion" => json!(self.region_by_id(str_param(p, "region_id")?).await?),
            "regionTransitions" | "regionNeighbors" => {
                let id = str_param(p, "region_id")?;
                let region = self.region_by_id(id).await?.ok_or_else(|| not_found(node::LATENT_REGION, id))?;
                if name == "regionTransitions" {
                    json!(self.region_transitions(region.id).await?)
                } else {
                    json!(self.region_neighbors(region.id).await?)
                }
            }
            "searchLatent" => {
                let space = str_param(p, "expert_space_id")?;
                let vector = vector_param(p, "vector")?;
                let k = opt_usize(p, "k", 20, MAX_SEARCH_K)?.max(1);
                json!(self.search_latent(space, &vector, k).await?)
            }
            "genomeParents" | "genomeFitness" | "genomeAncestry" => {
                let id = str_param(p, "genome_id")?;
                let genome = self.genome_by_id(id).await?.ok_or_else(|| not_found(node::GENOME, id))?;
                match name {
                    "genomeParents" => json!(self.genome_parents(genome.id).await?),
                    "genomeFitness" => json!(self.genome_fitness(genome.id).await?),
                    _ => {
                        let depth = opt_usize(p, "max_depth", 10, MAX_ANCESTRY_DEPTH)?.max(1);
                        json!(self.genome_ancestry(genome.id, depth).await?)
                    }
                }
            }
            "resolveNode" => json!(self.resolve_node(str_param(p, "id")?).await?),
            "neighborhood" => {
                let id = str_param(p, "node")?;
                let node = self.resolve_node(id).await?.ok_or_else(|| not_found("node", id))?;
                let depth = opt_usize(p, "depth", 1, MAX_NEIGHBORHOOD_DEPTH)?.max(1);
                json!(self.neighborhood(node.id, depth).await?)
            }
            _ => return Err(HubError::UnknownQuery(name.to_owned())),
        };
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_names_are_unique_and_known() {
        let mut n = names();
        n.sort_unstable();
        n.dedup();
        assert_eq!(n.len(), NAMED_QUERIES.len());
        assert!(lookup("neighborhood").is_some());
        assert!(lookup("addSnapshot").is_none(), "mutations are not in the whitelist");
        assert!(NAMED_QUERIES.iter().all(|q| !q.name.starts_with("add") && !q.name.starts_with("upsert")));
    }

    #[test]
    fn param_helpers_validate() {
        let p = json!({ "region_id": "r1", "k": 7, "vector": [1.0, 0.5], "bad": "x", "empty": [] });
        assert_eq!(str_param(&p, "region_id").unwrap(), "r1");
        assert!(matches!(str_param(&p, "missing"), Err(HubError::BadParam(_))));
        assert_eq!(opt_usize(&p, "k", 20, 500).unwrap(), 7);
        assert_eq!(opt_usize(&p, "k", 20, 5).unwrap(), 5, "clamped to max");
        assert_eq!(opt_usize(&p, "nope", 20, 500).unwrap(), 20);
        assert!(matches!(opt_usize(&p, "bad", 1, 5), Err(HubError::BadParam(_))));
        assert_eq!(vector_param(&p, "vector").unwrap(), vec![1.0f32, 0.5]);
        assert!(matches!(vector_param(&p, "empty"), Err(HubError::BadParam(_))));
    }

    #[tokio::test]
    async fn unknown_name_and_bad_params_fail_before_any_request() {
        // Port 1 is never listening; the checks below must fail without touching the network.
        let c = HubClient::new("http://127.0.0.1:1");
        assert!(matches!(c.run_named("dropEverything", &json!({})).await, Err(HubError::UnknownQuery(_))));
        assert!(matches!(c.run_named("getRegion", &json!({})).await, Err(HubError::BadParam(_))));
        assert!(matches!(c.run_named("getRegion", &json!([1])).await, Err(HubError::BadParam(_))));
    }
}
