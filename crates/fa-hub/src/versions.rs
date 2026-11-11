//! Event-time versions of the values Helix updates in place (spec 20 §6.3, spec 33 §5).
//!
//! `Transition` edges (posterior count / mean / interval / kernel params) and the lifecycle
//! `status` of `LatentRegion` are updated in place by model-runtime, so the edge or node alone
//! only knows its *current* value. Every such write now also appends a version node in the same
//! write batch (one transaction):
//!
//! - `TransitionVersion {edge_key, from_region_id, to_region_id, sampler_id, count, mean, lower,
//!   upper, kernel_params, ts}` on every [`HubClient::upsert_transitions`] row (insert or update);
//! - `RegionStatusVersion {region_id, status, related_region_id, reason, ts}` on every lifecycle
//!   mark ([`HubClient::mark_region_status`], `mark_region_died`) and when
//!   [`HubClient::upsert_region`] creates a region or revives one that was not `live`.
//!
//! `ts` is the server time of the write (`NOW`, the record time). It is never earlier than the
//! observation it summarises (model-runtime coalesces writes for up to `--hub-flush-secs`), so an
//! as-of read can show a value slightly late but never before it was known.
//!
//! As-of rule ([`HubClient::transitions_as_of`], [`HubClient::regions_as_of`], used by the
//! `*_as_of` reads): an edge or region whose `updated_at ≤ t` carries its value at `t` (nothing
//! was written after `t`). Otherwise the latest version with `ts ≤ t` supplies the value; with
//! none, the edge did not exist yet at `t` and is dropped, and a region's `status` /
//! `related_region_id` are reported as unknown (`None`). Rows written before versioning existed
//! (no version rows) therefore read correctly as of any `t` after their last write and as
//! unknown/invisible before it. Region statistics (`mass`, `occupancy`, `spread`, `age`) stay
//! current values (not versioned).
//!
//! Volume: transition writes are coalesced to one per edge per flush (10 s), status marks are
//! rare. [`HubClient::prune_space_versions`] (run with the observation retention, see
//! [`crate::retention`]) keeps every version newer than the cut-off plus the latest older one per
//! edge / region, so as-of reads before the cut-off keep the value that was in force.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::ast::{Batch, Input, NodeRef, Op, Pred, Val};
use crate::queries::{LatentRegion, Transition, TransitionUpsert};
use crate::retention::rfc3339_ms;
use crate::{HubClient, Result};

/// Labels introduced by this module (mirrored in `helix/schema.hx`).
pub mod labels {
    pub const TRANSITION_VERSION: &str = "TransitionVersion";
    pub const REGION_STATUS_VERSION: &str = "RegionStatusVersion";
}

/// Point lookups per read batch (as-of overlays, pruning).
pub const LOOKUP_CHUNK: usize = 100;

/// `from|to|sampler`: the business key of a `Transition` edge (region ids never contain `|`).
pub fn edge_key(from_region_id: &str, to_region_id: &str, sampler_id: &str) -> String {
    format!("{from_region_id}|{to_region_id}|{sampler_id}")
}

/// A `TransitionVersion` row.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TransitionVersion {
    #[serde(rename = "$id")]
    pub id: u64,
    pub edge_key: String,
    pub from_region_id: String,
    pub to_region_id: String,
    pub sampler_id: String,
    pub count: i64,
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
    #[serde(default)]
    pub kernel_params: String,
    #[serde(default)]
    pub ts: Option<String>,
}

/// A `RegionStatusVersion` row.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RegionStatusVersion {
    #[serde(rename = "$id")]
    pub id: u64,
    pub region_id: String,
    pub status: String,
    #[serde(default)]
    pub related_region_id: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub ts: Option<String>,
}

#[derive(Deserialize)]
struct IdRow {
    #[serde(rename = "$id")]
    id: u64,
}

fn s(v: &str) -> Input {
    Input::Lit(Val::Str(v.to_owned()))
}

fn at_or_before(t_ms: i64) -> Pred {
    Pred::lte("ts", Input::Lit(Val::DateTime(t_ms)))
}

/// `AddN<TransitionVersion>` for one upsert row (`ts = NOW`).
pub(crate) fn transition_version_node(r: &TransitionUpsert) -> Op {
    let t = &r.transition;
    Op::add_n(
        labels::TRANSITION_VERSION,
        vec![
            ("edge_key", s(&edge_key(&r.from_region_id, &r.to_region_id, &t.sampler_id))),
            ("from_region_id", s(&r.from_region_id)),
            ("to_region_id", s(&r.to_region_id)),
            ("sampler_id", s(&t.sampler_id)),
            ("count", t.count.into()),
            ("mean", t.mean.into()),
            ("lower", t.lower.into()),
            ("upper", t.upper.into()),
            ("kernel_params", s(&t.kernel_params)),
            ("ts", Input::Now),
        ],
    )
}

/// `AddN<RegionStatusVersion>` (`ts = NOW`).
pub(crate) fn status_version_node(region_id: &str, status: &str, related_region_id: &str, reason: &str) -> Op {
    Op::add_n(
        labels::REGION_STATUS_VERSION,
        vec![
            ("region_id", s(region_id)),
            ("status", s(status)),
            ("related_region_id", s(related_region_id)),
            ("reason", s(reason)),
            ("ts", Input::Now),
        ],
    )
}

/// Whether a row last written at `updated_at` already carries its value as of `t_ms`.
fn unchanged_since(updated_at: Option<&str>, t_ms: i64) -> bool {
    updated_at.and_then(rfc3339_ms).is_some_and(|u| u <= t_ms)
}

/// Overlay a version's values onto an edge row.
pub(crate) fn apply_transition_version(t: &mut Transition, v: &TransitionVersion) {
    t.count = v.count;
    t.mean = v.mean;
    t.lower = v.lower;
    t.upper = v.upper;
    t.kernel_params = v.kernel_params.clone();
    t.updated_at = v.ts.clone();
}

impl HubClient {
    /// `transitionVersions`: versions of one edge, newest first, at most `limit`; `as_of_ms`
    /// keeps `ts ≤ t`.
    pub async fn transition_versions(
        &self,
        from_region_id: &str,
        to_region_id: &str,
        sampler_id: &str,
        as_of_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<TransitionVersion>> {
        let mut pred = vec![Pred::eq("edge_key", edge_key(from_region_id, to_region_id, sampler_id).as_str())];
        pred.extend(as_of_ms.map(at_or_before));
        let op = Op::nodes_with_label_where(labels::TRANSITION_VERSION, Pred::and(pred))
            .order_by("ts", true)
            .limit(limit.max(1))
            .value_map(None);
        let mut resp = self.batch(&Batch::read().named("transition_versions").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// `regionStatusVersions`: lifecycle versions of one region, newest first, at most `limit`;
    /// `as_of_ms` keeps `ts ≤ t`.
    pub async fn region_status_versions(
        &self,
        region_id: &str,
        as_of_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<RegionStatusVersion>> {
        let mut pred = vec![Pred::eq("region_id", region_id)];
        pred.extend(as_of_ms.map(at_or_before));
        let op = Op::nodes_with_label_where(labels::REGION_STATUS_VERSION, Pred::and(pred))
            .order_by("ts", true)
            .limit(limit.max(1))
            .value_map(None);
        let mut resp = self.batch(&Batch::read().named("region_status_versions").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// Latest version with `ts ≤ t` for each key, `None` where there is none. One read batch of
    /// indexed point lookups per [`LOOKUP_CHUNK`] keys.
    async fn latest_versions<T: serde::de::DeserializeOwned>(
        &self,
        label: &str,
        key_prop: &str,
        keys: &[String],
        t_ms: i64,
    ) -> Result<Vec<Option<T>>> {
        let mut out = Vec::with_capacity(keys.len());
        for chunk in keys.chunks(LOOKUP_CHUNK) {
            let mut b = Batch::read().named("latest_versions");
            for (i, k) in chunk.iter().enumerate() {
                b = b.ret(
                    &format!("v{i}"),
                    Op::nodes_with_label_where(label, Pred::and([Pred::eq(key_prop, k.as_str()), at_or_before(t_ms)]))
                        .order_by("ts", true)
                        .limit(1)
                        .value_map(None),
                );
            }
            let mut resp = self.batch(&b).await?;
            for i in 0..chunk.len() {
                out.push(resp.take_opt::<T>(&format!("v{i}"))?);
            }
        }
        Ok(out)
    }

    /// Edges as they were at `t_ms` (module docs): unchanged ones as read, the others from their
    /// latest version `≤ t`, dropped when they had none.
    pub async fn transitions_as_of(&self, edges: Vec<Transition>, t_ms: i64) -> Result<Vec<Transition>> {
        let (mut out, stale): (Vec<Transition>, Vec<Transition>) =
            edges.into_iter().partition(|e| unchanged_since(e.updated_at.as_deref(), t_ms));
        let (keyed, _unkeyed): (Vec<Transition>, Vec<Transition>) =
            stale.into_iter().partition(|e| e.from_region_id.is_some() && e.to_region_id.is_some());
        let keys: Vec<String> = keyed
            .iter()
            .map(|e| {
                edge_key(
                    e.from_region_id.as_deref().unwrap_or(""),
                    e.to_region_id.as_deref().unwrap_or(""),
                    &e.sampler_id,
                )
            })
            .collect();
        let versions: Vec<Option<TransitionVersion>> =
            self.latest_versions(labels::TRANSITION_VERSION, "edge_key", &keys, t_ms).await?;
        for (mut e, v) in keyed.into_iter().zip(versions) {
            if let Some(v) = v {
                apply_transition_version(&mut e, &v);
                out.push(e);
            }
        }
        out.sort_by_key(|e| e.id);
        Ok(out)
    }

    /// Regions with their lifecycle `status` / `related_region_id` as of `t_ms` (module docs);
    /// unknown → `None`. Visibility (`created_at ≤ t`) is the caller's filter.
    pub async fn regions_as_of(&self, regions: Vec<LatentRegion>, t_ms: i64) -> Result<Vec<LatentRegion>> {
        let (mut out, stale): (Vec<LatentRegion>, Vec<LatentRegion>) =
            regions.into_iter().partition(|r| unchanged_since(r.updated_at.as_deref(), t_ms));
        let keys: Vec<String> = stale.iter().map(|r| r.region_id.clone()).collect();
        let versions: Vec<Option<RegionStatusVersion>> =
            self.latest_versions(labels::REGION_STATUS_VERSION, "region_id", &keys, t_ms).await?;
        for (mut r, v) in stale.into_iter().zip(versions) {
            match v {
                Some(v) => {
                    r.status = Some(v.status);
                    r.related_region_id = Some(v.related_region_id);
                    r.updated_at = v.ts;
                }
                None => {
                    r.status = None;
                    r.related_region_id = None;
                }
            }
            out.push(r);
        }
        out.sort_by_key(|r| r.id);
        Ok(out)
    }

    /// `pruneSpaceVersions`: for every transition edge and region of the expert space `space`
    /// (node id), drop the versions with `ts < cutoff_ms` except the newest of them (the value in
    /// force at the cut-off). Returns the number of version rows dropped.
    pub async fn prune_space_versions(&self, space: u64, cutoff_ms: i64) -> Result<u64> {
        let graph = self.region_graph(space).await?;
        let edge_keys: BTreeSet<String> = graph
            .transitions
            .iter()
            .filter_map(|t| Some(edge_key(t.from_region_id.as_deref()?, t.to_region_id.as_deref()?, &t.sampler_id)))
            .collect();
        let region_ids: BTreeSet<String> = graph.regions.iter().map(|r| r.region_id.clone()).collect();
        let mut dropped = 0;
        for (label, key_prop, keys) in [
            (labels::TRANSITION_VERSION, "edge_key", edge_keys.into_iter().collect::<Vec<_>>()),
            (labels::REGION_STATUS_VERSION, "region_id", region_ids.into_iter().collect::<Vec<_>>()),
        ] {
            dropped += self.prune_versions(label, key_prop, &keys, cutoff_ms).await?;
        }
        Ok(dropped)
    }

    async fn prune_versions(&self, label: &str, key_prop: &str, keys: &[String], cutoff_ms: i64) -> Result<u64> {
        let mut dropped = 0u64;
        for chunk in keys.chunks(LOOKUP_CHUNK) {
            let mut b = Batch::read().named("prune_versions_read");
            for (i, k) in chunk.iter().enumerate() {
                b = b.ret(
                    &format!("v{i}"),
                    Op::nodes_with_label_where(
                        label,
                        Pred::and([
                            Pred::eq(key_prop, k.as_str()),
                            Pred::lt("ts", Input::Lit(Val::DateTime(cutoff_ms))),
                        ]),
                    )
                    .order_by("ts", true)
                    .value_map(Some(&["$id"])),
                );
            }
            let mut resp = self.batch(&b).await?;
            let mut ids = Vec::new();
            for i in 0..chunk.len() {
                let rows: Vec<IdRow> = resp.take(&format!("v{i}"))?;
                // newest first: keep the first (in force at the cut-off), drop the rest
                ids.extend(rows.into_iter().skip(1).map(|r| r.id));
            }
            if ids.is_empty() {
                continue;
            }
            dropped += ids.len() as u64;
            for page in ids.chunks(1_000) {
                let b = Batch::write()
                    .named("prune_versions_drop")
                    .var("drop", Op::nodes(NodeRef::Ids(page.to_vec())).drop());
                self.batch(&b).await?;
            }
        }
        Ok(dropped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queries::NewTransition;

    fn edge(updated_at: &str) -> Transition {
        Transition {
            id: 1,
            from: 2,
            to: 3,
            from_region_id: Some("a".into()),
            to_region_id: Some("b".into()),
            sampler_id: "conjugate".into(),
            count: 9,
            mean: 0.9,
            lower: 0.8,
            upper: 0.95,
            kernel_params: "{}".into(),
            updated_at: Some(updated_at.into()),
        }
    }

    #[test]
    fn edge_key_and_unchanged_rule() {
        assert_eq!(edge_key("a", "b", "s"), "a|b|s");
        let t = rfc3339_ms("2026-11-11T05:00:00.000Z").unwrap();
        assert!(unchanged_since(Some("2026-11-11T05:00:00.000Z"), t), "inclusive");
        assert!(!unchanged_since(Some("2026-11-11T05:00:00.001Z"), t));
        assert!(!unchanged_since(None, t), "no timestamp: not known at t");
    }

    #[test]
    fn version_overlay_replaces_every_mutable_value() {
        let mut e = edge("2026-11-11T06:00:00.000Z");
        let v = TransitionVersion {
            id: 7,
            edge_key: "a|b|conjugate".into(),
            from_region_id: "a".into(),
            to_region_id: "b".into(),
            sampler_id: "conjugate".into(),
            count: 2,
            mean: 0.5,
            lower: 0.1,
            upper: 0.9,
            kernel_params: "{\"k\":1}".into(),
            ts: Some("2026-11-11T05:00:00.000Z".into()),
        };
        apply_transition_version(&mut e, &v);
        assert_eq!((e.count, e.mean, e.lower, e.upper), (2, 0.5, 0.1, 0.9));
        assert_eq!(e.kernel_params, "{\"k\":1}");
        assert_eq!(e.updated_at.as_deref(), Some("2026-11-11T05:00:00.000Z"));
        assert_eq!((e.id, e.from, e.to), (1, 2, 3), "identity kept");
    }

    #[test]
    fn version_nodes_carry_key_values_and_now() {
        let r = TransitionUpsert {
            from_region_id: "a".into(),
            to_region_id: "b".into(),
            transition: NewTransition {
                sampler_id: "s".into(),
                count: 3,
                mean: 0.25,
                lower: 0.1,
                upper: 0.4,
                kernel_params: "{}".into(),
            },
        };
        let j = serde_json::to_string(&transition_version_node(&r).into_json()).unwrap();
        for needle in ["TransitionVersion", "a|b|s", "edge_key", "\"ts\"", "kernel_params"] {
            assert!(j.contains(needle), "{needle} missing in {j}");
        }
        let j = serde_json::to_string(&status_version_node("r1", "retired", "r2", "merged").into_json()).unwrap();
        for needle in ["RegionStatusVersion", "r1", "retired", "r2", "merged"] {
            assert!(j.contains(needle), "{needle} missing in {j}");
        }
    }
}
