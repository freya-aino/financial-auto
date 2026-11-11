//! As-of variants of the snapshot-scoped reads (spec 20 §6.3: "Helix: all nodes carry
//! `created_at`; snapshot-scoped queries filter by snapshot id and `created_at ≤ t`").
//!
//! `as_of_ms` is unix milliseconds; rows without the timestamp property (written before it
//! existed) are treated as not visible. `Transition` posteriors and region lifecycle `status`
//! are updated in place, so the reads overlay their event-time versions ([`crate::versions`]):
//! edges carry the posterior in force at `t` (edges first written after `t` are dropped), regions
//! the `status` / `related_region_id` in force at `t` (`None` when unknown, i.e. a pre-versioning
//! row changed after `t`). Limits: `ExpertSpace` has no `created_at` and is visible with its
//! snapshot; `Snapshot.status` and region statistics (`mass`, `occupancy`, …) are current values.

use std::collections::HashSet;

use crate::ast::{Batch, Input, NodeRef, Op, Pred, Val};
use crate::queries::{LatentRegion, LatentRow, Snapshot, SnapshotGraph, Transition};
use crate::schema::{LATENT_TENANT, edge, node};
use crate::{HubClient, Result};

/// `created_at <= as_of_ms`.
pub fn created_before(as_of_ms: i64) -> Pred {
    Pred::lte("created_at", Input::Lit(Val::DateTime(as_of_ms)))
}

impl HubClient {
    /// `listSnapshotsAsOf`: snapshots created at or before `as_of_ms`, by `snapshot_id`.
    pub async fn list_snapshots_as_of(&self, as_of_ms: i64) -> Result<Vec<Snapshot>> {
        let op = Op::nodes_with_label_where(node::SNAPSHOT, created_before(as_of_ms))
            .order_by("snapshot_id", false)
            .value_map(None);
        let mut resp = self.batch(&Batch::read().named("list_snapshots_as_of").ret("rows", op)).await?;
        resp.take("rows")
    }

    /// `spaceRegionsAsOf`: regions of an expert space (`$id`) created at or before `as_of_ms`.
    pub async fn space_regions_as_of(&self, space: u64, as_of_ms: i64) -> Result<Vec<LatentRegion>> {
        let op = Op::nodes(space).r#in(Some(edge::IN_SPACE)).where_(created_before(as_of_ms)).value_map(None);
        let mut resp = self.batch(&Batch::read().named("space_regions_as_of").ret("rows", op)).await?;
        self.regions_as_of(resp.take("rows")?, as_of_ms).await
    }

    /// `snapshotGraphAsOf`: [`Self::snapshot_graph`] restricted to what existed at `as_of_ms`:
    /// `None` when the snapshot is unknown or created later; regions with `created_at <= t`;
    /// transitions whose both endpoints are visible, with the posterior and region status in force
    /// at `t` (see module docs).
    pub async fn snapshot_graph_as_of(&self, snapshot_id: &str, as_of_ms: i64) -> Result<Option<SnapshotGraph>> {
        let batch = Batch::read()
            .named("snapshot_graph_as_of")
            .var(
                "snap",
                Op::nodes_with_label_where(
                    node::SNAPSHOT,
                    Pred::and([Pred::eq("snapshot_id", snapshot_id), created_before(as_of_ms)]),
                ),
            )
            .ret("snap_row", Op::nodes(NodeRef::var("snap")).value_map(None))
            .var("spaces", Op::nodes(NodeRef::var("snap")).out(Some(edge::HAS_SPACE)))
            .ret("space_rows", Op::nodes(NodeRef::var("spaces")).value_map(None))
            .var(
                "regions",
                Op::nodes(NodeRef::var("spaces")).r#in(Some(edge::IN_SPACE)).where_(created_before(as_of_ms)),
            )
            .ret("region_rows", Op::nodes(NodeRef::var("regions")).value_map(None))
            .ret("transitions", Op::nodes(NodeRef::var("regions")).out_e(Some(edge::TRANSITION)).edge_properties());
        let mut resp = self.batch(&batch).await?;
        let Some(snapshot) = resp.take_opt::<Snapshot>("snap_row")? else {
            return Ok(None);
        };
        let regions: Vec<LatentRegion> = resp.take("region_rows")?;
        let visible: HashSet<u64> = regions.iter().map(|r| r.id).collect();
        let mut transitions: Vec<Transition> = resp.take("transitions")?;
        transitions.retain(|t| visible.contains(&t.to));
        let spaces = resp.take("space_rows")?;
        let transitions = self.transitions_as_of(transitions, as_of_ms).await?;
        let regions = self.regions_as_of(regions, as_of_ms).await?;
        Ok(Some(SnapshotGraph { snapshot, spaces, regions, transitions }))
    }

    /// `spaceLatentsAsOf`: `LatentVec` rows of one space with `ts <= as_of_ms`, newest first, at
    /// most `limit`; `kind` filters on `centroid | observation | window | aggregate`.
    pub async fn space_latents_as_of(
        &self,
        expert_space_id: &str,
        kind: Option<&str>,
        as_of_ms: i64,
        limit: usize,
    ) -> Result<Vec<LatentRow>> {
        let mut preds =
            vec![Pred::eq(LATENT_TENANT, expert_space_id), Pred::lte("ts", Input::Lit(Val::DateTime(as_of_ms)))];
        if let Some(k) = kind.filter(|k| !k.is_empty()) {
            preds.push(Pred::eq("kind", k));
        }
        let op = Op::nodes_with_label_where(node::LATENT_VEC, Pred::and(preds))
            .order_by("ts", true)
            .limit(limit.max(1))
            .value_map(None);
        let mut resp = self.batch(&Batch::read().named("space_latents_as_of").ret("rows", op)).await?;
        resp.take("rows")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn created_before_is_an_inclusive_datetime_bound() {
        assert_eq!(
            created_before(42).into_json(),
            json!({ "lte": { "left": { "property": "created_at" }, "right": { "constant": { "date_time": 42 } } } })
        );
    }
}
