//! Volume control and retention of state observations in Helix (spec 20 §6.4, §6.5: "Helix
//! state observations: per timescale … older pruned to region-level aggregates").
//!
//! `LatentVec` rows with `kind = observation` (`ref_id` = the region the observation fell in,
//! D-048) older than a horizon are folded into **one `kind = aggregate` row per `(expert space,
//! ref_id)`**: `obs_count`, the running mean vector (in `vec<dim>`, so aggregates stay searchable
//! with [`crate::KnnFilter`] `kind = "aggregate"`), mean `uncertainty` and `ts` = newest folded
//! observation. The folded rows are dropped in the same write batch as the aggregate update, so a
//! page is either fully folded or untouched. Regions, transitions, centroids and windows are never
//! touched; version rows of transitions and region status ([`crate::versions`]) older than the
//! same cut-off are thinned to the newest one per edge / region.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::Deserialize;

use crate::ast::{Batch, Input, NodeRef, Op, Pred, Val};
use crate::queries::LatentRow;
use crate::schema::{LATENT_TENANT, latent_vec_property, node};
use crate::vectors::validate_vector;
use crate::{HubClient, Result};

/// `kind` of the folded rows.
pub const AGGREGATE_KIND: &str = "aggregate";
/// Observations read and folded per write batch.
pub const PRUNE_PAGE: usize = 500;

/// Horizon per timescale (spec 20 §6.5 "2× backfill depth"; values are deployment config).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RetentionPolicy {
    /// `timescale` (as on `Snapshot.timescale`, e.g. `"1m"`) → how long observations are kept.
    pub horizons: BTreeMap<String, Duration>,
    /// Horizon of timescales not listed; `None` keeps them forever.
    pub default: Option<Duration>,
}

impl RetentionPolicy {
    pub fn horizon_for(&self, timescale: &str) -> Option<Duration> {
        self.horizons.get(timescale).copied().or(self.default)
    }
    /// Cut-off (unix ms) for `timescale` at `now_ms`; `None` = keep everything.
    pub fn cutoff_ms(&self, timescale: &str, now_ms: i64) -> Option<i64> {
        let h = self.horizon_for(timescale)?;
        Some(now_ms.saturating_sub(i64::try_from(h.as_millis()).unwrap_or(i64::MAX)))
    }
}

/// What a prune did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PruneReport {
    /// Observation rows dropped.
    pub pruned: u64,
    /// Aggregate rows created.
    pub aggregates_created: u64,
    /// Aggregate rows updated.
    pub aggregates_updated: u64,
    /// `TransitionVersion` / `RegionStatusVersion` rows dropped ([`crate::versions`]).
    pub versions_pruned: u64,
}

impl PruneReport {
    fn add(&mut self, o: &PruneReport) {
        self.pruned += o.pruned;
        self.aggregates_created += o.aggregates_created;
        self.aggregates_updated += o.aggregates_updated;
        self.versions_pruned += o.versions_pruned;
    }
}

#[derive(Debug, Deserialize)]
struct AggRow {
    #[serde(rename = "$id")]
    id: u64,
    ref_id: String,
    #[serde(default)]
    obs_count: u64,
    #[serde(default)]
    uncertainty: f64,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

/// Running fold of one `ref_id`'s observations.
#[derive(Debug, Clone, Default, PartialEq)]
struct Fold {
    count: u64,
    sum: Vec<f64>,
    unc_sum: f64,
    max_ts: Option<String>,
    instrument_id: String,
}

impl Fold {
    fn push(&mut self, v: Option<&[f32]>, uncertainty: f64, ts: Option<&str>, instrument: &str) {
        if let Some(v) = v {
            if self.sum.is_empty() {
                self.sum = vec![0.0; v.len()];
            }
            if self.sum.len() == v.len() {
                self.sum.iter_mut().zip(v).for_each(|(s, x)| *s += f64::from(*x));
            }
        }
        self.count += 1;
        self.unc_sum += uncertainty;
        if ts.is_some_and(|t| self.max_ts.as_deref().is_none_or(|m| t > m)) {
            self.max_ts = ts.map(str::to_owned);
        }
        if self.instrument_id.is_empty() {
            self.instrument_id = instrument.to_owned();
        }
    }

    /// Merge with an existing aggregate (`count` rows with mean `mean`, mean uncertainty `unc`).
    fn merged_with(&self, prev_count: u64, prev_mean: Option<&[f32]>, prev_unc: f64) -> (u64, Option<Vec<f32>>, f64) {
        let n = self.count + prev_count;
        let mut sum = self.sum.clone();
        if let Some(m) = prev_mean {
            if sum.is_empty() {
                sum = vec![0.0; m.len()];
            }
            if sum.len() == m.len() {
                sum.iter_mut().zip(m).for_each(|(s, x)| *s += f64::from(*x) * prev_count as f64);
            }
        }
        let nf = n.max(1) as f64;
        let mean: Vec<f32> = sum.iter().map(|s| (s / nf) as f32).collect();
        let unc = (self.unc_sum + prev_unc * prev_count as f64) / nf;
        (n, (!mean.is_empty() && validate_vector(&mean).is_ok()).then_some(mean), unc)
    }
}

/// Parse the server's RFC 3339 rendering (`YYYY-MM-DDTHH:MM:SS[.fff]Z`) into unix ms.
pub(crate) fn rfc3339_ms(ts: &str) -> Option<i64> {
    let b = ts.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let num = |r: std::ops::Range<usize>| ts.get(r)?.parse::<i64>().ok();
    let (y, mo, d, h, mi, sec) = (num(0..4)?, num(5..7)?, num(8..10)?, num(11..13)?, num(14..16)?, num(17..19)?);
    let rest = ts.get(19..)?.strip_suffix('Z')?;
    let ms = match rest.strip_prefix('.') {
        Some(frac) if !frac.is_empty() && frac.bytes().all(|c| c.is_ascii_digit()) => {
            let f = &frac[..frac.len().min(3)];
            f.parse::<i64>().ok()? * 10i64.pow(3 - f.len() as u32)
        }
        None if rest.is_empty() => 0,
        _ => return None,
    };
    // Days from civil (H. Hinnant).
    let y = if mo <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(((days * 24 + h) * 60 + mi) * 60_000 + sec * 1000 + ms)
}

impl HubClient {
    /// Fold `observation` latents of one expert space with `ts < cutoff_ms` into per-`ref_id`
    /// aggregates and drop them (see the module docs). Pages of [`PRUNE_PAGE`] rows, one write
    /// batch per page; stops when nothing older than the cut-off is left.
    pub async fn prune_space_observations(&self, expert_space_id: &str, cutoff_ms: i64) -> Result<PruneReport> {
        let mut total = PruneReport::default();
        loop {
            let page = self.prune_page(expert_space_id, cutoff_ms).await?;
            if page.pruned == 0 {
                return Ok(total);
            }
            total.add(&page);
        }
    }

    async fn prune_page(&self, expert_space_id: &str, cutoff_ms: i64) -> Result<PruneReport> {
        let old = Pred::and([
            Pred::eq(LATENT_TENANT, expert_space_id),
            Pred::eq("kind", "observation"),
            Pred::lt("ts", Input::Lit(Val::DateTime(cutoff_ms))),
        ]);
        let read = Batch::read()
            .named("prune_read")
            .ret(
                "rows",
                Op::nodes_with_label_where(node::LATENT_VEC, old)
                    .order_by("ts", false)
                    .limit(PRUNE_PAGE)
                    .value_map(None),
            )
            .ret(
                "aggs",
                Op::nodes_with_label_where(
                    node::LATENT_VEC,
                    Pred::and([Pred::eq(LATENT_TENANT, expert_space_id), Pred::eq("kind", AGGREGATE_KIND)]),
                )
                .value_map(None),
            );
        let mut resp = self.batch(&read).await?;
        let rows: Vec<LatentRow> = resp.take("rows")?;
        if rows.is_empty() {
            return Ok(PruneReport::default());
        }
        let aggs: Vec<AggRow> = resp.take("aggs")?;
        let aggs: BTreeMap<&str, &AggRow> = aggs.iter().map(|a| (a.ref_id.as_str(), a)).collect();

        let mut folds: BTreeMap<&str, Fold> = BTreeMap::new();
        for r in &rows {
            let v = r.vector();
            folds.entry(r.ref_id.as_str()).or_default().push(
                v.as_deref(),
                r.uncertainty,
                r.ts.as_deref(),
                &r.instrument_id,
            );
        }

        let mut report = PruneReport { pruned: rows.len() as u64, ..PruneReport::default() };
        let mut batch = Batch::write().named("prune_fold");
        for (i, (ref_id, fold)) in folds.iter().enumerate() {
            let prev = aggs.get(ref_id);
            let prev_mean = prev.and_then(|a| latent_vector(&a.extra));
            let (n, mean, unc) = fold.merged_with(
                prev.map_or(0, |a| a.obs_count),
                prev_mean.as_deref(),
                prev.map_or(0.0, |a| a.uncertainty),
            );
            let n = i64::try_from(n).unwrap_or(i64::MAX);
            // `ts` of an aggregate = newest folded observation (never older than a previous fold).
            let ts_ms = fold.max_ts.as_deref().and_then(rfc3339_ms).unwrap_or(cutoff_ms - 1);
            let ts_ms =
                prev.and_then(|a| a.extra.get("ts")?.as_str().and_then(rfc3339_ms)).map_or(ts_ms, |p| p.max(ts_ms));
            let ts = Input::Lit(Val::DateTime(ts_ms));
            let name = format!("a{i}");
            if let Some(a) = prev {
                let mut op = Op::nodes(a.id)
                    .set_property("obs_count", n)
                    .set_property("uncertainty", unc)
                    .set_property("ts", ts);
                if let Some(m) = mean {
                    op = op.set_property(&latent_vec_property(m.len()), m);
                }
                batch = batch.var(&name, op);
                report.aggregates_updated += 1;
            } else {
                let mut props: Vec<(String, Input)> = vec![
                    (LATENT_TENANT.into(), (*expert_space_id).into()),
                    ("kind".into(), AGGREGATE_KIND.into()),
                    ("ref_id".into(), (*ref_id).into()),
                    ("instrument_id".into(), fold.instrument_id.as_str().into()),
                    ("obs_count".into(), n.into()),
                    ("uncertainty".into(), unc.into()),
                    ("ts".into(), ts),
                ];
                if let Some(m) = mean {
                    props.push((latent_vec_property(m.len()), m.into()));
                }
                let props = props.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
                batch = batch.var(&name, Op::add_n(node::LATENT_VEC, props));
                report.aggregates_created += 1;
            }
        }
        let ids: Vec<u64> = rows.iter().map(|r| r.id).collect();
        batch = batch.var("drop", Op::nodes(NodeRef::Ids(ids)).drop());
        self.batch(&batch).await?;
        Ok(report)
    }

    /// Apply `policy` to every expert space of one snapshot (cut-off from `Snapshot.timescale`
    /// at `now_ms`): observations are folded, and transition / status versions older than the
    /// cut-off are thinned to the one in force at the cut-off ([`HubClient::prune_space_versions`]).
    /// A timescale without a horizon is left alone.
    pub async fn prune_snapshot_observations(
        &self,
        snapshot_id: &str,
        now_ms: i64,
        policy: &RetentionPolicy,
    ) -> Result<PruneReport> {
        let mut total = PruneReport::default();
        let Some(snap) = self.snapshot_by_id(snapshot_id).await? else { return Ok(total) };
        let Some(cutoff) = policy.cutoff_ms(&snap.timescale, now_ms) else { return Ok(total) };
        for space in self.snapshot_spaces(snap.id).await? {
            total.add(&self.prune_space_observations(&space.expert_space_id, cutoff).await?);
            total.versions_pruned += self.prune_space_versions(space.id, cutoff).await?;
        }
        Ok(total)
    }

    /// [`Self::prune_snapshot_observations`] for every snapshot in Helix.
    pub async fn prune_observations(&self, now_ms: i64, policy: &RetentionPolicy) -> Result<PruneReport> {
        let mut total = PruneReport::default();
        for s in self.list_snapshots().await? {
            total.add(&self.prune_snapshot_observations(&s.snapshot_id, now_ms, policy).await?);
        }
        Ok(total)
    }
}

fn latent_vector(extra: &serde_json::Map<String, serde_json::Value>) -> Option<Vec<f32>> {
    extra.iter().find_map(|(k, v)| {
        if !k.starts_with("vec") || !k[3..].chars().all(|c| c.is_ascii_digit()) || k.len() == 3 {
            return None;
        }
        v.as_array()?.iter().map(|x| x.as_f64().map(|f| f as f32)).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_server_timestamps() {
        assert_eq!(rfc3339_ms("1970-01-01T00:00:01.000Z"), Some(1000));
        assert_eq!(rfc3339_ms("2025-11-08T16:40:00.000Z"), Some(1_762_620_000_000));
        assert_eq!(rfc3339_ms("2026-11-08T17:51:51.867Z"), Some(1_794_160_311_867));
        assert_eq!(rfc3339_ms("2000-02-29T00:00:00Z"), Some(951_782_400_000));
        assert_eq!(rfc3339_ms("garbage"), None);
        assert_eq!(rfc3339_ms("2026-11-08T17:51:51+01:00"), None);
    }

    #[test]
    fn policy_cutoffs() {
        let mut p = RetentionPolicy::default();
        assert_eq!(p.cutoff_ms("1m", 10_000), None);
        p.horizons.insert("1m".into(), Duration::from_secs(2));
        p.default = Some(Duration::from_secs(5));
        assert_eq!(p.cutoff_ms("1m", 10_000), Some(8_000));
        assert_eq!(p.cutoff_ms("1h", 10_000), Some(5_000));
    }

    #[test]
    fn fold_and_merge_keep_a_weighted_mean() {
        let mut f = Fold::default();
        f.push(Some(&[1.0, 0.0]), 0.2, Some("2026-01-01T00:00:00.000Z"), "BTC");
        f.push(Some(&[0.0, 1.0]), 0.4, Some("2026-01-02T00:00:00.000Z"), "ETH");
        assert_eq!(f.count, 2);
        assert_eq!(f.max_ts.as_deref(), Some("2026-01-02T00:00:00.000Z"));
        assert_eq!(f.instrument_id, "BTC");
        let (n, mean, unc) = f.merged_with(0, None, 0.0);
        assert_eq!(n, 2);
        assert_eq!(mean, Some(vec![0.5, 0.5]));
        assert!((unc - 0.3).abs() < 1e-12);
        // An existing aggregate of 2 rows at [1, 1] (unc 0.0): mean over 4 rows.
        let (n, mean, unc) = f.merged_with(2, Some(&[1.0, 1.0]), 0.0);
        assert_eq!(n, 4);
        assert_eq!(mean, Some(vec![0.75, 0.75]));
        assert!((unc - 0.15).abs() < 1e-12);
        // Opposite vectors cancel to zero norm: no vector is written.
        let mut z = Fold::default();
        z.push(Some(&[1.0]), 0.0, None, "");
        z.push(Some(&[-1.0]), 0.0, None, "");
        assert_eq!(z.merged_with(0, None, 0.0).1, None);
    }

    #[test]
    fn latent_vector_finds_vec_property() {
        let m = serde_json::json!({ "vec2": [1.0, 2.0], "vector_x": 1, "vec": [3.0] });
        assert_eq!(latent_vector(m.as_object().unwrap()), Some(vec![1.0, 2.0]));
    }
}
