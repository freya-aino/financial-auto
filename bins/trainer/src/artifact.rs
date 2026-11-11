//! Where a fit lands: the snapshot folder (local `--out` and/or Garage through `fa_blob`) and
//! its Helix mirror (`Snapshot` candidate, `ExpertSpace`s, `LatentRegion`s, `SnapshotParent`
//! and `RegionMatch` edges), plus the parent's probe assignment read back from Helix and the
//! job checkpoint in `fa-ckpt/<job_id>/state.json`.

use std::path::PathBuf;

use fa_blob::{BlobStore, ObjectKey, S3BlobStore};
use fa_encoders::{MANIFEST_FILE, REGION_MAP_FILE, REGIONS_FILE, STATISTICAL_FAMILY, SnapshotManifest, WEIGHTS_FILE};
use fa_hub::{HubClient, NewExpertSpace, NewRegion, NewSnapshot, SnapshotGraph};
use fa_patterns::RegionMap;
use serde::Serialize;

use crate::{
    data::{BarHistory, Source},
    fit::{Bundle, Fit, ParentAssignment, ProbeKey},
};

/// The stores a run may write to; each is optional so the CLI works offline.
pub struct Stores {
    pub blob: Option<S3BlobStore>,
    pub hub: Option<HubClient>,
    pub out_dir: Option<PathBuf>,
}

/// What [`write_bundle`] managed to persist.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Written {
    pub local_dir: Option<String>,
    /// Garage key of the manifest (`Snapshot.garage_key`).
    pub manifest_key: Option<String>,
    pub garage: bool,
    pub helix: bool,
    pub helix_error: Option<String>,
    pub regions_written: usize,
    pub region_match_edges: usize,
}

fn files(bundle: &Bundle) -> anyhow::Result<Vec<(&'static str, &'static str, Vec<u8>)>> {
    Ok(vec![
        (MANIFEST_FILE, fa_blob::JSON, bundle.manifest.to_json()?),
        (WEIGHTS_FILE, fa_blob::OCTET_STREAM, bundle.weights()),
        (REGIONS_FILE, fa_blob::JSON, serde_json::to_vec_pretty(&bundle.regions)?),
        (REGION_MAP_FILE, fa_blob::JSON, bundle.region_map.to_json()?),
    ])
}

/// Writes the four snapshot files locally and/or to Garage, then mirrors the snapshot in Helix.
/// `parent` (the parent's graph) is needed for `RegionMatch` edges; `op` labels the lineage edge.
pub async fn write_bundle(
    stores: &Stores,
    bundle: &Bundle,
    parent: Option<&SnapshotGraph>,
    op: &str,
) -> anyhow::Result<Written> {
    let mut w = Written::default();
    let files = files(bundle)?;
    let id = &bundle.manifest.snapshot_id;
    if let Some(dir) = &stores.out_dir {
        let dir = dir.join(id);
        std::fs::create_dir_all(&dir)?;
        for (name, _, bytes) in &files {
            std::fs::write(dir.join(name), bytes)?;
        }
        w.local_dir = Some(dir.display().to_string());
    }
    if let Some(blob) = &stores.blob {
        for (name, ct, bytes) in &files {
            blob.put(&fa_blob::snapshot_file(id, name), bytes.clone(), ct).await?;
        }
        w.garage = true;
        w.manifest_key = Some(fa_blob::snapshot_file(id, MANIFEST_FILE).key);
    }
    if let Some(hub) = &stores.hub {
        match mirror(hub, bundle, w.manifest_key.clone().unwrap_or_default(), parent, op).await {
            Ok((regions, edges)) => {
                w.helix = true;
                w.regions_written = regions;
                w.region_match_edges = edges;
            }
            Err(e) => w.helix_error = Some(e.to_string()),
        }
    }
    Ok(w)
}

async fn mirror(
    hub: &HubClient,
    bundle: &Bundle,
    garage_key: String,
    parent: Option<&SnapshotGraph>,
    op: &str,
) -> anyhow::Result<(usize, usize)> {
    let m = &bundle.manifest;
    let snap = hub
        .ensure_snapshot(&NewSnapshot {
            snapshot_id: m.snapshot_id.clone(),
            timescale: m.timescale.to_string(),
            encoder_family: if m.encoder_family.is_empty() {
                STATISTICAL_FAMILY.into()
            } else {
                m.encoder_family.clone()
            },
            generation: m.generation,
            garage_key,
            status: "candidate".into(),
        })
        .await?;
    // The promotion checklist (control-api) reads the manifest and carried mass from the node.
    let manifest_json = String::from_utf8(m.to_json()?).unwrap_or_default();
    hub.set_snapshot_manifest(snap.id, &manifest_json, bundle.region_map.carried_mass()).await?;
    let mut n_regions = 0;
    let mut n_edges = 0;
    // Latent dimension of every expert space: the neural genome's `expert_dim`, else the statistical 8.
    let dim = m
        .params
        .get("expert_dim")
        .and_then(serde_json::Value::as_u64)
        .map_or(fa_encoders::EXPERT_DIM as u32, |d| d as u32);
    for (k, regions) in bundle.regions.iter().enumerate() {
        let space_id = format!("{}-e{k}", m.snapshot_id);
        let space = hub
            .ensure_expert_space(
                snap.id,
                &NewExpertSpace {
                    expert_space_id: space_id.clone(),
                    snapshot_id: m.snapshot_id.clone(),
                    index: k as u32,
                    dim,
                },
            )
            .await?;
        for r in regions {
            let row = hub
                .upsert_region(
                    space.id,
                    &NewRegion {
                        region_id: r.region_id.clone(),
                        lineage_id: r.lineage_id.clone(),
                        expert_space_id: space_id.clone(),
                        method: "leader".into(),
                        radius: f64::from(r.radius),
                        density: r.mass,
                    },
                )
                .await?;
            n_regions += 1;
            if let (Some(pid), Some(g)) = (&r.parent_region_id, parent)
                && let Some(prev) = g.regions.iter().find(|x| &x.region_id == pid)
            {
                hub.match_regions(prev.id, row.id, r.score).await?;
                n_edges += 1;
            }
        }
    }
    if let Some(g) = parent {
        hub.link_snapshot_parent(g.snapshot.id, snap.id, op).await?;
    }
    Ok((n_regions, n_edges))
}

/// The parent snapshot's graph (regions with lineage) by business id.
pub async fn parent_graph(hub: Option<&HubClient>, parent_id: Option<&str>) -> Option<SnapshotGraph> {
    let (hub, id) = (hub?, parent_id?);
    match hub.snapshot_graph(id).await {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(parent = id, error = %e, "parent snapshot graph unavailable");
            None
        }
    }
}

/// The parent's assignment of the probe windows, from its stored `LatentVec` observations
/// (`kind = observation`, `ref_id = region_id`, `ts` = bar close) of the fitted instruments.
pub async fn parent_assignment(
    hub: Option<&HubClient>,
    parent: Option<&SnapshotGraph>,
    histories: &[BarHistory],
    keys: &[ProbeKey],
) -> ParentAssignment {
    let (Some(hub), Some(g)) = (hub, parent) else {
        return ParentAssignment::none("no parent snapshot in Helix: every region opens a new lineage");
    };
    let since = histories.iter().map(|h| h.start_ms).min().unwrap_or_default();
    let tokens: std::collections::HashSet<String> =
        histories.iter().map(|h| h.instrument.to_ascii_lowercase()).collect();
    // The parent's expert count (statistical: 3; neural: its pruned count).
    let n_experts = if g.spaces.is_empty() { fa_encoders::EXPERT_NAMES.len() } else { g.spaces.len() };
    let mut obs = Vec::new();
    for k in 0..n_experts {
        let space_id = format!("{}-e{k}", g.snapshot.snapshot_id);
        let rows = match hub.space_latents(&space_id, Some("observation"), Some(since), 50_000).await {
            Ok(r) => r,
            Err(e) => return ParentAssignment::none(format!("parent observations unreadable: {e}")),
        };
        for r in rows {
            let token = r.instrument_id.replace('-', "").to_ascii_lowercase();
            if !tokens.contains(&token) {
                continue;
            }
            let Some(ts) = r.ts.as_deref().and_then(parse_rfc3339_ms) else { continue };
            let lineage = g
                .regions
                .iter()
                .find(|x| x.region_id == r.ref_id)
                .map_or_else(|| r.ref_id.clone(), |x| x.lineage_id.clone());
            obs.push((k as u32, r.ref_id, lineage, token, ts));
        }
    }
    let mut a = ParentAssignment::from_observations(&g.snapshot.snapshot_id, n_experts, keys, obs);
    if a.covered == 0 {
        a.note = Some(format!(
            "parent {} has no stored observations for the {} fitted instrument(s) in the window: every region opens a new lineage",
            g.snapshot.snapshot_id,
            histories.len()
        ));
    }
    a
}

pub fn parse_rfc3339_ms(s: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(s).ok().map(|t| t.timestamp_millis())
}

/// Reads a snapshot's manifest and weights (`region_map.json` when present) from Garage.
pub async fn load_snapshot(
    blob: &S3BlobStore,
    snapshot_id: &str,
) -> anyhow::Result<(SnapshotManifest, Vec<u8>, Option<RegionMap>)> {
    let manifest = SnapshotManifest::from_json(&blob.get(&fa_blob::snapshot_file(snapshot_id, MANIFEST_FILE)).await?)?;
    let weights = blob.get(&fa_blob::snapshot_file(snapshot_id, WEIGHTS_FILE)).await?;
    let map = match blob.get(&fa_blob::snapshot_file(snapshot_id, REGION_MAP_FILE)).await {
        Ok(b) => Some(RegionMap::from_json(&b)?),
        Err(fa_blob::BlobError::NotFound(..)) => None,
        Err(e) => return Err(e.into()),
    };
    Ok((manifest, weights, map))
}

// ── checkpoints ─────────────────────────────────────────────────────────────

pub fn checkpoint_key(job_id: &str) -> ObjectKey {
    fa_blob::job_state(job_id)
}

/// Saves the resumable fit (`Fit` + the data window) as `fa-ckpt/<job_id>/state.json`.
pub async fn save_checkpoint(blob: &S3BlobStore, job_id: &str, state: &Checkpoint) -> anyhow::Result<String> {
    let key = checkpoint_key(job_id);
    blob.put(&key, serde_json::to_vec(state)?, fa_blob::JSON).await?;
    Ok(key.key)
}

pub async fn load_checkpoint(blob: &S3BlobStore, key: &str) -> anyhow::Result<Option<Checkpoint>> {
    let key = ObjectKey { bucket: fa_blob::buckets::CKPT, key: key.to_owned() };
    match blob.get(&key).await {
        Ok(b) => Ok(Some(serde_json::from_slice(&b)?)),
        Err(fa_blob::BlobError::NotFound(..)) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Checkpoint payload: the fit state and the exact data window and sources it was built on,
/// so a resumed job reads the same bars.
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct Checkpoint {
    pub job_id: String,
    pub kind: String,
    /// `(instrument token, source)` per history, in fit order.
    pub pins: Vec<(String, Source)>,
    pub start_ms: i64,
    pub end_ms: i64,
    pub fit: Fit,
    pub saved_ms: i64,
}

// ── adapters ────────────────────────────────────────────────────────────────

/// Mirrors an adapter artifact as an `AdapterState` node linked from its snapshot. Returns
/// `Ok(None)` when the snapshot is not in Helix (baseline without a row).
pub async fn write_adapter_state(
    hub: &HubClient,
    a: &fa_hub::NewAdapterState,
) -> anyhow::Result<Option<fa_hub::AdapterState>> {
    let Some(snap) = hub.snapshot_by_id(&a.snapshot_id).await? else { return Ok(None) };
    Ok(Some(hub.add_adapter_state(snap.id, a).await?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_to_ms() {
        assert_eq!(parse_rfc3339_ms("2026-11-10T06:29:00.000Z"), Some(1_794_292_140_000));
        assert_eq!(parse_rfc3339_ms("1970-01-01T00:00:01Z"), Some(1000));
        assert_eq!(parse_rfc3339_ms("nope"), None);
        assert_eq!(checkpoint_key("j").key, "j/state.json");
    }
}
