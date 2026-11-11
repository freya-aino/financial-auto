//! Which region-discovery method a serving snapshot's live region stage runs, and how its regions
//! are labelled in Helix (spec 30 §6.2/§6.3, D-080).
//!
//! Each encoder family declares its discovery (`FamilyDescriptor::region_discovery`: `leader` for
//! the statistical family, `density` for `hier_vae_flow` / `latent_diffusion`, `mapper` for
//! `jepa_vae`, `codebook` for `fsq_hybrid`); the trainer records it in the manifest as
//! `params.region_discovery` and fits `region_map.json` with it. The manifest wins over the
//! descriptor (it is what the region map was actually fitted with); an unknown name falls back to
//! `leader` with a warning, like `fa_patterns::discovery_for_method`.
//!
//! Online, [`fa_patterns::online_form`] decides what streams: only leader clustering has an
//! online form, so a batch method (density, Mapper, codebook) streams as the leader set seeded
//! with its regions. The Helix labels say exactly that:
//!
//! - `RegionSet.method` = the declared method when the set was seeded from the snapshot's region
//!   map, else the online form (`leader`);
//! - `Region.method` = the declared method for a seeded region, the online form for a region born
//!   or split live.
//!
//! The method travels with the snapshot's [`crate::hub_sink::SnapshotDesc`], so a reload
//! (promotion, rollback, explicit swap) re-labels from the next snapshot on.

use std::collections::BTreeSet;

use fa_patterns::{DiscoveryMethod, RegionMap, online_form};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegionMethod {
    /// What the snapshot's family declares (manifest first, then the encoder descriptor).
    pub declared: DiscoveryMethod,
    /// What the streaming stage runs ([`online_form`] of `declared`).
    pub online: DiscoveryMethod,
    /// Region ids seeded from the snapshot's region map (labelled with `declared`).
    pub seeded: BTreeSet<String>,
}

impl Default for RegionMethod {
    /// The statistical baseline: leader everywhere, nothing seeded.
    fn default() -> Self {
        Self { declared: DiscoveryMethod::Leader, online: DiscoveryMethod::Leader, seeded: BTreeSet::new() }
    }
}

impl RegionMethod {
    /// Resolves the method of a snapshot from its manifest `params` (when it has one), its encoder
    /// descriptor's `region_discovery` and its region map. Logs when the online form differs from
    /// the declared method (a batch method streaming as seeded leader) and when names disagree.
    pub fn resolve(
        snapshot_id: &str,
        manifest_params: Option<&Value>,
        descriptor: &str,
        region_map: Option<&RegionMap>,
    ) -> Self {
        let recorded = manifest_params.and_then(|p| p.get("region_discovery")).and_then(Value::as_str);
        if let Some(r) = recorded
            && r != descriptor
        {
            tracing::warn!(snapshot = %snapshot_id, manifest = %r, descriptor = %descriptor, "manifest region_discovery differs from the family descriptor; using the manifest's");
        }
        let name = recorded.unwrap_or(descriptor);
        let declared = DiscoveryMethod::from_name(name).unwrap_or_else(|| {
            tracing::warn!(snapshot = %snapshot_id, method = %name, "unknown region discovery method; labelling as leader");
            DiscoveryMethod::Leader
        });
        let online = online_form(declared);
        let seeded: BTreeSet<String> = region_map
            .map(|m| m.experts.iter().flat_map(|e| e.regions.iter().map(|r| r.region_id.clone())).collect())
            .unwrap_or_default();
        if online != declared {
            tracing::info!(
                snapshot = %snapshot_id,
                declared = declared.as_str(),
                online = online.as_str(),
                seeded_regions = seeded.len(),
                "declared region discovery has no online form; streaming as {} seeded with the snapshot's {} regions",
                online.as_str(),
                declared.as_str()
            );
        }
        Self { declared, online, seeded }
    }

    /// `RegionSet.method`: the declared method when its regions came from the snapshot's map.
    pub fn set_label(&self) -> &'static str {
        if self.seeded.is_empty() { self.online.as_str() } else { self.declared.as_str() }
    }

    /// `Region.method` of one region: declared when seeded from the map, else the online form.
    pub fn region_label(&self, region_id: &str) -> &'static str {
        if self.seeded.contains(region_id) { self.declared.as_str() } else { self.online.as_str() }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn map(ids: &[&str]) -> RegionMap {
        let regions: Vec<Value> = ids
            .iter()
            .map(|id| json!({"region_id": id, "lineage_id": id, "expert_index": 0, "centroid": [0.0], "radius": 1.0, "mass": 1.0}))
            .collect();
        serde_json::from_value(json!({
            "snapshot_id": "s", "probe_set_version": "v", "n_probes": 0, "carry_threshold": 0.5,
            "experts": [{"expert_index": 0, "regions": regions, "parent_regions": 0, "matched": 0, "carried_mass": 0.0}]
        }))
        .unwrap()
    }

    #[test]
    fn statistical_snapshots_stay_leader() {
        let m = RegionMethod::resolve("stat-v1-1m", None, "leader", None);
        assert_eq!(m, RegionMethod::default());
        assert_eq!(m.set_label(), "leader");
        assert_eq!(m.region_label("1m-e0-r000"), "leader");
    }

    #[test]
    fn neural_snapshots_label_seeded_regions_with_their_declared_method() {
        let rm = map(&["1m-e0-r000", "1m-e0-r001"]);
        for (family_method, label) in [("density", "density"), ("mapper", "mapper"), ("codebook", "codebook")] {
            let m = RegionMethod::resolve("n1", None, family_method, Some(&rm));
            assert_eq!(m.online, DiscoveryMethod::Leader, "no online form → seeded leader");
            assert_eq!(m.set_label(), label);
            assert_eq!(m.region_label("1m-e0-r001"), label);
            assert_eq!(m.region_label("1m-e0-r007"), "leader", "born live");
        }
        // without a region map nothing came from the declared method
        assert_eq!(RegionMethod::resolve("n1", None, "density", None).set_label(), "leader");
    }

    #[test]
    fn the_manifest_wins_over_the_descriptor_and_unknown_names_fall_back() {
        let rm = map(&["r"]);
        let m = RegionMethod::resolve("n1", Some(&json!({"region_discovery": "codebook"})), "density", Some(&rm));
        assert_eq!(m.declared, DiscoveryMethod::Codebook);
        let m = RegionMethod::resolve("n1", Some(&json!({"other": 1})), "mapper", Some(&rm));
        assert_eq!(m.declared, DiscoveryMethod::Mapper);
        let m = RegionMethod::resolve("n1", Some(&json!({"region_discovery": "hdbscan"})), "density", Some(&rm));
        assert_eq!(m.declared, DiscoveryMethod::Leader);
        assert_eq!(m.set_label(), "leader");
    }
}
