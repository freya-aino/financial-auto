//! Region lifecycle / lineage / kernel persistence against a running HelixDB (fa-hub `regions`).
//! Skipped unless `FA_HELIX_URL` is set, e.g.
//! `FA_HELIX_URL=http://127.0.0.1:6969 cargo test -p fa-hub --test regions_live -- --nocapture`.
//! Every row carries a `test-<nanos>-` prefix and is dropped at the end.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fa_hub::regions::labels;
use fa_hub::schema::node;
use fa_hub::*;

fn client() -> Option<HubClient> {
    match std::env::var(URL_ENV) {
        Ok(url) => Some(HubClient::new(url)),
        Err(_) => {
            eprintln!("skipping: {URL_ENV} not set");
            None
        }
    }
}

async fn cleanup(c: &HubClient, p: &str) {
    for (label, key) in [
        (node::SNAPSHOT, "snapshot_id"),
        (node::EXPERT_SPACE, "expert_space_id"),
        (node::LATENT_REGION, "region_id"),
        (node::LATENT_VEC, "expert_space_id"),
        (labels::REGION_SET, "region_set_id"),
        (labels::KERNEL_MODEL, "kernel_id"),
        (labels::CROSS_PRIOR, "source_instrument_id"),
    ] {
        c.drop_nodes(label, Pred::starts_with(key, p)).await.expect("cleanup drop");
    }
}

#[tokio::test]
async fn lifecycle_lineage_hierarchy_kernel_and_cross_priors() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(Duration::from_secs(60)).await.expect("schema");
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let p = format!("test-{nanos}-");
    let snap_id = format!("{p}snap");
    let snap = c
        .ensure_snapshot(&NewSnapshot {
            snapshot_id: snap_id.clone(),
            timescale: "1m".into(),
            encoder_family: "statistical".into(),
            generation: 0,
            garage_key: String::new(),
            status: "active".into(),
        })
        .await
        .unwrap();
    let mut spaces = Vec::new();
    for k in 0..2u32 {
        let id = format!("{snap_id}-e{k}");
        let sp = c
            .ensure_expert_space(
                snap.id,
                &NewExpertSpace { expert_space_id: id.clone(), snapshot_id: snap_id.clone(), index: k, dim: 4 },
            )
            .await
            .unwrap();
        spaces.push((id, sp.id));
    }
    let r = |n: &str| format!("{p}{n}");
    for (name, space) in [("parent", 0), ("child", 0), ("split", 0), ("coarse", 1)] {
        c.upsert_region(
            spaces[space].1,
            &NewRegion {
                region_id: r(name),
                lineage_id: if name == "child" { r("parent") } else { r(name) },
                expert_space_id: spaces[space].0.clone(),
                method: "leader".into(),
                radius: 1.0,
                density: 0.0,
            },
        )
        .await
        .unwrap();
    }

    // stats + death + centroid
    c.update_region_stats(&[RegionStatsUpdate {
        region_id: r("child"),
        mass: 12.0,
        occupancy: 0.4,
        spread: 0.3,
        age: 99,
    }])
    .await
    .unwrap();
    let row = c.region_by_id(&r("child")).await.unwrap().unwrap();
    assert_eq!(row.lineage_id, r("parent"));
    let full = c.resolve_node(&r("child")).await.unwrap().unwrap();
    assert_eq!(full.properties["mass"], 12.0);
    assert_eq!(full.properties["age"], 99);
    c.mark_region_died(&r("parent"), "support", "").await.unwrap();
    let dead = c.resolve_node(&r("parent")).await.unwrap().unwrap();
    assert_eq!(dead.properties["status"], "retired");
    assert_eq!(dead.properties["retire_reason"], "support");
    assert!(dead.properties.contains_key("died_at"));
    assert!(c.mark_region_died(&r("nope"), "support", "").await.is_err());
    c.ensure_latent_index_ready(4, Duration::from_secs(60)).await.unwrap();
    assert!(c.upsert_centroid(&spaces[0].0, &r("child"), &[0.1, 0.2, 0.3, 0.4]).await.unwrap());
    assert!(c.upsert_centroid(&spaces[0].0, &r("child"), &[0.2, 0.2, 0.3, 0.4]).await.unwrap());
    assert!(!c.upsert_centroid(&spaces[0].0, &r("child"), &[0.0; 4]).await.unwrap());
    let cents = c.space_latents(&spaces[0].0, Some("centroid"), None, 10).await.unwrap();
    assert_eq!(cents.len(), 1, "centroid replaced, not appended");

    // lineage: child carries parent, split descends from parent (twice upserted = one edge)
    let d = |child: &str, kind: &str, overlap: f64| NewDescent {
        child_region_id: r(child),
        parent_region_id: r("parent"),
        kind: kind.into(),
        overlap,
    };
    c.upsert_descents(&[d("child", "carry", 0.7), d("split", "split", 0.25)]).await.unwrap();
    c.upsert_descents(&[d("split", "split", 0.3), d("ghost", "split", 0.1)]).await.unwrap();
    let lin = c.region_lineage(&r("parent"), 3).await.unwrap().unwrap();
    assert!(lin.ancestors.is_empty());
    let mut kids: Vec<(String, String, f64)> = lin
        .descendants
        .iter()
        .map(|l| (l.region.properties["region_id"].as_str().unwrap().to_owned(), l.kind.clone(), l.overlap))
        .collect();
    kids.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(kids, vec![(r("child"), "carry".into(), 0.7), (r("split"), "split".into(), 0.3)]);
    assert_eq!(lin.lineage.len(), 2, "parent and child share the lineage");
    let up = c.region_lineage(&r("split"), 3).await.unwrap().unwrap();
    assert_eq!(up.ancestors.len(), 1);
    assert_eq!(up.ancestors[0].via, r("split"));
    assert!(c.region_lineage(&r("nope"), 2).await.unwrap().is_none());

    // hierarchy
    let h = NewHierEdge {
        fine_region_id: r("child"),
        coarse_region_id: r("coarse"),
        p_coarse_given_fine: 0.8,
        p_fine_given_coarse: 0.5,
        support: 40.0,
        pmi_bits: 1.2,
        mi_bits: 0.3,
    };
    c.upsert_hier_edges(std::slice::from_ref(&h)).await.unwrap();
    c.upsert_hier_edges(&[NewHierEdge { support: 50.0, ..h }]).await.unwrap();
    let hier = c.snapshot_hierarchy(&snap_id).await.unwrap();
    assert_eq!(hier.len(), 1);
    assert_eq!(hier[0].properties["support"], 50.0);

    // region set + kernel model
    for round in 0..2 {
        c.upsert_region_set(&NewRegionSet {
            expert_space_id: spaces[0].0.clone(),
            snapshot_id: snap_id.clone(),
            timescale: "1m".into(),
            expert_index: 0,
            family: "statistical".into(),
            method: "leader".into(),
            n_regions: 3 + round,
            cfg_hash: "abc".into(),
            state: format!("{{\"round\":{round}}}"),
            kernel_state: "{}".into(),
        })
        .await
        .unwrap();
        c.upsert_kernel_model(&NewKernelModel {
            expert_space_id: spaces[0].0.clone(),
            snapshot_id: snap_id.clone(),
            timescale: "1m".into(),
            expert_index: 0,
            kernel_type: "dirichlet".into(),
            serving_sampler: "conjugate".into(),
            scores: "[]".into(),
            stationary: "[]".into(),
            relaxation_steps: 3.5,
            drift_kl: 0.01 * f64::from(round),
            n_rows: 3,
            observation_mass: 10.0,
            live_regions: 3,
        })
        .await
        .unwrap();
    }
    let rs = c.region_set(&spaces[0].0).await.unwrap().unwrap();
    assert_eq!((rs.n_regions, rs.state.as_str()), (4, "{\"round\":1}"));
    let ks = c.snapshot_kernels(&snap_id).await.unwrap();
    assert_eq!(ks.len(), 1);
    assert_eq!(ks[0].properties["drift_kl"], 0.01);
    let nb = c.neighborhood(c.space_by_id(&spaces[0].0).await.unwrap().unwrap().id, 1).await.unwrap();
    assert!(nb.edges.iter().any(|e| e.label == labels::HAS_KERNEL));
    assert!(nb.edges.iter().any(|e| e.label == labels::HAS_REGION_SET));

    // cross priors
    let cp = |src: &str, score: f64, rank: u32| NewCrossPrior {
        timescale: format!("{p}ts"),
        source_instrument_id: format!("{p}{src}"),
        target_instrument_id: format!("{p}t"),
        n: 100,
        correlation: 0.2,
        shrinkage: 0.1,
        lead_lag: 1,
        lead_corr: 0.3,
        predictive_gain: 0.001,
        uncertainty: 0.0002,
        score,
        rank,
    };
    c.upsert_cross_priors(&[cp("a", 5.0, 1), cp("b", 2.0, 2)]).await.unwrap();
    c.upsert_cross_priors(&[cp("b", 7.0, 1)]).await.unwrap();
    let rows = c.cross_priors(&format!("{p}ts"), 10).await.unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].properties["source_instrument_id"], format!("{p}b"));

    cleanup(&c, &p).await;
}
