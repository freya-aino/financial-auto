//! Integration tests against a running HelixDB. Skipped unless `FA_HELIX_URL` is set, e.g.
//! `FA_HELIX_URL=http://127.0.0.1:6969 cargo test -p fa-hub --test live -- --nocapture`.
//!
//! Every row written carries a `test-<nanos>-` prefixed business id and is dropped at the end,
//! so a shared dev database is left as found (indexes created by `ensure_schema` stay).

use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

fn prefix() -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock").as_nanos();
    format!("test-{nanos}-")
}

const WAIT: Duration = Duration::from_secs(20);

async fn cleanup(c: &HubClient, p: &str) {
    for (label, key) in [
        (node::SNAPSHOT, "snapshot_id"),
        (node::EXPERT_SPACE, "expert_space_id"),
        (node::LATENT_REGION, "region_id"),
        (node::LATENT_VEC, "expert_space_id"),
        (node::INSTRUMENT, "instrument_id"),
        (node::GENOME, "genome_id"),
        (node::FITNESS_RECORD, "genome_id"),
    ] {
        c.drop_nodes(label, Pred::starts_with(key, p)).await.expect("cleanup drop");
        let left = c.query(Op::nodes_with_label_where(label, Pred::starts_with(key, p)).count()).await.expect("count");
        assert_eq!(left, 0, "{label} rows with prefix {p} remain");
    }
}

#[tokio::test]
async fn health_ready_diagnostics_and_schema() {
    let Some(c) = client() else { return };
    let h = c.health().await.expect("healthz");
    assert!(h.ready, "server not ready: {h:?}");
    assert!(c.ready().await.expect("readyz"));
    let d = c.diagnostics().await.expect("diagnostics");
    assert!(d.get("checks").and_then(|v| v.as_array()).is_some(), "diagnostics: {d}");

    let receipts = c.ensure_schema().await.expect("ensure_schema");
    assert_eq!(receipts.len(), fa_hub::schema::static_indexes().len());
    assert!(
        receipts.iter().all(|r| matches!(r.kind.as_str(), "accepted" | "already_active" | "existing_operation")),
        "{receipts:?}"
    );
    c.wait_for_receipts(&receipts, WAIT).await.expect("index builds");
    // Second run is a no-op: everything is already active.
    let again = c.ensure_schema().await.expect("ensure_schema again");
    assert!(again.iter().all(IndexReceipt::is_active), "{again:?}");
}

#[tokio::test]
async fn snapshot_regions_transition_roundtrip() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();

    let snap = c
        .add_snapshot(&NewSnapshot {
            snapshot_id: format!("{p}snap"),
            timescale: "1h".into(),
            encoder_family: "moe".into(),
            generation: 3,
            garage_key: "snapshots/x.bin".into(),
            status: "candidate".into(),
        })
        .await
        .expect("add_snapshot");
    assert_eq!(snap.generation, 3);
    assert!(snap.created_at.as_deref().is_some_and(|t| t.ends_with('Z')), "created_at: {:?}", snap.created_at);

    // Unique index enforces snapshot_id.
    let dup = c
        .add_snapshot(&NewSnapshot {
            snapshot_id: format!("{p}snap"),
            timescale: "1h".into(),
            encoder_family: "moe".into(),
            generation: 4,
            garage_key: String::new(),
            status: "candidate".into(),
        })
        .await;
    assert!(matches!(&dup, Err(e) if e.is_unique_violation()), "{dup:?}");

    let space = c
        .add_expert_space(
            snap.id,
            &NewExpertSpace {
                expert_space_id: format!("{p}space"),
                snapshot_id: snap.snapshot_id.clone(),
                index: 0,
                dim: 4,
            },
        )
        .await
        .expect("add_expert_space");
    let spaces = c.snapshot_spaces(snap.id).await.expect("snapshot_spaces");
    assert_eq!(spaces, vec![space.clone()]);

    let mk = |n: &str| NewRegion {
        region_id: format!("{p}r{n}"),
        lineage_id: format!("{p}l{n}"),
        expert_space_id: space.expert_space_id.clone(),
        method: "codebook".into(),
        radius: 0.5,
        density: 1.5,
    };
    let r1 = c.add_region(space.id, &mk("1")).await.expect("add_region");
    let r2 = c.add_region(space.id, &mk("2")).await.expect("add_region");
    let mut regions = c.space_regions(space.id).await.expect("space_regions");
    regions.sort_by_key(|r| r.id);
    assert_eq!(regions, vec![r1.clone(), r2.clone()]);

    let t = c
        .add_transition(
            r1.id,
            r2.id,
            &NewTransition {
                sampler_id: format!("{p}sampler"),
                count: 7,
                mean: 0.25,
                lower: 0.1,
                upper: 0.4,
                kernel_params: "{\"h\":1}".into(),
            },
        )
        .await
        .expect("add_transition");
    assert_eq!((t.from, t.to, t.count, t.mean), (r1.id, r2.id, 7, 0.25));
    assert_eq!(c.region_transitions(r1.id).await.expect("region_transitions"), vec![t.clone()]);
    assert_eq!(c.region_neighbors(r1.id).await.expect("region_neighbors"), vec![r2.clone()]);
    assert!(c.region_transitions(r2.id).await.expect("region_transitions").is_empty());

    let g = c.region_graph(space.id).await.expect("region_graph");
    assert_eq!(g.regions.len(), 2);
    assert_eq!(g.transitions, vec![t]);

    let fetched = c.snapshot_by_id(&snap.snapshot_id).await.expect("snapshot_by_id");
    assert_eq!(fetched.as_ref(), Some(&snap));
    let promoted = c.set_snapshot_status(snap.id, "active").await.expect("set_snapshot_status");
    assert_eq!(promoted.status, "active");
    assert!(c.list_snapshots().await.expect("list_snapshots").iter().any(|s| s.id == snap.id));

    let nb = c.neighborhood(r1.id, 2).await.expect("neighborhood");
    let ids: Vec<u64> = nb.nodes.iter().map(|n| n.id).collect();
    assert!(
        ids.contains(&r1.id) && ids.contains(&r2.id) && ids.contains(&space.id) && ids.contains(&snap.id),
        "{ids:?}"
    );
    assert!(nb.edges.iter().any(|e| e.label == "Transition"));

    cleanup(&c, &p).await;
}

#[tokio::test]
async fn instrument_upsert_and_genome_lineage() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();

    let a = c
        .upsert_instrument(&NewInstrument {
            instrument_id: format!("{p}BTC"),
            asset_class: "crypto".into(),
            kind: "spot".into(),
            name: "Bitcoin".into(),
        })
        .await
        .expect("upsert create");
    let b = c
        .upsert_instrument(&NewInstrument {
            instrument_id: format!("{p}BTC"),
            asset_class: "crypto".into(),
            kind: "spot".into(),
            name: "Bitcoin (renamed)".into(),
        })
        .await
        .expect("upsert update");
    assert_eq!(a.id, b.id);
    assert_eq!(b.name, "Bitcoin (renamed)");
    assert_eq!(c.get_instrument(&a.instrument_id).await.expect("get_instrument"), Some(b));
    assert_eq!(c.get_instrument(&format!("{p}none")).await.expect("get_instrument"), None);

    let mk = |n: &str| NewGenome {
        genome_id: format!("{p}g{n}"),
        kind: "decoder".into(),
        family: "retrieval".into(),
        params: "{}".into(),
    };
    let g0 = c.add_genome(&mk("0")).await.expect("add_genome");
    let g1 = c.add_genome(&mk("1")).await.expect("add_genome");
    let g2 = c.add_genome(&mk("2")).await.expect("add_genome");
    c.add_lineage(g0.id, g1.id, "mutate").await.expect("add_lineage");
    c.add_lineage(g1.id, g2.id, "finetune").await.expect("add_lineage");
    assert_eq!(c.genome_parents(g2.id).await.expect("genome_parents"), vec![g1.clone()]);
    let anc = c.genome_ancestry(g2.id, 5).await.expect("genome_ancestry");
    assert_eq!(anc, vec![vec![g1.clone()], vec![g0.clone()]]);

    let f = c
        .add_fitness(
            g2.id,
            &NewFitness {
                genome_id: g2.genome_id.clone(),
                generation: 9,
                objectives: "{\"forecast\":0.4}".into(),
                pareto_rank: 1,
            },
        )
        .await
        .expect("add_fitness");
    assert_eq!(c.genome_fitness(g2.id).await.expect("genome_fitness"), vec![f]);
    assert_eq!(c.genome_by_id(&g1.genome_id).await.expect("genome_by_id"), Some(g1));

    cleanup(&c, &p).await;
}

#[tokio::test]
async fn latent_vector_search_is_scoped_to_the_expert_space() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    c.ensure_latent_index_ready(4, WAIT).await.expect("latent index");
    let p = prefix();
    let space_a = format!("{p}space-a");
    let space_b = format!("{p}space-b");

    let mk = |space: &str, ref_id: &str, v: [f32; 4]| NewLatent {
        vector: v.to_vec(),
        expert_space_id: space.to_owned(),
        kind: "window".into(),
        ref_id: format!("{p}{ref_id}"),
        instrument_id: "BTC".into(),
        ts_ms: 1_762_620_000_000,
        uncertainty: 0.1,
    };
    let ids = c
        .add_latents(&[
            mk(&space_a, "w1", [1.0, 0.0, 0.0, 0.0]),
            mk(&space_a, "w2", [0.9, 0.1, 0.0, 0.0]),
            mk(&space_a, "w3", [0.0, 0.0, 1.0, 0.0]),
            mk(&space_b, "w4", [1.0, 0.0, 0.0, 0.0]),
        ])
        .await
        .expect("add_latents");
    assert_eq!(ids.len(), 4);
    let single = c.add_latent(&mk(&space_b, "w5", [0.0, 1.0, 0.0, 0.0])).await.expect("add_latent");
    assert!(!ids.contains(&single));

    let hits = c.search_latent(&space_a, &[1.0, 0.0, 0.0, 0.0], 10).await.expect("search_latent");
    let refs: Vec<&str> = hits.iter().map(|h| h.ref_id.as_str()).collect();
    assert_eq!(refs, vec![format!("{p}w1"), format!("{p}w2"), format!("{p}w3")], "{hits:?}");
    assert!(
        hits[0].distance.abs() < 1e-6 && hits[0].distance <= hits[1].distance && hits[1].distance <= hits[2].distance
    );
    assert!(hits.iter().all(|h| h.expert_space_id == space_a && h.kind == "window" && h.ts.is_some()));

    let hits_b = c.search_latent(&space_b, &[1.0, 0.0, 0.0, 0.0], 1).await.expect("search_latent");
    assert_eq!(hits_b.len(), 1);
    assert_eq!(hits_b[0].ref_id, format!("{p}w4"));

    let none = c.search_latent(&format!("{p}unknown"), &[1.0, 0.0, 0.0, 0.0], 3).await.expect("unknown tenant");
    assert!(none.is_empty());

    cleanup(&c, &p).await;
}
