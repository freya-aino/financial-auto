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

#[tokio::test]
async fn ensure_upsert_and_lifecycle_queries_are_idempotent() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();

    let new_snap = NewSnapshot {
        snapshot_id: format!("{p}stat"),
        timescale: "1m".into(),
        encoder_family: "statistical".into(),
        generation: 1,
        garage_key: String::new(),
        status: "active".into(),
    };
    let snap = c.ensure_snapshot(&new_snap).await.expect("ensure_snapshot create");
    let again = c.ensure_snapshot(&NewSnapshot { generation: 99, ..new_snap.clone() }).await.expect("ensure again");
    assert_eq!(again, snap, "existing row is returned unchanged");

    let new_space = NewExpertSpace {
        expert_space_id: format!("{p}stat-e0"),
        snapshot_id: snap.snapshot_id.clone(),
        index: 0,
        dim: 8,
    };
    let space = c.ensure_expert_space(snap.id, &new_space).await.expect("ensure_expert_space");
    assert_eq!(c.ensure_expert_space(snap.id, &new_space).await.expect("ensure again"), space);
    assert_eq!(c.snapshot_spaces(snap.id).await.expect("spaces").len(), 1, "no duplicate HasSpace");
    assert_eq!(c.space_by_id(&space.expert_space_id).await.expect("space_by_id"), Some(space.clone()));

    let mk = |n: &str, radius: f64| NewRegion {
        region_id: format!("{p}r{n}"),
        lineage_id: format!("{p}r{n}"),
        expert_space_id: space.expert_space_id.clone(),
        method: "leader".into(),
        radius,
        density: 1.0,
    };
    let r1 = c.upsert_region(space.id, &mk("1", 1.0)).await.expect("upsert_region create");
    assert_eq!(r1.status.as_deref(), Some("live"));
    let r2 = c.upsert_region(space.id, &mk("2", 1.0)).await.expect("upsert_region create");
    let r1b = c.upsert_region(space.id, &mk("1", 2.0)).await.expect("upsert_region revive");
    assert_eq!((r1b.id, r1b.radius), (r1.id, 2.0));
    assert_eq!(c.space_regions(space.id).await.expect("regions").len(), 2, "no duplicate region / InSpace");

    let merged = c.mark_region_status(&r1.region_id, "merged", &r2.region_id).await.expect("mark merged");
    assert_eq!(
        (merged.status.as_deref(), merged.related_region_id.as_deref()),
        (Some("merged"), Some(r2.region_id.as_str()))
    );
    assert!(merged.updated_at.is_some());
    let missing = c.mark_region_status(&format!("{p}nope"), "retired", "").await;
    assert!(matches!(missing, Err(HubError::NotFound { .. })), "{missing:?}");
    assert_eq!(c.region_by_id(&r2.region_id).await.expect("region_by_id"), Some(r2.clone()));

    let row = |to: &LatentRegion, count: i64, mean: f64| TransitionUpsert {
        from_region_id: r1.region_id.clone(),
        to_region_id: to.region_id.clone(),
        transition: NewTransition {
            sampler_id: "conjugate".into(),
            count,
            mean,
            lower: mean - 0.05,
            upper: mean + 0.05,
            kernel_params: "{}".into(),
        },
    };
    let t = c.upsert_transition(&row(&r2, 1, 0.5)).await.expect("upsert_transition create");
    assert_eq!((t.from, t.to, t.count, t.to_region_id.as_deref()), (r1.id, r2.id, 1, Some(r2.region_id.as_str())));
    let both = c.upsert_transitions(&[row(&r2, 2, 0.6), row(&r1, 1, 0.4)]).await.expect("upsert_transitions");
    assert_eq!(both.len(), 2);
    assert_eq!((both[0].id, both[0].count, both[0].mean), (t.id, 2, 0.6), "same edge updated in place");
    assert_eq!((both[1].from, both[1].to), (r1.id, r1.id), "self loop created");
    assert_eq!(c.region_transitions(r1.id).await.expect("transitions").len(), 2);
    let unknown = c.upsert_transition(&TransitionUpsert { to_region_id: format!("{p}nope"), ..row(&r2, 1, 0.1) }).await;
    assert!(matches!(unknown, Err(HubError::NotFound { .. })), "{unknown:?}");

    let g = c.snapshot_graph(&snap.snapshot_id).await.expect("snapshot_graph").expect("exists");
    assert_eq!((g.snapshot.id, g.spaces.len(), g.regions.len(), g.transitions.len()), (snap.id, 1, 2, 2));
    assert!(c.snapshot_graph(&format!("{p}nope")).await.expect("snapshot_graph").is_none());

    // resolve_node by business id and by numeric id.
    let n = c.resolve_node(&r2.region_id).await.expect("resolve").expect("found");
    assert_eq!((n.id, n.label.as_str()), (r2.id, "LatentRegion"));
    assert_eq!(c.resolve_node(&snap.id.to_string()).await.expect("resolve").map(|n| n.id), Some(snap.id));
    assert!(c.resolve_node(&format!("{p}nothing")).await.expect("resolve").is_none());

    // Named-query catalog.
    let v = c.run_named("snapshotGraph", &serde_json::json!({ "snapshot_id": snap.snapshot_id })).await.expect("named");
    assert_eq!(v["regions"].as_array().map(Vec::len), Some(2));
    let v = c.run_named("regionTransitions", &serde_json::json!({ "region_id": r1.region_id })).await.expect("named");
    assert_eq!(v.as_array().map(Vec::len), Some(2));
    let v = c.run_named("neighborhood", &serde_json::json!({ "node": r1.region_id, "depth": 1 })).await.expect("named");
    assert!(v["nodes"].as_array().is_some_and(|n| n.len() >= 2));
    let e = c.run_named("getRegion", &serde_json::json!({})).await;
    assert!(matches!(e, Err(HubError::BadParam(_))), "{e:?}");
    let e = c.run_named("spaceRegions", &serde_json::json!({ "expert_space_id": format!("{p}nope") })).await;
    assert!(matches!(e, Err(HubError::NotFound { .. })), "{e:?}");

    cleanup(&c, &p).await;
}

#[tokio::test]
async fn genome_business_id_writes_for_evo_worker() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();
    let mk = |n: &str| NewGenome {
        genome_id: format!("{p}g{n}"),
        kind: "decoder".into(),
        family: "forecast".into(),
        params: "{\"h\":5}".into(),
    };
    let g0 = c.upsert_genome(&mk("0")).await.expect("upsert_genome");
    assert_eq!(c.upsert_genome(&mk("0")).await.expect("upsert again"), g0);
    let g1 = c.upsert_genome(&mk("1")).await.expect("upsert_genome");
    c.link_parent(&g0.genome_id, &g1.genome_id, "mutate").await.expect("link_parent");
    let bad = c.link_parent(&g0.genome_id, &format!("{p}nope"), "mutate").await;
    assert!(matches!(bad, Err(HubError::NotFound { .. })), "{bad:?}");
    let f = c
        .add_fitness_record(&NewFitness {
            genome_id: g1.genome_id.clone(),
            generation: 2,
            objectives: "{\"crps_skill\":0.1}".into(),
            pareto_rank: 0,
        })
        .await
        .expect("add_fitness_record");
    assert_eq!(c.genome_fitness(g1.id).await.expect("fitness"), vec![f]);
    assert_eq!(c.genome_parents(g1.id).await.expect("parents"), vec![g0.clone()]);
    let v = c.run_named("genomeAncestry", &serde_json::json!({ "genome_id": g1.genome_id })).await.expect("named");
    assert_eq!(v[0][0]["genome_id"], g0.genome_id);
    let bad = c
        .add_fitness_record(&NewFitness {
            genome_id: format!("{p}nope"),
            generation: 0,
            objectives: "{}".into(),
            pareto_rank: 0,
        })
        .await;
    assert!(matches!(bad, Err(HubError::NotFound { .. })), "{bad:?}");
    cleanup(&c, &p).await;
}

#[tokio::test]
async fn space_latents_read_back_vectors_newest_first() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    c.ensure_latent_index_ready(4, WAIT).await.expect("latent index");
    let p = prefix();
    let space = format!("{p}space-l");
    let mk = |ref_id: &str, ts_ms: i64, v: [f32; 4]| NewLatent {
        vector: v.to_vec(),
        expert_space_id: space.clone(),
        kind: "observation".into(),
        ref_id: format!("{p}{ref_id}"),
        instrument_id: "BTC".into(),
        ts_ms,
        uncertainty: 0.1,
    };
    c.add_latents(&[
        mk("a", 1_762_620_000_000, [1.0, 0.0, 0.0, 0.0]),
        mk("b", 1_762_620_060_000, [0.0, 1.0, 0.0, 0.0]),
        mk("c", 1_762_620_120_000, [0.0, 0.0, 1.0, 0.0]),
    ])
    .await
    .expect("add_latents");
    assert_eq!(c.count_space_latents(&space).await.expect("count"), 3);

    let rows = c.space_latents(&space, Some("observation"), None, 10).await.expect("space_latents");
    let refs: Vec<&str> = rows.iter().map(|r| r.ref_id.as_str()).collect();
    assert_eq!(refs, vec![format!("{p}c"), format!("{p}b"), format!("{p}a")], "{rows:?}");
    assert_eq!(rows[0].vector(), Some(vec![0.0, 0.0, 1.0, 0.0]), "{:?}", rows[0]);
    assert!(rows.iter().all(|r| r.ts.is_some()));

    let recent = c.space_latents(&space, None, Some(1_762_620_060_000), 10).await.expect("since");
    assert_eq!(recent.len(), 2, "{recent:?}");
    let limited = c.space_latents(&space, None, None, 1).await.expect("limit");
    assert_eq!(limited.len(), 1);

    cleanup(&c, &p).await;
}

#[tokio::test]
async fn snapshot_lineage_and_status_lookup() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();
    let mk = |g: u32, status: &str| NewSnapshot {
        snapshot_id: format!("{p}snap-g{g}"),
        timescale: format!("{p}ts"),
        encoder_family: "statistical".into(),
        generation: g,
        garage_key: format!("snapshots/{p}snap-g{g}/manifest.json"),
        status: status.into(),
    };
    let parent = c.ensure_snapshot(&mk(0, "active")).await.expect("parent");
    let child = c.ensure_snapshot(&mk(1, "candidate")).await.expect("child");
    assert!(c.snapshot_parent(child.id).await.expect("no parent yet").is_none());
    c.link_snapshot_parent(parent.id, child.id, "pretrain").await.expect("link");
    let got = c.snapshot_parent(child.id).await.expect("parent lookup").expect("linked");
    assert_eq!(got.id, parent.id);
    let kids = c.snapshot_children(parent.id).await.expect("children");
    assert_eq!(kids.len(), 1);
    assert_eq!(kids[0].snapshot_id, child.snapshot_id);

    let ts = format!("{p}ts");
    let active = c.active_snapshot(&ts, "statistical").await.expect("active").expect("one active");
    assert_eq!(active.id, parent.id);
    let cands = c.snapshots_by_status(&ts, "statistical", "candidate").await.expect("candidates");
    assert_eq!(cands.len(), 1);
    assert!(c.active_snapshot(&ts, "fsq_hybrid").await.expect("other family").is_none());
    cleanup(&c, &p).await;
}
