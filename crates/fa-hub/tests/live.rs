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
        (node::ADAPTER_STATE, "snapshot_id"),
        (node::EXPERT_SPACE, "expert_space_id"),
        (node::LATENT_REGION, "region_id"),
        (node::LATENT_VEC, "expert_space_id"),
        (node::INSTRUMENT, "instrument_id"),
        (node::ENTITY, "entity_id"),
        (node::VENUE, "venue_id"),
        (node::GENOME, "genome_id"),
        (node::FITNESS_RECORD, "genome_id"),
        (node::NEWS_DOC, "doc_id"),
        (node::MEDIA_SEGMENT, "segment_id"),
        (fa_hub::versions::labels::TRANSITION_VERSION, "edge_key"),
        (fa_hub::versions::labels::REGION_STATUS_VERSION, "region_id"),
    ] {
        // label-wide prefix scans of parallel tests conflict; back off beyond the client's retries
        let mut attempt = 0;
        while let Err(e) = c.drop_nodes(label, Pred::starts_with(key, p)).await {
            attempt += 1;
            assert!(attempt < 10 && matches!(&e, HubError::Server { status: 409, .. }), "cleanup drop: {e:?}");
            tokio::time::sleep(Duration::from_millis(100 * attempt)).await;
        }
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
async fn reference_mirror_upserts_prunes_and_retires() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();
    let ent =
        RefEntity { entity_id: format!("{p}e1"), lei: "LEI1".into(), name: "Issuer".into(), ..Default::default() };
    let ven = |n: &str| RefVenue {
        venue_id: format!("{p}v{n}"),
        code: format!("v{n}"),
        name: format!("Venue {n}"),
        kind: "exchange".into(),
        timezone: "UTC".into(),
        ..Default::default()
    };
    let listing = |v: &str, sym: &str| RefListing {
        venue_id: format!("{p}v{v}"),
        venue_symbol: sym.into(),
        source_id: "src".into(),
    };
    let mut spot = RefInstrument {
        instrument_id: format!("{p}spot"),
        asset_class: "crypto".into(),
        kind: "crypto_spot".into(),
        name: "BTC spot".into(),
        status: "active".into(),
        issuer_id: Some(ent.entity_id.clone()),
        listings: vec![listing("1", "BTC-USD"), listing("2", "XBTUSD")],
        exposures: vec![RefExposure { to_instrument_id: format!("{p}perp"), relation: "crypto_spot-perp".into() }],
        ..Default::default()
    };
    let perp = RefInstrument {
        instrument_id: format!("{p}perp"),
        asset_class: "crypto".into(),
        kind: "perp".into(),
        name: "BTC perp".into(),
        status: "active".into(),
        listings: vec![listing("2", "PF_XBTUSD")],
        ..Default::default()
    };
    let rep = c
        .sync_reference(std::slice::from_ref(&ent), &[ven("1"), ven("2")], &[spot.clone(), perp.clone()], false)
        .await
        .expect("sync");
    assert_eq!((rep.instruments, rep.listed_on, rep.issued_by, rep.same_exposure), (2, 3, 1, 1));
    // idempotent: a second identical sync leaves exactly the same edges
    c.sync_reference(std::slice::from_ref(&ent), &[ven("1"), ven("2")], &[spot.clone(), perp.clone()], false)
        .await
        .expect("sync again");
    let r = c.instrument_reference(&spot.instrument_id).await.expect("read").expect("present");
    assert_eq!(r.listings.len(), 2, "{r:?}");
    assert_eq!(r.venues.len(), 2);
    assert_eq!(r.issuer.len(), 1);
    assert_eq!(r.exposures.len(), 1);
    assert_eq!(r.exposures[0].properties["relation"], "crypto_spot-perp");

    // a listing closes and the issuer goes away: stale edges are pruned
    spot.listings.truncate(1);
    spot.issuer_id = None;
    spot.name = "BTC spot (renamed)".into();
    c.upsert_ref_instruments(std::slice::from_ref(&spot)).await.expect("incremental");
    let r = c.instrument_reference(&spot.instrument_id).await.expect("read").expect("present");
    assert_eq!((r.listings.len(), r.issuer.len(), r.exposures.len()), (1, 0, 1), "{r:?}");
    assert_eq!(r.listings[0].properties["venue_symbol"], "BTC-USD");
    assert_eq!(r.instrument["name"], "BTC spot (renamed)");
    let named = c
        .run_named("instrumentReference", &serde_json::json!({ "instrument_id": perp.instrument_id }))
        .await
        .expect("named");
    assert_eq!(named["listings"][0]["venue_symbol"], "PF_XBTUSD");

    // retired: inactive, no out-edges, node kept
    c.retire_ref_instruments(std::slice::from_ref(&spot.instrument_id)).await.expect("retire");
    let r = c.instrument_reference(&spot.instrument_id).await.expect("read").expect("present");
    assert_eq!(r.instrument["status"], "inactive");
    assert!(r.listings.is_empty() && r.exposures.is_empty());
    cleanup(&c, &p).await;
}

#[tokio::test]
async fn trained_on_and_feedback_from_edges_are_upserted() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();
    let snap = c
        .ensure_snapshot(&NewSnapshot {
            snapshot_id: format!("{p}snap"),
            timescale: "1m".into(),
            encoder_family: "statistical".into(),
            generation: 1,
            garage_key: String::new(),
            status: "candidate".into(),
        })
        .await
        .expect("snapshot");
    let g = c
        .upsert_genome(&NewGenome {
            genome_id: format!("{p}dec"),
            kind: "decoder".into(),
            family: "nn_forecast".into(),
            params: "{}".into(),
        })
        .await
        .expect("genome");
    let link = TrainedOnLink { genome_id: g.genome_id.clone(), snapshot_id: snap.snapshot_id.clone(), generation: 3 };
    let missing = TrainedOnLink { genome_id: format!("{p}nope"), snapshot_id: snap.snapshot_id.clone(), generation: 3 };
    c.link_trained_on(&[link.clone(), missing]).await.expect("trained_on");
    c.link_trained_on(&[TrainedOnLink { generation: 4, ..link }]).await.expect("trained_on again");
    let snaps = c.genome_trained_on(&g.genome_id).await.expect("read");
    assert_eq!(snaps.len(), 1, "upsert, not duplicate: {snaps:?}");
    assert_eq!(snaps[0].snapshot_id, snap.snapshot_id);

    let fb = FeedbackLink {
        snapshot_id: snap.snapshot_id.clone(),
        genome_id: g.genome_id.clone(),
        weight: 0.25,
        output_family: "forecast".into(),
        score: 0.1,
    };
    c.link_feedback_from(std::slice::from_ref(&fb)).await.expect("feedback");
    c.link_feedback_from(&[FeedbackLink { weight: 0.5, ..fb }]).await.expect("feedback again");
    let (edges, genomes) = c.snapshot_feedback(&snap.snapshot_id).await.expect("read");
    assert_eq!((edges.len(), genomes.len()), (1, 1), "{edges:?}");
    assert_eq!(edges[0].properties["weight"], 0.5);
    assert_eq!(genomes[0].genome_id, g.genome_id);
    cleanup(&c, &p).await;
}

#[tokio::test]
async fn genome_status_mirrors_and_creates_missing_genomes() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();
    let st = |id: &str, status: &str, since: i64| GenomeStatus {
        genome_id: format!("{p}{id}"),
        kind: "strategy".into(),
        family: "gp".into(),
        params: "{ id = \"x\" }".into(),
        strategy_id: format!("{p}{id}"),
        status: status.into(),
        reason: format!("test {status}"),
        since_ms: since,
    };
    // existing genome (written by the evo-worker) gets the status properties
    let g = c
        .upsert_genome(&NewGenome {
            genome_id: format!("{p}a"),
            kind: "strategy".into(),
            family: "gp".into(),
            params: "{}".into(),
        })
        .await
        .expect("upsert");
    assert_eq!(g.status, None);
    let a = c.set_genome_status(&st("a", "candidate", 1)).await.expect("set status");
    assert_eq!((a.id, a.status.as_deref(), a.status_since_ms), (g.id, Some("candidate"), Some(1)));
    assert_eq!(a.params, "{}", "params of an existing genome are kept");
    let a = c.set_genome_status(&st("a", "shadow", 2)).await.expect("transition");
    assert_eq!((a.id, a.status.as_deref(), a.status_reason.as_deref()), (g.id, Some("shadow"), Some("test shadow")));
    // a missing genome is created with the status
    let b = c.set_genome_status(&st("b", "retired", 3)).await.expect("create");
    assert_eq!((b.kind.as_str(), b.status.as_deref()), ("strategy", Some("retired")));
    let read = c.genome_by_id(&format!("{p}b")).await.expect("read").expect("exists");
    assert_eq!(read.status.as_deref(), Some("retired"));
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

#[tokio::test]
async fn adapter_state_round_trip_and_snapshot_manifest() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();
    let snap = c
        .ensure_snapshot(&NewSnapshot {
            snapshot_id: format!("{p}snap"),
            timescale: format!("{p}ts"),
            encoder_family: "statistical".into(),
            generation: 1,
            garage_key: format!("snapshots/{p}snap/manifest.json"),
            status: "candidate".into(),
        })
        .await
        .expect("snapshot");
    assert!(snap.manifest.is_none() && snap.carried_mass.is_none());
    assert!(c.latest_adapter(&snap.snapshot_id).await.expect("none yet").is_none());

    let row = c
        .set_snapshot_manifest(snap.id, r#"{"snapshot_id":"x","metrics":{"carried_mass":0.75}}"#, 0.75)
        .await
        .expect("manifest");
    assert_eq!(row.carried_mass, Some(0.75));
    assert!(row.manifest.as_deref().unwrap_or("").contains("carried_mass"));
    let back = c.snapshot_by_id(&snap.snapshot_id).await.expect("lookup").expect("row");
    assert_eq!(back.carried_mass, Some(0.75));

    let mk = |version: u64, drift: Option<f64>, frozen: bool| NewAdapterState {
        snapshot_id: snap.snapshot_id.clone(),
        timescale: format!("{p}ts"),
        version,
        garage_key: format!("adapters/{p}snap/{version}.json"),
        drift,
        max_drift: 1.0,
        frozen,
        n_windows: 50 + version,
        window_start_ms: 1_700_000_000_000,
        window_end_ms: 1_700_003_600_000,
    };
    let a1 = c.add_adapter_state(snap.id, &mk(1, None, false)).await.expect("adapter 1");
    assert_eq!(a1.adapter_id, format!("{p}snap@1"));
    assert_eq!(a1.drift, None, "no reference statistics → no drift property");
    assert!(!a1.frozen);
    assert_eq!(a1.n_windows, 51);
    assert!(a1.window_start.as_deref().unwrap_or("").starts_with("2023-11-14"));
    let a2 = c.add_adapter_state(snap.id, &mk(2, Some(1.7), true)).await.expect("adapter 2");
    assert_eq!(a2.drift, Some(1.7));
    assert!(a2.frozen);
    // duplicate version → unique violation
    let dup = c.add_adapter_state(snap.id, &mk(2, Some(0.1), false)).await;
    assert!(dup.err().is_some_and(|e| e.is_unique_violation()), "duplicate adapter_id must be refused");

    let latest = c.latest_adapter(&snap.snapshot_id).await.expect("latest").expect("two versions");
    assert_eq!(latest.id, a2.id);
    let all = c.snapshot_adapters(&snap.snapshot_id, 10).await.expect("list");
    assert_eq!(all.iter().map(|a| a.version).collect::<Vec<_>>(), vec![2, 1]);
    // edge Snapshot → AdapterState
    let out: Vec<AdapterState> = {
        let mut resp = c
            .batch(&Batch::read().ret("rows", Op::nodes(snap.id).out(Some("HasAdapter")).value_map(None)))
            .await
            .expect("edge read");
        resp.take("rows").expect("rows")
    };
    assert_eq!(out.len(), 2);
    cleanup(&c, &p).await;
}

fn now_ms() -> i64 {
    i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH).expect("clock").as_millis()).expect("ms")
}

/// A registered snapshot + expert space of `dim` (business ids under `p`).
async fn space_of_dim(c: &HubClient, p: &str, name: &str, dim: u32) -> (Snapshot, ExpertSpace) {
    let snap = c
        .ensure_snapshot(&NewSnapshot {
            snapshot_id: format!("{p}{name}-snap"),
            timescale: "1m".into(),
            encoder_family: "statistical".into(),
            generation: 1,
            garage_key: String::new(),
            status: "candidate".into(),
        })
        .await
        .expect("snapshot");
    let space = c
        .ensure_expert_space(
            snap.id,
            &NewExpertSpace {
                expert_space_id: format!("{p}{name}-e0"),
                snapshot_id: snap.snapshot_id.clone(),
                index: 0,
                dim,
            },
        )
        .await
        .expect("space");
    (snap, space)
}

#[tokio::test]
async fn schema_version_marker_is_stamped_and_never_lowered() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema + migrate");
    let m = c.schema_marker().await.expect("marker").expect("stamped");
    assert_eq!(m.meta_id, fa_hub::meta::META_ID);
    assert_eq!(m.version, SCHEMA_VERSION);
    assert_eq!(m.index_fingerprint, fa_hub::meta::index_fingerprint());
    // Idempotent: a second migrate writes nothing.
    let again = c.migrate_schema().await.expect("migrate again");
    assert_eq!(again, m);
    // Stamping a lower version leaves the marker alone.
    let low = c.stamp_schema_version(0, &m.index_fingerprint).await.expect("stamp low");
    assert_eq!((low.id, low.version), (m.id, SCHEMA_VERSION));
    assert_eq!(c.count_nodes(node::SCHEMA_META).await.expect("count"), 1, "exactly one marker row");
}

#[tokio::test]
async fn knn_validates_dims_filters_and_caches_the_index() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();
    let (_snap, space) = space_of_dim(&c, &p, "knn", 4).await;
    let sid = space.expert_space_id.clone();
    let mk = |space: &str, r: &str, kind: &str, ts_ms: i64, v: Vec<f32>| NewLatent {
        vector: v,
        expert_space_id: space.to_owned(),
        kind: kind.into(),
        ref_id: format!("{p}{r}"),
        instrument_id: if r.ends_with('e') { "ETH".into() } else { "BTC".into() },
        ts_ms,
        uncertainty: 0.1,
    };

    // Validation before any write.
    let e = c.add_latents_checked(&[mk(&sid, "x", "window", 1, vec![1.0, 0.0, 0.0])]).await;
    assert!(matches!(e, Err(HubError::DimMismatch { expected: 4, got: 3, .. })), "{e:?}");
    let e = c.add_latents_checked(&[mk(&format!("{p}unregistered"), "x", "window", 1, vec![1.0; 4])]).await;
    assert!(matches!(e, Err(HubError::NotFound { .. })), "{e:?}");
    let e = c.add_latents_checked(&[mk(&sid, "x", "window", 1, vec![0.0; 4])]).await;
    assert!(matches!(e, Err(HubError::InvalidVector(_))), "{e:?}");
    let e = c.add_latent(&mk(&sid, "x", "window", 1, vec![f32::NAN, 1.0, 0.0, 0.0])).await;
    assert!(matches!(e, Err(HubError::InvalidVector(_))), "{e:?}");

    let ids = c
        .add_latents_checked(&[
            mk(&sid, "w1", "window", 1_000, vec![1.0, 0.0, 0.0, 0.0]),
            mk(&sid, "w2", "window", 2_000, vec![0.95, 0.05, 0.0, 0.0]),
            mk(&sid, "w3e", "window", 3_000, vec![0.9, 0.1, 0.0, 0.0]),
            mk(&sid, "c1", "centroid", 3_000, vec![1.0, 0.01, 0.0, 0.0]),
        ])
        .await
        .expect("add_latents_checked");
    assert_eq!(ids.len(), 4);
    assert!(c.vector_index_cached(node::LATENT_VEC, "vec4"), "index remembered after the checked write");

    let q = [1.0, 0.0, 0.0, 0.0];
    let all = c.knn(&sid, &q, 10, &KnnFilter::default()).await.expect("knn");
    assert_eq!(all.len(), 4);
    assert!(all.windows(2).all(|w| w[0].distance <= w[1].distance));
    let windows = c
        .knn(&sid, &q, 10, &KnnFilter { kind: Some("window".into()), ..KnnFilter::default() })
        .await
        .expect("knn kind");
    assert_eq!(windows.len(), 3);
    let as_of = c
        .knn(&sid, &q, 10, &KnnFilter { kind: Some("window".into()), as_of_ms: Some(2_000), ..KnnFilter::default() })
        .await
        .expect("knn as_of");
    let refs: Vec<&str> = as_of.iter().map(|h| h.ref_id.as_str()).collect();
    assert_eq!(refs, vec![format!("{p}w1"), format!("{p}w2")], "as-of excludes ts > t");
    let f = KnnFilter {
        exclude_ref_ids: vec![format!("{p}w1")],
        instrument_id: Some("BTC".into()),
        ..KnnFilter::default()
    };
    let ex = c.knn(&sid, &q, 1, &f).await.expect("knn exclude");
    assert_eq!(ex.len(), 1);
    assert_ne!(ex[0].ref_id, format!("{p}w1"));
    let near = c.knn(&sid, &q, 10, &KnnFilter { max_distance: Some(1e-6), ..KnnFilter::default() }).await.expect("max");
    assert_eq!(near.len(), 1);
    let e = c.knn(&sid, &[1.0, 0.0], 3, &KnnFilter::default()).await;
    assert!(matches!(e, Err(HubError::DimMismatch { .. })), "{e:?}");

    // A registered dim without any index yet answers empty instead of `index_not_found`.
    let (_s7, space7) = space_of_dim(&c, &p, "knn7", 7).await;
    let none = c.knn(&space7.expert_space_id, &[1.0; 7], 3, &KnnFilter::default()).await.expect("no index yet");
    assert!(none.is_empty());

    // Catalog goes through knn: dim mismatch is a bad parameter, unknown space not found.
    let v = c
        .run_named("searchLatent", &serde_json::json!({ "expert_space_id": sid, "vector": q, "kind": "centroid" }))
        .await
        .expect("named knn");
    assert_eq!(v.as_array().map(Vec::len), Some(1));
    let e = c.run_named("searchLatent", &serde_json::json!({ "expert_space_id": sid, "vector": [1.0] })).await;
    assert!(matches!(e, Err(HubError::BadParam(_))), "{e:?}");
    let e =
        c.run_named("searchLatent", &serde_json::json!({ "expert_space_id": format!("{p}nope"), "vector": q })).await;
    assert!(matches!(e, Err(HubError::NotFound { .. })), "{e:?}");

    cleanup(&c, &p).await;
}

#[tokio::test]
async fn as_of_reads_hide_later_rows() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();
    let before_all = now_ms() - 60_000;
    let (snap, space) = space_of_dim(&c, &p, "asof", 4).await;
    let mk = |n: &str| NewRegion {
        region_id: format!("{p}r{n}"),
        lineage_id: format!("{p}r{n}"),
        expert_space_id: space.expert_space_id.clone(),
        method: "leader".into(),
        radius: 1.0,
        density: 1.0,
    };
    let r1 = c.upsert_region(space.id, &mk("1")).await.expect("r1");
    tokio::time::sleep(Duration::from_millis(50)).await;
    let between = now_ms();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let r2 = c.upsert_region(space.id, &mk("2")).await.expect("r2");
    let row = |from: &LatentRegion, to: &LatentRegion| TransitionUpsert {
        from_region_id: from.region_id.clone(),
        to_region_id: to.region_id.clone(),
        transition: NewTransition {
            sampler_id: "conjugate".into(),
            count: 1,
            mean: 0.5,
            lower: 0.4,
            upper: 0.6,
            kernel_params: "{}".into(),
        },
    };
    c.upsert_transitions(&[row(&r1, &r2), row(&r1, &r1)]).await.expect("transitions");

    let regions = c.space_regions_as_of(space.id, between).await.expect("regions as of");
    assert_eq!(regions.iter().map(|r| r.id).collect::<Vec<_>>(), vec![r1.id]);
    let g = c.snapshot_graph_as_of(&snap.snapshot_id, between).await.expect("graph").expect("snapshot visible");
    // the transitions were first written after `between` (event-time versions, fa-hub `versions`)
    assert_eq!((g.regions.len(), g.transitions.len()), (1, 0), "only r1, no edge yet");
    let g = c.snapshot_graph_as_of(&snap.snapshot_id, now_ms() + 60_000).await.expect("graph").expect("visible");
    assert_eq!((g.regions.len(), g.transitions.len()), (2, 2));
    assert!(c.snapshot_graph_as_of(&snap.snapshot_id, before_all).await.expect("graph").is_none());
    let snaps = c.list_snapshots_as_of(before_all).await.expect("list as of");
    assert!(!snaps.iter().any(|s| s.id == snap.id));
    assert!(c.list_snapshots_as_of(now_ms() + 60_000).await.expect("list").iter().any(|s| s.id == snap.id));

    c.ensure_latent_index_ready(4, WAIT).await.expect("index");
    let l = |r: &str, ts_ms| NewLatent {
        vector: vec![1.0, 0.0, 0.0, 0.0],
        expert_space_id: space.expert_space_id.clone(),
        kind: "observation".into(),
        ref_id: format!("{p}{r}"),
        instrument_id: "BTC".into(),
        ts_ms,
        uncertainty: 0.0,
    };
    c.add_latents_checked(&[l("a", 1_000), l("b", 2_000), l("c", 3_000)]).await.expect("latents");
    let rows = c.space_latents_as_of(&space.expert_space_id, Some("observation"), 2_000, 10).await.expect("as of");
    assert_eq!(rows.iter().map(|r| r.ref_id.clone()).collect::<Vec<_>>(), vec![format!("{p}b"), format!("{p}a")]);

    cleanup(&c, &p).await;
}

#[tokio::test]
async fn prune_folds_old_observations_into_region_aggregates() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();
    let (snap, space) = space_of_dim(&c, &p, "prune", 4).await;
    let sid = space.expert_space_id.clone();
    let l = |r: &str, kind: &str, ts_ms: i64, v: [f32; 4]| NewLatent {
        vector: v.to_vec(),
        expert_space_id: sid.clone(),
        kind: kind.into(),
        ref_id: format!("{p}{r}"),
        instrument_id: "BTC".into(),
        ts_ms,
        uncertainty: 0.2,
    };
    c.add_latents_checked(&[
        l("ra", "observation", 1_000, [1.0, 0.0, 0.0, 0.0]),
        l("ra", "observation", 2_000, [0.0, 1.0, 0.0, 0.0]),
        l("rb", "observation", 3_000, [0.0, 0.0, 1.0, 0.0]),
        l("ra", "observation", 9_000, [1.0, 1.0, 0.0, 0.0]),
        l("ra", "centroid", 500, [1.0, 0.0, 0.0, 0.0]),
    ])
    .await
    .expect("latents");

    let r = c.prune_space_observations(&sid, 5_000).await.expect("prune");
    assert_eq!(r, PruneReport { pruned: 3, aggregates_created: 2, aggregates_updated: 0, versions_pruned: 0 });
    let rows = c.space_latents(&sid, None, None, 50).await.expect("rows");
    let kinds: Vec<(String, String)> =
        rows.iter().map(|r| (r.kind.clone(), r.ref_id.trim_start_matches(&p).to_owned())).collect();
    assert_eq!(rows.len(), 4, "1 recent observation + 1 centroid + 2 aggregates: {kinds:?}");
    let agg_a = rows.iter().find(|r| r.kind == "aggregate" && r.ref_id == format!("{p}ra")).expect("aggregate ra");
    assert_eq!(agg_a.extra.get("obs_count").and_then(serde_json::Value::as_u64), Some(2));
    assert_eq!(agg_a.vector(), Some(vec![0.5, 0.5, 0.0, 0.0]));
    assert_eq!(agg_a.ts.as_deref(), Some("1970-01-01T00:00:02.000Z"), "newest folded observation");

    // Aggregates stay searchable.
    let hits = c
        .knn(&sid, &[1.0, 1.0, 0.0, 0.0], 5, &KnnFilter { kind: Some("aggregate".into()), ..KnnFilter::default() })
        .await
        .expect("knn aggregates");
    assert_eq!(hits.first().map(|h| h.ref_id.clone()), Some(format!("{p}ra")));

    // A later prune merges into the existing aggregate (policy path, per timescale).
    let policy = RetentionPolicy {
        horizons: [("1m".to_owned(), Duration::from_millis(1))].into_iter().collect(),
        default: None,
    };
    let r = c.prune_snapshot_observations(&snap.snapshot_id, 10_001, &policy).await.expect("policy prune");
    assert_eq!(r, PruneReport { pruned: 1, aggregates_created: 0, aggregates_updated: 1, versions_pruned: 0 });
    let aggs = c.space_latents(&sid, Some("aggregate"), None, 10).await.expect("aggs");
    let a = aggs.iter().find(|r| r.ref_id == format!("{p}ra")).expect("ra");
    assert_eq!(a.extra.get("obs_count").and_then(serde_json::Value::as_u64), Some(3));
    let v = a.vector().expect("vector");
    assert!((v[0] - 2.0 / 3.0).abs() < 1e-5 && (v[1] - 2.0 / 3.0).abs() < 1e-5, "{v:?}");
    assert_eq!(c.space_latents(&sid, Some("observation"), None, 10).await.expect("obs").len(), 0);
    assert_eq!(c.prune_space_observations(&sid, 1_000_000).await.expect("noop"), PruneReport::default());

    cleanup(&c, &p).await;
}

#[tokio::test]
async fn news_and_media_embeddings_round_trip() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    let p = prefix();
    let e = |id: &str, ts_ms: i64, v: [f32; 4]| NewEmbedding {
        id: format!("{p}{id}"),
        source: "test-source".into(),
        ts_ms,
        vector: v.to_vec(),
    };
    let ids = c
        .add_news_docs(&[e("d1", 1_000, [1.0, 0.0, 0.0, 0.0]), e("d2", 5_000, [0.9, 0.1, 0.0, 0.0])])
        .await
        .expect("news");
    assert_eq!(ids.len(), 2);
    assert!(c.vector_index_cached(node::NEWS_DOC, "vec4"));
    // The index is shared with other writers; filter our rows.
    let hits = c.search_news_docs(&[1.0, 0.0, 0.0, 0.0], 50, None).await.expect("search");
    let mine: Vec<&EmbeddingHit> = hits.iter().filter(|h| h.id.starts_with(&p)).collect();
    assert_eq!(mine.len(), 2);
    assert_eq!(mine[0].id, format!("{p}d1"));
    assert_eq!(mine[0].source, "test-source");
    let past = c.search_news_docs(&[1.0, 0.0, 0.0, 0.0], 50, Some(2_000)).await.expect("as of");
    assert!(past.iter().filter(|h| h.id.starts_with(&p)).all(|h| h.id == format!("{p}d1")));
    let mixed = c
        .add_news_docs(&[e("d3", 1, [1.0, 0.0, 0.0, 0.0]), NewEmbedding { vector: vec![1.0], ..e("d4", 1, [1.0; 4]) }])
        .await;
    assert!(matches!(mixed, Err(HubError::InvalidVector(_))), "{mixed:?}");

    c.add_media_segments(&[e("s1", 1_000, [0.0, 1.0, 0.0, 0.0])]).await.expect("media");
    let hits = c.search_media_segments(&[0.0, 1.0, 0.0, 0.0], 50, None).await.expect("media search");
    assert!(hits.iter().any(|h| h.id == format!("{p}s1") && h.source == "test-source"));
    // A dimension nobody wrote yet: no index, empty answer.
    assert!(c.search_media_segments(&[1.0; 11], 3, None).await.expect("no index").is_empty());

    cleanup(&c, &p).await;
}

#[tokio::test]
async fn write_buffer_flushes_queued_batches_in_order() {
    let Some(c) = client() else { return };
    c.ensure_schema_ready(WAIT).await.expect("schema");
    c.ensure_latent_index_ready(4, WAIT).await.expect("index");
    let p = prefix();
    let sid = format!("{p}buf");
    let l = |r: &str| NewLatent {
        vector: vec![1.0, 0.0, 0.0, 0.0],
        expert_space_id: sid.clone(),
        kind: "observation".into(),
        ref_id: format!("{p}{r}"),
        instrument_id: "BTC".into(),
        ts_ms: 1,
        uncertainty: 0.0,
    };
    let mut buf = WriteBuffer::new(c.clone(), 2);
    buf.push(fa_hub::vectors::add_latents_batch(&[l("a")]).expect("batch"));
    buf.push(fa_hub::vectors::add_latents_batch(&[l("b")]).expect("batch"));
    assert!(buf.push(fa_hub::vectors::add_latents_batch(&[l("c")]).expect("batch")), "oldest dropped");
    // A batch the server rejects (bad id reference) is discarded, not retried forever.
    buf.push(Batch::write().ret("x", Op::nodes(u64::MAX).set_property("k", 1i64).value_map(None)));
    let r = buf.flush().await;
    assert_eq!(r.remaining, 0, "{r:?}");
    assert_eq!(r.sent + r.rejected, 2);
    assert_eq!(buf.stats().dropped, 2);
    let refs: Vec<String> =
        c.space_latents(&sid, None, None, 10).await.expect("rows").into_iter().map(|r| r.ref_id).collect();
    assert!(refs.contains(&format!("{p}c")) && !refs.contains(&format!("{p}a")), "{refs:?}");
    assert!(buf.send_or_queue(fa_hub::vectors::add_latents_batch(&[l("d")]).expect("batch")).await.expect("sent"));
    cleanup(&c, &p).await;
}
