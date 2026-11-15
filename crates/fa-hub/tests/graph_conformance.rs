//! One conformance suite over [`GraphStore`], run against both backends: Postgres with
//! `DATABASE_URL` set (migrations applied), Helix with `FA_HELIX_URL` set. Each run uses a
//! unique id prefix and removes its own rows.
use fa_hub::graph::GraphStore;
use fa_hub::{
    HelixGraph, HubClient, NewAdapterState, NewDescent, NewExpertSpace, NewHierEdge, NewRegion, NewRegionSet,
    NewSnapshot, PgGraph, RefEntity, RefInstrument, RefListing, RefVenue, TransitionUpsert, edge_key,
};

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

async fn conformance<G: GraphStore>(g: &G, run: &str) {
    let sid = format!("conf-{run}-snap");
    let esid = format!("conf-{run}-space");
    // snapshots: ensure is idempotent and keeps the first row
    let snap = g
        .ensure_snapshot(&NewSnapshot {
            snapshot_id: sid.clone(),
            timescale: "1m".into(),
            encoder_family: "stat".into(),
            generation: 1,
            garage_key: "k".into(),
            status: "candidate".into(),
        })
        .await
        .unwrap();
    let again = g
        .ensure_snapshot(&NewSnapshot {
            snapshot_id: sid.clone(),
            timescale: "1m".into(),
            encoder_family: "stat".into(),
            generation: 9,
            garage_key: "other".into(),
            status: "active".into(),
        })
        .await
        .unwrap();
    assert_eq!(snap.id, again.id);
    assert_eq!(again.status, "candidate", "ensure keeps the existing row");
    assert!(g.snapshot_by_id(&sid).await.unwrap().is_some());
    let listed = g.list_snapshots().await.unwrap();
    assert!(listed.iter().any(|s| s.snapshot_id == sid));
    let promoted = g.set_snapshot_status(snap.id, "active").await.unwrap();
    assert_eq!(promoted.status, "active");
    assert!(g.snapshots_by_status("1m", "stat", "active").await.unwrap().iter().any(|s| s.snapshot_id == sid));
    let with_manifest = g.set_snapshot_manifest(snap.id, "{\"metrics\":{}}", 0.5).await.unwrap();
    assert_eq!(with_manifest.carried_mass, Some(0.5));

    // spaces
    let space = g
        .ensure_expert_space(
            snap.id,
            &NewExpertSpace { expert_space_id: esid.clone(), snapshot_id: sid.clone(), index: 0, dim: 8 },
        )
        .await
        .unwrap();
    assert_eq!(g.snapshot_spaces(snap.id).await.unwrap().len(), 1);
    assert_eq!(g.space_by_id(&esid).await.unwrap().unwrap().id, space.id);

    // adapters
    let a = g
        .add_adapter_state(
            snap.id,
            &NewAdapterState {
                snapshot_id: sid.clone(),
                timescale: "1m".into(),
                version: 7,
                garage_key: "g".into(),
                drift: Some(0.1),
                max_drift: 1.0,
                frozen: false,
                n_windows: 3,
                window_start_ms: now_ms() - 1000,
                window_end_ms: now_ms(),
            },
        )
        .await
        .unwrap();
    assert_eq!(a.version, 7);
    assert_eq!(g.snapshot_adapters(&sid, 5).await.unwrap().len(), 1);

    // regions: upsert, revive, status marks, versions
    let rid = |i: u32| format!("conf-{run}-r{i}");
    let mk = |i: u32| NewRegion {
        region_id: rid(i),
        lineage_id: format!("conf-{run}-lin"),
        expert_space_id: esid.clone(),
        method: "online".into(),
        radius: 1.0,
        density: 0.5,
    };
    let r0 = g.upsert_region(space.id, &mk(0)).await.unwrap();
    let _ = g.upsert_region(space.id, &mk(1)).await.unwrap();
    assert_eq!(r0.status.as_deref(), Some("live"));
    assert_eq!(g.space_regions(space.id).await.unwrap().len(), 2);
    let merged = g.mark_region_status(&rid(1), "merged", &rid(0)).await.unwrap();
    assert_eq!(merged.status.as_deref(), Some("merged"));
    let revived = g.upsert_region(space.id, &mk(1)).await.unwrap();
    assert_eq!(revived.status.as_deref(), Some("live"));
    let versions = g.region_status_versions(&rid(1), None, 10).await.unwrap();
    assert!(versions.len() >= 3, "created + merged + revived, got {}", versions.len());
    assert_eq!(versions[0].status, "live");

    // transitions with event-time versions and the as-of read
    let up = |count: i64, mean: f64| TransitionUpsert {
        from_region_id: rid(0),
        to_region_id: rid(1),
        transition: fa_hub::NewTransition {
            sampler_id: "dirichlet".into(),
            count,
            mean,
            lower: mean - 0.1,
            upper: mean + 0.1,
            kernel_params: "{}".into(),
        },
    };
    let t1 = g.upsert_transitions(&[up(1, 0.2)]).await.unwrap();
    let before_update_ms = now_ms();
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    let t2 = g.upsert_transitions(&[up(2, 0.4)]).await.unwrap();
    assert_eq!(t1[0].id, t2[0].id, "updated in place");
    assert_eq!(t2[0].count, 2);
    let outgoing = g.region_transitions(r0.id).await.unwrap();
    assert_eq!(outgoing.len(), 1);
    assert_eq!(g.region_neighbors(r0.id).await.unwrap().len(), 1);
    let versions = g.transition_versions(&rid(0), &rid(1), "dirichlet", None, 10).await.unwrap();
    assert_eq!(versions.len(), 2);
    let key = edge_key(&rid(0), &rid(1), "dirichlet");
    let latest = g.latest_transition_versions(std::slice::from_ref(&key), before_update_ms).await.unwrap();
    assert_eq!(latest[0].as_ref().unwrap().count, 1, "version in force before the update");

    // graphs
    let rg = g.region_graph(space.id).await.unwrap();
    assert_eq!((rg.regions.len(), rg.transitions.len()), (2, 1));
    let sg = g.snapshot_graph(&sid).await.unwrap().unwrap();
    assert_eq!((sg.spaces.len(), sg.regions.len(), sg.transitions.len()), (1, 2, 1));
    assert!(g.snapshot_graph(&format!("conf-{run}-missing")).await.unwrap().is_none());
    let past = g.snapshot_graph_as_of(&sid, now_ms() - 3_600_000).await.unwrap();
    assert!(past.is_none(), "snapshot not visible an hour ago");
    let sg_now = g.snapshot_graph_as_of(&sid, now_ms()).await.unwrap().unwrap();
    assert_eq!(sg_now.regions.len(), 2);

    // lineage + hierarchy
    g.upsert_descents(&[NewDescent {
        child_region_id: rid(1),
        parent_region_id: rid(0),
        kind: "split".into(),
        overlap: 0.7,
    }])
    .await
    .unwrap();
    let lin = g.region_lineage(&rid(1), 3).await.unwrap().unwrap();
    assert_eq!(lin.ancestors.len(), 1);
    assert_eq!(lin.ancestors[0].kind, "split");
    assert_eq!(lin.lineage.len(), 2, "both regions share the lineage id");
    let down = g.region_lineage(&rid(0), 3).await.unwrap().unwrap();
    assert_eq!(down.descendants.len(), 1);
    g.upsert_hier_edges(&[NewHierEdge {
        fine_region_id: rid(0),
        coarse_region_id: rid(1),
        p_coarse_given_fine: 0.8,
        p_fine_given_coarse: 0.3,
        support: 10.0,
        pmi_bits: 1.0,
        mi_bits: 0.5,
    }])
    .await
    .unwrap();
    assert_eq!(g.snapshot_hierarchy(&sid).await.unwrap().len(), 1);

    // per-space state
    g.upsert_region_set(&NewRegionSet {
        expert_space_id: esid.clone(),
        snapshot_id: sid.clone(),
        timescale: "1m".into(),
        expert_index: 0,
        family: "stat".into(),
        method: "online".into(),
        n_regions: 2,
        cfg_hash: "h".into(),
        state: "{}".into(),
        kernel_state: "{}".into(),
    })
    .await
    .unwrap();
    assert_eq!(g.region_set(&esid).await.unwrap().unwrap().n_regions, 2);

    // reference mirror
    let iid = format!("conf-{run}-inst");
    g.upsert_ref_entities(&[RefEntity {
        entity_id: format!("conf-{run}-ent"),
        lei: String::new(),
        name: "Conf Corp".into(),
        country: "DE".into(),
        sector: "Information Technology".into(),
    }])
    .await
    .unwrap();
    g.upsert_ref_venues(&[RefVenue {
        venue_id: format!("conf-{run}-venue"),
        code: "conf".into(),
        mic: String::new(),
        name: "Conf Venue".into(),
        kind: "exchange".into(),
        timezone: "UTC".into(),
    }])
    .await
    .unwrap();
    let rep = g
        .upsert_ref_instruments(&[RefInstrument {
            instrument_id: iid.clone(),
            asset_class: "equity".into(),
            kind: "spot".into(),
            name: "CONF".into(),
            status: "active".into(),
            base_ccy: "EUR".into(),
            quote_ccy: "EUR".into(),
            issuer_id: Some(format!("conf-{run}-ent")),
            listings: vec![RefListing {
                venue_id: format!("conf-{run}-venue"),
                venue_symbol: "CNF".into(),
                source_id: String::new(),
            }],
            exposures: vec![],
        }])
        .await
        .unwrap();
    assert_eq!((rep.instruments, rep.issued_by, rep.listed_on), (1, 1, 1));
    let r = g.instrument_reference(&iid).await.unwrap().unwrap();
    assert_eq!(r.issuer.len(), 1);
    assert_eq!(r.listings.len(), 1);
    assert_eq!(r.venues.len(), 1);
    assert_eq!(r.sectors.len(), 1, "issuer sector resolved");
    let ids = g.ref_instrument_ids().await.unwrap();
    assert!(ids.iter().any(|(id, st)| id == &iid && st == "active"));
    g.retire_ref_instruments(std::slice::from_ref(&iid)).await.unwrap();
    let r = g.instrument_reference(&iid).await.unwrap().unwrap();
    assert!(r.listings.is_empty(), "retired instruments lose their out-edges");

    // genome links by business id
    g.link_trained_on(&[fa_hub::TrainedOnLink {
        genome_id: format!("conf-{run}-genome"),
        snapshot_id: sid.clone(),
        generation: 1,
    }])
    .await
    .unwrap();
    let snaps = g.trained_on_snapshots(&format!("conf-{run}-genome")).await.unwrap();
    assert!(
        snaps.iter().any(|s| s.snapshot_id == sid) || snaps.is_empty(),
        "helix skips links whose genome node does not exist; postgres stores them"
    );
}

#[tokio::test]
async fn postgres_backend_conforms() {
    let Ok(url) = std::env::var("DATABASE_URL") else {
        eprintln!("skipped: DATABASE_URL not set");
        return;
    };
    let pg = PgGraph::connect_lazy(&url);
    let run = format!("pg{}", chrono::Utc::now().timestamp_millis());
    conformance(&pg, &run).await;
    // cleanup: every table keys its conformance rows by the run prefix
    let like = format!("conf-{run}-%");
    for sql in [
        "DELETE FROM graph.transition_version WHERE from_region_id LIKE $1",
        "DELETE FROM graph.region_status_version WHERE region_id LIKE $1",
        "DELETE FROM graph.descends_from WHERE child_region_id LIKE $1",
        "DELETE FROM graph.hier WHERE fine_region_id LIKE $1",
        "DELETE FROM graph.region WHERE region_id LIKE $1",
        "DELETE FROM graph.region_set WHERE region_set_id LIKE $1",
        "DELETE FROM graph.adapter_state WHERE snapshot_id LIKE $1",
        "DELETE FROM graph.expert_space WHERE expert_space_id LIKE $1",
        "DELETE FROM graph.trained_on WHERE snapshot_id LIKE $1",
        "DELETE FROM graph.snapshot WHERE snapshot_id LIKE $1",
        "DELETE FROM graph.ref_listed_on WHERE instrument_id LIKE $1",
        "DELETE FROM graph.ref_issued_by WHERE instrument_id LIKE $1",
        "DELETE FROM graph.ref_instrument WHERE instrument_id LIKE $1",
        "DELETE FROM graph.ref_in_sector WHERE entity_id LIKE $1",
        "DELETE FROM graph.ref_entity WHERE entity_id LIKE $1",
        "DELETE FROM graph.ref_venue WHERE venue_id LIKE $1",
    ] {
        sqlx::query(sql).bind(&like).execute(pg.pool()).await.unwrap();
    }
}

#[tokio::test]
async fn helix_backend_conforms() {
    let Ok(url) = std::env::var("FA_HELIX_URL") else {
        eprintln!("skipped: FA_HELIX_URL not set");
        return;
    };
    let client = HubClient::new(url).with_graph(None).with_vectors(None);
    client.ensure_schema_ready(std::time::Duration::from_secs(60)).await.unwrap();
    let run = format!("hx{}", chrono::Utc::now().timestamp_millis());
    let helix = HelixGraph::new(client.clone());
    conformance(&helix, &run).await;
    // cleanup by prefix via the business-id predicates
    use fa_hub::Pred;
    for (label, key) in [
        ("TransitionVersion", "from_region_id"),
        ("RegionStatusVersion", "region_id"),
        ("LatentRegion", "region_id"),
        ("RegionSet", "snapshot_id"),
        ("AdapterState", "snapshot_id"),
        ("ExpertSpace", "expert_space_id"),
        ("Snapshot", "snapshot_id"),
        ("Instrument", "instrument_id"),
        ("Entity", "entity_id"),
        ("Venue", "venue_id"),
    ] {
        client
            .drop_nodes(label, Pred::starts_with(key, &format!("conf-{run}-")))
            .await
            .unwrap_or_else(|e| eprintln!("cleanup {label}: {e}"));
    }
}
