//! Live transport tests for the Dgraph backend (ADR 0011, D-110). Skipped unless `FA_DGRAPH_URL`
//! points at an alpha, like the other `*_live.rs` suites:
//!
//! ```sh
//! FA_DGRAPH_URL=http://127.0.0.1:18180 just cargo test -p fa-hub --test dgraph_live
//! ```
//!
//! These cover the four primitives `GraphStore` is built on — schema alter, set-and-read, the
//! upsert block that replaces `ON CONFLICT`, and delete — not the trait itself.
use std::collections::BTreeMap;

use fa_hub::dgraph::{DgraphClient, uid_to_u64};
use serde_json::{Value as Json, json};

fn client() -> Option<DgraphClient> {
    let url = std::env::var("FA_DGRAPH_URL").ok().filter(|u| !u.is_empty())?;
    Some(DgraphClient::new(&url))
}

/// A unique business-id prefix per run, so repeated runs against a live alpha never collide.
fn run_id() -> String {
    format!("dgl-{}", chrono::Utc::now().timestamp_millis())
}

const SCHEMA: &str = r#"
    test_key: string @index(exact) @upsert .
    test_n: int .
    test_label: string .
    type TestNode {
        test_key
        test_n
        test_label
    }
"#;

#[tokio::test]
async fn transport_round_trip() {
    let Some(dg) = client() else {
        println!("FA_DGRAPH_URL not set; skipping");
        return;
    };
    assert!(dg.healthy().await, "alpha not healthy at {}", dg.url());
    dg.alter(SCHEMA).await.expect("alter");

    let run = run_id();
    let key = format!("{run}-a");

    // set → uid assigned and returned
    let uids = dg
        .mutate(vec![json!({"uid": "_:a", "dgraph.type": "TestNode", "test_key": key, "test_n": 1})])
        .await
        .expect("mutate");
    let uid = uids.get("a").expect("blank node uid").clone();
    assert!(uid_to_u64(&uid).is_some_and(|n| n > 0), "uid `{uid}` is not a u64");

    // read it back by business key, with variables
    let vars = BTreeMap::from([("$k".to_owned(), key.clone())]);
    let data = dg
        .query_vars("query q($k: string) { q(func: eq(test_key, $k)) { uid test_key test_n } }", &vars)
        .await
        .expect("query_vars");
    let rows = data["q"].as_array().expect("q array");
    assert_eq!(rows.len(), 1, "expected exactly one node, got {rows:?}");
    assert_eq!(rows[0]["test_n"], json!(1));

    // upsert block: same key updates in place rather than inserting a second node
    let q = format!("{{ v as var(func: eq(test_key, \"{key}\")) }}");
    dg.upsert(&q, vec![json!({"uid": "uid(v)", "test_n": 2, "test_label": "updated"})], vec![])
        .await
        .expect("upsert update");
    let data = dg.query(&format!("{{ q(func: eq(test_key, \"{key}\")) {{ uid test_n test_label }} }}")).await.unwrap();
    let rows = data["q"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "upsert must not create a duplicate: {rows:?}");
    assert_eq!(rows[0]["test_n"], json!(2));
    assert_eq!(rows[0]["test_label"], json!("updated"));

    // upsert block on a missing key inserts
    let key_b = format!("{run}-b");
    let q = format!("{{ v as var(func: eq(test_key, \"{key_b}\")) }}");
    dg.upsert(&q, vec![json!({"uid": "uid(v)", "dgraph.type": "TestNode", "test_key": key_b, "test_n": 7})], vec![])
        .await
        .expect("upsert insert");
    let data = dg.query(&format!("{{ q(func: eq(test_key, \"{key_b}\")) {{ test_n }} }}")).await.unwrap();
    assert_eq!(data["q"].as_array().unwrap().len(), 1);

    // cleanup: delete both nodes of this run. `eq` takes a value list on an exact index; prefix
    // matching would need a trigram index, which matters for the conformance suite's
    // `LIKE 'prefix%'` cleanup — see dgraph_schema.rs.
    let both = format!("[\"{key}\", \"{key_b}\"]");
    let data = dg.query(&format!("{{ q(func: eq(test_key, {both})) {{ uid }} }}")).await.unwrap();
    let del: Vec<Json> = data["q"].as_array().unwrap().iter().map(|r| json!({"uid": r["uid"]})).collect();
    let n = del.len();
    dg.delete(del).await.expect("delete");
    assert_eq!(n, 2, "expected to clean up 2 nodes");
    let data = dg.query(&format!("{{ q(func: eq(test_key, {both})) {{ uid }} }}")).await.unwrap();
    assert!(data["q"].as_array().unwrap().is_empty(), "nodes survived delete");
}

#[tokio::test]
async fn query_errors_become_server_errors() {
    let Some(dg) = client() else {
        println!("FA_DGRAPH_URL not set; skipping");
        return;
    };
    // `errors` arrives with HTTP 200 from Dgraph, so the client must surface it rather than
    // returning an empty result set.
    let err = dg.query("{ this is not dql }").await.expect_err("malformed DQL must error");
    match err {
        fa_hub::HubError::Server { status, .. } => assert!((400..500).contains(&status), "status {status}"),
        other => panic!("expected Server, got {other}"),
    }
}

/// The real schema must apply cleanly, and the prefix trick the conformance cleanup depends on
/// must actually work on an `exact` index.
#[tokio::test]
async fn platform_schema_applies() {
    let Some(dg) = client() else {
        println!("FA_DGRAPH_URL not set; skipping");
        return;
    };
    dg.alter(fa_hub::dgraph_schema::SCHEMA).await.expect("alter platform schema");

    let run = run_id();
    let a = format!("{run}-r1");
    let b = format!("{run}-r2");
    dg.mutate(vec![
        json!({"uid": "_:a", "dgraph.type": "LatentRegion", "region.region_id": a, "region.status": "live"}),
        json!({"uid": "_:b", "dgraph.type": "LatentRegion", "region.region_id": b, "region.status": "retired"}),
    ])
    .await
    .expect("mutate regions");

    // prefix_filter stands in for the suite's `LIKE 'prefix%'` cleanup
    let f = fa_hub::dgraph_schema::prefix_filter("region.region_id", &run);
    let q = format!("{{ q(func: type(LatentRegion)) @filter({f}) {{ uid region.region_id region.status }} }}");
    let data = dg.query(&q).await.expect("prefix query");
    let rows = data["q"].as_array().expect("rows");
    assert_eq!(rows.len(), 2, "prefix filter should match both regions of this run: {rows:?}");

    let del: Vec<Json> = rows.iter().map(|r| json!({"uid": r["uid"]})).collect();
    dg.delete(del).await.expect("cleanup");
    let left = dg.query(&q).await.unwrap();
    assert!(left["q"].as_array().unwrap().is_empty());
}

/// Genomes, fitness, lineage, Pareto fronts, evaluations, experts, the schema marker and the
/// graph explorer — the surface that was Helix-only before D-110 phase 3.
#[tokio::test]
async fn evolution_and_experts_round_trip() {
    let Some(dg) = client() else {
        println!("FA_DGRAPH_URL not set; skipping");
        return;
    };
    let g = fa_hub::DgraphGraph::new(dg);
    g.ensure_schema().await.expect("schema");
    let run = run_id();
    let gid = |n: u32| format!("{run}-g{n}");

    // genomes: add rejects duplicates, upsert does not overwrite
    let mk = |n: u32| fa_hub::NewGenome {
        genome_id: gid(n),
        kind: "strategy".into(),
        family: "momentum".into(),
        params: format!("{{\"n\":{n}}}"),
    };
    let g0 = g.dg_add_genome(&mk(0)).await.expect("add_genome");
    assert_eq!(g0.genome_id, gid(0));
    assert!(g.dg_add_genome(&mk(0)).await.is_err(), "duplicate genome_id must be rejected");
    let same = g.dg_upsert_genome(&fa_hub::NewGenome { params: "{\"n\":999}".into(), ..mk(0) }).await.unwrap();
    assert_eq!(same.params, "{\"n\":0}", "upsert must not overwrite an existing genome");
    let g1 = g.dg_add_genome(&mk(1)).await.unwrap();
    let g2 = g.dg_add_genome(&mk(2)).await.unwrap();

    // status mirrors the strategy lifecycle, creating the genome when absent
    let st = g
        .dg_set_genome_status(&fa_hub::GenomeStatus {
            genome_id: gid(9),
            kind: "strategy".into(),
            family: "momentum".into(),
            params: "{}".into(),
            strategy_id: format!("{run}-strat"),
            status: "live".into(),
            reason: "promoted".into(),
            since_ms: 1_763_000_000_000,
        })
        .await
        .expect("set_genome_status");
    assert_eq!(st.status.as_deref(), Some("live"));
    assert_eq!(st.strategy_id.as_deref(), Some(format!("{run}-strat").as_str()));

    // lineage: g2 → g1 → g0
    g.dg_link_parent(&gid(1), &gid(2), "mutate").await.expect("link_parent");
    g.dg_add_lineage(g0.id, g1.id, "crossover").await.expect("add_lineage");
    let parents = g.dg_genome_parents(g2.id).await.unwrap();
    assert_eq!(parents.len(), 1);
    assert_eq!(parents[0].genome_id, gid(1));
    let ancestry = g.dg_genome_ancestry(g2.id, 3).await.unwrap();
    assert_eq!(ancestry.len(), 2, "two generations of ancestors: {ancestry:?}");
    assert_eq!(ancestry[1][0].genome_id, gid(0));

    // fitness
    let f = g
        .dg_add_fitness_record(&fa_hub::NewFitness {
            genome_id: gid(0),
            generation: 3,
            objectives: "{\"sharpe\":1.2}".into(),
            pareto_rank: 1,
        })
        .await
        .expect("fitness");
    assert_eq!(f.generation, 3);
    assert_eq!(g.dg_genome_fitness(g0.id).await.unwrap().len(), 1);
    let detail = g.dg_genome_detail(&gid(2)).await.unwrap().expect("detail");
    assert_eq!(detail.parents.len(), 1);

    // pareto front + evaluation
    let front = fa_hub::evolution::NewParetoFront {
        generation_id: format!("{run}-gen"),
        gen_index: 4,
        kind: "decoder".into(),
        snapshot_id: format!("{run}-snap"),
        keys: vec!["sharpe".into(), "dd".into()],
        members: vec![fa_hub::evolution::FrontMember { genome_id: gid(0), rank: 1, hv: 0.8, crowding: 0.2, score: 0.9 }],
        hypervolume: 0.8,
        final_loop: true,
    };
    assert!(g.dg_upsert_pareto_front(&front).await.unwrap() > 0);
    g.dg_upsert_pareto_front(&front).await.unwrap();
    let fronts = g.dg_pareto_fronts(&format!("{run}-gen")).await.unwrap();
    assert_eq!(fronts.len(), 1, "front is keyed by generation+kind, so a re-run updates");
    assert!(fronts[0].members.contains(&gid(0)), "members JSON kept: {}", fronts[0].members);

    g.dg_record_evaluation(&fa_hub::evolution::Evaluation {
        genome_id: gid(0),
        snapshot_id: format!("{run}-snap"),
        generation: format!("{run}-gen"),
        gen_index: 4,
        rung: 1,
        score: 0.7,
        objectives: "{}".into(),
        job_id: format!("{run}-job"),
        experts: Some(vec![0, 2]),
    })
    .await
    .expect("record_evaluation");
    // empty ids are ignored rather than erroring, as in the Helix path
    g.dg_record_evaluation(&fa_hub::evolution::Evaluation {
        genome_id: String::new(),
        snapshot_id: String::new(),
        generation: String::new(),
        gen_index: 0,
        rung: 0,
        score: 0.0,
        objectives: String::new(),
        job_id: String::new(),
        experts: None,
    })
    .await
    .expect("empty evaluation is a no-op");

    // experts: the name is derived, alignment requires both to exist
    let snap = format!("{run}-snap");
    for e in 0..2u32 {
        g.dg_upsert_expert(&fa_hub::experts::NewExpert {
            snapshot_id: snap.clone(),
            timescale: "1m".into(),
            expert_id: e,
            dim: 8,
            usage: 0.5,
            skill: 0.6,
            born_gen: 1,
            specialisation: "{}".into(),
        })
        .await
        .expect("upsert_expert");
    }
    let experts = g.dg_snapshot_experts(&snap).await.unwrap();
    assert_eq!(experts.len(), 2);
    assert_eq!(experts[0].name, fa_hub::experts::expert_name(&snap, 0), "name derived from expert_uid");
    assert_ne!(experts[0].name, experts[1].name, "adjacent experts get different names");
    assert!(experts[0].name.contains('-'), "attribute-animal shape: {}", experts[0].name);

    let align = fa_hub::experts::NewAlignment {
        child_snapshot_id: snap.clone(),
        child_expert_id: 1,
        parent_snapshot_id: snap.clone(),
        parent_expert_id: 0,
        cka: 0.9,
        kind: "carry".into(),
    };
    g.dg_align_expert(&align).await.expect("align");
    assert_eq!(g.dg_expert_parents(&snap, 1).await.unwrap().len(), 1);
    assert_eq!(g.dg_expert_alignments(&snap, 1).await.unwrap()[0].kind, "carry");
    assert!(
        g.dg_align_expert(&fa_hub::experts::NewAlignment { parent_expert_id: 99, ..align }).await.is_err(),
        "aligning to a missing expert must error"
    );
    g.dg_retire_expert(&snap, 1, 7).await.expect("retire");
    let retired = g.dg_snapshot_experts(&snap).await.unwrap();
    assert_eq!(retired.iter().find(|e| e.expert_id == 1).unwrap().died_gen, Some(7));

    // schema marker: monotonic, and a newer store is refused
    g.dg_stamp_schema(3).await.expect("stamp");
    assert_eq!(g.dg_schema_version().await.unwrap(), Some(3));
    g.dg_stamp_schema(3).await.expect("re-stamping the same version is a no-op");
    assert!(matches!(g.dg_stamp_schema(2).await, Err(fa_hub::HubError::SchemaTooNew { stored: 3, supported: 2 })));

    // explorer: resolve by business key and by node id, then expand
    let resolved = g.dg_resolve_node(&gid(0)).await.unwrap().expect("resolve by key");
    assert_eq!(resolved.label, "Genome");
    assert_eq!(resolved.properties.get("genome_id").and_then(|v| v.as_str()), Some(gid(0).as_str()));
    let by_id = g.dg_resolve_node(&g0.id.to_string()).await.unwrap().expect("resolve by uid");
    assert_eq!(by_id.id, g0.id);
    assert!(g.dg_resolve_node(&format!("{run}-nope")).await.unwrap().is_none());
    let hood = g.dg_neighborhood(g0.id, 1).await.unwrap();
    assert_eq!(hood.nodes.first().map(|n| n.id), Some(g0.id), "the start node comes first");
    // reified edges come back as edges, not nodes: fitness is an entity, the parent link is an edge
    assert!(hood.nodes.iter().any(|n| n.label == "FitnessRecord"), "nodes: {:?}", hood.nodes.len());
    assert!(
        hood.edges.iter().any(|e| e.label == "GenomeParent"),
        "the parent link must be an edge, got {:?}",
        hood.edges.iter().map(|e| &e.label).collect::<Vec<_>>()
    );
    assert!(hood.edges.iter().any(|e| e.label == "EvaluatedOn"));

    // cleanup
    for (ty, pred) in [
        ("FitnessRecord", "fitness_record.genome_id"),
        ("GenomeParent", "genome_parent.child_genome_id"),
        ("EvaluatedOn", "evaluated_on.genome_id"),
        ("ParetoFront", "pareto_front.generation_id"),
        ("ExpertAlign", "expert_align.child_expert_uid"),
        ("Expert", "expert.snapshot_id"),
        ("Genome", "genome.genome_id"),
    ] {
        let f = fa_hub::dgraph_schema::prefix_filter(pred, &run);
        let q = format!("{{ q(func: type({ty})) @filter({f}) {{ uid }} }}");
        let data = g.client().query(&q).await.unwrap();
        let del: Vec<Json> =
            data["q"].as_array().map(|a| a.iter().map(|r| json!({"uid": r["uid"]})).collect()).unwrap_or_default();
        g.client().delete(del).await.unwrap();
    }
}

/// The D-110 cutover: `HubClient` must serve the whole surface from Dgraph.
///
/// The client's own base URL points at a dead port, so anything that still fell through to the
/// Helix query layer would fail rather than quietly working — which is the only way to be sure the
/// routing is complete and not merely present.
#[tokio::test]
async fn hub_client_routes_everything_to_dgraph() {
    let Some(dg) = client() else {
        println!("FA_DGRAPH_URL not set; skipping");
        return;
    };
    let graph = fa_hub::DgraphGraph::new(dg);
    graph.ensure_schema().await.expect("schema");
    let hub = fa_hub::HubClient::new("http://127.0.0.1:1").with_graph(Some(graph.clone())).with_vectors(None);
    assert_eq!(hub.graph_backend(), "dgraph");

    let run = run_id();
    let sid = format!("{run}-snap");
    let gid = format!("{run}-genome");

    // schema readiness goes through /alter + the marker, with no index receipts to wait on
    hub.ensure_schema_ready(std::time::Duration::from_secs(10)).await.expect("ensure_schema_ready");

    // model-state graph (GraphStore surface)
    let snap = hub
        .ensure_snapshot(&fa_hub::NewSnapshot {
            snapshot_id: sid.clone(),
            timescale: "1m".into(),
            encoder_family: "stat".into(),
            generation: 1,
            garage_key: String::new(),
            status: "candidate".into(),
        })
        .await
        .expect("ensure_snapshot via hub");
    assert!(hub.snapshot_by_id(&sid).await.unwrap().is_some());

    // genomes + fitness + lineage (was Helix-only)
    let g = hub
        .upsert_genome(&fa_hub::NewGenome {
            genome_id: gid.clone(),
            kind: "strategy".into(),
            family: "momentum".into(),
            params: "{}".into(),
        })
        .await
        .expect("upsert_genome via hub");
    hub.add_fitness_record(&fa_hub::NewFitness {
        genome_id: gid.clone(),
        generation: 1,
        objectives: "{}".into(),
        pareto_rank: 1,
    })
    .await
    .expect("fitness via hub");
    assert_eq!(hub.genome_fitness(g.id).await.unwrap().len(), 1);
    assert!(hub.genome_detail(&gid).await.unwrap().is_some());

    // experts (was Helix-only), including the D-111 derived name
    hub.upsert_expert(&fa_hub::experts::NewExpert {
        snapshot_id: sid.clone(),
        timescale: "1m".into(),
        expert_id: 0,
        dim: 8,
        usage: 0.1,
        skill: 0.2,
        born_gen: 1,
        specialisation: "{}".into(),
    })
    .await
    .expect("upsert_expert via hub");
    let experts = hub.snapshot_experts(&sid).await.expect("snapshot_experts via hub");
    assert_eq!(experts.len(), 1);
    assert_eq!(experts[0].name, fa_hub::experts::expert_name(&sid, 0));

    // evolution (was Helix-only)
    hub.record_evaluation(&fa_hub::evolution::Evaluation {
        genome_id: gid.clone(),
        snapshot_id: sid.clone(),
        generation: format!("{run}-gen"),
        gen_index: 1,
        rung: 0,
        score: 0.5,
        objectives: "{}".into(),
        job_id: String::new(),
        experts: None,
    })
    .await
    .expect("record_evaluation via hub");
    hub.upsert_pareto_front(&fa_hub::evolution::NewParetoFront {
        generation_id: format!("{run}-gen"),
        gen_index: 1,
        kind: "decoder".into(),
        snapshot_id: sid.clone(),
        keys: vec!["sharpe".into()],
        members: vec![],
        hypervolume: 0.1,
        final_loop: false,
    })
    .await
    .expect("upsert_pareto_front via hub");
    assert_eq!(hub.pareto_fronts(&format!("{run}-gen")).await.unwrap().len(), 1);

    // explorer (was Helix-only, and label-free there)
    let node = hub.resolve_node(&gid).await.expect("resolve_node via hub").expect("found");
    assert_eq!(node.label, "Genome");
    let hood = hub.neighborhood(g.id, 1).await.expect("neighborhood via hub");
    assert!(!hood.nodes.is_empty());

    // the named-query catalogue the UI drives
    let listed = hub.run_named("listSnapshots", &json!({})).await.expect("run_named via hub");
    assert!(listed.to_string().contains(&sid), "listSnapshots must see the snapshot");

    let _ = snap;
    // cleanup
    for (ty, pred) in [
        ("FitnessRecord", "fitness_record.genome_id"),
        ("EvaluatedOn", "evaluated_on.genome_id"),
        ("ParetoFront", "pareto_front.generation_id"),
        ("Expert", "expert.snapshot_id"),
        ("Genome", "genome.genome_id"),
        ("RegionStatusVersion", "region_status_version.region_id"),
        ("Snapshot", "snapshot.snapshot_id"),
    ] {
        let f = fa_hub::dgraph_schema::prefix_filter(pred, &run);
        let q = format!("{{ q(func: type({ty})) @filter({f}) {{ uid }} }}");
        let data = graph.client().query(&q).await.unwrap();
        let del: Vec<Json> =
            data["q"].as_array().map(|a| a.iter().map(|r| json!({"uid": r["uid"]})).collect()).unwrap_or_default();
        graph.client().delete(del).await.unwrap();
    }
}
