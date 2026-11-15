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
