//! Event-time versions of transitions and region status (fa-hub `versions`, spec 20 §6.3)
//! against a running HelixDB. Skipped unless `FA_HELIX_URL` is set, e.g.
//! `FA_HELIX_URL=http://127.0.0.1:6969 cargo test -p fa-hub --test versions_live -- --nocapture`.
//! Every row carries a `test-<nanos>-` prefix and is dropped at the end.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fa_hub::schema::node;
use fa_hub::versions::labels;
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

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64
}

/// A unix-ms instant strictly between two server writes.
async fn tick() -> i64 {
    tokio::time::sleep(Duration::from_millis(40)).await;
    let t = now_ms();
    tokio::time::sleep(Duration::from_millis(40)).await;
    t
}

async fn cleanup(c: &HubClient, p: &str) {
    for (label, key) in [
        (node::SNAPSHOT, "snapshot_id"),
        (node::EXPERT_SPACE, "expert_space_id"),
        (node::LATENT_REGION, "region_id"),
        (labels::TRANSITION_VERSION, "edge_key"),
        (labels::REGION_STATUS_VERSION, "region_id"),
    ] {
        let mut attempt = 0;
        while let Err(e) = c.drop_nodes(label, Pred::starts_with(key, p)).await {
            attempt += 1;
            assert!(attempt < 10 && matches!(&e, HubError::Server { status: 409, .. }), "cleanup drop: {e:?}");
            tokio::time::sleep(Duration::from_millis(100 * attempt)).await;
        }
    }
}

fn tr(p: &str, from: &str, to: &str, count: i64, mean: f64) -> TransitionUpsert {
    TransitionUpsert {
        from_region_id: format!("{p}{from}"),
        to_region_id: format!("{p}{to}"),
        transition: NewTransition {
            sampler_id: "conjugate".into(),
            count,
            mean,
            lower: mean / 2.0,
            upper: mean.midpoint(1.0),
            kernel_params: format!("{{\"n\":{count}}}"),
        },
    }
}

#[tokio::test]
async fn as_of_reads_see_values_in_force_at_t_and_pruning_keeps_the_last_one() {
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
    let space_id = format!("{snap_id}-e0");
    let space = c
        .ensure_expert_space(
            snap.id,
            &NewExpertSpace { expert_space_id: space_id.clone(), snapshot_id: snap_id.clone(), index: 0, dim: 4 },
        )
        .await
        .unwrap();
    let region = |n: &str| NewRegion {
        region_id: format!("{p}{n}"),
        lineage_id: format!("{p}{n}"),
        expert_space_id: space_id.clone(),
        method: "leader".into(),
        radius: 1.0,
        density: 0.0,
    };
    for n in ["a", "b"] {
        c.upsert_region(space.id, &region(n)).await.unwrap();
    }
    let t0 = tick().await; // regions exist, no edge yet
    c.upsert_transition(&tr(&p, "a", "b", 1, 0.2)).await.unwrap();
    let t1 = tick().await;
    c.upsert_transitions(&[tr(&p, "a", "b", 5, 0.6), tr(&p, "b", "a", 2, 0.3)]).await.unwrap();
    let t2 = tick().await;
    c.mark_region_status(&format!("{p}b"), "merged", &format!("{p}a")).await.unwrap();
    let t3 = tick().await;

    let graph = |g: Option<SnapshotGraph>| {
        let g = g.expect("snapshot visible");
        let edges: Vec<(String, i64, f64)> = g
            .transitions
            .iter()
            .map(|t| (t.to_region_id.clone().unwrap_or_default().replace(&p, ""), t.count, t.mean))
            .collect();
        let b = g.regions.iter().find(|r| r.region_id == format!("{p}b")).cloned().expect("b visible");
        (edges, b.status)
    };
    let (e, b) = graph(c.snapshot_graph_as_of(&snap_id, t0).await.unwrap());
    assert!(e.is_empty(), "no edge before its first write: {e:?}");
    assert_eq!(b.as_deref(), Some("live"));
    let (e, _) = graph(c.snapshot_graph_as_of(&snap_id, t1).await.unwrap());
    assert_eq!(e, vec![("b".to_owned(), 1, 0.2)], "first posterior, not the current one");
    let (mut e, b) = graph(c.snapshot_graph_as_of(&snap_id, t2).await.unwrap());
    e.sort_by(|x, y| x.0.cmp(&y.0));
    assert_eq!(e, vec![("a".to_owned(), 2, 0.3), ("b".to_owned(), 5, 0.6)]);
    assert_eq!(b.as_deref(), Some("live"), "merge mark came later");
    let (_, b) = graph(c.snapshot_graph_as_of(&snap_id, t3).await.unwrap());
    assert_eq!(b.as_deref(), Some("merged"));
    let regions = c.space_regions_as_of(space.id, t2).await.unwrap();
    assert_eq!(regions.iter().find(|r| r.region_id == format!("{p}b")).unwrap().status.as_deref(), Some("live"));

    let v = c.transition_versions(&format!("{p}a"), &format!("{p}b"), "conjugate", None, 10).await.unwrap();
    assert_eq!(v.iter().map(|x| x.count).collect::<Vec<_>>(), vec![5, 1], "newest first");
    assert_eq!(v[0].kernel_params, "{\"n\":5}");
    let v = c.transition_versions(&format!("{p}a"), &format!("{p}b"), "conjugate", Some(t1), 10).await.unwrap();
    assert_eq!(v.len(), 1);
    let s = c.region_status_versions(&format!("{p}b"), None, 10).await.unwrap();
    assert_eq!(
        s.iter().map(|x| (x.status.as_str(), x.reason.as_str())).collect::<Vec<_>>(),
        vec![("merged", ""), ("live", "created")]
    );

    // a revival of the merged region is versioned; re-upserting a live one is not
    c.upsert_region(space.id, &region("b")).await.unwrap();
    c.upsert_region(space.id, &region("a")).await.unwrap();
    assert_eq!(c.region_status_versions(&format!("{p}b"), None, 10).await.unwrap()[0].reason, "revived");
    assert_eq!(c.region_status_versions(&format!("{p}a"), None, 10).await.unwrap().len(), 1);
    // death mark
    c.mark_region_died(&format!("{p}a"), "support", "").await.unwrap();
    let a = c.region_status_versions(&format!("{p}a"), None, 10).await.unwrap();
    assert_eq!((a[0].status.as_str(), a[0].reason.as_str()), ("retired", "support"));

    // pruning at t3: per edge / region only the newest version older than the cut-off survives
    let dropped = c.prune_space_versions(space.id, t3).await.unwrap();
    assert_eq!(dropped, 2, "a->b count 1 and b's creation mark");
    let v = c.transition_versions(&format!("{p}a"), &format!("{p}b"), "conjugate", None, 10).await.unwrap();
    assert_eq!(v.iter().map(|x| x.count).collect::<Vec<_>>(), vec![5]);
    let (mut e, _) = graph(c.snapshot_graph_as_of(&snap_id, t2).await.unwrap());
    e.sort_by(|x, y| x.0.cmp(&y.0));
    assert_eq!(e, vec![("a".to_owned(), 2, 0.3), ("b".to_owned(), 5, 0.6)], "value in force at the cut-off kept");
    assert_eq!(c.prune_space_versions(space.id, t3).await.unwrap(), 0, "idempotent");

    cleanup(&c, &p).await;
    assert!(c.region_status_versions(&format!("{p}b"), None, 10).await.unwrap().is_empty());
}
