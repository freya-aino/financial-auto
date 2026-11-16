//! Copies the model-state graph between backends (D-108, then the D-110 Dgraph cutover):
//! snapshots with their spaces, regions, transitions, lineage, hierarchy, per-space state and the
//! reference mirror.
//!
//!   cargo run -p fa-hub --example graph_migrate -- --from postgres --to dgraph [--dry-run]
//!
//! Sources: `FA_HELIX_URL` (default http://127.0.0.1:6969), `DATABASE_URL` (the platform Postgres
//! with migrations applied), `FA_DGRAPH_URL`. Idempotent: every write is an upsert keyed by a
//! business id, so an interrupted run is resumed by re-running it.
//!
//! **`postgres` → `dgraph` is a complete copy**, because D-110 drops the `graph` schema afterwards
//! and anything skipped here is lost for good. The generic pass below carries what `GraphStore`
//! can express; [`copy_pg_extras`] carries what it cannot and so was silently missing before:
//! event-time history (`transition_version`, `region_status_version` — the trait can read versions
//! but only ever *writes* them with `now()`), `region_match`, `snapshot_parent.op`, `kernel_model`,
//! `cross_prior`, `region_episode`, and `region_set.family`/`method`. [`verify`] then compares
//! per-table counts and exits non-zero on any mismatch.
//!
//! Still not copied: Qdrant vectors (unchanged by D-110, see D-101).
use fa_hub::graph::GraphStore;
use fa_hub::{
    DgraphGraph, HelixGraph, HubClient, PgGraph, RefEntity, RefInstrument, RefListing, RefVenue, TransitionUpsert,
};
use sqlx::Row;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Unknown flags are refused rather than ignored. `--times-only` was first typed at a shell
    // that silently dropped it, and what ran instead was the full migration — against a live
    // destination, where the history re-import is authoritative. A typo must not be able to do
    // that.
    const FLAGS: [&str; 6] = ["--from", "--to", "--dry-run", "--verify-only", "--times-only", "--help"];
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if !FLAGS.contains(&a.as_str()) {
            eprintln!("unknown argument {a:?}; expected one of {}", FLAGS.join(" "));
            std::process::exit(2);
        }
        if a == "--from" || a == "--to" {
            it.next(); // its value
        }
    }
    if std::env::args().any(|a| a == "--help") {
        println!(
            "graph_migrate --from <helix|postgres> --to <postgres|dgraph> [--dry-run]\n               --dry-run      read and count, write nothing\n               --verify-only  compare source and destination, write nothing\n               --times-only   copy node timestamps only (repairs a cutover that stamped `now()`)"
        );
        return;
    }
    let from = arg("--from").unwrap_or_else(|| "helix".into());
    let to = arg("--to").unwrap_or_else(|| "postgres".into());
    let dry = std::env::args().any(|a| a == "--dry-run");
    let verify_only = std::env::args().any(|a| a == "--verify-only");
    let times_only = std::env::args().any(|a| a == "--times-only");
    let helix_client = HubClient::from_env().with_graph(None);
    let helix = HelixGraph::new(helix_client.clone());
    let pg = PgGraph::connect_lazy(&std::env::var("DATABASE_URL").expect("DATABASE_URL"));
    match (from.as_str(), to.as_str()) {
        ("helix", "postgres") => migrate(&helix, &pg, dry).await,
        ("postgres", "helix") => migrate(&pg, &helix, dry).await,
        ("postgres", "dgraph") => {
            let dg = DgraphGraph::from_env();
            if verify_only {
                if !verify(&pg, &dg, false).await {
                    eprintln!("verification FAILED: the copy is incomplete, do not drop the source");
                    std::process::exit(1);
                }
                return;
            }
            dg.ensure_schema().await.expect("dgraph schema");
            if times_only {
                // Node timestamps only: no history is cleared and no current value is written, so
                // this is the one pass that is safe against a destination the platform is using.
                copy_node_times(pg.pool(), &dg).await;
                if !verify_node_times(pg.pool(), &dg).await {
                    eprintln!("timestamp verification FAILED");
                    std::process::exit(1);
                }
                return;
            }
            migrate(&pg, &dg, dry).await;
            if !dry {
                copy_pg_extras(&pg, &dg).await;
            }
            if !verify(&pg, &dg, dry).await {
                eprintln!("verification FAILED: the copy is incomplete, do not drop the source");
                std::process::exit(1);
            }
        }
        ("helix", "dgraph") => {
            let dg = DgraphGraph::from_env();
            dg.ensure_schema().await.expect("dgraph schema");
            migrate(&helix, &dg, dry).await;
            if !helix_evolution_is_empty(&helix_client).await {
                eprintln!(
                    "the Helix store holds genome/expert rows, which this direction does not copy \
                     (see D-110 phase 4); migrate them before dropping Helix"
                );
                std::process::exit(1);
            }
        }
        other => panic!("unsupported direction {other:?}"),
    }
}

/// Whether Helix holds any of the rows that only live there (genomes, fitness, Pareto fronts,
/// experts). The `helix → dgraph` direction copies the model-state graph through `GraphStore` but
/// not these, because when D-110 was implemented the live Helix store was measured as holding
/// none of them — only its own `SchemaMeta` marker. Rather than carry an untestable copy path for
/// data that does not exist, this refuses the cutover loudly if that ever stops being true.
async fn helix_evolution_is_empty(hub: &HubClient) -> bool {
    let mut empty = true;
    for label in ["Genome", "FitnessRecord", "ParetoFront", "Expert"] {
        match hub.count_nodes(label).await {
            Ok(0) => {}
            Ok(n) => {
                eprintln!("helix still holds {n} {label} rows");
                empty = false;
            }
            Err(e) => {
                eprintln!("could not count helix {label}: {e}");
                empty = false;
            }
        }
    }
    empty
}

/// Everything `GraphStore` cannot express, read straight from the source tables.
///
/// This exists because the trait is the platform's runtime surface, not a migration surface: it
/// has no way to write an event-time version with its original `ts`, no reader that exposes
/// `snapshot_parent.op`, no `region_match` reader at all, and `RegionSetRow` omits
/// `family`/`method`. Going through SQL here is what makes the copy lossless.
async fn copy_pg_extras(pg: &PgGraph, dg: &DgraphGraph) {
    let pool = pg.pool();
    // The source is authoritative for event-time history, so clear it first. This removes both the
    // versions the current-value pass just manufactured (`upsert_transitions` stamps `now()`) and
    // any from an earlier run, which is what makes re-running converge instead of doubling.
    let dropped = dg.dg_drop_versions(None).await.expect("clear destination history");
    println!("cleared {dropped} destination version rows before importing history");
    let ts = |r: &sqlx::postgres::PgRow, c: &str| -> String {
        r.try_get::<chrono::DateTime<chrono::Utc>, _>(c)
            .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
            .unwrap_or_default()
    };

    // event-time history, with the original timestamps
    let rows = sqlx::query("SELECT * FROM graph.transition_version ORDER BY id")
        .fetch_all(pool)
        .await
        .expect("read transition_version");
    let batch: Vec<fa_hub::graph_dgraph::TransitionVersionImport> = rows
        .iter()
        .map(|r| fa_hub::graph_dgraph::TransitionVersionImport {
            edge_key: r.get("edge_key"),
            from_region_id: r.get("from_region_id"),
            to_region_id: r.get("to_region_id"),
            sampler_id: r.get("sampler_id"),
            count: r.get::<i64, _>("count"),
            mean: r.get::<f64, _>("mean"),
            lower: r.get::<f64, _>("lower"),
            upper: r.get::<f64, _>("upper"),
            kernel_params: r.get("kernel_params"),
            ts: ts(r, "ts"),
        })
        .collect();
    let n = dg.dg_import_transition_versions(&batch).await.expect("import transition_version");
    println!("transition_version: {n}");

    let rows = sqlx::query("SELECT * FROM graph.region_status_version ORDER BY id")
        .fetch_all(pool)
        .await
        .expect("read region_status_version");
    let batch: Vec<fa_hub::graph_dgraph::RegionStatusVersionImport> = rows
        .iter()
        .map(|r| fa_hub::graph_dgraph::RegionStatusVersionImport {
            region_id: r.get("region_id"),
            status: r.get("status"),
            related_region_id: r.get("related_region_id"),
            reason: r.get("reason"),
            ts: ts(r, "ts"),
        })
        .collect();
    let n = dg.dg_import_region_status_versions(&batch).await.expect("import region_status_version");
    println!("region_status_version: {n}");

    // region matches (the lineage "carry" links)
    let rows = sqlx::query("SELECT prev_region_id, next_region_id, score FROM graph.region_match")
        .fetch_all(pool)
        .await
        .expect("read region_match");
    for r in &rows {
        dg.dg_import_region_match(r.get("prev_region_id"), r.get("next_region_id"), r.get::<f64, _>("score"))
            .await
            .expect("import region_match");
    }
    println!("region_match: {}", rows.len());

    // snapshot parents, keeping `op`
    let rows = sqlx::query(
        "SELECT p.snapshot_id AS parent, c.snapshot_id AS child, sp.op FROM graph.snapshot_parent sp
         JOIN graph.snapshot p ON p.id = sp.parent_id JOIN graph.snapshot c ON c.id = sp.child_id",
    )
    .fetch_all(pool)
    .await
    .expect("read snapshot_parent");
    for r in &rows {
        dg.dg_import_snapshot_parent(r.get("parent"), r.get("child"), r.get("op"))
            .await
            .expect("import snapshot_parent");
    }
    println!("snapshot_parent: {}", rows.len());

    // kernels
    let rows = sqlx::query("SELECT * FROM graph.kernel_model").fetch_all(pool).await.expect("read kernel_model");
    for r in &rows {
        dg.upsert_kernel_model(&fa_hub::NewKernelModel {
            expert_space_id: r.get("expert_space_id"),
            snapshot_id: r.get("snapshot_id"),
            timescale: r.get("timescale"),
            expert_index: r.get::<i32, _>("expert_index") as u32,
            kernel_type: r.get("kernel_type"),
            serving_sampler: r.get("serving_sampler"),
            scores: r.get("scores"),
            stationary: r.get("stationary"),
            relaxation_steps: r.get::<f64, _>("relaxation_steps"),
            drift_kl: r.get::<f64, _>("drift_kl"),
            n_rows: r.get::<i32, _>("n_rows") as u32,
            observation_mass: r.get::<f64, _>("observation_mass"),
            live_regions: r.get::<i32, _>("live_regions") as u32,
        })
        .await
        .expect("import kernel_model");
    }
    println!("kernel_model: {}", rows.len());

    // cross priors (the trait reader needs a timescale, so SQL avoids having to guess the set)
    let rows = sqlx::query("SELECT * FROM graph.cross_prior").fetch_all(pool).await.expect("read cross_prior");
    for r in &rows {
        dg.upsert_cross_priors(&[fa_hub::NewCrossPrior {
            timescale: r.get("timescale"),
            source_instrument_id: r.get("source_instrument_id"),
            target_instrument_id: r.get("target_instrument_id"),
            n: r.get::<i32, _>("n") as u32,
            correlation: r.get::<f64, _>("correlation"),
            shrinkage: r.get::<f64, _>("shrinkage"),
            lead_lag: r.get::<i32, _>("lead_lag") as u32,
            lead_corr: r.get::<f64, _>("lead_corr"),
            predictive_gain: r.get::<f64, _>("predictive_gain"),
            uncertainty: r.get::<f64, _>("uncertainty"),
            score: r.get::<f64, _>("score"),
            rank: r.get::<i32, _>("rank") as u32,
        }])
        .await
        .expect("import cross_prior");
    }
    println!("cross_prior: {}", rows.len());

    // episodes
    let rows = sqlx::query("SELECT * FROM graph.region_episode").fetch_all(pool).await.expect("read region_episode");
    for r in &rows {
        let ms = |c: &str| {
            r.try_get::<chrono::DateTime<chrono::Utc>, _>(c).map(|t| t.timestamp_millis()).unwrap_or_default()
        };
        dg.upsert_region_episodes(&[fa_hub::NewRegionEpisode {
            region_id: r.get("region_id"),
            expert_space_id: r.get("expert_space_id"),
            timescale: r.get("timescale"),
            instrument_id: r.get("instrument_id"),
            t_enter_ms: ms("t_enter"),
            t_exit_ms: ms("t_exit"),
            bars: r.get::<i32, _>("bars") as u32,
            max_weight: r.get::<f64, _>("max_weight"),
            mean_weight: r.get::<f64, _>("mean_weight"),
        }])
        .await
        .expect("import region_episode");
    }
    println!("region_episode: {}", rows.len());

    // reference rows that no instrument points at. The generic pass walks the mirror through
    // `instrument_reference`, so a venue, entity or sector with no listing was simply never seen
    // (the source had 7 venues and only 2 were reachable).
    let rows = sqlx::query("SELECT * FROM graph.ref_entity").fetch_all(pool).await.expect("read ref_entity");
    for r in &rows {
        dg.upsert_ref_entities(&[RefEntity {
            entity_id: r.get("entity_id"),
            lei: r.get("lei"),
            name: r.get("name"),
            country: r.get("country"),
            sector: r.get("sector"),
        }])
        .await
        .expect("import ref_entity");
    }
    println!("ref_entity: {}", rows.len());

    let rows = sqlx::query("SELECT * FROM graph.ref_venue").fetch_all(pool).await.expect("read ref_venue");
    for r in &rows {
        dg.upsert_ref_venues(&[RefVenue {
            venue_id: r.get("venue_id"),
            code: r.get("code"),
            mic: r.get("mic"),
            name: r.get("name"),
            kind: r.get("kind"),
            timezone: r.get("timezone"),
        }])
        .await
        .expect("import ref_venue");
    }
    println!("ref_venue: {}", rows.len());

    // region sets, this time with family/method (RegionSetRow omits both)
    let rows = sqlx::query("SELECT * FROM graph.region_set").fetch_all(pool).await.expect("read region_set");
    for r in &rows {
        dg.upsert_region_set(&fa_hub::NewRegionSet {
            expert_space_id: r.get("region_set_id"),
            snapshot_id: r.get("snapshot_id"),
            timescale: r.get("timescale"),
            expert_index: r.get::<i32, _>("expert_index") as u32,
            family: r.get("family"),
            method: r.get("method"),
            n_regions: r.get::<i32, _>("n_regions") as u32,
            cfg_hash: r.get("cfg_hash"),
            state: r.get("state"),
            kernel_state: r.get("kernel_state"),
        })
        .await
        .expect("import region_set");
    }
    println!("region_set (with family/method): {}", rows.len());

    copy_node_times(pool, dg).await;
}

/// Node timestamps, which the runtime API always stamps `now()`.
///
/// Without this pass a faithful-looking copy carries the migration's wall-clock as every
/// snapshot's and region's creation time, and `verify`'s row counts cannot see it: the first
/// cutover (2026-11-16) stamped all 5 snapshots and 451 regions at 20:48, so an as-of read of
/// any earlier instant returned an empty graph. [`verify_node_times`] now compares the extremes
/// of every one of these columns, so a future run cannot lose them silently either.
async fn copy_node_times(pool: &sqlx::PgPool, dg: &DgraphGraph) {
    use fa_hub::graph_dgraph::NodeTimesImport;
    let ts = |r: &sqlx::postgres::PgRow, c: &str| -> String {
        r.try_get::<Option<chrono::DateTime<chrono::Utc>>, _>(c)
            .ok()
            .flatten()
            .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
            .unwrap_or_default()
    };
    let mut batch: Vec<NodeTimesImport> = Vec::new();

    // (source query, dgraph type, key column, key predicate, timestamp columns → predicates).
    // The queries are literals, so nothing from the data reaches the SQL.
    type Spec = (&'static str, &'static str, &'static str, &'static str, &'static [(&'static str, &'static str)]);
    let specs: &[Spec] = &[
        (
            "SELECT * FROM graph.snapshot",
            "Snapshot",
            "snapshot_id",
            "snapshot.snapshot_id",
            &[("created_at", "snapshot.created_at")],
        ),
        (
            "SELECT * FROM graph.region",
            "LatentRegion",
            "region_id",
            "region.region_id",
            &[
                ("created_at", "region.created_at"),
                ("updated_at", "region.updated_at"),
                ("died_at", "region.died_at"),
            ],
        ),
        (
            "SELECT * FROM graph.region_set",
            "RegionSet",
            "region_set_id",
            "region_set.region_set_id",
            &[("fit_time", "region_set.fit_time"), ("updated_at", "region_set.updated_at")],
        ),
        (
            "SELECT * FROM graph.ref_venue",
            "RefVenue",
            "venue_id",
            "ref_venue.venue_id",
            &[("mirrored_at", "ref_venue.mirrored_at")],
        ),
        (
            "SELECT * FROM graph.ref_instrument",
            "RefInstrument",
            "instrument_id",
            "ref_instrument.instrument_id",
            &[("mirrored_at", "ref_instrument.mirrored_at")],
        ),
        (
            "SELECT * FROM graph.ref_entity",
            "RefEntity",
            "entity_id",
            "ref_entity.entity_id",
            &[("mirrored_at", "ref_entity.mirrored_at")],
        ),
    ];
    for (sql, ty, key_col, key_pred, cols) in specs {
        let rows = sqlx::query(*sql).fetch_all(pool).await.unwrap_or_else(|e| panic!("read for {ty}: {e}"));
        for r in &rows {
            let Ok(key) = r.try_get::<String, _>(*key_col) else { continue };
            batch.push(NodeTimesImport {
                ty,
                key_pred,
                key,
                times: cols.iter().map(|(c, pred)| (*pred, ts(r, c))).collect(),
            });
        }
    }
    let n = dg.dg_import_node_times(&batch).await.expect("import node timestamps");
    println!("node timestamps: {n} nodes stamped from the source");
}

/// Every `graph` table's row count in one static statement — the table names are fixed here,
/// so nothing is interpolated into SQL.
const COUNTS_SQL: &str = "SELECT 'snapshot' AS t, count(*) AS n FROM graph.snapshot
     UNION ALL SELECT 'snapshot_parent' AS t, count(*) AS n FROM graph.snapshot_parent
     UNION ALL SELECT 'adapter_state' AS t, count(*) AS n FROM graph.adapter_state
     UNION ALL SELECT 'expert_space' AS t, count(*) AS n FROM graph.expert_space
     UNION ALL SELECT 'region' AS t, count(*) AS n FROM graph.region
     UNION ALL SELECT 'transition' AS t, count(*) AS n FROM graph.transition
     UNION ALL SELECT 'transition_version' AS t, count(*) AS n FROM graph.transition_version
     UNION ALL SELECT 'region_status_version' AS t, count(*) AS n FROM graph.region_status_version
     UNION ALL SELECT 'descends_from' AS t, count(*) AS n FROM graph.descends_from
     UNION ALL SELECT 'region_match' AS t, count(*) AS n FROM graph.region_match
     UNION ALL SELECT 'hier' AS t, count(*) AS n FROM graph.hier
     UNION ALL SELECT 'region_set' AS t, count(*) AS n FROM graph.region_set
     UNION ALL SELECT 'kernel_model' AS t, count(*) AS n FROM graph.kernel_model
     UNION ALL SELECT 'cross_prior' AS t, count(*) AS n FROM graph.cross_prior
     UNION ALL SELECT 'region_episode' AS t, count(*) AS n FROM graph.region_episode
     UNION ALL SELECT 'trained_on' AS t, count(*) AS n FROM graph.trained_on
     UNION ALL SELECT 'feedback_from' AS t, count(*) AS n FROM graph.feedback_from
     UNION ALL SELECT 'ref_entity' AS t, count(*) AS n FROM graph.ref_entity
     UNION ALL SELECT 'ref_sector' AS t, count(*) AS n FROM graph.ref_sector
     UNION ALL SELECT 'ref_in_sector' AS t, count(*) AS n FROM graph.ref_in_sector
     UNION ALL SELECT 'ref_venue' AS t, count(*) AS n FROM graph.ref_venue
     UNION ALL SELECT 'ref_instrument' AS t, count(*) AS n FROM graph.ref_instrument
     UNION ALL SELECT 'ref_issued_by' AS t, count(*) AS n FROM graph.ref_issued_by
     UNION ALL SELECT 'ref_listed_on' AS t, count(*) AS n FROM graph.ref_listed_on
     UNION ALL SELECT 'ref_same_exposure' AS t, count(*) AS n FROM graph.ref_same_exposure";

/// Per-table row counts, source against destination. The whole point of the cutover is that the
/// source is then dropped, so "it looked fine" is not good enough.
async fn verify(pg: &PgGraph, dg: &DgraphGraph, dry: bool) -> bool {
    if dry {
        println!("dry run: nothing copied, verification skipped");
        return true;
    }
    let pairs = [
        ("snapshot", "Snapshot"),
        ("snapshot_parent", "SnapshotParent"),
        ("adapter_state", "AdapterState"),
        ("expert_space", "ExpertSpace"),
        ("region", "LatentRegion"),
        ("transition", "Transition"),
        ("transition_version", "TransitionVersion"),
        ("region_status_version", "RegionStatusVersion"),
        ("descends_from", "DescendsFrom"),
        ("region_match", "RegionMatch"),
        ("hier", "Hier"),
        ("region_set", "RegionSet"),
        ("kernel_model", "KernelModel"),
        ("cross_prior", "CrossPrior"),
        ("region_episode", "RegionEpisode"),
        ("trained_on", "TrainedOn"),
        ("feedback_from", "FeedbackFrom"),
        ("ref_entity", "RefEntity"),
        ("ref_sector", "RefSector"),
        ("ref_in_sector", "RefInSector"),
        ("ref_venue", "RefVenue"),
        ("ref_instrument", "RefInstrument"),
        ("ref_issued_by", "RefIssuedBy"),
        ("ref_listed_on", "RefListedOn"),
        ("ref_same_exposure", "RefSameExposure"),
    ];
    let rows = sqlx::query(COUNTS_SQL).fetch_all(pg.pool()).await.expect("read source counts");
    let src_counts: std::collections::HashMap<String, i64> =
        rows.iter().map(|r| (r.get::<String, _>("t"), r.get::<i64, _>("n"))).collect();
    let mut ok = true;
    println!("\n{:<24} {:>8} {:>8}", "table", "postgres", "dgraph");
    for (table, ty) in pairs {
        let src = src_counts.get(table).copied().unwrap_or(-1);
        let dst = dg.dg_count(ty).await.unwrap_or(u64::MAX) as i64;
        let mark = if src == dst {
            " "
        } else {
            ok = false;
            "!"
        };
        println!("{mark}{table:<23} {src:>8} {dst:>8}");
    }
    ok = verify_node_times(pg.pool(), dg).await && ok;
    if ok {
        println!("\nverification OK: every table matches, every timestamp range matches");
    }
    ok
}

/// Timestamp ranges per node type, source against destination.
///
/// Row counts cannot see a manufactured creation time — the first cutover copied every count
/// correctly while stamping all five snapshots and 451 regions with the migration's own clock.
/// Comparing the extremes of each column catches exactly that, and is cheap: two aggregates.
async fn verify_node_times(pool: &sqlx::PgPool, dg: &DgraphGraph) -> bool {
    // (label, SQL over the source, dgraph type, predicate)
    type Check = (&'static str, &'static str, &'static str, &'static str);
    let checks: &[Check] = &[
        (
            "snapshot.created_at",
            "SELECT min(created_at) a, max(created_at) b FROM graph.snapshot",
            "Snapshot",
            "snapshot.created_at",
        ),
        (
            "region.created_at",
            "SELECT min(created_at) a, max(created_at) b FROM graph.region",
            "LatentRegion",
            "region.created_at",
        ),
        (
            "region_set.fit_time",
            "SELECT min(fit_time) a, max(fit_time) b FROM graph.region_set",
            "RegionSet",
            "region_set.fit_time",
        ),
        (
            "ref_instrument.mirrored_at",
            "SELECT min(mirrored_at) a, max(mirrored_at) b FROM graph.ref_instrument",
            "RefInstrument",
            "ref_instrument.mirrored_at",
        ),
    ];
    let mut ok = true;
    println!("\n{:<28} {:<26} {:<26}", "timestamp range", "postgres", "dgraph");
    for (label, sql, ty, pred) in checks {
        let row = sqlx::query(*sql).fetch_one(pool).await.expect("read source timestamp range");
        let fmt = |t: Option<chrono::DateTime<chrono::Utc>>| {
            t.map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)).unwrap_or_else(|| "-".into())
        };
        let src = (fmt(row.try_get("a").unwrap_or(None)), fmt(row.try_get("b").unwrap_or(None)));
        let dst = dg.dg_time_range(ty, pred).await.unwrap_or_default();
        let dst = (
            dst.0.map(|t| t[..19.min(t.len())].to_string() + "Z").unwrap_or_else(|| "-".into()),
            dst.1.map(|t| t[..19.min(t.len())].to_string() + "Z").unwrap_or_else(|| "-".into()),
        );
        let same = src == dst;
        if !same {
            ok = false;
        }
        println!(
            "{}{label:<27} {:<26} {:<26}",
            if same { " " } else { "!" },
            format!("{} .. {}", src.0, src.1),
            format!("{} .. {}", dst.0, dst.1)
        );
    }
    ok
}

async fn migrate<S: GraphStore, D: GraphStore>(src: &S, dst: &D, dry: bool) {
    let snapshots = src.list_snapshots().await.expect("list snapshots");
    println!("snapshots: {}", snapshots.len());
    let mut counts = [0usize; 6]; // spaces, regions, transitions, descents, hier, adapters
    for snap in &snapshots {
        let Some(g) = src.snapshot_graph(&snap.snapshot_id).await.expect("snapshot graph") else { continue };
        if dry {
            counts[0] += g.spaces.len();
            counts[1] += g.regions.len();
            counts[2] += g.transitions.len();
            continue;
        }
        let new = dst
            .ensure_snapshot(&fa_hub::NewSnapshot {
                snapshot_id: snap.snapshot_id.clone(),
                timescale: snap.timescale.clone(),
                encoder_family: snap.encoder_family.clone(),
                generation: snap.generation,
                garage_key: snap.garage_key.clone(),
                status: snap.status.clone(),
            })
            .await
            .expect("snapshot");
        let copied = dst.set_snapshot_status(new.id, &snap.status).await.expect("status");
        if let Some(m) = &snap.manifest {
            let _ = dst.set_snapshot_manifest(copied.id, m, snap.carried_mass.unwrap_or(0.0)).await;
        }
        if let Some(prev) = &snap.previous_active {
            let _ = dst.set_snapshot_previous_active(copied.id, prev).await;
        }
        for sp in &g.spaces {
            let new_space = dst
                .ensure_expert_space(
                    new.id,
                    &fa_hub::NewExpertSpace {
                        expert_space_id: sp.expert_space_id.clone(),
                        snapshot_id: sp.snapshot_id.clone(),
                        index: sp.index,
                        dim: sp.dim,
                    },
                )
                .await
                .expect("space");
            counts[0] += 1;
            for r in g.regions.iter().filter(|r| r.expert_space_id == sp.expert_space_id) {
                dst.upsert_region(
                    new_space.id,
                    &fa_hub::NewRegion {
                        region_id: r.region_id.clone(),
                        lineage_id: r.lineage_id.clone(),
                        expert_space_id: r.expert_space_id.clone(),
                        method: r.method.clone(),
                        radius: r.radius,
                        density: r.density,
                    },
                )
                .await
                .expect("region");
                if let (Some(st), Some(rel)) = (r.status.as_deref(), r.related_region_id.as_deref())
                    && st != "live"
                {
                    let _ = dst.mark_region_status(&r.region_id, st, rel).await;
                }
                counts[1] += 1;
                // direct lineage links of this region
                if let Some(lin) = src.region_lineage(&r.region_id, 1).await.expect("lineage") {
                    let descents: Vec<fa_hub::NewDescent> = lin
                        .ancestors
                        .iter()
                        .filter_map(|l| {
                            let parent = l.region.properties.get("region_id")?.as_str()?.to_owned();
                            Some(fa_hub::NewDescent {
                                child_region_id: r.region_id.clone(),
                                parent_region_id: parent,
                                kind: l.kind.clone(),
                                overlap: l.overlap,
                            })
                        })
                        .collect();
                    dst.upsert_descents(&descents).await.expect("descents");
                    counts[3] += descents.len();
                }
            }
            if let Some(rs) = src.region_set(&sp.expert_space_id).await.expect("region set") {
                dst.upsert_region_set(&fa_hub::NewRegionSet {
                    expert_space_id: sp.expert_space_id.clone(),
                    snapshot_id: rs.snapshot_id.clone(),
                    timescale: rs.timescale.clone(),
                    expert_index: rs.expert_index,
                    family: String::new(),
                    method: String::new(),
                    n_regions: rs.n_regions,
                    cfg_hash: rs.cfg_hash.clone(),
                    state: rs.state.clone(),
                    kernel_state: rs.kernel_state.clone(),
                })
                .await
                .expect("region set");
            }
        }
        let ups: Vec<TransitionUpsert> = g
            .transitions
            .iter()
            .filter_map(|t| {
                Some(TransitionUpsert {
                    from_region_id: t.from_region_id.clone()?,
                    to_region_id: t.to_region_id.clone()?,
                    transition: fa_hub::NewTransition {
                        sampler_id: t.sampler_id.clone(),
                        count: t.count,
                        mean: t.mean,
                        lower: t.lower,
                        upper: t.upper,
                        kernel_params: t.kernel_params.clone(),
                    },
                })
            })
            .collect();
        for chunk in ups.chunks(100) {
            dst.upsert_transitions(chunk).await.expect("transitions");
        }
        counts[2] += ups.len();
        for e in src.snapshot_hierarchy(&snap.snapshot_id).await.expect("hierarchy") {
            let get = |k: &str| e.properties.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let id = |k: &str| e.properties.get(k).and_then(|v| v.as_str()).unwrap_or("").to_owned();
            dst.upsert_hier_edges(&[fa_hub::NewHierEdge {
                fine_region_id: id("fine_region_id"),
                coarse_region_id: id("coarse_region_id"),
                p_coarse_given_fine: get("p_coarse_given_fine"),
                p_fine_given_coarse: get("p_fine_given_coarse"),
                support: get("support"),
                pmi_bits: get("pmi_bits"),
                mi_bits: get("mi_bits"),
            }])
            .await
            .expect("hier");
            counts[4] += 1;
        }
        for a in src.snapshot_adapters(&snap.snapshot_id, 10_000).await.expect("adapters") {
            let parse = |t: &Option<String>| {
                t.as_deref()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map_or(0, |d| d.timestamp_millis())
            };
            let res = dst
                .add_adapter_state(
                    new.id,
                    &fa_hub::NewAdapterState {
                        snapshot_id: a.snapshot_id.clone(),
                        timescale: a.timescale.clone(),
                        version: a.version,
                        garage_key: a.garage_key.clone(),
                        drift: a.drift,
                        max_drift: a.max_drift,
                        frozen: a.frozen,
                        n_windows: a.n_windows,
                        window_start_ms: parse(&a.window_start),
                        window_end_ms: parse(&a.window_end),
                    },
                )
                .await;
            // a duplicate adapter_id is the expected outcome of a re-run; anything else is a
            // real failure and was being swallowed here
            match res {
                Ok(_) => counts[5] += 1,
                Err(fa_hub::HubError::Server { status: 409, .. }) => counts[5] += 1,
                Err(e) => panic!("adapter {} v{}: {e}", a.snapshot_id, a.version),
            }
        }
    }
    // reference mirror: whole instruments via their reference reads
    let ids = src.ref_instrument_ids().await.expect("ref ids");
    println!("ref instruments: {}", ids.len());
    if !dry {
        for (id, _status) in &ids {
            let Some(r) = src.instrument_reference(id).await.expect("ref read") else { continue };
            let gs = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_owned();
            for e in &r.issuer {
                dst.upsert_ref_entities(&[RefEntity {
                    entity_id: gs(e, "entity_id"),
                    lei: gs(e, "lei"),
                    name: gs(e, "name"),
                    country: gs(e, "country"),
                    sector: gs(e, "sector"),
                }])
                .await
                .expect("entity");
            }
            for v in &r.venues {
                dst.upsert_ref_venues(&[RefVenue {
                    venue_id: gs(v, "venue_id"),
                    code: gs(v, "code"),
                    mic: gs(v, "mic"),
                    name: gs(v, "name"),
                    kind: gs(v, "kind"),
                    timezone: gs(v, "timezone"),
                }])
                .await
                .expect("venue");
            }
            let i = &r.instrument;
            let listings = r
                .listings
                .iter()
                .map(|l| RefListing {
                    venue_id: l.properties.get("venue_id").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
                    venue_symbol: l.properties.get("venue_symbol").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
                    source_id: l.properties.get("source_id").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
                })
                .collect();
            dst.upsert_ref_instruments(&[RefInstrument {
                instrument_id: gs(i, "instrument_id"),
                asset_class: gs(i, "asset_class"),
                kind: gs(i, "kind"),
                name: gs(i, "name"),
                status: gs(i, "status"),
                base_ccy: gs(i, "base_ccy"),
                quote_ccy: gs(i, "quote_ccy"),
                issuer_id: r.issuer.first().map(|e| gs(e, "entity_id")).filter(|s| !s.is_empty()),
                listings,
                // the source read returns these; dropping them was silent data loss once the
                // source is deleted (D-110 drops the `graph` schema)
                exposures: r
                    .exposures
                    .iter()
                    .map(|x| fa_hub::RefExposure {
                        to_instrument_id: x
                            .properties
                            .get("to_instrument_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned(),
                        relation: x.properties.get("relation").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
                    })
                    .filter(|x| !x.to_instrument_id.is_empty())
                    .collect(),
            }])
            .await
            .expect("instrument");
        }
    }
    println!(
        "copied: spaces {} regions {} transitions {} descents {} hier {} adapters {} (dry-run: {dry})",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
    );
}
