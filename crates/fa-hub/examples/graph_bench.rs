//! Graph-store benchmark (ADR 0010): the heaviest graph shapes fa-hub runs on Helix today,
//! replayed on PostgreSQL (recursive CTEs, LATERAL) and Dgraph (DQL over HTTP), on the same
//! deterministic synthetic graph at roughly 10× the live size.
//!
//! Graph (defaults): 100 snapshots (5 timescales × 20 generations) × 4 expert spaces × 64 regions
//! = 25 600 regions; 8 `Transition`s per region (205 k) with 3 `TransitionVersion`s each (614 k);
//! `DescendsFrom` from every region of generation g to 1–2 regions of g − 1; 50 000 genomes in
//! 500 generations of 100, each with two `ParentOf` parents among the previous generation's 30
//! elites; reference mirror of 20 000 instruments, 8 000 issuers, 60 venues, 40 sectors.
//!
//! Shapes (one in flight, 200 measured after 20 warm-up):
//! - `ancestry`   genome ancestors up to depth 10 (`genomeAncestry`), distinct count;
//! - `lineage`    region ancestors + descendants over `DescendsFrom`, depth 6 (`regionLineage`);
//! - `trans2hop`  2-hop transition neighbourhood of a region with summed counts (`neighborhood` +
//!   `regionTransitions`);
//! - `graphasof`  one snapshot's regions created ≤ t with each transition's newest version ≤ t
//!   (`snapshotGraphAsOf` + `transitionVersions`);
//! - `instref`    instrument → issuer → sector, listings → venues, exposures (`instrumentReference`).
//! ```sh
//! graph_bench --pg postgres://bench:bench@fa-bench-pg/bench --dgraph http://fa-bench-dgraph:8080
//! ```
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sqlx::Row;

const T0: i64 = 1_790_000_000_000;
const DAY: i64 = 86_400_000;
const TIMESCALES: usize = 5;
const SPACES: usize = 4;
const REGIONS: usize = 64;
const FANOUT: usize = 8;
const VERSIONS: usize = 3;
const GEN_POP: usize = 100;
const ELITES: usize = 30;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct Graph {
    snapshots: usize,
    // regions
    r_snap: Vec<usize>,
    r_space: Vec<usize>,
    r_created: Vec<i64>,
    // transitions (reified): from, to, count, versions (ts, count)
    t_from: Vec<usize>,
    t_to: Vec<usize>,
    t_cnt: Vec<i64>,
    v_tid: Vec<usize>,
    v_ts: Vec<i64>,
    v_cnt: Vec<i64>,
    // descends: (child, parent)
    desc: Vec<(usize, usize)>,
    genomes: usize,
    parent_of: Vec<(usize, usize)>, // (parent, child)
    // reference
    instruments: usize,
    entities: usize,
    venues: usize,
    sectors: usize,
    issuer: Vec<usize>,
    entity_sector: Vec<usize>,
    listings: Vec<(usize, usize)>,  // (instrument, venue)
    exposures: Vec<(usize, usize)>, // (from, to)
}

fn build(gens: usize, genomes: usize, instruments: usize) -> Graph {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let snapshots = TIMESCALES * gens;
    let per_snap = SPACES * REGIONS;
    let n = snapshots * per_snap;
    let mut g = Graph {
        snapshots,
        r_snap: Vec::with_capacity(n),
        r_space: Vec::with_capacity(n),
        r_created: Vec::with_capacity(n),
        t_from: vec![],
        t_to: vec![],
        t_cnt: vec![],
        v_tid: vec![],
        v_ts: vec![],
        v_cnt: vec![],
        desc: vec![],
        genomes,
        parent_of: vec![],
        instruments,
        entities: instruments * 2 / 5,
        venues: 60,
        sectors: 40,
        issuer: vec![],
        entity_sector: vec![],
        listings: vec![],
        exposures: vec![],
    };
    for r in 0..n {
        let s = r / per_snap;
        g.r_snap.push(s);
        g.r_space.push(r / REGIONS);
        // generation (s / TIMESCALES) is one day apart; regions appear over the first hours
        g.r_created.push(T0 + (s / TIMESCALES) as i64 * DAY + (r % REGIONS) as i64 * 60_000 * 7);
    }
    for r in 0..n {
        let base = (r / REGIONS) * REGIONS;
        let mut seen = HashSet::new();
        while seen.len() < FANOUT {
            let to = base + rng.below(REGIONS);
            if seen.insert(to) {
                let tid = g.t_from.len();
                g.t_from.push(r);
                g.t_to.push(to);
                let c = 1 + rng.below(500) as i64;
                g.t_cnt.push(c);
                let start = g.r_created[r].max(g.r_created[to]);
                for v in 0..VERSIONS {
                    g.v_tid.push(tid);
                    g.v_ts.push(start + (v as i64 + 1) * DAY / 4 + rng.below(3_600_000) as i64);
                    g.v_cnt.push(c * (v as i64 + 1) / VERSIONS as i64);
                }
            }
        }
        let s = r / per_snap;
        if s >= TIMESCALES {
            let parent = r - TIMESCALES * per_snap;
            g.desc.push((r, parent));
            if rng.below(10) < 3 {
                let base = (parent / REGIONS) * REGIONS;
                let p2 = base + (parent % REGIONS + 1) % REGIONS;
                g.desc.push((r, p2));
            }
        }
    }
    for c in GEN_POP..genomes {
        let prev = (c / GEN_POP - 1) * GEN_POP;
        let a = prev + rng.below(ELITES);
        let mut b = prev + rng.below(ELITES);
        if b == a {
            b = prev + (a - prev + 1) % ELITES;
        }
        g.parent_of.push((a, c));
        g.parent_of.push((b, c));
    }
    for i in 0..instruments {
        g.issuer.push(rng.below(g.entities));
        g.listings.push((i, rng.below(g.venues)));
        if rng.below(2) == 0 {
            g.listings.push((i, rng.below(g.venues)));
        }
        if rng.below(5) == 0 {
            g.exposures.push((i, rng.below(instruments)));
        }
    }
    for _ in 0..g.entities {
        g.entity_sector.push(rng.below(g.sectors));
    }
    g
}

/// Parameters drawn the same way for both stores.
struct Params {
    genome: Vec<usize>,
    region: Vec<usize>,
    snapshot: Vec<(usize, i64)>,
    instrument: Vec<usize>,
}

fn params(g: &Graph, count: usize) -> Params {
    let mut rng = Rng(77);
    let gens = g.snapshots / TIMESCALES;
    let n_regions = g.r_snap.len();
    Params {
        // genomes of the newest 50 generations: deep ancestry
        genome: (0..count).map(|_| g.genomes - 1 - rng.below(50 * GEN_POP)).collect(),
        // regions of a middle generation: both ancestors and descendants exist
        region: (0..count)
            .map(|_| (gens / 2) * TIMESCALES * SPACES * REGIONS + rng.below(n_regions / gens * 2))
            .collect(),
        snapshot: (0..count)
            .map(|_| {
                let s = rng.below(g.snapshots);
                (s, T0 + (s / TIMESCALES) as i64 * DAY + rng.below(DAY as usize * 2) as i64)
            })
            .collect(),
        instrument: (0..count).map(|_| rng.below(g.instruments)).collect(),
    }
}

fn pct(sorted: &[Duration], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize].as_secs_f64() * 1e3
}

fn report(store: &str, shape: &str, mut lat: Vec<Duration>, rows: usize) {
    lat.sort_unstable();
    println!(
        "{store:>7} {shape:>10} {:>8.2} {:>8.2} {:>8.2} {:>10.1}",
        pct(&lat, 0.5),
        pct(&lat, 0.95),
        pct(&lat, 0.99),
        rows as f64 / lat.len().max(1) as f64
    );
}

// ── PostgreSQL ────────────────────────────────────────────────────────────

async fn pg_load(pool: &sqlx::PgPool, g: &Graph) {
    let ddl = r#"
DROP SCHEMA IF EXISTS gb CASCADE; CREATE SCHEMA gb; SET search_path = gb;
CREATE TABLE snapshot (id int PRIMARY KEY, snapshot_id text UNIQUE, created_at bigint);
CREATE TABLE space (id int PRIMARY KEY, snapshot int REFERENCES snapshot);
CREATE TABLE region (id int PRIMARY KEY, region_id text UNIQUE, space int REFERENCES space, snapshot int, created_at bigint);
CREATE TABLE transition (id int PRIMARY KEY, from_id int, to_id int, count bigint);
CREATE TABLE tversion (tid int, ts bigint, count bigint);
CREATE TABLE descends (child int, parent int);
CREATE TABLE genome (id int PRIMARY KEY, genome_id text UNIQUE, generation int);
CREATE TABLE parent_of (parent int, child int);
CREATE TABLE sector (id int PRIMARY KEY, name text);
CREATE TABLE entity (id int PRIMARY KEY, name text, sector int);
CREATE TABLE venue (id int PRIMARY KEY, code text);
CREATE TABLE instrument (id int PRIMARY KEY, instrument_id text UNIQUE, issuer int);
CREATE TABLE listing (instrument int, venue int, symbol text);
CREATE TABLE exposure (from_id int, to_id int, relation text);
"#;
    sqlx::raw_sql(ddl).execute(pool).await.expect("ddl");
    let ids = |n: usize| (0..n as i32).collect::<Vec<i32>>();
    let ins = |sql: &'static str| sqlx::query(sql);
    let s = g.snapshots;
    ins("INSERT INTO gb.snapshot SELECT i, 'snap-'||i, $2::bigint + (i/5)::bigint*86400000 FROM unnest($1::int[]) i")
        .bind(ids(s))
        .bind(T0)
        .execute(pool)
        .await
        .expect("snapshot");
    ins("INSERT INTO gb.space SELECT i, i/4 FROM unnest($1::int[]) i")
        .bind(ids(s * SPACES))
        .execute(pool)
        .await
        .expect("space");
    let rs: Vec<i32> = g.r_space.iter().map(|&x| x as i32).collect();
    let rsn: Vec<i32> = g.r_snap.iter().map(|&x| x as i32).collect();
    ins("INSERT INTO gb.region SELECT i, 'reg-'||i, sp, sn, c FROM unnest($1::int[], $2::int[], $3::int[], $4::bigint[]) AS u(i, sp, sn, c)")
        .bind(ids(rs.len()))
        .bind(rs)
        .bind(rsn)
        .bind(g.r_created.clone())
        .execute(pool)
        .await
        .expect("region");
    let to32 = |v: &[usize]| v.iter().map(|&x| x as i32).collect::<Vec<i32>>();
    for (lo, hi) in chunks(g.t_from.len(), 100_000) {
        ins("INSERT INTO gb.transition SELECT * FROM unnest($1::int[], $2::int[], $3::int[], $4::bigint[])")
            .bind((lo as i32..hi as i32).collect::<Vec<_>>())
            .bind(to32(&g.t_from[lo..hi]))
            .bind(to32(&g.t_to[lo..hi]))
            .bind(g.t_cnt[lo..hi].to_vec())
            .execute(pool)
            .await
            .expect("transition");
    }
    for (lo, hi) in chunks(g.v_tid.len(), 200_000) {
        ins("INSERT INTO gb.tversion SELECT * FROM unnest($1::int[], $2::bigint[], $3::bigint[])")
            .bind(to32(&g.v_tid[lo..hi]))
            .bind(g.v_ts[lo..hi].to_vec())
            .bind(g.v_cnt[lo..hi].to_vec())
            .execute(pool)
            .await
            .expect("tversion");
    }
    let (dc, dp): (Vec<i32>, Vec<i32>) = g.desc.iter().map(|&(c, p)| (c as i32, p as i32)).unzip();
    ins("INSERT INTO gb.descends SELECT * FROM unnest($1::int[], $2::int[])")
        .bind(dc)
        .bind(dp)
        .execute(pool)
        .await
        .expect("descends");
    ins("INSERT INTO gb.genome SELECT i, 'gen-'||i, i/100 FROM unnest($1::int[]) i")
        .bind(ids(g.genomes))
        .execute(pool)
        .await
        .expect("genome");
    let (pp, pc): (Vec<i32>, Vec<i32>) = g.parent_of.iter().map(|&(p, c)| (p as i32, c as i32)).unzip();
    ins("INSERT INTO gb.parent_of SELECT * FROM unnest($1::int[], $2::int[])")
        .bind(pp)
        .bind(pc)
        .execute(pool)
        .await
        .expect("parent_of");
    ins("INSERT INTO gb.sector SELECT i, 'sector-'||i FROM unnest($1::int[]) i")
        .bind(ids(g.sectors))
        .execute(pool)
        .await
        .expect("sector");
    ins("INSERT INTO gb.entity SELECT i, 'entity-'||i, s FROM unnest($1::int[], $2::int[]) AS u(i, s)")
        .bind(ids(g.entities))
        .bind(to32(&g.entity_sector))
        .execute(pool)
        .await
        .expect("entity");
    ins("INSERT INTO gb.venue SELECT i, 'V'||i FROM unnest($1::int[]) i")
        .bind(ids(g.venues))
        .execute(pool)
        .await
        .expect("venue");
    ins("INSERT INTO gb.instrument SELECT i, 'inst-'||i, e FROM unnest($1::int[], $2::int[]) AS u(i, e)")
        .bind(ids(g.instruments))
        .bind(to32(&g.issuer))
        .execute(pool)
        .await
        .expect("instrument");
    let (li, lv): (Vec<i32>, Vec<i32>) = g.listings.iter().map(|&(i, v)| (i as i32, v as i32)).unzip();
    ins("INSERT INTO gb.listing SELECT i, v, 'SYM'||i FROM unnest($1::int[], $2::int[]) AS u(i, v)")
        .bind(li)
        .bind(lv)
        .execute(pool)
        .await
        .expect("listing");
    let (ef, et): (Vec<i32>, Vec<i32>) = g.exposures.iter().map(|&(f, t)| (f as i32, t as i32)).unzip();
    ins("INSERT INTO gb.exposure SELECT f, t, 'spot-perp' FROM unnest($1::int[], $2::int[]) AS u(f, t)")
        .bind(ef)
        .bind(et)
        .execute(pool)
        .await
        .expect("exposure");
    sqlx::raw_sql(
        r#"SET search_path = gb;
CREATE INDEX ON region (space); CREATE INDEX ON region (snapshot, created_at);
CREATE INDEX ON transition (from_id); CREATE INDEX ON transition (to_id);
CREATE INDEX ON tversion (tid, ts DESC);
CREATE INDEX ON descends (child); CREATE INDEX ON descends (parent);
CREATE INDEX ON parent_of (child); CREATE INDEX ON parent_of (parent);
CREATE INDEX ON listing (instrument); CREATE INDEX ON exposure (from_id);
ANALYZE;"#,
    )
    .execute(pool)
    .await
    .expect("indexes");
}

fn chunks(n: usize, size: usize) -> Vec<(usize, usize)> {
    (0..n.div_ceil(size)).map(|i| (i * size, ((i + 1) * size).min(n))).collect()
}

const PG_ANCESTRY: &str = "WITH RECURSIVE anc(id, d) AS (
    SELECT p.parent, 1 FROM gb.parent_of p JOIN gb.genome g ON g.id = p.child WHERE g.genome_id = $1
  UNION
    SELECT p.parent, a.d + 1 FROM gb.parent_of p JOIN anc a ON p.child = a.id WHERE a.d < 10)
SELECT count(DISTINCT id) AS n FROM anc";

const PG_LINEAGE: &str = "WITH RECURSIVE
  start AS (SELECT id FROM gb.region WHERE region_id = $1),
  up(id, d) AS (SELECT d.parent, 1 FROM gb.descends d JOIN start s ON d.child = s.id
     UNION SELECT d.parent, u.d + 1 FROM gb.descends d JOIN up u ON d.child = u.id WHERE u.d < 6),
  down(id, d) AS (SELECT d.child, 1 FROM gb.descends d JOIN start s ON d.parent = s.id
     UNION SELECT d.child, w.d + 1 FROM gb.descends d JOIN down w ON d.parent = w.id WHERE w.d < 6)
SELECT r.region_id, 'up' AS dir FROM gb.region r WHERE r.id IN (SELECT id FROM up)
UNION ALL SELECT r.region_id, 'down' FROM gb.region r WHERE r.id IN (SELECT id FROM down)";

const PG_TRANS2HOP: &str = "SELECT r2.region_id, sum(t2.count)::bigint AS c, count(*) AS paths
FROM gb.region r JOIN gb.transition t1 ON t1.from_id = r.id
JOIN gb.transition t2 ON t2.from_id = t1.to_id JOIN gb.region r2 ON r2.id = t2.to_id
WHERE r.region_id = $1 GROUP BY r2.region_id";

const PG_GRAPHASOF: &str = "SELECT r.region_id, r2.region_id AS to_region, v.ts, v.count
FROM gb.snapshot s JOIN gb.region r ON r.snapshot = s.id AND r.created_at <= $2
JOIN gb.transition t ON t.from_id = r.id
JOIN gb.region r2 ON r2.id = t.to_id AND r2.created_at <= $2
CROSS JOIN LATERAL (SELECT ts, count FROM gb.tversion v WHERE v.tid = t.id AND v.ts <= $2 ORDER BY ts DESC LIMIT 1) v
WHERE s.snapshot_id = $1 AND s.created_at <= $2";

const PG_INSTREF: &str = "SELECT i.instrument_id, e.name AS issuer, sc.name AS sector,
  (SELECT json_agg(json_build_object('venue', v.code, 'symbol', l.symbol)) FROM gb.listing l JOIN gb.venue v ON v.id = l.venue WHERE l.instrument = i.id) AS listings,
  (SELECT json_agg(i2.instrument_id) FROM gb.exposure x JOIN gb.instrument i2 ON i2.id = x.to_id WHERE x.from_id = i.id) AS exposures
FROM gb.instrument i JOIN gb.entity e ON e.id = i.issuer JOIN gb.sector sc ON sc.id = e.sector
WHERE i.instrument_id = $1";

async fn pg_bench(url: &str, g: &Graph, p: &Params, warm: usize) {
    let pool = sqlx::postgres::PgPoolOptions::new().max_connections(2).connect(url).await.expect("pg connect");
    let t = Instant::now();
    pg_load(&pool, g).await;
    println!("postgres load + index: {:.1} s", t.elapsed().as_secs_f64());
    let mut conn = pool.acquire().await.expect("conn");
    macro_rules! run {
        ($shape:expr, $iter:expr, |$x:ident| $q:expr) => {{
            let mut lat = vec![];
            let mut rows = 0usize;
            for (i, $x) in $iter.enumerate() {
                let t = Instant::now();
                let r = $q.fetch_all(&mut *conn).await.expect($shape);
                let el = t.elapsed();
                if i >= warm {
                    lat.push(el);
                    rows += r.len();
                }
            }
            report("pg", $shape, lat, rows);
        }};
    }
    run!("ancestry", p.genome.iter(), |x| sqlx::query(PG_ANCESTRY).bind(format!("gen-{x}")));
    run!("lineage", p.region.iter(), |x| sqlx::query(PG_LINEAGE).bind(format!("reg-{x}")));
    run!("trans2hop", p.region.iter(), |x| sqlx::query(PG_TRANS2HOP).bind(format!("reg-{x}")));
    run!("graphasof", p.snapshot.iter(), |x| sqlx::query(PG_GRAPHASOF).bind(format!("snap-{}", x.0)).bind(x.1));
    run!("instref", p.instrument.iter(), |x| sqlx::query(PG_INSTREF).bind(format!("inst-{x}")));
    // sanity: one ancestry count
    let n: i64 =
        sqlx::query(PG_ANCESTRY).bind(format!("gen-{}", p.genome[0])).fetch_one(&mut *conn).await.expect("a").get("n");
    println!("     pg   ancestry of gen-{} = {n} distinct genomes", p.genome[0]);
}

// ── Dgraph ────────────────────────────────────────────────────────────────

struct Dg {
    http: reqwest::Client,
    url: String,
}
impl Dg {
    async fn post(&self, path: &str, ctype: &str, body: String) -> Value {
        let r = self
            .http
            .post(format!("{}{path}", self.url))
            .header("Content-Type", ctype)
            .body(body)
            .send()
            .await
            .expect("dgraph");
        let st = r.status();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        assert!(st.is_success() && v.get("errors").is_none(), "{path}: {st} {v}");
        v
    }
    async fn mutate(&self, set: Vec<Value>) -> HashMap<String, String> {
        let v = self.post("/mutate?commitNow=true", "application/json", json!({ "set": set }).to_string()).await;
        v["data"]["uids"].as_object().map_or_else(HashMap::new, |m| {
            m.iter().map(|(k, x)| (k.clone(), x.as_str().unwrap_or("").to_owned())).collect()
        })
    }
    async fn query(&self, q: String) -> Value {
        self.post("/query", "application/dql", q).await
    }
}

/// Creates `n` nodes (`mk(i)` without uid) in batches, returns their uids.
async fn dg_nodes(dg: &Dg, n: usize, mk: impl Fn(usize) -> Value) -> Vec<String> {
    let mut uids = vec![String::new(); n];
    for (lo, hi) in chunks(n, 5_000) {
        let set: Vec<Value> = (lo..hi)
            .map(|i| {
                let mut o = mk(i);
                o["uid"] = json!(format!("_:n{i}"));
                o
            })
            .collect();
        let m = dg.mutate(set).await;
        for (j, u) in uids[lo..hi].iter_mut().enumerate() {
            *u = m.get(&format!("n{}", lo + j)).cloned().expect("uid");
        }
    }
    uids
}

async fn dg_load(dg: &Dg, g: &Graph) -> Duration {
    let t = Instant::now();
    dg.post("/alter", "application/json", json!({"drop_all": true}).to_string()).await;
    let schema = r#"
sid: string @index(exact) .
rid: string @index(exact) .
gid: string @index(exact) .
iid: string @index(exact) .
created: int @index(int) .
ts: int @index(int) .
cnt: int .
gen: int .
name: string .
code: string .
symbol: string .
has_space: [uid] .
in_space: uid @reverse .
out: [uid] .
to: uid .
versions: [uid] .
descends: [uid] @reverse .
parent_of: [uid] @reverse .
issued_by: uid .
in_sector: uid .
listed_on: [uid] .
venue: uid .
same_exposure: [uid] .
"#;
    dg.post("/alter", "application/dql", schema.to_owned()).await;
    let spaces_total = g.snapshots * SPACES;
    let space_uids = dg_nodes(dg, spaces_total, |_| json!({"dgraph.type": "ExpertSpace"})).await;
    for (lo, hi) in chunks(g.snapshots, 2_000) {
        let set: Vec<Value> = (lo..hi)
            .map(|s| {
                json!({"sid": format!("snap-{s}"), "created": T0 + (s / TIMESCALES) as i64 * DAY,
                       "has_space": (0..SPACES).map(|k| json!({"uid": space_uids[s * SPACES + k]})).collect::<Vec<_>>()})
            })
            .collect();
        dg.mutate(set).await;
    }
    let ru = dg_nodes(dg, g.r_snap.len(), |r| {
        json!({"rid": format!("reg-{r}"), "created": g.r_created[r], "in_space": {"uid": space_uids[g.r_space[r]]}})
    })
    .await;
    // reified transitions with versions, nested under their source region
    let per_region = FANOUT;
    for (lo, hi) in chunks(g.r_snap.len(), 400) {
        let set: Vec<Value> = (lo..hi)
            .map(|r| {
                let outs: Vec<Value> = (r * per_region..(r + 1) * per_region)
                    .map(|tid| {
                        let vers: Vec<Value> = (tid * VERSIONS..(tid + 1) * VERSIONS)
                            .map(|v| json!({"ts": g.v_ts[v], "cnt": g.v_cnt[v]}))
                            .collect();
                        json!({"to": {"uid": ru[g.t_to[tid]]}, "cnt": g.t_cnt[tid], "versions": vers})
                    })
                    .collect();
                json!({"uid": ru[r], "out": outs})
            })
            .collect();
        dg.mutate(set).await;
    }
    for c in chunks(g.desc.len(), 10_000) {
        let set: Vec<Value> =
            g.desc[c.0..c.1].iter().map(|&(ch, p)| json!({"uid": ru[ch], "descends": [{"uid": ru[p]}]})).collect();
        dg.mutate(set).await;
    }
    let gu = dg_nodes(dg, g.genomes, |i| json!({"gid": format!("gen-{i}"), "gen": (i / GEN_POP) as i64})).await;
    for c in chunks(g.parent_of.len(), 10_000) {
        let set: Vec<Value> = g.parent_of[c.0..c.1]
            .iter()
            .map(|&(p, ch)| json!({"uid": gu[p], "parent_of": [{"uid": gu[ch]}]}))
            .collect();
        dg.mutate(set).await;
    }
    let su = dg_nodes(dg, g.sectors, |i| json!({"name": format!("sector-{i}")})).await;
    let eu = dg_nodes(
        dg,
        g.entities,
        |i| json!({"name": format!("entity-{i}"), "in_sector": {"uid": su[g.entity_sector[i]]}}),
    )
    .await;
    let vu = dg_nodes(dg, g.venues, |i| json!({"code": format!("V{i}")})).await;
    let iu =
        dg_nodes(dg, g.instruments, |i| json!({"iid": format!("inst-{i}"), "issued_by": {"uid": eu[g.issuer[i]]}}))
            .await;
    for c in chunks(g.listings.len(), 10_000) {
        let set: Vec<Value> = g.listings[c.0..c.1]
            .iter()
            .map(|&(i, v)| json!({"uid": iu[i], "listed_on": [{"venue": {"uid": vu[v]}, "symbol": format!("SYM{i}")}]}))
            .collect();
        dg.mutate(set).await;
    }
    let set: Vec<Value> =
        g.exposures.iter().map(|&(f, to)| json!({"uid": iu[f], "same_exposure": [{"uid": iu[to]}]})).collect();
    dg.mutate(set).await;
    t.elapsed()
}

async fn dg_bench(url: &str, g: &Graph, p: &Params, warm: usize) {
    let dg = Dg {
        http: reqwest::Client::builder().timeout(Duration::from_secs(600)).build().expect("client"),
        url: url.to_owned(),
    };
    let el = dg_load(&dg, g).await;
    println!("dgraph load: {:.1} s", el.as_secs_f64());
    tokio::time::sleep(Duration::from_secs(5)).await;
    async fn run(dg: &Dg, shape: &str, qs: Vec<String>, warm: usize, count: impl Fn(&Value) -> usize) -> Value {
        let mut lat = vec![];
        let mut rows = 0;
        let mut first = Value::Null;
        for (i, q) in qs.into_iter().enumerate() {
            let t = Instant::now();
            let v = dg.query(q).await;
            let e = t.elapsed();
            if i >= warm {
                lat.push(e);
                rows += count(&v["data"]);
            }
            if i == 0 {
                first = v["data"].clone();
            }
        }
        report("dgraph", shape, lat, rows);
        first
    }
    let len = |v: &Value, k: &str| v[k].as_array().map_or(0, |a| a.len());
    let a = run(
        &dg,
        "ancestry",
        p.genome
            .iter()
            .map(|x| format!(r#"{{ var(func: eq(gid, "gen-{x}")) @recurse(depth: 10, loop: false) {{ a as ~parent_of }} q(func: uid(a)) {{ n: count(uid) }} }}"#))
            .collect(),
        warm,
        |_| 1,
    )
    .await;
    println!("  dgraph   ancestry of gen-{} = {} distinct genomes", p.genome[0], a["q"][0]["n"]);
    run(
        &dg,
        "lineage",
        p.region
            .iter()
            .map(|x| {
                format!(
                    r#"{{ var(func: eq(rid, "reg-{x}")) @recurse(depth: 6, loop: false) {{ u as descends }}
                        var(func: eq(rid, "reg-{x}")) @recurse(depth: 6, loop: false) {{ d as ~descends }}
                        up(func: uid(u)) {{ rid }} down(func: uid(d)) {{ rid }} }}"#
                )
            })
            .collect(),
        warm,
        |v| len(v, "up") + len(v, "down"),
    )
    .await;
    run(
        &dg,
        "trans2hop",
        p.region
            .iter()
            .map(|x| {
                format!(r#"{{ q(func: eq(rid, "reg-{x}")) {{ out {{ to {{ out {{ cnt to {{ rid }} }} }} }} }} }}"#)
            })
            .collect(),
        warm,
        |v| {
            // client-side group by target region, summing counts (what PG's GROUP BY returns)
            let mut agg: HashMap<String, i64> = HashMap::new();
            for o1 in v["q"][0]["out"].as_array().into_iter().flatten() {
                for o2 in o1["to"]["out"].as_array().into_iter().flatten() {
                    *agg.entry(o2["to"]["rid"].as_str().unwrap_or("").to_owned()).or_default() +=
                        o2["cnt"].as_i64().unwrap_or(0);
                }
            }
            agg.len()
        },
    )
    .await;
    run(
        &dg,
        "graphasof",
        p.snapshot
            .iter()
            .map(|(s, t)| {
                format!(
                    r#"{{ q(func: eq(sid, "snap-{s}")) @filter(le(created, {t})) {{ has_space {{ ~in_space @filter(le(created, {t})) {{ rid
                        out @cascade {{ to @filter(le(created, {t})) {{ rid }} versions @filter(le(ts, {t})) (orderdesc: ts, first: 1) {{ ts cnt }} }} }} }} }} }}"#
                )
            })
            .collect(),
        warm,
        |v| {
            let mut n = 0;
            for sp in v["q"][0]["has_space"].as_array().into_iter().flatten() {
                for r in sp["~in_space"].as_array().into_iter().flatten() {
                    n += r["out"].as_array().map_or(0, |a| a.len());
                }
            }
            n
        },
    )
    .await;
    run(
        &dg,
        "instref",
        p.instrument
            .iter()
            .map(|x| {
                format!(
                    r#"{{ q(func: eq(iid, "inst-{x}")) {{ iid issued_by {{ name in_sector {{ name }} }} listed_on {{ symbol venue {{ code }} }} same_exposure {{ iid }} }} }}"#
                )
            })
            .collect(),
        warm,
        |v| len(v, "q"),
    )
    .await;
}

#[tokio::main]
async fn main() {
    let (mut pg, mut dgraph, mut gens, mut genomes, mut instruments, mut queries) =
        (None, None, 20usize, 50_000usize, 20_000usize, 200usize);
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut v = || it.next().expect("flag value");
        match f.as_str() {
            "--pg" => pg = Some(v()),
            "--dgraph" => dgraph = Some(v()),
            "--gens" => gens = v().parse().expect("gens"),
            "--genomes" => genomes = v().parse().expect("genomes"),
            "--instruments" => instruments = v().parse().expect("instruments"),
            "--queries" => queries = v().parse().expect("queries"),
            other => panic!("unknown flag {other}"),
        }
    }
    let g = build(gens, genomes, instruments);
    println!(
        "graph: {} snapshots, {} regions, {} transitions, {} versions, {} descends, {} genomes, {} parent_of, {} instruments, {} listings",
        g.snapshots,
        g.r_snap.len(),
        g.t_from.len(),
        g.v_tid.len(),
        g.desc.len(),
        g.genomes,
        g.parent_of.len(),
        g.instruments,
        g.listings.len()
    );
    let warm = 20;
    let p = params(&g, queries + warm);
    println!("{:>7} {:>10} {:>8} {:>8} {:>8} {:>10}", "store", "shape", "p50 ms", "p95 ms", "p99 ms", "rows/q");
    if let Some(url) = pg {
        pg_bench(&url, &g, &p, warm).await;
    }
    if let Some(url) = dgraph {
        dg_bench(&url, &g, &p, warm).await;
    }
}
