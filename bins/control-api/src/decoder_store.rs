//! Decoder outputs in Postgres (`wh.decoder_output`, migration `20261114090000`), so the
//! region detail view can show what the decoders said while an instrument sat in a region
//! (`GET /api/latent/regions/{id}/reconstructions`, spec 51 §6.3).
//!
//! - **Writer:** `live` hands every `decoder.output.>` message it ingests to a bounded queue
//!   ([`QUEUE`]) with `try_send`, so the bus collector never waits on Postgres; a full queue
//!   drops the output (counted and logged in `live`). The writer waits for the pool, then
//!   inserts in batches of up to [`BATCH`] rows (one `unnest` insert per batch).
//! - **Retention:** rows with `as_of` older than [`RETENTION_DAYS`] are deleted every hour.
//! - **Read:** [`for_episodes`] answers the outputs of many episodes in one query (newest first,
//!   at most `per_episode` each).

use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use sqlx::PgPool;
use tokio::sync::mpsc;

use crate::{live::DecoderOutputView, state::AppState};

/// Outputs waiting for Postgres before new ones are dropped.
pub const QUEUE: usize = 4_096;
/// Rows per insert.
pub const BATCH: usize = 256;
/// Rows older than this (by `as_of`) are deleted.
pub const RETENTION_DAYS: i32 = 30;
const RETENTION_EVERY: Duration = Duration::from_secs(3_600);

/// The episode an output query is for: snapshot + timescale + instrument + `[t0, t1]` (ms).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeKey {
    pub snapshot_id: String,
    pub timescale: String,
    pub instrument_id: String,
    pub t0_ms: i64,
    pub t1_ms: i64,
}

/// Snapshot id of an expert space id (`<snapshot>-e<expert>`, model-runtime's `expert_space_id`).
pub fn snapshot_of_space(space: &str) -> Option<&str> {
    let (snap, k) = space.rsplit_once("-e")?;
    (!snap.is_empty() && !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit())).then_some(snap)
}

/// Installs the sink in `live` and starts the writer and the retention task.
pub fn spawn(state: &AppState) {
    let (tx, rx) = mpsc::channel(QUEUE);
    state.live_mut().set_decoder_sink(tx);
    let s = state.clone();
    tokio::spawn(async move {
        let pool = wait_for_pool(&s).await;
        writer(pool, rx).await;
    });
    let s = state.clone();
    tokio::spawn(async move {
        let pool = wait_for_pool(&s).await;
        let mut tick = tokio::time::interval(RETENTION_EVERY);
        loop {
            tick.tick().await;
            match prune(&pool, RETENTION_DAYS).await {
                Ok(n) if n > 0 => tracing::info!(deleted = n, "decoder-output retention"),
                Ok(_) => {}
                Err(e) => tracing::warn!(error = %e, "decoder-output retention failed"),
            }
        }
    });
}

async fn wait_for_pool(state: &AppState) -> PgPool {
    loop {
        if let Some(p) = state.pg.get() {
            return p.clone();
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn writer(pool: PgPool, mut rx: mpsc::Receiver<DecoderOutputView>) {
    let mut buf = Vec::with_capacity(BATCH);
    while rx.recv_many(&mut buf, BATCH).await > 0 {
        if let Err(e) = insert(&pool, &buf).await {
            tracing::warn!(error = %e, rows = buf.len(), "decoder outputs not stored");
        }
        buf.clear();
    }
}

/// The `output` column: everything that is not a key column.
fn output_json(o: &DecoderOutputView) -> Value {
    json!({ "output_refs": o.output_refs, "quantiles": o.quantiles, "analogs": o.analogs })
}

/// Inserts a batch of outputs (one statement).
pub async fn insert(pool: &PgPool, rows: &[DecoderOutputView]) -> sqlx::Result<u64> {
    if rows.is_empty() {
        return Ok(0);
    }
    let col = |f: fn(&DecoderOutputView) -> String| rows.iter().map(f).collect::<Vec<_>>();
    let inst: Vec<Option<String>> =
        rows.iter().map(|o| Some(o.instrument_id.clone()).filter(|i| !i.is_empty())).collect();
    let as_of: Vec<DateTime<Utc>> = rows.iter().map(|o| o.as_of.unwrap_or(o.received_at)).collect();
    let recorded: Vec<DateTime<Utc>> = rows.iter().map(|o| o.received_at).collect();
    let output: Vec<Value> = rows.iter().map(output_json).collect();
    let r = sqlx::query(
        "INSERT INTO wh.decoder_output (decoder, genome, family, snapshot, timescale, instrument_id, as_of, output, recorded_at)
         SELECT * FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::text[], $6::text[],
                              $7::timestamptz[], $8::jsonb[], $9::timestamptz[])",
    )
    .bind(col(|o| o.decoder_id.clone()))
    .bind(col(|o| o.genome_id.clone()))
    .bind(col(|o| o.family.clone()))
    .bind(col(|o| o.snapshot_id.clone()))
    .bind(col(|o| o.timescale.clone()))
    .bind(inst)
    .bind(as_of)
    .bind(output)
    .bind(recorded)
    .execute(pool)
    .await?;
    Ok(r.rows_affected())
}

/// Deletes rows whose `as_of` is older than `days`.
pub async fn prune(pool: &PgPool, days: i32) -> sqlx::Result<u64> {
    let r = sqlx::query("DELETE FROM wh.decoder_output WHERE as_of < now() - make_interval(days => $1)")
        .bind(days)
        .execute(pool)
        .await?;
    Ok(r.rows_affected())
}

type Row = (i64, String, String, String, String, String, Option<String>, DateTime<Utc>, Value, DateTime<Utc>);

fn view_of(r: Row) -> (usize, DecoderOutputView) {
    let (idx, decoder_id, genome_id, family, snapshot_id, timescale, inst, as_of, output, recorded_at) = r;
    let strings = |k: &str| -> Vec<String> {
        output[k]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
            .unwrap_or_default()
    };
    let view = DecoderOutputView {
        received_at: recorded_at,
        decoder_id,
        genome_id,
        family,
        snapshot_id,
        timescale,
        as_of: Some(as_of),
        instrument_id: inst.unwrap_or_default(),
        output_refs: strings("output_refs"),
        quantiles: output["quantiles"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default(),
        analogs: output["analogs"].as_array().cloned().unwrap_or_default(),
    };
    (usize::try_from(idx).unwrap_or(0), view)
}

/// Stored outputs of each episode (same order as `eps`), newest first, at most `per_episode`
/// each. Outputs without a snapshot match any snapshot of their timescale.
pub async fn for_episodes(
    pool: &PgPool,
    eps: &[EpisodeKey],
    per_episode: usize,
) -> sqlx::Result<Vec<Vec<DecoderOutputView>>> {
    let mut out = vec![Vec::new(); eps.len()];
    if eps.is_empty() {
        return Ok(out);
    }
    let ms = |t: i64| DateTime::from_timestamp_millis(t).unwrap_or_default();
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT e.idx - 1, d.decoder, d.genome, d.family, d.snapshot, d.timescale, d.instrument_id, d.as_of, d.output, d.recorded_at
         FROM unnest($1::text[], $2::text[], $3::text[], $4::timestamptz[], $5::timestamptz[])
              WITH ORDINALITY AS e(snap, ts, inst, t0, t1, idx)
         CROSS JOIN LATERAL (
             SELECT * FROM wh.decoder_output d
             WHERE d.snapshot IN (e.snap, '') AND d.timescale = e.ts AND d.instrument_id = e.inst
               AND d.as_of BETWEEN e.t0 AND e.t1
             ORDER BY d.as_of DESC
             LIMIT $6
         ) d
         ORDER BY e.idx, d.as_of DESC",
    )
    .bind(eps.iter().map(|e| e.snapshot_id.clone()).collect::<Vec<_>>())
    .bind(eps.iter().map(|e| e.timescale.clone()).collect::<Vec<_>>())
    .bind(eps.iter().map(|e| e.instrument_id.clone()).collect::<Vec<_>>())
    .bind(eps.iter().map(|e| ms(e.t0_ms)).collect::<Vec<_>>())
    .bind(eps.iter().map(|e| ms(e.t1_ms)).collect::<Vec<_>>())
    .bind(i64::try_from(per_episode).unwrap_or(i64::MAX))
    .fetch_all(pool)
    .await?;
    for r in rows {
        let (i, v) = view_of(r);
        if let Some(slot) = out.get_mut(i) {
            slot.push(v);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(inst: &str, snap: &str, as_of_ms: i64) -> DecoderOutputView {
        DecoderOutputView {
            received_at: DateTime::from_timestamp_millis(as_of_ms).unwrap(),
            decoder_id: "d1".into(),
            genome_id: "g1".into(),
            family: "retrieval".into(),
            snapshot_id: snap.into(),
            timescale: "1m".into(),
            as_of: DateTime::from_timestamp_millis(as_of_ms),
            instrument_id: inst.into(),
            output_refs: vec![format!("inst:{inst}")],
            quantiles: vec![0.1, 0.5],
            analogs: vec![json!({ "window_id": "w1", "similarity": 0.9 })],
        }
    }

    #[test]
    fn expert_space_ids_name_their_snapshot() {
        assert_eq!(snapshot_of_space("stat-v1-1m-e0"), Some("stat-v1-1m"));
        assert_eq!(snapshot_of_space("snap-e12"), Some("snap"));
        assert_eq!(snapshot_of_space("snap-ex"), None);
        assert_eq!(snapshot_of_space("-e1"), None);
        assert_eq!(snapshot_of_space("nope"), None);
    }

    #[test]
    fn stored_rows_round_trip_into_views() {
        let v = view("i1", "s", 5_000);
        let row: Row = (
            2,
            v.decoder_id.clone(),
            v.genome_id.clone(),
            v.family.clone(),
            v.snapshot_id.clone(),
            v.timescale.clone(),
            Some("i1".into()),
            v.as_of.unwrap(),
            output_json(&v),
            v.received_at,
        );
        let (i, back) = view_of(row);
        assert_eq!(i, 2);
        assert_eq!(serde_json::to_value(&back).unwrap(), serde_json::to_value(&v).unwrap());
    }

    #[test]
    fn episode_match_needs_instrument_timescale_snapshot_and_window() {
        let v = view("i1", "s", 5_000);
        assert!(v.in_episode("s", "1m", "i1", 1_000, 5_000));
        assert!(!v.in_episode("s", "1m", "i1", 5_001, 9_000), "outside the window");
        assert!(!v.in_episode("s", "1h", "i1", 1_000, 9_000), "other timescale");
        assert!(!v.in_episode("t", "1m", "i1", 1_000, 9_000), "other snapshot");
        assert!(!v.in_episode("s", "1m", "i2", 1_000, 9_000), "other instrument");
        assert!(view("i1", "", 5_000).in_episode("t", "1m", "i1", 1_000, 9_000), "unbound output");
        assert!(!view("", "s", 5_000).in_episode("s", "1m", "", 1_000, 9_000), "market-wide output");
    }

    /// Needs `DATABASE_URL` with migration 20261114090000 applied; skipped otherwise.
    #[tokio::test]
    async fn writes_and_joins_on_postgres() {
        let Ok(url) = std::env::var("DATABASE_URL") else { return };
        let pool = sqlx::postgres::PgPoolOptions::new().max_connections(2).connect(&url).await.unwrap();
        if sqlx::query("SELECT 1 FROM wh.decoder_output LIMIT 1").execute(&pool).await.is_err() {
            return;
        }
        let inst = format!("test-{}", uuid::Uuid::now_v7());
        let t = Utc::now().timestamp_millis();
        let rows = [view(&inst, "s", t - 3_000), view(&inst, "", t - 2_000), view(&inst, "s", t - 1_000)];
        assert_eq!(insert(&pool, &rows).await.unwrap(), 3);
        let key = |t0, t1| EpisodeKey {
            snapshot_id: "s".into(),
            timescale: "1m".into(),
            instrument_id: inst.clone(),
            t0_ms: t0,
            t1_ms: t1,
        };
        let got =
            for_episodes(&pool, &[key(t - 10_000, t), key(t - 2_500, t - 1_500), key(t + 1, t + 5)], 2).await.unwrap();
        assert_eq!(got.iter().map(Vec::len).collect::<Vec<_>>(), [2, 1, 0]);
        assert_eq!(got[0][0].as_of.unwrap().timestamp_millis(), t - 1_000, "newest first");
        assert_eq!(got[1][0].snapshot_id, "");
        sqlx::query("DELETE FROM wh.decoder_output WHERE instrument_id = $1").bind(&inst).execute(&pool).await.unwrap();
    }
}
