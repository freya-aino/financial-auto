//! Scheduled reference-data maintenance (spec 12 §7 `calendar.refresh`,
//! `adjustment.materialize_daily`), run beside the Helix reference mirror because control-api
//! already holds the Postgres pool and is off the trading path.
//!
//! Every `FA_REFDATA_MAINT_SECS` (default 86 400 = daily; `0` disables), once Postgres is up:
//!
//! - **Calendar refresh:** `fa_warehouse::calendar::refresh` materialises the next 400 days of
//!   every venue with a built-in session rule (24/7 venues excluded). Idempotent: unchanged
//!   sessions keep their `recorded_at`, venue-published rows are never overwritten. When rows
//!   changed, `ref.changed.calendar` is announced and audited (`ref.calendar.refreshed`).
//! - **Adjusted daily bars:** `fa_warehouse::adjusted::refresh` rewrites `wh.adjusted_bar_1d`
//!   (both modes) for every instrument with splits or dividends, from the raw daily bars in
//!   VictoriaMetrics (`fa_bar_*{tf="1d"}` over the last two years at an hourly step, so each bar
//!   keeps the trading day of its close).
//!
use std::time::Duration;

use chrono::Utc;
use fa_warehouse::{adjusted, calendar, corpactions::BarPoint};
use serde_json::json;
use sqlx::PgPool;
use uuid::Uuid;

use crate::state::AppState;

const EVERY_ENV: &str = "FA_REFDATA_MAINT_SECS";
const DEFAULT_EVERY_SECS: u64 = 86_400;
/// Raw daily history read per instrument for the adjusted series.
const HISTORY_DAYS: i64 = 730;
const ACTOR: &str = "system:control-api";

pub fn spawn(state: &AppState) {
    let every = std::env::var(EVERY_ENV).ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(DEFAULT_EVERY_SECS);
    if every == 0 {
        tracing::info!("refdata maintenance disabled ({EVERY_ENV}=0)");
        return;
    }
    let s = state.clone();
    tokio::spawn(async move {
        let tsdb = fa_tsdb::Tsdb::new(s.cfg.victoriametrics.url.clone());
        loop {
            let Some(pool) = s.pg.get().cloned() else {
                tokio::time::sleep(Duration::from_secs(30)).await;
                continue;
            };
            run_once(&s, &pool, &tsdb).await;
            tokio::time::sleep(Duration::from_secs(every)).await;
        }
    });
}

async fn run_once(s: &AppState, pool: &PgPool, tsdb: &fa_tsdb::Tsdb) {
    match calendar::refresh(pool, Utc::now().date_naive(), calendar::REFRESH_HORIZON_DAYS).await {
        Ok(venues) => {
            let changed: Vec<String> = venues.iter().filter(|v| v.rows > 0).map(|v| v.venue.clone()).collect();
            tracing::info!(venues = venues.len(), changed = ?changed, "calendar refresh");
            if !changed.is_empty() {
                let payload = json!({ "venues": venues });
                if let Err(e) = crate::audit::append(
                    pool,
                    ACTOR,
                    "ref.calendar.refreshed",
                    changed.first().map(String::as_str),
                    payload,
                )
                .await
                {
                    tracing::warn!(error = %e, "audit append failed");
                }
                if let Some(nats) = &s.nats {
                    let n = fa_bus::ingest::ref_changed_notice("calendar", changed, "refreshed", ACTOR);
                    if let Err(e) =
                        nats.publish(fa_bus::subjects::ref_changed("calendar"), prost_encode(&n).into()).await
                    {
                        tracing::warn!(error = %e, "ref.changed publish failed");
                    }
                }
            }
        }
        Err(e) => tracing::warn!(error = %e, "calendar refresh failed"),
    }
    match adjusted::refresh(pool, |inst| daily_bars(tsdb, inst)).await {
        Ok(rep) => tracing::info!(?rep, "adjusted daily bars materialised"),
        Err(e) => tracing::warn!(error = %e, "adjusted daily bars not materialised"),
    }
}

fn prost_encode(n: &fa_proto::v1::RefChanged) -> Vec<u8> {
    use fa_proto::Message;
    n.encode_to_vec()
}

/// Raw daily bars of one instrument from VictoriaMetrics (longest series when several venues
/// carry it), dated by trading day.
async fn daily_bars(tsdb: &fa_tsdb::Tsdb, inst: Uuid) -> Result<Vec<BarPoint>, String> {
    let token = fa_tsdb::inst_token(&inst.to_string());
    let end = Utc::now().timestamp();
    let start = end - HISTORY_DAYS * 86_400;
    let mut results: [Vec<fa_tsdb::RangeSeries>; 6] = Default::default();
    for (col, metric) in fa_tsdb::metrics::BAR_FAMILY.iter().enumerate() {
        let q = format!("last_over_time({metric}{{inst=\"{token}\",tf=\"1d\"}}[1h])");
        results[col] = tsdb.query_range(&q, start, end, "1h").await.map_err(|e| e.to_string())?;
    }
    let Some(series) = fa_tsdb::join_bar_series(results).into_iter().next() else { return Ok(Vec::new()) };
    Ok(series
        .points
        .iter()
        .filter_map(|p| adjusted::daily_point(p.ts_ms, p.open, p.high, p.low, p.close, p.volume))
        .collect())
}
