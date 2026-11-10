//! Decoder feedback for `generation` jobs (spec 34 §6.6, spec 36 §6.3): the latest
//! `validation.generation_stats` row of the Postgres trial registry (D-052). The statistical
//! family has no loss to add it to, so the summary is only recorded in the manifest
//! (`feedback`) — the hook the neural families will consume.

use serde_json::{Value, json};
use sqlx::{Row, postgres::PgPoolOptions};

/// Latest generation statistics, or `None` when Postgres / the table is unavailable.
pub async fn latest_generation_stats(pg_url: &str) -> Option<Value> {
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(std::time::Duration::from_secs(3))
        .connect(pg_url)
        .await
        .map_err(|e| tracing::warn!(error = %e, "postgres unreachable; generation runs without decoder feedback"))
        .ok()?;
    let row = sqlx::query(
        "select generation, gen_index, kind, n_trials, sharpe_mean, sharpe_var, best_candidate, best_sharpe, \
         expected_max_sharpe, deflated_sharpe, pbo, pbo_trials, pbo_blocks, computed_at::text \
         from validation.generation_stats order by computed_at desc limit 1",
    )
    .fetch_optional(&pool)
    .await
    .map_err(|e| tracing::warn!(error = %e, "validation.generation_stats unreadable; no decoder feedback"))
    .ok()??;
    let f = |c: &str| row.try_get::<Option<f64>, _>(c).ok().flatten();
    Some(json!({
        "source": "postgres:validation.generation_stats",
        "generation": row.try_get::<String, _>("generation").ok(),
        "gen_index": row.try_get::<i32, _>("gen_index").ok(),
        "kind": row.try_get::<String, _>("kind").ok(),
        "n_trials": row.try_get::<i32, _>("n_trials").ok(),
        "sharpe_mean": f("sharpe_mean"),
        "sharpe_var": f("sharpe_var"),
        "best_candidate": row.try_get::<Option<String>, _>("best_candidate").ok().flatten(),
        "best_sharpe": f("best_sharpe"),
        "expected_max_sharpe": f("expected_max_sharpe"),
        "deflated_sharpe": f("deflated_sharpe"),
        "pbo": f("pbo"),
        "pbo_trials": row.try_get::<i32, _>("pbo_trials").ok(),
        "pbo_blocks": row.try_get::<i32, _>("pbo_blocks").ok(),
        "computed_at": row.try_get::<String, _>("computed_at").ok(),
    }))
}
