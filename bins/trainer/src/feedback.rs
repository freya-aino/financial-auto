//! Decoder feedback inputs for `generation` jobs (spec 34 §6.6, spec 36 §6.3), read from the
//! Postgres trial registry (D-052): the latest `validation.generation_stats` row (recorded as
//! `manifest.feedback.decoder_stats`) and the decoder skill reported for the parent snapshot
//! ([`parent_skill`]), which [`crate::search`] uses as the tie-breaker of its proxy.

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

/// Decoder skill reported for `snapshot_id` (the tie-breaker of [`crate::search`]):
/// 1. mean `objectives.crps_skill` of `validation.trial` rows whose `objectives.snapshot_id`
///    is the parent (decoders evaluated on its latents) — attributable, `keyed = true`;
/// 2. else the latest `validation.generation_stats.deflated_sharpe` — not attributable to a
///    snapshot today (trials do not record one), `keyed = false`;
/// 3. else none (the tie-breaker weight is then 0).
pub async fn parent_skill(pg_url: Option<&str>, snapshot_id: Option<&str>) -> crate::search::ParentSkill {
    use crate::search::ParentSkill;
    let none =
        |why: &str| ParentSkill { source: "none".into(), detail: json!({ "reason": why }), ..Default::default() };
    let Some(url) = pg_url else { return none("no postgres url") };
    let pool = match PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(std::time::Duration::from_secs(3))
        .connect(url)
        .await
    {
        Ok(p) => p,
        Err(e) => return none(&format!("postgres unreachable: {e}")),
    };
    if let Some(id) = snapshot_id {
        let keyed = sqlx::query(
            "select avg((objectives->>'crps_skill')::float8) as skill, count(*) as n \
             from validation.trial where objectives->>'snapshot_id' = $1 \
             and objectives ? 'crps_skill' and holdout_id is null",
        )
        .bind(id)
        .fetch_optional(&pool)
        .await;
        if let Ok(Some(row)) = keyed {
            let n: i64 = row.try_get("n").unwrap_or(0);
            let v: Option<f64> = row.try_get("skill").ok().flatten();
            if n > 0 && v.is_some() {
                return ParentSkill {
                    value: v,
                    source: "validation.trial".into(),
                    keyed: true,
                    n_trials: n,
                    detail: json!({ "objective": "crps_skill", "snapshot_id": id }),
                };
            }
        }
    }
    let row = sqlx::query(
        "select generation, n_trials, deflated_sharpe, computed_at::text as at \
         from validation.generation_stats where deflated_sharpe is not null order by computed_at desc limit 1",
    )
    .fetch_optional(&pool)
    .await;
    match row {
        Ok(Some(r)) => ParentSkill {
            value: r.try_get::<Option<f64>, _>("deflated_sharpe").ok().flatten(),
            source: "validation.generation_stats".into(),
            keyed: false,
            n_trials: i64::from(r.try_get::<i32, _>("n_trials").unwrap_or(0)),
            detail: json!({
                "objective": "deflated_sharpe",
                "generation": r.try_get::<String, _>("generation").ok(),
                "computed_at": r.try_get::<String, _>("at").ok(),
                "note": "latest evolution generation, not attributable to the parent snapshot",
            }),
        },
        Ok(None) => none("no trials for the parent and no generation_stats rows"),
        Err(e) => none(&format!("validation tables unreadable: {e}")),
    }
}
