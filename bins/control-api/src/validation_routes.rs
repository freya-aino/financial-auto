//! Validation and population routes (spec 50 §4.6, spec 41 §4; D-052), read from Postgres
//! (`validation.holdout`, `validation.trial`, `validation.generation_stats`, migration
//! `20261110000030_validation.sql`; `ledger.strategy_state` for the shadow view).
//!
//! | Route | Source |
//! |---|---|
//! | `GET /api/validation/holdout` | locked holdout rows per timescale |
//! | `GET /api/validation/reports/{candidate}` | trial rows of one candidate (genome text or `gp-…` strategy id) + its latest objectives |
//! | `GET /api/validation/shadow/{strategy}` | shadow / lifecycle row of the risk-engine (`ledger.strategy_state`); 503 while the table is missing |
//! | `GET /api/evolution/populations` | one row per generation: kind, loops, trials, latest deflated Sharpe / PBO |
//! | `GET /api/evolution/populations/{id}/pareto?gen=` | Pareto fronts of a generation loop recomputed from the latest trial per candidate (`fa_evolution`) |
//!
//! The pool is `AppState::pg` (set by `ledger_routes::install`); every route answers 503 while
//! it is unset.

use std::collections::BTreeMap;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use chrono::{DateTime, Utc};
use fa_evolution::{crowding_distance, non_dominated_sort, objectives};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::{PgPool, Row};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/validation/holdout", get(holdout))
        .route("/api/validation/reports/{candidate}", get(reports))
        .route("/api/validation/shadow/{strategy}", get(shadow))
        .route("/api/evolution/populations", get(populations))
        .route("/api/evolution/populations/{id}/pareto", get(pareto))
}

fn pool(s: &AppState) -> Result<PgPool, Box<Response>> {
    s.pg.get().cloned().ok_or_else(|| {
        Box::new((StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "postgres unavailable"}))).into_response())
    })
}

fn db_err(e: sqlx::Error) -> Response {
    if let sqlx::Error::Database(d) = &e
        && d.code().as_deref() == Some("42P01")
    {
        // undefined_table: the owning migration has not been applied yet
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": format!("table not available yet: {}", d.message()), "hint": "restart control-api so pending migrations apply"})),
        )
            .into_response();
    }
    (StatusCode::BAD_GATEWAY, Json(json!({"error": format!("postgres: {e}")}))).into_response()
}

/// Objective directions per job kind (the trial rows store objectives only).
pub fn directions_for(kind: &str) -> BTreeMap<String, String> {
    let pairs: &[(&str, &str)] = if kind.ends_with(".strategy") {
        &fa_backtest::replay::DIRECTIONS
    } else {
        // spec 34 §6.3 as reported by evo-worker (D-044)
        &[
            ("crps", "min"),
            ("crps_skill", "max"),
            ("coverage_err", "min"),
            ("hit_rate", "max"),
            ("trade_sharpe", "max"),
            ("trade_psr", "max"),
            ("compute_cost", "min"),
        ]
    };
    pairs.iter().map(|(k, d)| ((*k).to_owned(), (*d).to_owned())).collect()
}

/// Pareto fronts over `candidates` (`(id, objectives)`), returned as `fronts[rank] = [{candidate, rank, crowding, objectives}]`.
pub fn fronts_of(candidates: &[(String, BTreeMap<String, f64>)], kind: &str) -> (Vec<String>, Vec<Vec<Value>>) {
    let dirs = directions_for(kind);
    let keys = objectives::ranked_keys(&dirs);
    let ranked: Vec<&(String, BTreeMap<String, f64>)> =
        candidates.iter().filter(|(_, o)| keys.iter().any(|k| o.contains_key(k))).collect();
    if ranked.is_empty() {
        return (keys, Vec::new());
    }
    let vecs: Vec<Vec<f64>> = ranked.iter().map(|(_, o)| objectives::minimised(o, &dirs, &keys)).collect();
    let fronts = non_dominated_sort(&vecs)
        .into_iter()
        .enumerate()
        .map(|(rank, front)| {
            let cd = crowding_distance(&vecs, &front);
            front
                .iter()
                .zip(cd)
                .map(|(&i, c)| {
                    json!({"candidate": ranked[i].0, "rank": rank,
                           "crowding": if c.is_finite() { json!(c) } else { json!("inf") },
                           "objectives": ranked[i].1})
                })
                .collect()
        })
        .collect();
    (keys, fronts)
}

fn objectives_of(v: &Value) -> BTreeMap<String, f64> {
    v.as_object()
        .map(|m| m.iter().filter_map(|(k, v)| v.as_f64().map(|x| (k.clone(), x))).collect())
        .unwrap_or_default()
}

// ------------------------------------------------------------------ routes

async fn holdout(State(s): State<AppState>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let rows = match sqlx::query(
        "select timescale, start_at, end_at, locked_at, note from validation.holdout order by timescale, start_at",
    )
    .fetch_all(&pool)
    .await
    {
        Ok(r) => r,
        Err(e) => return db_err(e),
    };
    let holdouts: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "timescale": r.get::<String, _>("timescale"),
                "start": r.get::<DateTime<Utc>, _>("start_at"),
                "end": r.get::<DateTime<Utc>, _>("end_at"),
                "locked_at": r.get::<DateTime<Utc>, _>("locked_at"),
                "note": r.get::<Option<String>, _>("note"),
            })
        })
        .collect();
    Json(json!({"holdouts": holdouts, "immutable": true, "note": "selection replays intersecting a holdout are rejected (spec 41 §6.4)"}))
        .into_response()
}

#[derive(Debug, Deserialize)]
pub struct ReportQuery {
    pub limit: Option<i64>,
}

/// `candidate` is the genome text (URL-encoded) or a strategy id (`gp-…`), matched against the
/// inline-table prefix `{ id = "<id>"`.
async fn reports(State(s): State<AppState>, Path(candidate): Path<String>, Query(q): Query<ReportQuery>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let limit = q.limit.unwrap_or(50).clamp(1, 500);
    let rows = match sqlx::query(
        "select id, generation, gen_index, rung, job_id, candidate, kind, window_start, window_end, sharpe, n_obs, objectives, created_at \
         from validation.trial where candidate = $1 or candidate like ('{ id = \"' || $1 || '\"%') \
         order by created_at desc limit $2",
    )
    .bind(&candidate)
    .bind(limit)
    .fetch_all(&pool)
    .await
    {
        Ok(r) => r,
        Err(e) => return db_err(e),
    };
    if rows.is_empty() {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "no trial for candidate", "candidate": candidate})))
            .into_response();
    }
    let trials: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "id": r.get::<uuid::Uuid, _>("id"), "generation": r.get::<String, _>("generation"),
                "gen_index": r.get::<i32, _>("gen_index"), "rung": r.get::<i32, _>("rung"),
                "job_id": r.get::<String, _>("job_id"), "kind": r.get::<String, _>("kind"),
                "window_start": r.get::<Option<DateTime<Utc>>, _>("window_start"),
                "window_end": r.get::<Option<DateTime<Utc>>, _>("window_end"),
                "sharpe": r.get::<Option<f64>, _>("sharpe"), "n_obs": r.get::<i32, _>("n_obs"),
                "objectives": r.get::<Value, _>("objectives"), "created_at": r.get::<DateTime<Utc>, _>("created_at"),
                "report_ref": format!("fa-artifacts/evals/{}/{}.json", r.get::<String, _>("generation"), r.get::<String, _>("job_id")),
            })
        })
        .collect();
    let latest = &rows[0];
    Json(json!({
        "candidate": latest.get::<String, _>("candidate"),
        "kind": latest.get::<String, _>("kind"),
        "n_trials": trials.len(),
        "latest": {"objectives": latest.get::<Value, _>("objectives"), "sharpe": latest.get::<Option<f64>, _>("sharpe"),
                   "n_obs": latest.get::<i32, _>("n_obs"), "generation": latest.get::<String, _>("generation"),
                   "created_at": latest.get::<DateTime<Utc>, _>("created_at")},
        "directions": directions_for(&latest.get::<String, _>("kind")),
        "trials": trials,
    }))
    .into_response()
}

/// Shadow / lifecycle state of a strategy (`ledger.strategy_state`, owned by the risk-engine
/// workstream). 503 with a clear message while the table does not exist.
async fn shadow(State(s): State<AppState>, Path(strategy): Path<String>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let exists = sqlx::query("select to_regclass('ledger.strategy_state') is not null as ok").fetch_one(&pool).await;
    match exists {
        Ok(r) if r.get::<bool, _>("ok") => {}
        Ok(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "shadow tracking not available: ledger.strategy_state does not exist yet (risk-engine lifecycle migration pending)", "strategy": strategy})),
            )
                .into_response();
        }
        Err(e) => return db_err(e),
    }
    let row = match sqlx::query("select row_to_json(s) as row from ledger.strategy_state s where strategy_id = $1")
        .bind(&strategy)
        .fetch_optional(&pool)
        .await
    {
        Ok(r) => r,
        Err(e) => return db_err(e),
    };
    match row {
        Some(r) => Json(json!({"strategy": strategy, "state": r.get::<Value, _>("row")})).into_response(),
        None => (StatusCode::NOT_FOUND, Json(json!({"error": "strategy not tracked", "strategy": strategy})))
            .into_response(),
    }
}

async fn populations(State(s): State<AppState>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let rows = match sqlx::query(
        "with t as (select generation, kind, max(gen_index) as loops, count(*) as trials, count(distinct candidate) as candidates, \
                           min(created_at) as first_at, max(created_at) as last_at \
                    from validation.trial group by generation, kind), \
              g as (select distinct on (generation) generation, gen_index, n_trials, deflated_sharpe, pbo, pbo_trials, best_candidate, \
                           best_sharpe, sharpe_var, computed_at \
                    from validation.generation_stats order by generation, gen_index desc) \
         select t.generation, t.kind, t.loops, t.trials, t.candidates, t.first_at, t.last_at, g.gen_index as stats_loop, \
                g.n_trials, g.deflated_sharpe, g.pbo, g.pbo_trials, g.best_candidate, g.best_sharpe, g.sharpe_var, g.computed_at \
         from t left join g using (generation) order by t.last_at desc limit 200",
    )
    .fetch_all(&pool)
    .await
    {
        Ok(r) => r,
        Err(e) => return db_err(e),
    };
    let pops: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "id": r.get::<String, _>("generation"), "kind": r.get::<String, _>("kind"),
                "loops": r.get::<i32, _>("loops") + 1, "trials": r.get::<i64, _>("trials"),
                "candidates": r.get::<i64, _>("candidates"),
                "first_at": r.get::<DateTime<Utc>, _>("first_at"), "last_at": r.get::<DateTime<Utc>, _>("last_at"),
                "stats": r.get::<Option<i32>, _>("stats_loop").map(|loop_| json!({
                    "loop": loop_, "n_trials": r.get::<Option<i32>, _>("n_trials"),
                    "deflated_sharpe": r.get::<Option<f64>, _>("deflated_sharpe"), "pbo": r.get::<Option<f64>, _>("pbo"),
                    "pbo_trials": r.get::<Option<i32>, _>("pbo_trials"), "best_candidate": r.get::<Option<String>, _>("best_candidate"),
                    "best_sharpe": r.get::<Option<f64>, _>("best_sharpe"), "sharpe_var": r.get::<Option<f64>, _>("sharpe_var"),
                    "computed_at": r.get::<Option<DateTime<Utc>>, _>("computed_at"),
                })),
            })
        })
        .collect();
    Json(json!({"populations": pops})).into_response()
}

#[derive(Debug, Deserialize)]
pub struct ParetoQuery {
    /// Loop index inside the generation; default = the latest.
    #[serde(rename = "gen")]
    pub loop_index: Option<i32>,
}

async fn pareto(State(s): State<AppState>, Path(id): Path<String>, Query(q): Query<ParetoQuery>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let gen_index = match q.loop_index {
        Some(g) => g,
        None => match sqlx::query("select max(gen_index) as g from validation.trial where generation = $1")
            .bind(&id)
            .fetch_one(&pool)
            .await
        {
            Ok(r) => match r.get::<Option<i32>, _>("g") {
                Some(g) => g,
                None => {
                    return (StatusCode::NOT_FOUND, Json(json!({"error": "unknown generation", "id": id})))
                        .into_response();
                }
            },
            Err(e) => return db_err(e),
        },
    };
    // latest trial per candidate in that loop (the last rung it reached)
    let rows = match sqlx::query(
        "select distinct on (candidate) candidate, kind, rung, sharpe, n_obs, objectives, created_at \
         from validation.trial where generation = $1 and gen_index = $2 order by candidate, created_at desc",
    )
    .bind(&id)
    .bind(gen_index)
    .fetch_all(&pool)
    .await
    {
        Ok(r) => r,
        Err(e) => return db_err(e),
    };
    if rows.is_empty() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "no trials for generation loop", "id": id, "gen": gen_index})),
        )
            .into_response();
    }
    let kind = rows[0].get::<String, _>("kind");
    let cands: Vec<(String, BTreeMap<String, f64>)> = rows
        .iter()
        .map(|r| (r.get::<String, _>("candidate"), objectives_of(&r.get::<Value, _>("objectives"))))
        .collect();
    let (keys, fronts) = fronts_of(&cands, &kind);
    let stats = match sqlx::query(
        "select row_to_json(g) as row from validation.generation_stats g where generation = $1 and gen_index = $2",
    )
    .bind(&id)
    .bind(gen_index)
    .fetch_optional(&pool)
    .await
    {
        Ok(r) => r.map(|r| r.get::<Value, _>("row")),
        Err(e) => return db_err(e),
    };
    Json(json!({
        "id": id, "gen": gen_index, "kind": kind, "candidates": cands.len(),
        "objective_keys": keys, "directions": directions_for(&kind),
        "front_sizes": fronts.iter().map(Vec::len).collect::<Vec<_>>(),
        "pareto_front": fronts.first().cloned().unwrap_or_default(),
        "fronts": fronts,
        "trials": rows.iter().map(|r| json!({
            "candidate": r.get::<String, _>("candidate"), "rung": r.get::<i32, _>("rung"),
            "sharpe": r.get::<Option<f64>, _>("sharpe"), "n_obs": r.get::<i32, _>("n_obs"),
            "created_at": r.get::<DateTime<Utc>, _>("created_at"),
        })).collect::<Vec<_>>(),
        "stats": stats,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use fa_core::config::PlatformConfig;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn app() -> Router {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        router().with_state(AppState::new(cfg, format!("{root}/sources.yaml").into(), None))
    }

    #[tokio::test]
    async fn routes_answer_503_without_postgres() {
        for uri in [
            "/api/validation/holdout",
            "/api/validation/reports/gp-1",
            "/api/validation/shadow/baseline-momentum-1s",
            "/api/evolution/populations",
            "/api/evolution/populations/G1/pareto?gen=0",
        ] {
            let res = app().oneshot(Request::get(uri).body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE, "{uri}");
            let body = res.into_body().collect().await.unwrap().to_bytes();
            let v: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(v["error"], "postgres unavailable");
        }
    }

    #[test]
    fn fronts_follow_directions_per_kind() {
        let o = |s: f64, dd: f64| BTreeMap::from([("deflated_sharpe".to_owned(), s), ("max_drawdown".to_owned(), dd)]);
        let cands = vec![("a".to_owned(), o(0.9, 0.1)), ("b".to_owned(), o(0.5, 0.2)), ("c".to_owned(), o(0.95, 0.3))];
        let (keys, fronts) = fronts_of(&cands, "evo.eval.strategy");
        assert!(keys.contains(&"deflated_sharpe".to_owned()));
        assert_eq!(fronts.len(), 2);
        let front0: Vec<&str> = fronts[0].iter().map(|m| m["candidate"].as_str().unwrap()).collect();
        assert_eq!(front0, vec!["a", "c"], "b is dominated by a");
        assert_eq!(fronts[0][0]["crowding"], "inf");
        assert_eq!(directions_for("evo.eval.decoder")["crps"], "min");
        // candidates without any ranked objective are left out
        let (_, none) = fronts_of(&[("x".to_owned(), BTreeMap::new())], "evo.eval.decoder");
        assert!(none.is_empty());
    }
}
