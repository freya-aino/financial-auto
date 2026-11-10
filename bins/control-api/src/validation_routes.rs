//! Validation and population routes (spec 50 §4.6, spec 41 §4; D-052), read from Postgres
//! (`validation.holdout`, `validation.trial`, `validation.generation_stats`, migration
//! `20261110000030_validation.sql`; `ledger.strategy_state` for the shadow view).
//!
//! | Route | Source |
//! |---|---|
//! | `GET /api/validation/holdout` | locked holdout rows per timescale (with their `id`) |
//! | `POST /api/validation/holdout` | locks a holdout `{timescale, start, end, note}` (D-059): 409 when it overlaps a locked range of the timescale, 400 on a bad range; immutable afterwards (trigger); audited `validation.holdout.lock` |
//! | `POST /api/validation/holdout/{id}/evaluate` | the promotion-evaluation request (spec 41 §6.4): enqueues one `evo.eval.strategy.holdout` job `{candidate, snapshot?, budget_ms?, seed?}` on `jobs.evo.eval.strategy.holdout` for a locked holdout; 404 unknown holdout, 503 without NATS; audited `validation.holdout.evaluate`; the result lands in `reports/{candidate}.holdout` |
//! | `GET /api/validation/reports/{candidate}` | selection trial rows of one candidate (genome text or `gp-…` strategy id) + its latest objectives; holdout evaluations (`kind = holdout`) listed separately under `holdout` |
//! | `GET /api/validation/shadow/{strategy}?min_horizons=` | shadow / lifecycle row of the risk-engine (`ledger.strategy_state`) plus `consistency` = realised shadow Sharpe (fill-marked, from `ledger.fill_attribution`) vs the latest backtest Sharpe (spec 40 §6.6 item 4); 503 while the table is missing |
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
    routing::{get, post},
};
use chrono::{DateTime, TimeZone, Utc};
use fa_evolution::{crowding_distance, non_dominated_sort, objectives};
use fa_proto::Message;
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::{PgPool, Row};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/validation/holdout", get(holdout).post(lock_holdout))
        .route("/api/validation/holdout/{id}/evaluate", post(evaluate_holdout))
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
    let pairs: &[(&str, &str)] = if kind.ends_with(".strategy") || kind == "holdout" {
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
        "select id::text as id, timescale, start_at, end_at, locked_at, note from validation.holdout order by timescale, start_at",
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
                "id": r.get::<String, _>("id"),
                "timescale": r.get::<String, _>("timescale"),
                "start": r.get::<DateTime<Utc>, _>("start_at"),
                "end": r.get::<DateTime<Utc>, _>("end_at"),
                "locked_at": r.get::<DateTime<Utc>, _>("locked_at"),
                "note": r.get::<Option<String>, _>("note"),
            })
        })
        .collect();
    Json(json!({"holdouts": holdouts, "immutable": true, "note": "selection replays intersecting a holdout are rejected (spec 41 §6.4); promotion replays run only inside one (evo.eval.strategy.holdout)"}))
        .into_response()
}

/// Job kind of a promotion replay (evo-worker `holdout::KIND`).
pub const HOLDOUT_JOB_KIND: &str = "evo.eval.strategy.holdout";

/// `POST /api/validation/holdout/{id}/evaluate` body.
#[derive(Debug, Deserialize)]
pub struct EvaluateHoldout {
    /// Genome text (inline TOML table) or a `gp-…` strategy id known to the trial registry.
    pub candidate: String,
    #[serde(default)]
    pub snapshot: Option<String>,
    #[serde(default)]
    pub budget_ms: Option<u64>,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Builds the `Job` an evo-worker turns into a `Promotion` replay of `candidate` on holdout `id`.
pub fn holdout_job(id: &str, body: &EvaluateHoldout, job_id: &str) -> fa_proto::v1::Job {
    let spec = json!({"candidate": body.candidate.trim(), "holdout_id": id, "snapshot": body.snapshot});
    fa_proto::v1::Job {
        job_id: job_id.to_owned(),
        kind: HOLDOUT_JOB_KIND.into(),
        parent_job_id: format!("holdout-{id}"),
        priority: 0,
        snapshot_id: body.snapshot.clone().unwrap_or_default(),
        genome_id: body.candidate.trim().to_owned(),
        spec: serde_json::to_vec(&spec).unwrap_or_default(),
        input_refs: vec![],
        deadline: None,
        budget_gpu_ms: body.budget_ms.unwrap_or(0),
        checkpoint_ref: String::new(),
        seed: body.seed.unwrap_or(0),
    }
}

/// Requests a promotion evaluation of one candidate on one locked holdout: the only path to a
/// `ReplayPurpose::Promotion` replay (evo-worker refuses the purpose elsewhere). The job goes
/// on the `JOBS_EVO` work queue; its trial row (`kind = holdout`) shows up under
/// `GET /api/validation/reports/{candidate}` → `holdout`.
async fn evaluate_holdout(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<EvaluateHoldout>,
) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    if body.candidate.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "candidate is required"}))).into_response();
    }
    let row = match sqlx::query(
        "select id::text as id, timescale, start_at, end_at from validation.holdout where id::text = $1",
    )
    .bind(id.trim())
    .fetch_optional(&pool)
    .await
    {
        Ok(r) => r,
        Err(e) => return db_err(e),
    };
    let Some(row) = row else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "no locked holdout with that id", "id": id})))
            .into_response();
    };
    let Some(nats) = s.nats.clone() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "nats unavailable"}))).into_response();
    };
    let job_id = fa_core::JobId::new().0.to_string();
    let job = holdout_job(&row.get::<String, _>("id"), &body, &job_id);
    let subject = fa_bus::subjects::jobs(HOLDOUT_JOB_KIND);
    let js = fa_bus::async_nats::jetstream::new(nats);
    let published = match js.publish(subject.clone(), job.encode_to_vec().into()).await {
        Ok(ack) => ack.await,
        Err(e) => Err(e),
    };
    if let Err(e) = published {
        return (StatusCode::BAD_GATEWAY, Json(json!({"error": format!("publish on {subject}: {e}")}))).into_response();
    }
    let holdout = json!({
        "id": row.get::<String, _>("id"), "timescale": row.get::<String, _>("timescale"),
        "start": row.get::<DateTime<Utc>, _>("start_at"), "end": row.get::<DateTime<Utc>, _>("end_at"),
    });
    let audited = crate::config_routes::audit(
        &s,
        "validation.holdout.evaluate",
        &job_id,
        json!({"holdout": holdout, "candidate": body.candidate, "kind": HOLDOUT_JOB_KIND}),
    )
    .await;
    (
        StatusCode::ACCEPTED,
        Json(json!({
            "job_id": job_id, "kind": HOLDOUT_JOB_KIND, "subject": subject, "holdout": holdout,
            "candidate": body.candidate, "audited": audited,
            "result": "GET /api/validation/reports/{candidate} → holdout.trials once the evo-worker finishes",
        })),
    )
        .into_response()
}

/// `POST /api/validation/holdout` body. `start` / `end` are RFC 3339 strings or epoch ms.
#[derive(Debug, Deserialize)]
pub struct LockHoldout {
    pub timescale: String,
    pub start: Value,
    pub end: Value,
    #[serde(default)]
    pub note: Option<String>,
}

/// RFC 3339 string or epoch milliseconds.
pub fn parse_time(v: &Value) -> Option<DateTime<Utc>> {
    match v {
        Value::String(s) => DateTime::parse_from_rfc3339(s.trim()).ok().map(|t| t.with_timezone(&Utc)),
        Value::Number(n) => n.as_i64().and_then(|ms| Utc.timestamp_millis_opt(ms).single()),
        _ => None,
    }
}

/// Closed-interval overlap of two ranges.
pub fn overlaps(a_start: i64, a_end: i64, b_start: i64, b_end: i64) -> bool {
    a_start <= b_end && b_start <= a_end
}

/// Validates a lock request against the clock and the already locked ranges of the timescale.
/// `Err((status, message))`: 400 for a bad request, 409 for an overlap.
pub fn admit_lock(
    timescale: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    now: DateTime<Utc>,
    locked: &[(String, i64, i64)],
) -> Result<fa_core::Timescale, (StatusCode, String)> {
    let ts: fa_core::Timescale =
        timescale.parse().map_err(|_| (StatusCode::BAD_REQUEST, format!("unknown timescale `{timescale}`")))?;
    if end <= start {
        return Err((StatusCode::BAD_REQUEST, "end must be after start".into()));
    }
    if end > now {
        return Err((
            StatusCode::BAD_REQUEST,
            "a holdout is a completed segment: end must not be in the future".into(),
        ));
    }
    if let Some((id, s, e)) =
        locked.iter().find(|(_, s, e)| overlaps(start.timestamp_millis(), end.timestamp_millis(), *s, *e))
    {
        return Err((
            StatusCode::CONFLICT,
            format!(
                "overlaps locked holdout {id} [{}, {}] of timescale {ts}",
                Utc.timestamp_millis_opt(*s).single().map(|t| t.to_rfc3339()).unwrap_or_default(),
                Utc.timestamp_millis_opt(*e).single().map(|t| t.to_rfc3339()).unwrap_or_default()
            ),
        ));
    }
    Ok(ts)
}

/// Locks a holdout (spec 41 §6.4): the most recent completed segment the operator names.
/// Immutable afterwards (`validation.holdout_immutable` trigger); a re-lock is a new,
/// non-overlapping row. Audited as `validation.holdout.lock`.
async fn lock_holdout(State(s): State<AppState>, Json(body): Json<LockHoldout>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let (Some(start), Some(end)) = (parse_time(&body.start), parse_time(&body.end)) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "start and end must be RFC 3339 timestamps or epoch milliseconds"})),
        )
            .into_response();
    };
    let locked = match sqlx::query(
        "select id::text as id, start_at, end_at from validation.holdout where timescale = $1 order by start_at",
    )
    .bind(body.timescale.trim())
    .fetch_all(&pool)
    .await
    {
        Ok(rows) => rows
            .iter()
            .map(|r| {
                (
                    r.get::<String, _>("id"),
                    r.get::<DateTime<Utc>, _>("start_at").timestamp_millis(),
                    r.get::<DateTime<Utc>, _>("end_at").timestamp_millis(),
                )
            })
            .collect::<Vec<_>>(),
        Err(e) => return db_err(e),
    };
    let ts = match admit_lock(body.timescale.trim(), start, end, Utc::now(), &locked) {
        Ok(ts) => ts,
        Err((status, error)) => {
            return (status, Json(json!({"error": error, "timescale": body.timescale, "start": start, "end": end})))
                .into_response();
        }
    };
    let row = match sqlx::query(
        "insert into validation.holdout (timescale, start_at, end_at, note) values ($1, $2, $3, $4) \
         returning id::text as id, locked_at",
    )
    .bind(ts.as_str())
    .bind(start)
    .bind(end)
    .bind(&body.note)
    .fetch_one(&pool)
    .await
    {
        Ok(r) => r,
        Err(e) => return db_err(e),
    };
    let id = row.get::<String, _>("id");
    let locked_at = row.get::<DateTime<Utc>, _>("locked_at");
    let holdout = json!({
        "id": id, "timescale": ts, "start": start, "end": end, "locked_at": locked_at, "note": body.note,
    });
    let audited = crate::config_routes::audit(&s, "validation.holdout.lock", &id, holdout.clone()).await;
    tracing::info!(%id, %ts, %start, %end, "holdout locked");
    (StatusCode::CREATED, Json(json!({"holdout": holdout, "immutable": true, "audited": audited}))).into_response()
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
        "select id, generation, gen_index, rung, job_id, candidate, kind, window_start, window_end, sharpe, n_obs, objectives, created_at, \
                holdout_id::text as holdout_id \
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
    let row_json = |r: &sqlx::postgres::PgRow| {
        json!({
            "id": r.get::<uuid::Uuid, _>("id"), "generation": r.get::<String, _>("generation"),
            "gen_index": r.get::<i32, _>("gen_index"), "rung": r.get::<i32, _>("rung"),
            "job_id": r.get::<String, _>("job_id"), "kind": r.get::<String, _>("kind"),
            "window_start": r.get::<Option<DateTime<Utc>>, _>("window_start"),
            "window_end": r.get::<Option<DateTime<Utc>>, _>("window_end"),
            "sharpe": r.get::<Option<f64>, _>("sharpe"), "n_obs": r.get::<i32, _>("n_obs"),
            "objectives": r.get::<Value, _>("objectives"), "created_at": r.get::<DateTime<Utc>, _>("created_at"),
            "holdout_id": r.get::<Option<String>, _>("holdout_id"),
            "report_ref": format!("fa-artifacts/evals/{}/{}.json", r.get::<String, _>("generation"), r.get::<String, _>("job_id")),
        })
    };
    let is_holdout = |r: &sqlx::postgres::PgRow| r.get::<String, _>("kind") == "holdout";
    let trials: Vec<Value> = rows.iter().filter(|r| !is_holdout(r)).map(row_json).collect();
    let holdout_trials: Vec<Value> = rows.iter().filter(|r| is_holdout(r)).map(row_json).collect();
    let latest = rows.iter().find(|r| !is_holdout(r));
    let kind = latest.map_or_else(|| "evo.eval.strategy".to_owned(), |r| r.get::<String, _>("kind"));
    Json(json!({
        "candidate": rows[0].get::<String, _>("candidate"),
        "kind": kind,
        "n_trials": trials.len(),
        "latest": latest.map(|l| json!({"objectives": l.get::<Value, _>("objectives"), "sharpe": l.get::<Option<f64>, _>("sharpe"),
                   "n_obs": l.get::<i32, _>("n_obs"), "generation": l.get::<String, _>("generation"),
                   "created_at": l.get::<DateTime<Utc>, _>("created_at")})),
        "directions": directions_for(&kind),
        "trials": trials,
        // Promotion evidence (spec 41 §6.4): evaluated on a locked holdout, never part of selection.
        "holdout": {
            "n_trials": holdout_trials.len(),
            "latest": holdout_trials.first().cloned(),
            "trials": holdout_trials,
        },
    }))
    .into_response()
}

/// Spec 40 §6.6 placeholder: horizons a shadow strategy must have completed before its shadow
/// Sharpe is compared with the backtest.
pub const MIN_SHADOW_HORIZONS: i64 = 20;

#[derive(Debug, Deserialize)]
pub struct ShadowQuery {
    pub min_horizons: Option<i64>,
}

/// One attributed paper fill of a strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowFill {
    pub at_ms: i64,
    pub instrument: String,
    /// Signed quantity (buy +, sell −).
    pub qty: f64,
    /// Price in EUR.
    pub price_eur: f64,
    pub fee_eur: f64,
}

/// Per-bar returns of a strategy's attributed fills: positions are rebuilt from the fills, the
/// book is marked at each fill's own price (fill-marked: no market data here), and the equity
/// change per bar of `bar_ms` is divided by `capital` (the sleeve, or the peak gross notional
/// when the sleeve is unknown). Bars without a fill contribute 0.
pub fn shadow_step_returns(fills: &[ShadowFill], bar_ms: i64, capital: f64) -> Vec<f64> {
    if fills.len() < 2 || bar_ms <= 0 {
        return Vec::new();
    }
    let mut sorted: Vec<&ShadowFill> = fills.iter().collect();
    sorted.sort_by_key(|f| f.at_ms);
    let mut pos: BTreeMap<&str, (f64, f64)> = BTreeMap::new(); // instrument → (units, last price)
    let mut cash = 0.0;
    let mut peak_notional = 0.0f64;
    let equity = |cash: f64, pos: &BTreeMap<&str, (f64, f64)>| cash + pos.values().map(|(u, p)| u * p).sum::<f64>();
    let first_bar = sorted[0].at_ms.div_euclid(bar_ms);
    let last_bar = sorted[sorted.len() - 1].at_ms.div_euclid(bar_ms);
    let n_bars = usize::try_from(last_bar - first_bar + 1).unwrap_or(0).min(1_000_000);
    let mut per_bar_equity: Vec<Option<f64>> = vec![None; n_bars];
    for f in &sorted {
        cash -= f.qty * f.price_eur + f.fee_eur;
        let e = pos.entry(f.instrument.as_str()).or_insert((0.0, f.price_eur));
        e.0 += f.qty;
        e.1 = f.price_eur;
        peak_notional = peak_notional.max((e.0 * e.1).abs());
        let b = usize::try_from(f.at_ms.div_euclid(bar_ms) - first_bar).unwrap_or(0).min(n_bars - 1);
        per_bar_equity[b] = Some(equity(cash, &pos));
    }
    let cap = if capital > 0.0 { capital } else { peak_notional.max(1.0) };
    let mut out = Vec::with_capacity(n_bars);
    let mut prev = 0.0;
    for e in per_bar_equity {
        match e {
            Some(eq) => {
                out.push((eq - prev) / cap);
                prev = eq;
            }
            None => out.push(0.0),
        }
    }
    out
}

/// `timescale = "<ts>"` of a strategy genome's inline-table text.
pub fn timescale_of_candidate(text: &str) -> Option<fa_core::Timescale> {
    let i = text.find("timescale = \"")? + "timescale = \"".len();
    let rest = &text[i..];
    let j = rest.find('"')?;
    rest[..j].parse().ok()
}

/// Shadow / lifecycle state of a strategy (`ledger.strategy_state`, owned by the risk-engine
/// workstream) with the shadow-vs-backtest consistency statistic (spec 40 §6.6 item 4, spec 41
/// §6.6). 503 with a clear message while the table does not exist.
async fn shadow(State(s): State<AppState>, Path(strategy): Path<String>, Query(q): Query<ShadowQuery>) -> Response {
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
    let Some(row) = row else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "strategy not tracked", "strategy": strategy})))
            .into_response();
    };
    let state = row.get::<Value, _>("row");
    let consistency = shadow_consistency(&pool, &strategy, &state, q.min_horizons.unwrap_or(MIN_SHADOW_HORIZONS)).await;
    Json(json!({"strategy": strategy, "state": state, "consistency": consistency})).into_response()
}

/// `consistency{backtest_sharpe, shadow_sharpe, ratio, n_horizons}` for a shadow strategy with at
/// least `min_horizons` completed horizons; otherwise `available: false` with the reason.
async fn shadow_consistency(pool: &PgPool, strategy: &str, state: &Value, min_horizons: i64) -> Value {
    let n_horizons = state["horizons_completed"].as_i64().unwrap_or(0);
    let sleeve = state["sleeve_eur"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| state["sleeve_eur"].as_f64())
        .unwrap_or(0.0);
    let mut out = json!({
        "available": false, "n_horizons": n_horizons, "min_horizons": min_horizons,
        "source": "ledger.fill_attribution (fill-marked per-bar returns) vs latest validation.trial sharpe",
    });
    // latest selection trial of the strategy id
    let trial = sqlx::query(
        "select candidate, sharpe, n_obs, created_at from validation.trial \
         where candidate like ('{ id = \"' || $1 || '\"%') and kind = 'evo.eval.strategy' and sharpe is not null \
         order by created_at desc limit 1",
    )
    .bind(strategy)
    .fetch_optional(pool)
    .await;
    let trial = match trial {
        Ok(t) => t,
        Err(e) => {
            out["reason"] = json!(format!("trial registry: {e}"));
            return out;
        }
    };
    let Some(trial) = trial else {
        out["reason"] = json!("no backtest trial for the strategy id");
        return out;
    };
    let backtest_sharpe = trial.get::<Option<f64>, _>("sharpe").unwrap_or(f64::NAN);
    let timescale = timescale_of_candidate(&trial.get::<String, _>("candidate")).unwrap_or(fa_core::Timescale::Minute);
    out["backtest_sharpe"] = json!(backtest_sharpe);
    out["backtest_n_obs"] = json!(trial.get::<i32, _>("n_obs"));
    out["backtest_trial_at"] = json!(trial.get::<DateTime<Utc>, _>("created_at"));
    out["bar"] = json!(timescale);
    if n_horizons < min_horizons {
        out["reason"] = json!(format!("{n_horizons} of {min_horizons} shadow horizons completed"));
        return out;
    }
    let fills = sqlx::query(
        "select f.filled_at, oi.instrument_id::text as instrument, oi.side, a.qty::float8 as qty, \
                f.price::float8 as price, f.fx_rate_eur::float8 as fx, a.fee_share::float8 as fee \
         from ledger.fill_attribution a join ledger.fill f on f.id = a.fill_id \
         join ledger.venue_order vo on vo.client_order_id = f.client_order_id \
         join ledger.order_intent oi on oi.id = vo.intent_id \
         where a.strategy_id = $1 order by f.filled_at",
    )
    .bind(fa_ledger::pg::uuid_of(strategy))
    .fetch_all(pool)
    .await;
    let fills: Vec<ShadowFill> = match fills {
        Ok(rows) => rows
            .iter()
            .map(|r| {
                let sign = if r.get::<String, _>("side") == "sell" { -1.0 } else { 1.0 };
                ShadowFill {
                    at_ms: r.get::<DateTime<Utc>, _>("filled_at").timestamp_millis(),
                    instrument: r.get::<String, _>("instrument"),
                    qty: sign * r.get::<f64, _>("qty"),
                    price_eur: r.get::<f64, _>("price") * r.get::<f64, _>("fx"),
                    fee_eur: r.get::<f64, _>("fee"),
                }
            })
            .collect(),
        Err(e) => {
            out["reason"] = json!(format!("fills: {e}"));
            return out;
        }
    };
    out["n_fills"] = json!(fills.len());
    let bar_ms = timescale.duration().as_millis() as i64;
    let returns = shadow_step_returns(&fills, bar_ms, sleeve);
    if returns.len() < 3 {
        out["reason"] = json!(format!("{} attributed fills span {} bars; need ≥ 3", fills.len(), returns.len()));
        return out;
    }
    let shadow_sharpe = fa_backtest::stats::sharpe(&returns);
    let ratio = if backtest_sharpe.is_finite() && backtest_sharpe.abs() > 1e-12 {
        shadow_sharpe / backtest_sharpe
    } else {
        f64::NAN
    };
    out["available"] = json!(true);
    out["shadow_sharpe"] = json!(shadow_sharpe);
    out["n_shadow_obs"] = json!(returns.len());
    out["ratio"] = if ratio.is_finite() { json!(ratio) } else { Value::Null };
    out["capital_eur"] = json!(if sleeve > 0.0 { sleeve } else { f64::NAN });
    out
}

async fn populations(State(s): State<AppState>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let rows = match sqlx::query(
        "with t as (select generation, kind, max(gen_index) as loops, count(*) as trials, count(distinct candidate) as candidates, \
                           min(created_at) as first_at, max(created_at) as last_at \
                    from validation.trial where kind <> 'holdout' group by generation, kind), \
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
        None => match sqlx::query(
            "select max(gen_index) as g from validation.trial where generation = $1 and kind <> 'holdout'",
        )
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
         from validation.trial where generation = $1 and gen_index = $2 and kind <> 'holdout' order by candidate, created_at desc",
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
        let res = app()
            .oneshot(
                Request::post("/api/validation/holdout")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"timescale":"1s","start":0,"end":1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
        let res = app()
            .oneshot(
                Request::post("/api/validation/holdout/8a4bcdc7-0251-40c6-86cf-743862638987/evaluate")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"candidate":"gp-1"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
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
    fn holdout_job_names_candidate_and_holdout() {
        let body = EvaluateHoldout {
            candidate: " gp-1 ".into(),
            snapshot: Some("stat-v1-1s".into()),
            budget_ms: Some(5),
            seed: None,
        };
        let job = holdout_job("8a4bcdc7-0251-40c6-86cf-743862638987", &body, "J1");
        assert_eq!(
            (job.kind.as_str(), job.job_id.as_str(), job.budget_gpu_ms, job.seed),
            (HOLDOUT_JOB_KIND, "J1", 5, 0)
        );
        assert_eq!(job.parent_job_id, "holdout-8a4bcdc7-0251-40c6-86cf-743862638987");
        let spec: Value = serde_json::from_slice(&job.spec).unwrap();
        assert_eq!(
            spec,
            json!({"candidate": "gp-1", "holdout_id": "8a4bcdc7-0251-40c6-86cf-743862638987", "snapshot": "stat-v1-1s"})
        );
        assert_eq!(fa_bus::subjects::jobs(HOLDOUT_JOB_KIND), "jobs.evo.eval.strategy.holdout");
    }

    #[test]
    fn holdout_lock_admission_409_on_overlap_400_on_bad_range() {
        let t = |ms: i64| Utc.timestamp_millis_opt(ms).single().unwrap();
        let now = t(10_000_000);
        let locked = vec![("h1".to_owned(), 1_000_000, 2_000_000), ("h2".to_owned(), 5_000_000, 6_000_000)];
        assert_eq!(admit_lock("1m", t(2_000_001), t(4_999_999), now, &locked).unwrap(), fa_core::Timescale::Minute);
        let (code, msg) = admit_lock("1m", t(1_500_000), t(2_500_000), now, &locked).unwrap_err();
        assert_eq!(code, StatusCode::CONFLICT);
        assert!(msg.contains("overlaps locked holdout h1"), "{msg}");
        // touching the boundary counts as overlap (closed intervals)
        assert_eq!(admit_lock("1m", t(2_000_000), t(3_000_000), now, &locked).unwrap_err().0, StatusCode::CONFLICT);
        assert_eq!(
            admit_lock("1m", t(0), t(7_000_000), now, &locked).unwrap_err().0,
            StatusCode::CONFLICT,
            "spans both"
        );
        assert_eq!(admit_lock("1m", t(3_000_000), t(3_000_000), now, &locked).unwrap_err().0, StatusCode::BAD_REQUEST);
        assert_eq!(
            admit_lock("1m", t(9_000_000), t(11_000_000), now, &locked).unwrap_err().0,
            StatusCode::BAD_REQUEST,
            "future end"
        );
        assert_eq!(admit_lock("3m", t(0), t(1), now, &[]).unwrap_err().0, StatusCode::BAD_REQUEST, "unknown timescale");
        assert!(overlaps(0, 10, 10, 20) && !overlaps(0, 9, 10, 20));
        assert_eq!(parse_time(&json!("2026-11-10T06:00:00Z")).unwrap().timestamp_millis(), 1_794_290_400_000);
        assert_eq!(parse_time(&json!(1_794_290_400_000i64)).unwrap().to_rfc3339(), "2026-11-10T06:00:00+00:00");
        assert_eq!(parse_time(&json!(true)), None);
        assert_eq!(parse_time(&json!("yesterday")), None);
    }

    #[test]
    fn shadow_returns_are_fill_marked_per_bar() {
        let f = |at_ms: i64, inst: &str, qty: f64, px: f64| ShadowFill {
            at_ms,
            instrument: inst.into(),
            qty,
            price_eur: px,
            fee_eur: 1.0,
        };
        // buy 10 @ 100 in bar 0, sell 10 @ 110 in bar 2 (bar = 60 s): +99 − 1 fee at the exit, nothing in bar 1
        let fills = vec![f(10_000, "x", 10.0, 100.0), f(130_000, "x", -10.0, 110.0)];
        let r = shadow_step_returns(&fills, 60_000, 1_000.0);
        assert_eq!(r.len(), 3);
        assert!((r[0] - (-1.0 / 1_000.0)).abs() < 1e-12, "entry: only the fee moves equity (marked at the fill price)");
        assert_eq!(r[1], 0.0);
        assert!((r[2] - 99.0 / 1_000.0).abs() < 1e-12, "{r:?}");
        // unknown sleeve → peak gross notional (1 000 here) is the capital
        let r2 = shadow_step_returns(&fills, 60_000, 0.0);
        assert!((r2[2] - 99.0 / 1_000.0).abs() < 1e-12);
        assert!(shadow_step_returns(&fills[..1], 60_000, 1.0).is_empty());
        assert!(fa_backtest::stats::sharpe(&r) > 0.0);
        assert_eq!(
            timescale_of_candidate(r#"{ id = "gp-1", timescale = "1m", signal = "1" }"#),
            Some(fa_core::Timescale::Minute)
        );
        assert_eq!(timescale_of_candidate("forecast:h=1"), None);
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
        assert_eq!(directions_for("holdout")["deflated_sharpe"], "max", "holdout rows are strategy replays");
        // candidates without any ranked objective are left out
        let (_, none) = fronts_of(&[("x".to_owned(), BTreeMap::new())], "evo.eval.decoder");
        assert!(none.is_empty());
    }
}
