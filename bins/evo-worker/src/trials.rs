//! Trial registry and per-generation validation statistics in Postgres (spec 41 §5, §6.5;
//! migration `20261110000030_validation.sql`).
//!
//! Every evaluation the worker finishes (decoder or strategy) becomes one `validation.trial`
//! row: generation id (`Job.parent_job_id`), loop index and rung from the job envelope, the
//! candidate text, its per-step Sharpe, observation count, objective map and the returns
//! compressed into `fa_backtest::validation::N_BLOCKS` block sums. After the insert the worker
//! recomputes `fa_backtest::validation::generation_stats` over the generation's trial set,
//! upserts `validation.generation_stats` and returns the stats so they ride along in the
//! `JobResult` JSON (`generation_stats`), where the orchestrator folds them into
//! `generation.summary`. Locked holdouts (`validation.holdout`) are read per timescale before a
//! Selection replay. Holdout evaluations (`evo.eval.strategy.holdout`, D-059) are rows with
//! `kind = holdout` and a `holdout_id`; they are **excluded** from every generation statistic
//! ([`TrialStore::trials`] filters them), so promotion evidence never feeds selection.
//!
//! Everything here is best effort: a failed write only warns (the evaluation result is still
//! published) and a missing pool means no registry.

use std::time::Duration;

use chrono::{DateTime, TimeZone, Utc};
use fa_backtest::{
    replay::Holdout,
    validation::{GenerationStats, N_BLOCKS, PBO_BLOCKS, Trial, block_sums, generation_stats},
};
use fa_core::Timescale;
use serde_json::Value;
use sqlx::{PgPool, Row, postgres::PgPoolOptions};

/// One registry row.
#[derive(Debug, Clone)]
pub struct TrialRow {
    pub generation: String,
    pub gen_index: i32,
    pub rung: i32,
    pub job_id: String,
    pub candidate: String,
    pub kind: String,
    pub window_start_ms: i64,
    pub window_end_ms: i64,
    pub sharpe: f64,
    pub n_obs: usize,
    pub objectives: Value,
    /// Per-step net returns (compressed on write).
    pub returns: Vec<f64>,
    /// Locked holdout this row was evaluated on (`kind = holdout` only).
    pub holdout_id: Option<String>,
}

/// A locked holdout row with its id (`validation.holdout.id`, migration `20261110000050`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoldoutRow {
    pub id: String,
    pub holdout: Holdout,
    pub note: Option<String>,
}

#[derive(Clone)]
pub struct TrialStore {
    pool: PgPool,
}

fn ts(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms).single().unwrap_or_else(Utc::now)
}

impl TrialStore {
    /// Connects with a short timeout; `None` when Postgres is unreachable or not configured.
    pub async fn connect(url: &str) -> Option<Self> {
        if url.is_empty() {
            return None;
        }
        let connect = PgPoolOptions::new().max_connections(2).acquire_timeout(Duration::from_secs(3)).connect(url);
        match tokio::time::timeout(Duration::from_secs(5), connect).await {
            Ok(Ok(pool)) => Some(Self { pool }),
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "postgres unavailable; trial registry disabled");
                None
            }
            Err(_) => {
                tracing::warn!("postgres connect timed out; trial registry disabled");
                None
            }
        }
    }

    /// Appends an `audit.log` row (spec 42 §6, append-only) as `system:evo-worker`.
    pub async fn audit(&self, action: &str, subject: Option<&str>, payload: &Value) -> Result<(), sqlx::Error> {
        sqlx::query("INSERT INTO audit.log (actor, action, subject, payload) VALUES ($1, $2, $3, $4)")
            .bind(crate::holdout::AUDIT_ACTOR)
            .bind(action)
            .bind(subject)
            .bind(payload)
            .execute(&self.pool)
            .await
            .map(|_| ())
    }

    /// `validation.*` tables exist (control-api applies migrations on start).
    pub async fn ready(&self) -> bool {
        sqlx::query("select 1 from validation.trial limit 0").execute(&self.pool).await.is_ok()
    }

    /// Inserts a trial (idempotent on `job_id`).
    pub async fn record(&self, t: &TrialRow) -> Result<(), sqlx::Error> {
        let returns: Vec<f64> = if t.returns.is_empty() { Vec::new() } else { block_sums(&t.returns, N_BLOCKS) };
        sqlx::query(
            "insert into validation.trial \
             (generation, gen_index, rung, job_id, candidate, kind, window_start, window_end, sharpe, n_obs, objectives, returns, holdout_id) \
             values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::uuid) \
             on conflict (job_id) do update set sharpe = excluded.sharpe, n_obs = excluded.n_obs, \
             objectives = excluded.objectives, returns = excluded.returns, holdout_id = excluded.holdout_id",
        )
        .bind(&t.generation)
        .bind(t.gen_index)
        .bind(t.rung)
        .bind(&t.job_id)
        .bind(&t.candidate)
        .bind(&t.kind)
        .bind(ts(t.window_start_ms))
        .bind(ts(t.window_end_ms))
        .bind(if t.sharpe.is_finite() { Some(t.sharpe) } else { None })
        .bind(i32::try_from(t.n_obs).unwrap_or(i32::MAX))
        .bind(&t.objectives)
        .bind(&returns)
        .bind(&t.holdout_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// All selection trials of a generation (every loop and rung), oldest first. Holdout rows
    /// (`kind = holdout`) never take part in selection statistics.
    pub async fn trials(&self, generation: &str) -> Result<Vec<Trial>, sqlx::Error> {
        let rows = sqlx::query(
            "select candidate, sharpe, n_obs, returns from validation.trial \
             where generation = $1 and kind <> 'holdout' order by created_at",
        )
        .bind(generation)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| Trial {
                candidate: r.get::<String, _>("candidate"),
                sharpe: r.get::<Option<f64>, _>("sharpe").unwrap_or(f64::NAN),
                n_obs: usize::try_from(r.get::<i32, _>("n_obs")).unwrap_or(0),
                returns: r.get::<Option<Vec<f64>>, _>("returns").unwrap_or_default(),
            })
            .collect())
    }

    /// Block returns of the generation's current elite set for the diversification objective
    /// (spec 35 §6.2): the best `k` other candidates of `kind` by deflated Sharpe whose trial
    /// window overlaps `[start, end]` by ≥ 90 % (so their blocks are time-aligned), one row per
    /// candidate (its newest such trial).
    pub async fn elite_returns(
        &self,
        generation: &str,
        kind: &str,
        exclude_candidate: &str,
        window: (i64, i64),
        k: usize,
    ) -> Result<Vec<Vec<f64>>, sqlx::Error> {
        let len_s = ((window.1 - window.0).max(1) as f64) / 1000.0;
        let rows = sqlx::query(
            "select distinct on (candidate) candidate, returns, \
                    coalesce((objectives->>'deflated_sharpe')::float8, 0) as dsr \
             from validation.trial \
             where generation = $1 and kind = $2 and candidate <> $3 and returns is not null \
               and extract(epoch from (least(window_end, $5) - greatest(window_start, $4))) >= 0.9 * $6 \
             order by candidate, created_at desc",
        )
        .bind(generation)
        .bind(kind)
        .bind(exclude_candidate)
        .bind(ts(window.0))
        .bind(ts(window.1))
        .bind(len_s)
        .fetch_all(&self.pool)
        .await?;
        let mut v: Vec<(f64, Vec<f64>)> = rows
            .iter()
            .map(|r| (r.get::<f64, _>("dsr"), r.get::<Option<Vec<f64>>, _>("returns").unwrap_or_default()))
            .filter(|(_, r)| !r.is_empty())
            .collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        Ok(v.into_iter().take(k).map(|x| x.1).collect())
    }

    /// `(n_trials, sharpe variance)` of the generation so far: the deflated-Sharpe inputs of
    /// the next evaluation.
    pub async fn deflation_inputs(&self, generation: &str) -> (usize, f64) {
        match self.trials(generation).await {
            Ok(t) => {
                let s = generation_stats(&t, PBO_BLOCKS);
                (s.n_trials, s.sharpe_var)
            }
            Err(e) => {
                tracing::debug!(error = %e, "deflation inputs unavailable");
                (0, 0.0)
            }
        }
    }

    /// Recomputes and upserts the generation statistics; returns them.
    pub async fn refresh_generation_stats(
        &self,
        generation: &str,
        gen_index: i32,
        kind: &str,
    ) -> Result<GenerationStats, sqlx::Error> {
        let trials = self.trials(generation).await?;
        let s = generation_stats(&trials, PBO_BLOCKS);
        let f = |x: f64| if x.is_finite() { Some(x) } else { None };
        sqlx::query(
            "insert into validation.generation_stats \
             (generation, gen_index, kind, n_trials, sharpe_mean, sharpe_var, best_candidate, best_sharpe, \
              expected_max_sharpe, deflated_sharpe, pbo, pbo_trials, pbo_blocks, effective_trials, computed_at) \
             values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, now()) \
             on conflict (generation, gen_index) do update set kind = excluded.kind, n_trials = excluded.n_trials, \
             sharpe_mean = excluded.sharpe_mean, sharpe_var = excluded.sharpe_var, best_candidate = excluded.best_candidate, \
             best_sharpe = excluded.best_sharpe, expected_max_sharpe = excluded.expected_max_sharpe, \
             deflated_sharpe = excluded.deflated_sharpe, pbo = excluded.pbo, pbo_trials = excluded.pbo_trials, \
             pbo_blocks = excluded.pbo_blocks, effective_trials = excluded.effective_trials, computed_at = now()",
        )
        .bind(generation)
        .bind(gen_index)
        .bind(kind)
        .bind(i32::try_from(s.n_trials).unwrap_or(i32::MAX))
        .bind(f(s.sharpe_mean))
        .bind(f(s.sharpe_var))
        .bind(&s.best_candidate)
        .bind(f(s.best_sharpe))
        .bind(f(s.expected_max_sharpe))
        .bind(f(s.deflated_sharpe))
        .bind(f(s.pbo))
        .bind(i32::try_from(s.pbo_trials).unwrap_or(i32::MAX))
        .bind(i32::try_from(s.pbo_blocks).unwrap_or(16))
        .bind(f(s.effective_trials))
        .execute(&self.pool)
        .await?;
        Ok(s)
    }

    /// One locked holdout by id; `None` when unknown (or not a UUID).
    pub async fn holdout_by_id(&self, id: &str) -> Result<Option<HoldoutRow>, sqlx::Error> {
        if uuid_like(id).is_none() {
            return Ok(None);
        }
        let row = sqlx::query(
            "select id::text as id, timescale, start_at, end_at, locked_at, note, (to_jsonb(h) ->> 'embargo_ms')::bigint as embargo_ms \
             from validation.holdout h where id = $1::uuid",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.and_then(|r| {
            let timescale: Timescale = r.get::<String, _>("timescale").parse().ok()?;
            Some(HoldoutRow {
                id: r.get::<String, _>("id"),
                holdout: Holdout {
                    timescale,
                    start_ms: r.get::<DateTime<Utc>, _>("start_at").timestamp_millis(),
                    end_ms: r.get::<DateTime<Utc>, _>("end_at").timestamp_millis(),
                    locked_at_ms: r.get::<DateTime<Utc>, _>("locked_at").timestamp_millis(),
                    embargo_ms: r.get::<Option<i64>, _>("embargo_ms"),
                },
                note: r.get::<Option<String>, _>("note"),
            })
        }))
    }

    /// Latest genome text evaluated under a strategy id (`gp-…`, matched against the inline
    /// table prefix `{ id = "<id>"`), or the text itself when it already is one.
    pub async fn candidate_text(&self, candidate: &str) -> Result<Option<String>, sqlx::Error> {
        if candidate.trim_start().starts_with('{') {
            return Ok(Some(candidate.to_owned()));
        }
        let row = sqlx::query(
            "select candidate from validation.trial where candidate like ('{ id = \"' || $1 || '\"%') \
             order by created_at desc limit 1",
        )
        .bind(candidate)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| r.get::<String, _>("candidate")))
    }

    /// Locked holdouts of a timescale, each with the embargo it was locked with
    /// (`embargo_ms`, migration 20261113000040; read through `to_jsonb` so an unmigrated table
    /// gives NULL = the timescale default instead of failing the lookup — a failed lookup
    /// would disable the holdout check).
    pub async fn holdouts(&self, timescale: Timescale) -> Result<Vec<Holdout>, sqlx::Error> {
        let rows = sqlx::query(
            "select start_at, end_at, locked_at, (to_jsonb(h) ->> 'embargo_ms')::bigint as embargo_ms \
             from validation.holdout h where timescale = $1 order by start_at",
        )
        .bind(timescale.as_str())
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| Holdout {
                timescale,
                start_ms: r.get::<DateTime<Utc>, _>("start_at").timestamp_millis(),
                end_ms: r.get::<DateTime<Utc>, _>("end_at").timestamp_millis(),
                locked_at_ms: r.get::<DateTime<Utc>, _>("locked_at").timestamp_millis(),
                embargo_ms: r.get::<Option<i64>, _>("embargo_ms"),
            })
            .collect())
    }
}

/// Loose UUID check (36 chars, dashes in place) so a bad id is a clean "not found".
pub fn uuid_like(s: &str) -> Option<&str> {
    let b = s.as_bytes();
    (b.len() == 36
        && b.iter().enumerate().all(|(i, c)| match i {
            8 | 13 | 18 | 23 => *c == b'-',
            _ => c.is_ascii_hexdigit(),
        }))
    .then_some(s)
}

/// Records a holdout trial **without** touching the generation statistics (promotion evidence
/// never feeds selection). Returns whether the row was written.
pub async fn register_holdout(store: Option<&TrialStore>, row: &TrialRow) -> bool {
    let Some(store) = store else { return false };
    match tokio::time::timeout(Duration::from_secs(15), store.record(row)).await {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            tracing::warn!(error = %e, job = %row.job_id, "holdout trial write failed");
            false
        }
        Err(_) => {
            tracing::warn!(job = %row.job_id, "holdout trial write timed out");
            false
        }
    }
}

/// Records the trial and refreshes the generation stats; never fails the job. Returns the stats
/// JSON (`null` when no registry is configured or the write failed).
pub async fn register(store: Option<&TrialStore>, row: &TrialRow) -> Value {
    let Some(store) = store else { return Value::Null };
    let work = async {
        store.record(row).await?;
        store.refresh_generation_stats(&row.generation, row.gen_index, &row.kind).await
    };
    match tokio::time::timeout(Duration::from_secs(15), work).await {
        Ok(Ok(s)) => serde_json::to_value(&s).unwrap_or(Value::Null),
        Ok(Err(e)) => {
            tracing::warn!(error = %e, job = %row.job_id, "trial registry write failed");
            Value::Null
        }
        Err(_) => {
            tracing::warn!(job = %row.job_id, "trial registry write timed out");
            Value::Null
        }
    }
}

/// Live Postgres round trip; run with `FA_PG_TEST=1 cargo test -p evo-worker` against the
/// compose database (migrations applied by control-api).
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn live_registry_round_trip() {
        if std::env::var("FA_PG_TEST").ok().as_deref() != Some("1") {
            eprintln!("skipped: set FA_PG_TEST=1 (Postgres at FA_PG_URL or postgres://fa:fa@127.0.0.1:5432/fa)");
            return;
        }
        let url = std::env::var("FA_PG_URL").unwrap_or_else(|_| "postgres://fa:fa@127.0.0.1:5432/fa".into());
        let store = TrialStore::connect(&url).await.expect("connect");
        assert!(store.ready().await, "validation schema missing: start control-api once to migrate");
        let generation = format!("test-{}", ulid::Ulid::generate());
        for i in 0..3 {
            let returns: Vec<f64> = (0..200).map(|k| ((k + i) as f64 * 0.37).sin() * 0.001).collect();
            let row = TrialRow {
                generation: generation.clone(),
                gen_index: 0,
                rung: i,
                job_id: format!("{generation}-{i}"),
                candidate: format!("c{i}"),
                kind: "evo.eval.strategy".into(),
                window_start_ms: 0,
                window_end_ms: 1_000_000,
                sharpe: 0.01 * f64::from(i),
                n_obs: 200,
                objectives: serde_json::json!({"sharpe": 0.01 * f64::from(i)}),
                returns,
                holdout_id: None,
            };
            let v = register(Some(&store), &row).await;
            assert_eq!(v["n_trials"], serde_json::json!(i + 1), "{v}");
        }
        // a holdout row of the same generation is written but never counted
        let hold_row = TrialRow {
            job_id: format!("{generation}-holdout"),
            kind: "holdout".into(),
            holdout_id: None,
            rung: 0,
            ..TrialRow {
                generation: generation.clone(),
                gen_index: 0,
                rung: 0,
                job_id: String::new(),
                candidate: "c0".into(),
                kind: String::new(),
                window_start_ms: 0,
                window_end_ms: 1,
                sharpe: 9.0,
                n_obs: 10,
                objectives: serde_json::json!({}),
                returns: vec![0.0; 10],
                holdout_id: None,
            }
        };
        assert!(register_holdout(Some(&store), &hold_row).await);
        assert!(store.holdout_by_id("not-a-uuid").await.unwrap().is_none());
        assert_eq!(store.candidate_text("{ id = \"x\" }").await.unwrap().as_deref(), Some("{ id = \"x\" }"));
        assert!(store.candidate_text("gp-does-not-exist").await.unwrap().is_none());
        let (n, var) = store.deflation_inputs(&generation).await;
        assert_eq!(n, 3, "holdout rows do not feed selection statistics");
        assert!(var > 0.0);
        let trials = store.trials(&generation).await.unwrap();
        assert_eq!(trials.len(), 3);
        assert_eq!(trials[0].returns.len(), N_BLOCKS);
        let hold = store.holdouts(Timescale::Second).await.unwrap();
        assert!(hold.iter().all(|h| h.end_ms > h.start_ms));
        let _ = sqlx::query("delete from validation.trial where generation = $1")
            .bind(&generation)
            .execute(&store.pool)
            .await;
        let _ = sqlx::query("delete from validation.generation_stats where generation = $1")
            .bind(&generation)
            .execute(&store.pool)
            .await;
    }
}
