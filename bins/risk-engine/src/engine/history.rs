//! Risk-chain history (spec 40 §5): allocation snapshots, netted portfolio targets before and
//! after the risk chain, and promotion-gate reviews, persisted to `ledger.allocation_snapshot`,
//! `ledger.portfolio_target` and `ledger.promotion_review` (migration `20261114070000`).
//!
//! The tick hands every decision to [`Engine::record_tick`], which keeps one when it matters:
//! an order was sent, the active weights moved by more than [`WEIGHT_EPS`], a gated target moved
//! by more than the engine's `min_order_eur`, or `[history] snapshot_secs` passed since the last
//! kept record. Kept records queue (bounded, oldest dropped and counted) until the I/O layer
//! drains them ([`Engine::take_history`]) and writes them in one transaction ([`write`]); rows
//! older than `[history] retention_days` are pruned hourly ([`prune`]). Promotion reviews are
//! written by the I/O layer whenever the gate decides ([`write_review`]): every promote attempt
//! (operator or pipeline, pass or refusal), every change of a shadow strategy's gate outcome
//! and hourly while unchanged ([`ReviewDedup`]).
use std::collections::{BTreeMap, VecDeque};

use chrono::{DateTime, Utc};
use fa_core::InstrumentId;
use fa_ledger::money::dec;
use fa_risk::strategy::Checklist;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::PgPool;
use uuid::Uuid;

use super::Engine;

/// A weight change below this does not make a new allocation snapshot.
pub const WEIGHT_EPS: f64 = 1e-4;
/// Kept records waiting for the writer.
const MAX_QUEUE: usize = 2_000;

/// `[history]` in `risk.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HistorySettings {
    pub enabled: bool,
    /// A record is kept at least this often even when nothing changed.
    pub snapshot_secs: u64,
    /// Allocation snapshots and targets older than this are deleted (reviews are kept).
    pub retention_days: u32,
    /// A shadow strategy whose gate outcome is unchanged is reviewed at most this often.
    pub review_unchanged_secs: u64,
}

impl Default for HistorySettings {
    fn default() -> Self {
        Self { enabled: true, snapshot_secs: 60, retention_days: 30, review_unchanged_secs: 3_600 }
    }
}

/// One instrument's netted target through the chain.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TargetRow {
    pub instrument: InstrumentId,
    pub raw_eur: f64,
    pub capped_eur: f64,
    pub gated_eur: f64,
    pub current_eur: f64,
    /// Signed EUR contribution per strategy (allocation × exposure, before netting).
    pub contributors: BTreeMap<String, f64>,
    /// `trade` | `skip:<reason>` | `none` (no mark)
    pub decision: String,
    pub intent_id: Option<String>,
}

/// One decision tick worth keeping.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TickRecord {
    pub at_ms: i64,
    pub allocator: String,
    pub spec: Value,
    pub weights: BTreeMap<String, f64>,
    pub equity_eur: f64,
    pub deployable_eur: f64,
    pub fallback: bool,
    pub targets: Vec<TargetRow>,
}

/// What the last kept record looked like (change detection).
#[derive(Debug, Default)]
pub struct HistoryState {
    last_at_ms: Option<i64>,
    last_weights: BTreeMap<String, f64>,
    last_gated: BTreeMap<InstrumentId, f64>,
    queue: VecDeque<TickRecord>,
}

impl Engine {
    /// Offers a tick's decisions to the history; keeps it when something changed or the
    /// snapshot interval passed.
    pub(super) fn record_tick(&mut self, rec: TickRecord) {
        let cfg = &self.settings.history;
        if !cfg.enabled {
            return;
        }
        let h = &mut self.history;
        let traded = rec.targets.iter().any(|t| t.decision == "trade");
        let weights_moved = rec.weights.len() != h.last_weights.len()
            || rec.weights.iter().any(|(k, w)| h.last_weights.get(k).is_none_or(|o| (o - w).abs() > WEIGHT_EPS));
        let band = self.settings.min_order_eur.max(1.0);
        let targets_moved = rec.targets.iter().any(|t| {
            h.last_gated.get(&t.instrument).is_none_or(|o| (o - t.gated_eur).abs() > band)
        }) || h.last_gated.keys().any(|i| !rec.targets.iter().any(|t| t.instrument == *i));
        let due = h.last_at_ms.is_none_or(|t| rec.at_ms - t >= cfg.snapshot_secs as i64 * 1000);
        if !(traded || weights_moved || targets_moved || due) {
            return;
        }
        h.last_at_ms = Some(rec.at_ms);
        h.last_weights = rec.weights.clone();
        h.last_gated = rec.targets.iter().map(|t| (t.instrument, t.gated_eur)).collect();
        h.queue.push_back(rec);
        if h.queue.len() > MAX_QUEUE {
            h.queue.pop_front();
            *self.counters.entry("history.dropped").or_insert(0.0) += 1.0;
        }
        *self.counters.entry("history.kept").or_insert(0.0) += 1.0;
    }

    /// Kept records, oldest first (drained).
    pub fn take_history(&mut self) -> Vec<TickRecord> {
        self.history.queue.drain(..).collect()
    }
}

fn at(ms: i64) -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp_millis(ms).unwrap_or_default()
}

/// Writes kept records (one transaction; idempotent on `(book, at[, instrument])`).
pub async fn write(pool: &PgPool, book: &str, recs: &[TickRecord]) -> Result<usize, sqlx::Error> {
    let mut tx = pool.begin().await?;
    let mut n = 0;
    for r in recs {
        let sleeves: BTreeMap<&String, f64> = r.weights.iter().map(|(k, w)| (k, w * r.deployable_eur)).collect();
        n += sqlx::query(
            "INSERT INTO ledger.allocation_snapshot \
             (book, at, allocator, spec, weights, sleeves_eur, equity_eur, deployable_eur, fallback) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) ON CONFLICT DO NOTHING",
        )
        .bind(book)
        .bind(at(r.at_ms))
        .bind(&r.allocator)
        .bind(&r.spec)
        .bind(json!(r.weights))
        .bind(json!(sleeves))
        .bind(dec(r.equity_eur))
        .bind(dec(r.deployable_eur))
        .bind(r.fallback)
        .execute(&mut *tx)
        .await?
        .rows_affected() as usize;
        for t in &r.targets {
            sqlx::query(
                "INSERT INTO ledger.portfolio_target \
                 (book, at, instrument_id, raw_eur, capped_eur, gated_eur, current_eur, contributors, decision, intent_id) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) ON CONFLICT DO NOTHING",
            )
            .bind(book)
            .bind(at(r.at_ms))
            .bind(t.instrument.0)
            .bind(dec(t.raw_eur))
            .bind(dec(t.capped_eur))
            .bind(dec(t.gated_eur))
            .bind(dec(t.current_eur))
            .bind(json!(t.contributors))
            .bind(&t.decision)
            .bind(&t.intent_id)
            .execute(&mut *tx)
            .await?;
        }
    }
    tx.commit().await?;
    Ok(n)
}

/// Deletes allocation snapshots and targets older than `retention_days`.
pub async fn prune(pool: &PgPool, book: &str, retention_days: u32) -> Result<u64, sqlx::Error> {
    let cut = Utc::now() - chrono::Duration::days(i64::from(retention_days.max(1)));
    let a = sqlx::query("DELETE FROM ledger.allocation_snapshot WHERE book = $1 AND at < $2")
        .bind(book)
        .bind(cut)
        .execute(pool)
        .await?
        .rows_affected();
    let t = sqlx::query("DELETE FROM ledger.portfolio_target WHERE book = $1 AND at < $2")
        .bind(book)
        .bind(cut)
        .execute(pool)
        .await?
        .rows_affected();
    Ok(a + t)
}

/// One promotion-gate review.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Review<'a> {
    pub strategy_id: &'a str,
    pub at_ms: i64,
    /// `promote` (operator) | `pipeline` (automatic promote) | `gate_change` | `gate_hourly`
    pub trigger: &'a str,
    pub actor: &'a str,
    pub status_before: Option<&'a str>,
    pub status_after: Option<&'a str>,
    pub checklist: &'a Checklist,
    pub evidence: Value,
}

/// Appends a review to `ledger.promotion_review`.
pub async fn write_review(pool: &PgPool, book: &str, r: &Review<'_>) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO ledger.promotion_review \
         (id, book, strategy_id, at, trigger, actor, status_before, status_after, passed, missing, checklist, evidence) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
    )
    .bind(Uuid::now_v7())
    .bind(book)
    .bind(r.strategy_id)
    .bind(at(r.at_ms))
    .bind(r.trigger)
    .bind(r.actor)
    .bind(r.status_before)
    .bind(r.status_after)
    .bind(r.checklist.green)
    .bind(json!(r.checklist.missing))
    .bind(serde_json::to_value(r.checklist).unwrap_or_default())
    .bind(&r.evidence)
    .execute(pool)
    .await?;
    Ok(())
}

/// Decides whether a periodic gate evaluation of a shadow strategy is worth a review row: its
/// outcome (green and the set of failing items) changed, or `every_ms` passed.
#[derive(Debug, Default)]
pub struct ReviewDedup {
    last: BTreeMap<String, (String, i64)>,
}

impl ReviewDedup {
    /// `Some(trigger)` when a review should be written now.
    pub fn due(&mut self, id: &str, c: &Checklist, now_ms: i64, every_ms: i64) -> Option<&'static str> {
        let sig = format!("{}|{}", c.green, c.missing.join(","));
        let trigger = match self.last.get(id) {
            Some((s, _)) if *s != sig => "gate_change",
            Some((_, t)) if now_ms - t < every_ms => return None,
            Some(_) => "gate_hourly",
            None => "gate_change",
        };
        self.last.insert(id.to_owned(), (sig, now_ms));
        Some(trigger)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reviews_are_written_on_change_and_hourly() {
        let mut d = ReviewDedup::default();
        let red = Checklist { green: false, items: vec![], missing: vec!["min_shadow_days".into()] };
        let green = Checklist { green: true, items: vec![], missing: vec![] };
        assert_eq!(d.due("s", &red, 0, 3_600_000), Some("gate_change"));
        assert_eq!(d.due("s", &red, 60_000, 3_600_000), None);
        assert_eq!(d.due("s", &green, 120_000, 3_600_000), Some("gate_change"));
        assert_eq!(d.due("s", &green, 3_720_000, 3_600_000), Some("gate_hourly"));
    }
}
