//! Locked-holdout exclusion of encoder fits (spec 36 §6.1, acceptance 7; spec 41 §6.4).
//!
//! Every fit window goes through [`load_fit_set`]: the bars are loaded as before
//! ([`crate::data::load_set`]), then every bar whose end falls in a locked holdout of the
//! timescale **or in the embargo before it** (`[start − embargo, end]`, the embargo the holdout
//! was locked with, else the timescale default, at least the fit's horizon) is removed. A
//! history that a holdout cuts in two becomes two segments of the same instrument (each needs
//! [`crate::data::MIN_BARS`]), so no training window, next-step target or probe straddles the
//! holdout. **Every** holdout row of the timescale is excluded, also older ones after a re-lock
//! (spec 36 Q2, decided: holdouts are fixed by default and re-locked only explicitly; a period that
//! was ever a holdout is never trained on).
//!
//! Fail-closed: when Postgres is configured but `validation.holdout` cannot be read, the fit
//! refuses (`FA_TRAIN_HOLDOUT_OPTIONAL=1` downgrades that to a warning, for offline use).
//! Bootstrap / reproduction runs that pass frozen bars ([`crate::run::Services::frozen`]) apply
//! the same exclusion to them.
//!
//! What was excluded is recorded per run ([`DataAudit`]) and lands in the run manifest
//! ([`crate::run_manifest`]) together with a SHA-256 of the bars that were read.

use std::sync::Mutex;

use fa_backtest::{holdout::HoldoutPolicy, replay::Holdout};
use fa_core::Timescale;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{Row, postgres::PgPoolOptions};

use crate::{
    data::{self, BarHistory, MIN_BARS},
    fit::FitSpec,
    run::Services,
};

/// What the data loading of one run did (run manifest `data_audit`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DataAudit {
    pub timescale: Option<Timescale>,
    /// Requested `[start, end]` before exclusion.
    pub start_ms: i64,
    pub end_ms: i64,
    /// Holdouts of the timescale that were applied.
    pub holdouts: Vec<Holdout>,
    /// `[from, to]` actually excluded per holdout (embargo included).
    pub excluded_ranges: Vec<(i64, i64)>,
    pub excluded_bars: usize,
    /// Segments dropped because they kept fewer than `MIN_BARS` bars.
    pub dropped_segments: usize,
    /// Where the bars came from: `victoriametrics` or `frozen` (bootstrap / reproduction input).
    pub source: String,
    /// `holdouts_read` | `no_postgres` | `unreadable_optional: …`.
    pub holdout_lookup: String,
    /// SHA-256 over every bar (instrument, end, OHLCV) after exclusion: the data version.
    pub bars_sha256: String,
    pub n_bars: usize,
}

/// Locked holdouts of `ts` (`validation.holdout`), each with its recorded embargo.
pub async fn holdouts(pg_url: &str, ts: Timescale) -> anyhow::Result<Vec<Holdout>> {
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(std::time::Duration::from_secs(5))
        .connect(pg_url)
        .await?;
    // `to_jsonb` keeps the lookup working before the embargo column exists (NULL = default)
    let rows = sqlx::query(
        "select (extract(epoch from start_at) * 1000)::bigint as s, (extract(epoch from end_at) * 1000)::bigint as e, \
         (extract(epoch from locked_at) * 1000)::bigint as l, (to_jsonb(h) ->> 'embargo_ms')::bigint as embargo_ms \
         from validation.holdout h where timescale = $1 order by start_at",
    )
    .bind(ts.as_str())
    .fetch_all(&pool)
    .await?;
    pool.close().await;
    Ok(rows
        .iter()
        .map(|r| Holdout {
            timescale: ts,
            start_ms: r.get("s"),
            end_ms: r.get("e"),
            locked_at_ms: r.get("l"),
            embargo_ms: r.get::<Option<i64>, _>("embargo_ms"),
        })
        .collect())
}

/// Excluded range of a holdout for a fit with `horizon_ms`: `[start − max(embargo, horizon), end]`.
pub fn excluded_range(h: &Holdout, horizon_ms: i64) -> (i64, i64) {
    let embargo = h.embargo_ms.unwrap_or_else(|| HoldoutPolicy::default_for(h.timescale).embargo_ms).max(horizon_ms);
    (h.start_ms - embargo.max(0), h.end_ms)
}

/// Removes every bar whose end lies in an excluded range; splits histories at the cuts.
pub fn exclude(
    histories: Vec<BarHistory>,
    holdouts: &[Holdout],
    horizon_ms: i64,
    audit: &mut DataAudit,
) -> Vec<BarHistory> {
    let ranges: Vec<(i64, i64)> = holdouts.iter().map(|h| excluded_range(h, horizon_ms)).collect();
    audit.excluded_ranges = ranges.clone();
    if ranges.is_empty() {
        return histories;
    }
    let inside = |t: i64| ranges.iter().any(|(a, b)| t >= *a && t <= *b);
    let mut out = Vec::with_capacity(histories.len());
    for h in histories {
        let mut seg: Option<BarHistory> = None;
        let flush = |seg: &mut Option<BarHistory>, out: &mut Vec<BarHistory>, audit: &mut DataAudit| {
            if let Some(s) = seg.take() {
                if s.bars.len() >= MIN_BARS {
                    out.push(s);
                } else {
                    audit.dropped_segments += 1;
                    audit.excluded_bars += s.bars.len();
                }
            }
        };
        for (t, bar) in h.ends_ms.iter().zip(&h.bars) {
            if inside(*t) {
                audit.excluded_bars += 1;
                flush(&mut seg, &mut out, audit);
                continue;
            }
            let s = seg.get_or_insert_with(|| BarHistory {
                ends_ms: Vec::new(),
                bars: Vec::new(),
                start_ms: *t - data::step_ms(h.timescale),
                ..h.clone()
            });
            s.ends_ms.push(*t);
            s.bars.push(*bar);
            s.end_ms = *t;
        }
        flush(&mut seg, &mut out, audit);
    }
    out
}

/// SHA-256 over the bars (the data version of a run).
pub fn bars_sha256(histories: &[BarHistory]) -> String {
    let mut h = Sha256::new();
    for x in histories {
        h.update(x.instrument.as_bytes());
        h.update(x.source.as_str().as_bytes());
        for (t, b) in x.ends_ms.iter().zip(&x.bars) {
            h.update(t.to_le_bytes());
            for v in [b.open, b.high, b.low, b.close, b.volume] {
                h.update(v.to_le_bytes());
            }
            h.update(b.trade_count.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

/// `[start, end]` of a fit: `spec.end_ms` (pinned by reproduction / bootstrap) else `now`.
pub fn fit_window(spec: &FitSpec, now_ms: i64) -> (i64, i64) {
    data::window(spec.timescale, spec.window_hours, spec.end_ms.unwrap_or(now_ms))
}

/// The bars of a fit with every locked holdout excluded. `pins` = resume pins (instrument,
/// source); `horizon_steps` = steps the fit looks ahead (next-step target, JEPA future).
#[allow(clippy::too_many_arguments)]
pub async fn load_fit_set(
    svc: &Services<'_>,
    ts: Timescale,
    instruments: Option<&data::Instruments>,
    instrument: Option<&str>,
    start_ms: i64,
    end_ms: i64,
    pins: Option<&[(String, data::Source)]>,
    horizon_steps: usize,
) -> anyhow::Result<Vec<BarHistory>> {
    let mut audit = DataAudit { timescale: Some(ts), start_ms, end_ms, ..Default::default() };
    let loaded = match svc.frozen {
        Some(f) => {
            audit.source = "frozen".into();
            f.iter()
                .filter(|h| h.timescale == ts)
                .map(|h| {
                    let keep: Vec<usize> =
                        (0..h.ends_ms.len()).filter(|&i| h.ends_ms[i] > start_ms && h.ends_ms[i] <= end_ms).collect();
                    BarHistory {
                        ends_ms: keep.iter().map(|&i| h.ends_ms[i]).collect(),
                        bars: keep.iter().map(|&i| h.bars[i]).collect(),
                        ..h.clone()
                    }
                })
                .filter(|h| h.bars.len() >= MIN_BARS)
                .collect()
        }
        None => {
            audit.source = "victoriametrics".into();
            data::load_set(svc.tsdb, ts, instruments, instrument, start_ms, end_ms, pins).await?
        }
    };
    let hs = match svc.pg_url {
        Some(url) => match holdouts(url, ts).await {
            Ok(h) => {
                audit.holdout_lookup = "holdouts_read".into();
                h
            }
            Err(e) if optional() => {
                tracing::warn!(error = %e, "validation.holdout unreadable; FA_TRAIN_HOLDOUT_OPTIONAL=1, fitting without exclusion");
                audit.holdout_lookup = format!("unreadable_optional: {e}");
                Vec::new()
            }
            Err(e) => anyhow::bail!(
                "locked holdouts unreadable ({e}); refusing to fit a window that may contain one (FA_TRAIN_HOLDOUT_OPTIONAL=1 overrides)"
            ),
        },
        None => {
            audit.holdout_lookup = "no_postgres".into();
            Vec::new()
        }
    };
    let horizon_ms = data::step_ms(ts) * horizon_steps.max(1) as i64;
    let out = exclude(loaded, &hs, horizon_ms, &mut audit);
    if audit.excluded_bars > 0 {
        tracing::info!(%ts, excluded = audit.excluded_bars, holdouts = hs.len(), "locked holdout(s) excluded from the fit window");
    }
    audit.holdouts = hs;
    audit.bars_sha256 = bars_sha256(&out);
    audit.n_bars = out.iter().map(|h| h.bars.len()).sum();
    if let Some(slot) = svc.audit {
        *slot.lock().expect("audit lock") = Some(audit);
    }
    if out.is_empty() {
        anyhow::bail!("no bars left for {ts} in [{start_ms}, {end_ms}] after excluding locked holdouts");
    }
    Ok(out)
}

/// The adapter window (one instrument, the last hour(s)) with locked holdouts excluded: the
/// newest segment that remains. Adapters train too, so they never read a holdout either.
pub async fn load_adapter_window(
    svc: &Services<'_>,
    ts: Timescale,
    instrument: Option<&str>,
    start_ms: i64,
    end_ms: i64,
    horizon_steps: usize,
) -> anyhow::Result<BarHistory> {
    let h = data::load_one(svc.tsdb, ts, instrument, start_ms, end_ms, None).await?;
    let hs = match svc.pg_url {
        Some(url) => match holdouts(url, ts).await {
            Ok(h) => h,
            Err(e) if optional() => {
                tracing::warn!(error = %e, "validation.holdout unreadable; adapter window not checked");
                Vec::new()
            }
            Err(e) => anyhow::bail!("locked holdouts unreadable ({e}); refusing the adapter window"),
        },
        None => Vec::new(),
    };
    let mut audit = DataAudit::default();
    let horizon_ms = data::step_ms(ts) * horizon_steps.max(1) as i64;
    exclude(vec![h], &hs, horizon_ms, &mut audit)
        .into_iter()
        .max_by_key(|s| s.end_ms)
        .ok_or_else(|| anyhow::anyhow!("the adapter window [{start_ms}, {end_ms}] lies inside a locked holdout"))
}

pub fn optional() -> bool {
    std::env::var("FA_TRAIN_HOLDOUT_OPTIONAL").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Shared slot a run's data audit is written to.
pub type AuditSlot = Mutex<Option<DataAudit>>;

#[cfg(test)]
pub(crate) mod tests {
    use fa_encoders::MarketBar;

    use super::*;

    pub(crate) fn history(inst: &str, ts: Timescale, n: usize, t0: i64) -> BarHistory {
        let step = data::step_ms(ts);
        let mut rng = fa_core::rng::Rng::seed(inst.len() as u64);
        let mut p = 100.0;
        let mut bars = Vec::new();
        for _ in 0..n {
            let o = p;
            p *= 1.0 + 0.002 * rng.normal();
            bars.push(MarketBar {
                open: o,
                high: o.max(p) * 1.001,
                low: o.min(p) * 0.999,
                close: p,
                volume: 100.0 + 50.0 * rng.uniform(),
                trade_count: 10,
            });
        }
        BarHistory {
            instrument: inst.into(),
            timescale: ts,
            source: data::Source::Bars,
            start_ms: t0,
            end_ms: t0 + n as i64 * step,
            ends_ms: (1..=n as i64).map(|i| t0 + i * step).collect(),
            bars,
        }
    }

    /// Acceptance 7: no bar of a locked holdout (nor of its embargo) reaches a fit, and the
    /// history around it is split rather than spliced.
    #[test]
    fn holdout_bars_never_reach_a_fit() {
        let ts = Timescale::Minute;
        let h = history("aa", ts, 600, 0);
        let hold = Holdout {
            timescale: ts,
            start_ms: 200 * 60_000,
            end_ms: 320 * 60_000,
            locked_at_ms: 0,
            embargo_ms: Some(30 * 60_000),
        };
        let mut audit = DataAudit::default();
        let out = exclude(vec![h.clone()], std::slice::from_ref(&hold), 60_000, &mut audit);
        let (a, b) = excluded_range(&hold, 60_000);
        assert_eq!((a, b), (170 * 60_000, 320 * 60_000));
        assert_eq!(out.len(), 2, "split into the segment before and after");
        for s in &out {
            assert!(s.ends_ms.iter().all(|t| *t < a || *t > b), "a holdout bar leaked");
            assert_eq!(s.bars.len(), s.ends_ms.len());
            assert!(s.ends_ms.windows(2).all(|w| w[1] - w[0] == 60_000), "segments stay contiguous");
        }
        assert_eq!(audit.excluded_bars, 151);
        assert_eq!(out.iter().map(|s| s.bars.len()).sum::<usize>(), 600 - 151);
        // the horizon widens the embargo when it is longer
        assert_eq!(excluded_range(&hold, 3_600_000).0, 140 * 60_000);
        // short remnants are dropped, not trained on
        let hold2 = Holdout { start_ms: 10 * 60_000, ..hold };
        let mut audit = DataAudit::default();
        let out = exclude(vec![h], &[hold2], 60_000, &mut audit);
        assert_eq!(out.len(), 1);
        assert!(out[0].ends_ms[0] > 320 * 60_000);
        assert_eq!(audit.dropped_segments, 0, "nothing before minute 0 − 30 survives at all");
    }

    #[test]
    fn bars_hash_is_a_data_version() {
        let a = history("aa", Timescale::Minute, 50, 0);
        let mut b = a.clone();
        assert_eq!(bars_sha256(std::slice::from_ref(&a)), bars_sha256(std::slice::from_ref(&b)));
        b.bars[3].close += 1e-9;
        assert_ne!(bars_sha256(&[a]), bars_sha256(&[b]));
    }
}
