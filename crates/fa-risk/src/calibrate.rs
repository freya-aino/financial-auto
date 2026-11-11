//! Limit calibration rules (spec 40 §7, §11 "all numeric limit values are placeholders").
//!
//! The limits stay configuration (`risk.toml`) and non-evolvable; this module only *derives*
//! suggested values from what the paper book has realised, so the placeholders can be replaced
//! by numbers with a stated basis:
//!
//! | Limit | Rule |
//! |---|---|
//! | `max_daily_loss` | `1.5 × z99 × σ_day` of portfolio equity, clamped to [0.5 %, 5 %] |
//! | `max_drawdown` | `3 × max_daily_loss` (suggested), clamped to [3 %, 25 %] |
//! | `max_position_frac` | position whose one-day 99 % VaR is `position_var_budget` of equity: `budget / (z99 × σ_day,inst)` at the median instrument volatility, clamped to [1 %, 20 %] |
//!
//! `σ_day` comes from hourly log returns (`× sqrt(24)`), at least `min_hours` of them. The
//! result carries the suggestion and the **tighten-only** limit set (`min(current, suggested)`
//! per field), which is what `ctl.risk.limits {calibrate: {apply: true}}` may apply: a rule can
//! never loosen a limit at runtime (spec 40 §6.5).

use serde::{Deserialize, Serialize};

use crate::RiskConfig;

const Z99: f64 = 2.326_347_874;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRules {
    pub min_hours: usize,
    pub position_var_budget: f64,
}

impl Default for CalibrationRules {
    fn default() -> Self {
        Self { min_hours: 24, position_var_budget: 0.005 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub hours: usize,
    pub sigma_day: Option<f64>,
    pub median_instrument_sigma_day: Option<f64>,
    /// `field → suggested value` (only fields with enough data).
    pub suggested: std::collections::BTreeMap<String, f64>,
    /// Current limits with every suggestion that is stricter applied.
    pub tighten_only: RiskConfig,
    /// Fields the tighten-only set changes.
    pub tightened: Vec<String>,
    pub notes: Vec<String>,
}

fn sigma(values: &[f64]) -> Option<f64> {
    let r: Vec<f64> = values
        .windows(2)
        .filter(|w| w[0] > 0.0 && w[1] > 0.0)
        .map(|w| (w[1] / w[0]).ln())
        .filter(|x| x.is_finite())
        .collect();
    if r.len() < 2 {
        return None;
    }
    let m = r.iter().sum::<f64>() / r.len() as f64;
    Some((r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (r.len() - 1) as f64).sqrt())
}

/// Derives suggestions from hourly portfolio equity points and per-instrument daily volatilities.
pub fn calibrate(
    cfg: &RiskConfig,
    rules: &CalibrationRules,
    hourly_equity: &[f64],
    instrument_sigma_day: &[f64],
) -> Calibration {
    let mut suggested = std::collections::BTreeMap::new();
    let mut notes = Vec::new();
    let hours = hourly_equity.len().saturating_sub(1);
    let sigma_day = if hours >= rules.min_hours { sigma(hourly_equity).map(|s| s * 24f64.sqrt()) } else { None };
    match sigma_day {
        Some(s) => {
            let dl = (1.5 * Z99 * s).clamp(0.005, 0.05);
            suggested.insert("max_daily_loss".to_owned(), dl);
            suggested.insert("max_drawdown".to_owned(), (3.0 * dl).clamp(0.03, 0.25));
        }
        None => notes.push(format!("{hours} hourly equity returns < min_hours {}: no portfolio rule", rules.min_hours)),
    }
    let mut vols: Vec<f64> = instrument_sigma_day.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    vols.sort_by(f64::total_cmp);
    let median = (!vols.is_empty()).then(|| vols[vols.len() / 2]);
    match median {
        Some(v) => {
            suggested.insert("max_position_frac".to_owned(), (rules.position_var_budget / (Z99 * v)).clamp(0.01, 0.20));
        }
        None => notes.push("no instrument volatility yet: no position rule".into()),
    }
    let mut tighten_only = cfg.clone();
    let mut tightened = Vec::new();
    for (k, v) in &suggested {
        let slot = match k.as_str() {
            "max_daily_loss" => &mut tighten_only.max_daily_loss,
            "max_drawdown" => &mut tighten_only.max_drawdown,
            "max_position_frac" => &mut tighten_only.max_position_frac,
            _ => continue,
        };
        if *v < *slot {
            *slot = *v;
            tightened.push(k.clone());
        }
    }
    if tightened.iter().any(|k| k == "max_position_frac") {
        // keep the provisional cap at or below the new position cap
        if let Some(p) = tighten_only.provisional_position_frac {
            tighten_only.provisional_position_frac = Some(p.min(tighten_only.max_position_frac));
        }
    }
    Calibration { hours, sigma_day, median_instrument_sigma_day: median, suggested, tighten_only, tightened, notes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggestions_follow_volatility_and_only_tighten() {
        let cfg =
            RiskConfig { max_daily_loss: 0.02, max_drawdown: 0.10, max_position_frac: 0.05, ..RiskConfig::default() };
        // equity alternating ±0.5 % per hour → σ_day ≈ 0.005 × sqrt(24) × ~1 ≈ 2.4 %
        let eq: Vec<f64> = (0..49).map(|i| 100_000.0 * if i % 2 == 0 { 1.0 } else { 1.005 }).collect();
        let c = calibrate(&cfg, &CalibrationRules::default(), &eq, &[0.04, 0.02, 0.03]);
        let dl = c.suggested["max_daily_loss"];
        assert!(dl > 0.02 && dl <= 0.05, "{c:?}");
        // looser suggestion → kept at the current value
        assert_eq!(c.tighten_only.max_daily_loss, 0.02);
        assert!(!c.tightened.contains(&"max_daily_loss".to_owned()));
        // position: 0.005 / (2.33 × 0.03) ≈ 7.2 % > 5 % → not tightened
        assert!((c.suggested["max_position_frac"] - 0.005 / (Z99 * 0.03)).abs() < 1e-12);
        assert_eq!(c.tighten_only.max_position_frac, 0.05);
        // a very volatile universe tightens the position cap; the provisional cap follows
        let cfg2 = RiskConfig { provisional_position_frac: Some(0.04), ..cfg.clone() };
        let c = calibrate(&cfg2, &CalibrationRules::default(), &eq, &[0.2]);
        assert!(c.tightened.contains(&"max_position_frac".to_owned()));
        assert!(c.tighten_only.max_position_frac < 0.05);
        assert!(c.tighten_only.provisional_position_frac.unwrap() <= c.tighten_only.max_position_frac);
        assert!(cfg2.check_tightens(&c.tighten_only).is_empty(), "never loosens");
        // calm equity tightens the daily loss
        let calm: Vec<f64> = (0..49).map(|i| 100_000.0 + (i % 2) as f64).collect();
        let c = calibrate(&cfg, &CalibrationRules::default(), &calm, &[]);
        assert_eq!(c.suggested["max_daily_loss"], 0.005);
        assert_eq!(c.tighten_only.max_daily_loss, 0.005);
        assert_eq!(c.suggested["max_drawdown"], 0.03);
        assert!(c.notes.iter().any(|n| n.contains("no instrument volatility")));
    }

    #[test]
    fn too_little_history_suggests_nothing() {
        let cfg = RiskConfig::default();
        let c = calibrate(&cfg, &CalibrationRules::default(), &[1.0, 1.1, 1.0], &[]);
        assert!(c.suggested.is_empty() && c.tightened.is_empty() && c.sigma_day.is_none());
        assert_eq!(c.tighten_only, cfg);
    }
}
