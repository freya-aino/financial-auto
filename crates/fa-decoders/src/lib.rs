//! Decoder population members: families, trait, and scoring functions (spec 34).

use fa_expert_space::ExpertView;
use serde::{Deserialize, Serialize};

/// Decoder families; string forms match `fa.v1.DecoderOutput.family`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Family {
    VisualChart,
    VisualFrame,
    VisualFuture,
    Retrieval,
    TextNews,
    TextSummary,
    TextNarrative,
    RnnPrior,
}

impl Family {
    pub const ALL: [Family; 8] = [
        Self::VisualChart,
        Self::VisualFrame,
        Self::VisualFuture,
        Self::Retrieval,
        Self::TextNews,
        Self::TextSummary,
        Self::TextNarrative,
        Self::RnnPrior,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::VisualChart => "visual.chart",
            Self::VisualFrame => "visual.frame",
            Self::VisualFuture => "visual.future",
            Self::Retrieval => "retrieval",
            Self::TextNews => "text.news",
            Self::TextSummary => "text.summary",
            Self::TextNarrative => "text.narrative",
            Self::RnnPrior => "rnn_prior",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.as_str() == s)
    }
}

/// What a decoder produced for one expert view.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecodeOutput {
    /// Forecast quantiles (levels given by `quantile_levels`), if the family forecasts.
    pub quantiles: Vec<f64>,
    pub quantile_levels: Vec<f64>,
    /// Retrieved analogs (retrieval family).
    pub analogs: Vec<Analog>,
    /// Garage keys of rendered images / generated text.
    pub artifact_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analog {
    pub window_id: String,
    pub similarity: f64,
    pub time_distance_days: f64,
    /// Realised forward outcome of the analog window (e.g. return over the horizon).
    pub outcome: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("ML backend not enabled (build with feature `burn`/`candle`)")]
    BackendDisabled,
    #[error("expert view does not contain the experts this decoder needs")]
    MissingExperts,
}

pub trait Decoder: Send + Sync {
    fn id(&self) -> &str;
    fn family(&self) -> Family;
    fn decode(&self, view: &ExpertView) -> Result<DecodeOutput, DecodeError>;
}

/// CRPS of an ensemble forecast: E|X−y| − ½ E|X−X'| (lower is better).
pub fn crps_ensemble(members: &[f64], y: f64) -> f64 {
    let n = members.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut sorted = members.to_vec();
    sorted.sort_by(f64::total_cmp);
    let term1 = sorted.iter().map(|x| (x - y).abs()).sum::<f64>() / n as f64;
    // E|X−X'| via the sorted-sample identity: (2/n²) Σ_i (2i − n − 1) x_(i)
    let term2 = sorted.iter().enumerate().map(|(i, x)| (2.0 * (i as f64 + 1.0) - n as f64 - 1.0) * x).sum::<f64>()
        * 2.0
        / (n * n) as f64;
    term1 - 0.5 * term2
}

/// Pinball (quantile) loss.
pub fn pinball(q: f64, level: f64, y: f64) -> f64 {
    let d = y - q;
    if d >= 0.0 { level * d } else { (level - 1.0) * d }
}

/// CRPS approximation from quantile forecasts: 2 · mean pinball loss over levels.
pub fn crps_quantiles(quantiles: &[f64], levels: &[f64], y: f64) -> f64 {
    assert_eq!(quantiles.len(), levels.len());
    if quantiles.is_empty() {
        return f64::NAN;
    }
    2.0 * quantiles.iter().zip(levels).map(|(q, l)| pinball(*q, *l, y)).sum::<f64>() / quantiles.len() as f64
}

/// Retrieval weight of an analog: similarity discounted by time distance
/// (time distance is an explicit feature, spec 34): `sim · exp(−Δt / τ)`.
pub fn analog_weight(a: &Analog, tau_days: f64) -> f64 {
    a.similarity.max(0.0) * (-a.time_distance_days / tau_days).exp()
}

/// Weighted forecast from analogs; `None` if all weights are zero.
pub fn analog_forecast(analogs: &[Analog], tau_days: f64) -> Option<f64> {
    let (num, den) = analogs.iter().fold((0.0, 0.0), |(n, d), a| {
        let w = analog_weight(a, tau_days);
        (n + w * a.outcome, d + w)
    });
    (den > 0.0).then(|| num / den)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_strings_roundtrip() {
        for f in Family::ALL {
            assert_eq!(Family::parse(f.as_str()), Some(f));
        }
    }

    #[test]
    fn crps_ensemble_point_mass_is_abs_error() {
        assert!((crps_ensemble(&[2.0], 5.0) - 3.0).abs() < 1e-12);
        assert!((crps_ensemble(&[2.0, 2.0, 2.0], 5.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn crps_ensemble_matches_bruteforce() {
        let xs: [f64; 5] = [0.1, -0.4, 1.3, 0.7, 0.0];
        let y: f64 = 0.2;
        let n = xs.len() as f64;
        let t1: f64 = xs.iter().map(|x| (x - y).abs()).sum::<f64>() / n;
        let t2: f64 = xs.iter().flat_map(|a| xs.iter().map(move |b| (a - b).abs())).sum::<f64>() / (n * n);
        assert!((crps_ensemble(&xs, y) - (t1 - 0.5 * t2)).abs() < 1e-12);
    }

    #[test]
    fn sharper_correct_forecast_scores_better() {
        let levels = [0.1, 0.5, 0.9];
        let sharp = crps_quantiles(&[-0.1, 0.0, 0.1], &levels, 0.0);
        let wide = crps_quantiles(&[-1.0, 0.0, 1.0], &levels, 0.0);
        assert!(sharp < wide);
    }

    #[test]
    fn recent_analogs_dominate() {
        let a = |sim, dt, out| Analog { window_id: String::new(), similarity: sim, time_distance_days: dt, outcome: out };
        let f = analog_forecast(&[a(0.9, 1.0, 1.0), a(0.9, 3650.0, -1.0)], 365.0).unwrap();
        assert!(f > 0.99);
        assert_eq!(analog_forecast(&[a(0.0, 1.0, 1.0)], 365.0), None);
    }
}
