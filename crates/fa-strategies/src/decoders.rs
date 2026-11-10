//! Recency cache of `decoder.output.<family>` messages (spec 34) behind the spec 35 §4.2
//! decoder primitives.
//!
//! `evo-worker` publishes a `DecoderOutput` per evaluated genome and instrument on
//! `decoder.output.rnn_prior` (forecast `quantiles`) and `decoder.output.retrieval`
//! (`analogs`); model-runtime publishes the baseline text / visual decoders on
//! `decoder.output.text.<topic>` (`news`, `summary`, `narrative`) and
//! `decoder.output.visual.<kind>` (`chart`, `future`) with `quantiles[0]` = score in `[-1, 1]`.
//! The message has no instrument field; the instrument travels in `output_refs` as
//! `inst:<uuid>`, or `inst:*` for a market-wide output (outputs without either are treated as
//! market-wide too). Outputs are kept per `(family, instrument)` in event-time order, bounded,
//! and every query is point-in-time with a recency window: the latest output with
//! `as_of ≤ at_ms` and `as_of ≥ at_ms − max_age` wins; an instrument-specific output is
//! preferred over a market-wide one whenever one is fresh.
//!
//! **Horizon.** The message has no horizon field. [`Output::horizon`] is recovered from the
//! `output_refs` convention `h:<steps>`, else from a `genome:<text>` ref or the `genome_id`
//! itself when it is a decoder genome text carrying `h=<steps>` (kv form) or `"h"` /
//! `"horizon"` (JSON form); otherwise it is unknown. `fc_quantile` / `fc_dist` prefer the fresh
//! output whose horizon is closest to the asked `k`, latest first on ties; when no output has a
//! known horizon the latest wins (the pre-D-063 behaviour).
//!
//! | primitive | source |
//! |---|---|
//! | `fc_quantile(q, k)` | linear interpolation of the `rnn_prior` quantile vector at level `q` (levels are the decoder defaults `0.05 0.25 0.5 0.75 0.95` for five values, else `(i+1)/(n+1)`) of the output with horizon closest to `k` |
//! | `fc_dist(k)` | mean of that quantile vector (expected forecast return) |
//! | `analog_outcome(k)` | similarity-weighted mean realised `k`-bar log return after each analog's `window_end`, from the caller's price history ([`PriceLookup`]); analogs whose prices are not available are **skipped and reported** ([`AnalogOutcome::unresolved`]) so the caller can prefetch them; only outcomes that completed at or before the view time count (point-in-time) |
//! | `analog_age` | mean `time_distance_days` of the analogs |
//! | `score(family)` | freshness of the latest output of `family` (exact name, dotted families included): `1 − age / max_age` |
//! | `text_score(topic)` | `quantiles[0]` of the latest fresh `text.<topic>` output, clamped to `[-1, 1]`; `None` when nothing fresh |
//! | `visual_score(kind)` | the same for `visual.<kind>` |
use std::collections::{BTreeMap, VecDeque};

use fa_core::{InstrumentId, Timescale};
use fa_proto::v1::DecoderOutput;

/// Default quantile levels of the baseline forecast decoders (`fa_decoders::genome::DEFAULT_LEVELS`).
pub const DEFAULT_LEVELS: [f64; 5] = [0.05, 0.25, 0.5, 0.75, 0.95];
pub const MAX_OUTPUTS_PER_KEY: usize = 64;
/// Default recency window: outputs older than this are not used.
pub const DEFAULT_MAX_AGE_MS: i64 = 6 * 3_600_000;

pub const FAMILY_FORECAST: &str = "rnn_prior";
pub const FAMILY_RETRIEVAL: &str = "retrieval";
/// Family prefix of the text decoders (`text.news`, `text.summary`, `text.narrative`).
pub const FAMILY_TEXT_PREFIX: &str = "text.";
/// Family prefix of the visual decoders (`visual.chart`, `visual.future`).
pub const FAMILY_VISUAL_PREFIX: &str = "visual.";
/// `output_refs` entry marking a market-wide output.
pub const MARKET_WIDE_REF: &str = "inst:*";

fn ms(t: Option<&fa_proto::Timestamp>) -> Option<i64> {
    t.map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Analog {
    pub window_id: String,
    pub window_end_ms: Option<i64>,
    pub similarity: f64,
    pub time_distance_days: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub as_of_ms: i64,
    pub received_ms: i64,
    pub decoder_id: String,
    pub genome_id: String,
    pub family: String,
    pub snapshot_id: String,
    pub timescale: Option<Timescale>,
    pub instrument: Option<InstrumentId>,
    /// `inst:*` was present (explicitly market-wide, as opposed to merely unlabelled).
    pub market_wide: bool,
    /// Forecast horizon in steps of `timescale` when it can be recovered (module docs).
    pub horizon: Option<u32>,
    pub output_refs: Vec<String>,
    pub quantiles: Vec<f64>,
    pub analogs: Vec<Analog>,
}

/// Horizon encoded in a decoder genome text: `h=<n>` (kv form, `forecast:h=3;lb=8`) or
/// `"h": n` / `"horizon": n` (JSON form).
pub fn horizon_of_genome(text: &str) -> Option<u32> {
    let t = text.trim();
    if t.starts_with('{') {
        let v: serde_json::Value = serde_json::from_str(t).ok()?;
        let n = v.get("h").or_else(|| v.get("horizon")).or_else(|| v.get("params").and_then(|p| p.get("h")))?;
        return n.as_f64().filter(|x| *x >= 1.0).map(|x| x as u32);
    }
    t.split([';', ',', ' ', ':'])
        .find_map(|kv| kv.split_once('=').filter(|(k, _)| k.trim() == "h" || k.trim() == "horizon"))
        .and_then(|(_, v)| v.trim().parse::<f64>().ok())
        .filter(|x| *x >= 1.0)
        .map(|x| x as u32)
}

/// Horizon of a `DecoderOutput`: `h:<steps>` ref, else `genome:<text>` ref, else `genome_id`.
pub fn horizon_of(m: &DecoderOutput) -> Option<u32> {
    if let Some(h) = m.output_refs.iter().find_map(|r| r.strip_prefix("h:")).and_then(|h| h.trim().parse::<u32>().ok())
    {
        return (h >= 1).then_some(h);
    }
    m.output_refs
        .iter()
        .find_map(|r| r.strip_prefix("genome:").and_then(horizon_of_genome))
        .or_else(|| horizon_of_genome(&m.genome_id))
}

impl Output {
    /// Score of a text / visual output: `quantiles[0]` clamped to `[-1, 1]`.
    pub fn score_value(&self) -> Option<f64> {
        self.quantiles.first().filter(|v| v.is_finite()).map(|v| v.clamp(-1.0, 1.0))
    }

    /// Quantile levels the `quantiles` vector is sampled at.
    pub fn levels(&self) -> Vec<f64> {
        let n = self.quantiles.len();
        if n == DEFAULT_LEVELS.len() {
            DEFAULT_LEVELS.to_vec()
        } else {
            (1..=n).map(|i| i as f64 / (n + 1) as f64).collect()
        }
    }

    /// Forecast quantile at `q` by linear interpolation (clamped to the outer quantiles).
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.quantiles.is_empty() {
            return None;
        }
        let levels = self.levels();
        let q = q.clamp(0.0, 1.0);
        if q <= levels[0] {
            return Some(self.quantiles[0]);
        }
        for w in 0..levels.len() - 1 {
            let (l0, l1) = (levels[w], levels[w + 1]);
            if q <= l1 {
                let t = (q - l0) / (l1 - l0);
                return Some(self.quantiles[w] + t * (self.quantiles[w + 1] - self.quantiles[w]));
            }
        }
        self.quantiles.last().copied()
    }

    /// Expected value of the forecast distribution (mean of the quantile vector).
    pub fn mean(&self) -> Option<f64> {
        if self.quantiles.is_empty() {
            return None;
        }
        Some(self.quantiles.iter().sum::<f64>() / self.quantiles.len() as f64)
    }
}

/// Price history lookup used by `analog_outcome`: last price at or before `at_ms`.
pub trait PriceLookup {
    fn price_at(&self, at_ms: i64) -> Option<f64>;
}

impl<F: Fn(i64) -> Option<f64>> PriceLookup for F {
    fn price_at(&self, at_ms: i64) -> Option<f64> {
        self(at_ms)
    }
}

/// Cache of decoder outputs keyed by `(family, instrument)`.
#[derive(Debug, Default)]
pub struct DecoderOutputs {
    by_key: BTreeMap<(String, Option<InstrumentId>), VecDeque<Output>>,
    pub seen: u64,
}

impl DecoderOutputs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Instrument carried in `output_refs` as `inst:<uuid>`.
    pub fn instrument_of(m: &DecoderOutput) -> Option<InstrumentId> {
        m.output_refs.iter().find_map(|r| r.strip_prefix("inst:")).and_then(crate::graph::parse_inst)
    }

    /// Ingests one output; `received_ms` is the receive time (used when `as_of` is missing).
    pub fn on_output(&mut self, m: &DecoderOutput, received_ms: i64) -> bool {
        if m.family.is_empty() {
            return false;
        }
        let out = Output {
            as_of_ms: ms(m.as_of.as_ref()).unwrap_or(received_ms),
            received_ms,
            decoder_id: m.decoder_id.clone(),
            genome_id: m.genome_id.clone(),
            family: m.family.clone(),
            snapshot_id: m.snapshot_id.clone(),
            timescale: m.timescale.parse().ok(),
            instrument: Self::instrument_of(m),
            market_wide: m.output_refs.iter().any(|r| r == MARKET_WIDE_REF),
            horizon: horizon_of(m),
            output_refs: m.output_refs.clone(),
            quantiles: m.quantiles.clone(),
            analogs: m
                .analogs
                .iter()
                .map(|a| Analog {
                    window_id: a.window_id.clone(),
                    window_end_ms: ms(a.window_end.as_ref()),
                    similarity: a.similarity,
                    time_distance_days: a.time_distance_days,
                })
                .collect(),
        };
        let q = self.by_key.entry((out.family.clone(), out.instrument)).or_default();
        // keep ascending event time; a late output is inserted in order
        let pos = q.iter().rposition(|o| o.as_of_ms <= out.as_of_ms).map_or(0, |i| i + 1);
        q.insert(pos, out);
        while q.len() > MAX_OUTPUTS_PER_KEY {
            q.pop_front();
        }
        self.seen += 1;
        true
    }

    /// Families with any output.
    pub fn families(&self) -> Vec<String> {
        let mut v: Vec<String> = self.by_key.keys().map(|k| k.0.clone()).collect();
        v.dedup();
        v
    }

    pub fn len(&self) -> usize {
        self.by_key.values().map(VecDeque::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fresh outputs (`as_of ≤ at_ms`, age ≤ `max_age_ms`) of exactly `family` for `inst`;
    /// when the instrument has none, the market-wide (`inst:*` / unlabelled) ones.
    fn fresh<'a>(
        &'a self,
        family: &str,
        inst: InstrumentId,
        at_ms: i64,
        max_age_ms: i64,
    ) -> impl Iterator<Item = &'a Output> + 'a {
        let pick = move |key_inst: Option<InstrumentId>| {
            self.by_key.get(&(family.to_owned(), key_inst)).into_iter().flat_map(move |q| {
                q.iter().rev().filter(move |o| o.as_of_ms <= at_ms && at_ms - o.as_of_ms <= max_age_ms)
            })
        };
        let own: Vec<&Output> = pick(Some(inst)).collect();
        if own.is_empty() { pick(None).collect::<Vec<_>>().into_iter() } else { own.into_iter() }
    }

    /// Latest output of exactly `family` for `inst` as of `at_ms` within `max_age_ms`; an
    /// instrument-specific output wins over a market-wide one whenever one is fresh.
    pub fn latest(&self, family: &str, inst: InstrumentId, at_ms: i64, max_age_ms: i64) -> Option<&Output> {
        self.fresh(family, inst, at_ms, max_age_ms).max_by_key(|o| o.as_of_ms)
    }

    /// Like [`Self::latest`] but, among the fresh outputs, prefers the one whose [`Output::horizon`]
    /// is closest to `k` (outputs of unknown horizon rank after any known one; ties → latest).
    pub fn latest_for_horizon(
        &self,
        family: &str,
        inst: InstrumentId,
        at_ms: i64,
        max_age_ms: i64,
        k: u32,
    ) -> Option<&Output> {
        self.fresh(family, inst, at_ms, max_age_ms).min_by_key(|o| {
            let dist = o.horizon.map_or(u32::MAX, |h| h.abs_diff(k));
            (dist, std::cmp::Reverse(o.as_of_ms))
        })
    }

    /// Point-in-time view for one instrument.
    pub fn at(&self, inst: InstrumentId, at_ms: i64, max_age_ms: i64) -> DecodersAt<'_> {
        DecodersAt { d: self, inst, at_ms, max_age_ms }
    }
}

/// Decoder primitives for one instrument at one time (spec 35 §4.2).
pub struct DecodersAt<'a> {
    d: &'a DecoderOutputs,
    inst: InstrumentId,
    at_ms: i64,
    max_age_ms: i64,
}

impl DecodersAt<'_> {
    pub fn latest(&self, family: &str) -> Option<&Output> {
        self.d.latest(family, self.inst, self.at_ms, self.max_age_ms)
    }

    /// Forecast output whose horizon is closest to `k` (module docs).
    pub fn forecast_for(&self, k: u32) -> Option<&Output> {
        self.d.latest_for_horizon(FAMILY_FORECAST, self.inst, self.at_ms, self.max_age_ms, k)
    }

    /// `fc_quantile(q, k)`: quantile `q` of the forecast output with horizon closest to `k`.
    pub fn fc_quantile(&self, q: f64, k: u32) -> Option<f64> {
        self.forecast_for(k)?.quantile(q)
    }

    /// `fc_dist(k)` as the expected value of the forecast output with horizon closest to `k`.
    pub fn fc_dist(&self, k: u32) -> Option<f64> {
        self.forecast_for(k)?.mean()
    }

    /// `analog_outcome(k)`: similarity-weighted mean of `ln(p(window_end + k·bar) / p(window_end))`
    /// over analogs whose prices are available; `None` when none is.
    pub fn analog_outcome(&self, k: u32, bar: Timescale, prices: &dyn PriceLookup) -> Option<f64> {
        self.analog_outcome_detail(k, bar, prices).value
    }

    /// [`Self::analog_outcome`] with the resolution detail: how many analogs contributed and
    /// which ones could not be priced (so the caller can fetch them for the next step).
    /// Point-in-time rule: an analog counts only when its outcome window
    /// `[window_end, window_end + k·bar]` has completed at or before the view time; analogs
    /// still running are neither counted nor reported.
    pub fn analog_outcome_detail(&self, k: u32, bar: Timescale, prices: &dyn PriceLookup) -> AnalogOutcome {
        let mut res = AnalogOutcome::default();
        let Some(out) = self.latest(FAMILY_RETRIEVAL) else { return res };
        let k = k.max(1);
        let step = bar.duration().as_millis() as i64 * i64::from(k);
        let mut num = 0.0;
        let mut den = 0.0;
        for a in &out.analogs {
            let Some(end) = a.window_end_ms else { continue };
            if end + step > self.at_ms {
                res.pending += 1; // outcome not realised yet (point-in-time)
                continue;
            }
            let (Some(p0), Some(p1)) = (prices.price_at(end), prices.price_at(end + step)) else {
                res.unresolved.push(AnalogRequest { window_end_ms: end, k, bar });
                continue;
            };
            if p0 <= 0.0 || p1 <= 0.0 {
                res.unresolved.push(AnalogRequest { window_end_ms: end, k, bar });
                continue;
            }
            let w = a.similarity.max(0.0);
            num += w * (p1 / p0).ln();
            den += w;
            res.resolved.push(AnalogRequest { window_end_ms: end, k, bar });
        }
        res.value = (den > 0.0).then(|| num / den);
        res
    }

    /// `analog_age`: mean time distance of the analogs in days.
    pub fn analog_age(&self) -> Option<f64> {
        let out = self.latest(FAMILY_RETRIEVAL)?;
        if out.analogs.is_empty() {
            return None;
        }
        Some(out.analogs.iter().map(|a| a.time_distance_days).sum::<f64>() / out.analogs.len() as f64)
    }

    /// `score(family)`: freshness in [0, 1] of the latest output of `family` (1 = just published).
    pub fn score(&self, family: &str) -> Option<f64> {
        let out = self.latest(family)?;
        Some((1.0 - (self.at_ms - out.as_of_ms) as f64 / self.max_age_ms.max(1) as f64).clamp(0.0, 1.0))
    }

    /// `text_score(topic)`: score of the latest fresh `text.<topic>` output (instrument-specific
    /// preferred over market-wide); `None` when nothing fresh.
    pub fn text_score(&self, topic: &str) -> Option<f64> {
        self.latest(&format!("{FAMILY_TEXT_PREFIX}{topic}"))?.score_value()
    }

    /// `visual_score(kind)`: score of the latest fresh `visual.<kind>` output; `None` when nothing fresh.
    pub fn visual_score(&self, kind: &str) -> Option<f64> {
        self.latest(&format!("{FAMILY_VISUAL_PREFIX}{kind}"))?.score_value()
    }
}

/// One analog whose outcome prices were missing: the caller needs `p(window_end)` and
/// `p(window_end + k·bar)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AnalogRequest {
    pub window_end_ms: i64,
    pub k: u32,
    pub bar: Timescale,
}

impl AnalogRequest {
    /// End of the outcome window.
    pub fn outcome_end_ms(&self) -> i64 {
        self.window_end_ms + self.bar.duration().as_millis() as i64 * i64::from(self.k.max(1))
    }
}

/// Result of [`DecodersAt::analog_outcome_detail`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnalogOutcome {
    pub value: Option<f64>,
    /// Analogs whose outcome was priced and counted.
    pub resolved: Vec<AnalogRequest>,
    /// Analogs whose outcome window has not completed yet (never counted).
    pub pending: u32,
    /// Analogs skipped for missing prices.
    pub unresolved: Vec<AnalogRequest>,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn ts(ms: i64) -> fa_proto::Timestamp {
        fa_proto::Timestamp { seconds: ms.div_euclid(1000), nanos: (ms.rem_euclid(1000) * 1_000_000) as i32 }
    }

    pub fn forecast(inst: Option<InstrumentId>, at_ms: i64, q: &[f64]) -> DecoderOutput {
        DecoderOutput {
            decoder_id: "baseline-forecast-1".into(),
            genome_id: "g1".into(),
            family: FAMILY_FORECAST.into(),
            snapshot_id: "snap".into(),
            timescale: "1s".into(),
            as_of: Some(ts(at_ms)),
            output_refs: inst.map(|i| vec![format!("inst:{i}"), "job:j1".into()]).unwrap_or_default(),
            quantiles: q.to_vec(),
            analogs: vec![],
        }
    }

    pub fn retrieval(inst: InstrumentId, at_ms: i64, analogs: &[(i64, f64, f64)]) -> DecoderOutput {
        DecoderOutput {
            decoder_id: "baseline-retrieval-1".into(),
            genome_id: "g2".into(),
            family: FAMILY_RETRIEVAL.into(),
            snapshot_id: "snap".into(),
            timescale: "1s".into(),
            as_of: Some(ts(at_ms)),
            output_refs: vec![format!("inst:{inst}")],
            quantiles: vec![],
            analogs: analogs
                .iter()
                .map(|(end, sim, days)| fa_proto::v1::Analog {
                    window_id: format!("w{end}"),
                    window_end: Some(ts(*end)),
                    similarity: *sim,
                    time_distance_days: *days,
                })
                .collect(),
        }
    }

    /// A text / visual output on the model-runtime contract: `family` is `text.<topic>` or
    /// `visual.<kind>`, `quantiles[0]` the score, `inst:<uuid>` or `inst:*` in `output_refs`.
    pub fn scored(family: &str, inst: Option<InstrumentId>, at_ms: i64, score: f64) -> DecoderOutput {
        let inst_ref = inst.map_or_else(|| MARKET_WIDE_REF.to_owned(), |i| format!("inst:{i}"));
        DecoderOutput {
            decoder_id: format!("baseline-{family}"),
            genome_id: format!("{family}-baseline"),
            family: family.into(),
            snapshot_id: "snap".into(),
            timescale: "1s".into(),
            as_of: Some(ts(at_ms)),
            output_refs: vec![inst_ref, "topic:news".into(), "src:gnews".into()],
            quantiles: vec![score],
            analogs: vec![],
        }
    }

    #[test]
    fn keyed_by_family_and_instrument_with_recency() {
        let inst = InstrumentId::new();
        let other = InstrumentId::new();
        let mut d = DecoderOutputs::new();
        assert!(d.on_output(&forecast(Some(inst), 10_000, &[-0.02, -0.01, 0.0, 0.01, 0.02]), 10_000));
        assert!(d.on_output(&forecast(Some(inst), 20_000, &[-0.01, 0.0, 0.01, 0.02, 0.03]), 20_000));
        assert!(d.on_output(&forecast(None, 15_000, &[0.1, 0.2, 0.3]), 15_000));
        assert_eq!(d.seen, 3);
        assert_eq!(d.len(), 3);
        assert_eq!(d.families(), vec![FAMILY_FORECAST.to_owned()]);
        // latest as of 25 s → the 20 s output; median interpolates exactly
        let v = d.at(inst, 25_000, DEFAULT_MAX_AGE_MS);
        assert_eq!(v.fc_quantile(0.5, 5), Some(0.01));
        assert_eq!(v.fc_quantile(0.0, 5), Some(-0.01));
        assert_eq!(v.fc_quantile(1.0, 5), Some(0.03));
        let q = v.fc_quantile(0.375, 5).unwrap(); // halfway between 0.25 and 0.5 levels
        assert!((q - 0.005).abs() < 1e-12, "{q}");
        assert_eq!(v.fc_dist(5), Some(0.01));
        assert!(v.score(FAMILY_FORECAST).unwrap() > 0.99);
        // point in time: at 12 s only the first output is visible
        assert_eq!(d.at(inst, 12_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5, 5), Some(0.0));
        // recency window: nothing within 1 s of 30 s
        assert_eq!(d.at(inst, 30_000, 1_000).fc_quantile(0.5, 5), None);
        // another instrument falls back to the instrument-less output (levels (i+1)/(n+1) → median 0.2)
        assert_eq!(d.at(other, 25_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5, 5), Some(0.2));
        assert_eq!(d.at(other, 14_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5, 5), None);
        // nothing published for text/visual
        assert_eq!(v.text_score("news"), None);
        assert_eq!(v.visual_score("future"), None);
        assert_eq!(v.analog_age(), None);
        // exact family match: a prefix is not a family
        assert_eq!(v.score("rnn"), None);
        // late arrival keeps order
        assert!(d.on_output(&forecast(Some(inst), 5_000, &[0.0; 5]), 30_000));
        assert_eq!(d.at(inst, 6_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5, 5), Some(0.0));
        assert_eq!(d.at(inst, 25_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5, 5), Some(0.01));
    }

    #[test]
    fn text_and_visual_scores_prefer_instrument_over_market_wide_and_respect_recency() {
        let inst = InstrumentId::new();
        let other = InstrumentId::new();
        let mut d = DecoderOutputs::new();
        // market-wide news at 10 s, instrument-specific news at 8 s (older but specific), chart for inst at 9 s
        assert!(d.on_output(&scored("text.news", None, 10_000, -0.6), 10_000));
        assert!(d.on_output(&scored("text.news", Some(inst), 8_000, 0.4), 8_000));
        assert!(d.on_output(&scored("visual.chart", Some(inst), 9_000, 1.7), 9_000)); // clamped
        assert!(d.on_output(&scored("text.summary", None, 9_500, 0.2), 9_500));
        assert_eq!(d.families(), vec!["text.news".to_owned(), "text.summary".into(), "visual.chart".into()]);
        let v = d.at(inst, 12_000, DEFAULT_MAX_AGE_MS);
        assert_eq!(v.text_score("news"), Some(0.4), "instrument-specific beats the newer market-wide output");
        assert_eq!(v.text_score("summary"), Some(0.2), "market-wide serves every instrument");
        assert_eq!(v.text_score("narrative"), None, "topic without an output");
        assert_eq!(v.visual_score("chart"), Some(1.0), "clamped to [-1, 1]");
        assert_eq!(v.visual_score("future"), None);
        // another instrument only sees the market-wide output
        assert_eq!(d.at(other, 12_000, DEFAULT_MAX_AGE_MS).text_score("news"), Some(-0.6));
        assert_eq!(d.at(other, 12_000, DEFAULT_MAX_AGE_MS).visual_score("chart"), None);
        // recency: with a 3 s window at 12 s the 8 s instrument output is stale → market-wide (10 s) wins
        assert_eq!(d.at(inst, 12_000, 3_000).text_score("news"), Some(-0.6));
        // with a 1 s window nothing is fresh
        assert_eq!(d.at(inst, 12_000, 1_000).text_score("news"), None);
        // point in time: before the market-wide output existed, the instrument one is all there is
        assert_eq!(d.at(other, 9_000, DEFAULT_MAX_AGE_MS).text_score("news"), None);
        // `(score family)` with the dotted family: freshness of the preferred output
        let s = v.score("text.news").unwrap();
        assert!((s - (1.0 - 4_000.0 / DEFAULT_MAX_AGE_MS as f64)).abs() < 1e-9, "{s}");
        assert!(v.score("visual.chart").is_some() && v.score("visual.future").is_none());
        // market-wide flag and refs are kept
        let o = d.latest("text.summary", inst, 12_000, DEFAULT_MAX_AGE_MS).unwrap();
        assert!(o.market_wide && o.instrument.is_none());
        assert!(o.output_refs.iter().any(|r| r == "src:gnews"));
        // a non-finite score is no score
        d.on_output(&scored("text.narrative", Some(inst), 11_000, f64::NAN), 11_000);
        assert_eq!(d.at(inst, 12_000, DEFAULT_MAX_AGE_MS).text_score("narrative"), None);
    }

    #[test]
    fn forecast_horizon_from_refs_or_genome_text_and_closest_k_wins() {
        assert_eq!(horizon_of_genome("forecast:h=3;lb=8"), Some(3));
        assert_eq!(horizon_of_genome("retrieval:k=4"), None);
        assert_eq!(horizon_of_genome(r#"{"id":"j","family":"forecast","h":7}"#), Some(7));
        assert_eq!(horizon_of_genome(r#"{"horizon":2.0}"#), Some(2));
        assert_eq!(horizon_of_genome("gp-abc"), None);
        let inst = InstrumentId::new();
        let mut d = DecoderOutputs::new();
        let mut h3 = forecast(Some(inst), 10_000, &[0.3; 5]);
        h3.genome_id = "forecast:h=3;lb=8".into();
        let mut h8 = forecast(Some(inst), 11_000, &[0.8; 5]);
        h8.output_refs.push("h:8".into());
        let mut unknown = forecast(Some(inst), 12_000, &[0.0; 5]); // latest, horizon unknown
        unknown.genome_id = "abc".into();
        for m in [&h3, &h8, &unknown] {
            d.on_output(m, 12_000);
        }
        assert_eq!(horizon_of(&h3), Some(3));
        assert_eq!(horizon_of(&h8), Some(8));
        assert_eq!(horizon_of(&unknown), None);
        let v = d.at(inst, 13_000, DEFAULT_MAX_AGE_MS);
        assert_eq!(v.fc_quantile(0.5, 2), Some(0.3), "k = 2 → h = 3");
        assert_eq!(v.fc_quantile(0.5, 6), Some(0.8), "k = 6 → h = 8");
        assert_eq!(v.fc_dist(100), Some(0.8));
        assert_eq!(v.latest(FAMILY_FORECAST).map(|o| o.as_of_ms), Some(12_000), "`latest` is still the newest");
        // only unknown horizons → latest wins (previous behaviour)
        let mut d2 = DecoderOutputs::new();
        d2.on_output(&forecast(Some(inst), 10_000, &[0.1; 5]), 10_000);
        d2.on_output(&forecast(Some(inst), 11_000, &[0.2; 5]), 11_000);
        assert_eq!(d2.at(inst, 12_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5, 5), Some(0.2));
    }

    #[test]
    fn analog_outcome_uses_realised_prices_point_in_time() {
        let inst = InstrumentId::new();
        let mut d = DecoderOutputs::new();
        // analogs ending at 10 s (sim 1) and 20 s (sim 3); prices rise 1 %/s from 100 at 0 s
        d.on_output(&retrieval(inst, 30_000, &[(10_000, 1.0, 2.0), (20_000, 3.0, 4.0)]), 30_000);
        let prices = |at: i64| -> Option<f64> { (at >= 0).then(|| 100.0 * (1.0 + at as f64 / 1000.0 * 0.01)) };
        let v = d.at(inst, 30_000, DEFAULT_MAX_AGE_MS);
        let o = v.analog_outcome(5, Timescale::Second, &prices).unwrap();
        let r1 = (115.0f64 / 110.0).ln();
        let r2 = (125.0f64 / 120.0).ln();
        assert!((o - (r1 + 3.0 * r2) / 4.0).abs() < 1e-12);
        assert_eq!(v.analog_age(), Some(3.0));
        // k = 15 s: the second analog's outcome (35 s) is not realised at 30 s → only the first counts
        let det = v.analog_outcome_detail(15, Timescale::Second, &prices);
        assert!((det.value.unwrap() - (125.0f64 / 110.0).ln()).abs() < 1e-12);
        assert_eq!((det.resolved.len(), det.pending, det.unresolved.len()), (1, 1, 0));
        assert_eq!(det.resolved[0].window_end_ms, 10_000);
        // no prices → None, and both analogs are reported for prefetch
        let det = v.analog_outcome_detail(1, Timescale::Second, &|_: i64| None);
        assert_eq!(det.value, None);
        assert_eq!(
            det.unresolved,
            vec![
                AnalogRequest { window_end_ms: 10_000, k: 1, bar: Timescale::Second },
                AnalogRequest { window_end_ms: 20_000, k: 1, bar: Timescale::Second },
            ]
        );
        assert_eq!(det.unresolved[0].outcome_end_ms(), 11_000);
        // prices only after 15 s (short in-memory history): the first analog is unresolved, the second counts
        let late = |at: i64| -> Option<f64> { (at >= 15_000).then(|| prices(at).unwrap()) };
        let det = v.analog_outcome_detail(5, Timescale::Second, &late);
        assert!((det.value.unwrap() - r2).abs() < 1e-12);
        assert_eq!((det.resolved.len(), det.unresolved.len()), (1, 1));
        assert_eq!(det.unresolved[0].window_end_ms, 10_000);
        assert_eq!(DecoderOutputs::instrument_of(&retrieval(inst, 0, &[])), Some(inst));
        assert_eq!(DecoderOutputs::instrument_of(&scored("text.news", None, 0, 0.0)), None);
    }
}
