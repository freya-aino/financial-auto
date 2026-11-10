//! Recency cache of `decoder.output.<family>` messages (spec 34) behind the spec 35 §4.2
//! decoder primitives.
//!
//! `evo-worker` publishes a `DecoderOutput` per evaluated genome and instrument on
//! `decoder.output.rnn_prior` (forecast `quantiles`) and `decoder.output.retrieval`
//! (`analogs`). The message has no instrument field; the instrument travels in
//! `output_refs` as `inst:<uuid>` (outputs without it are kept under the "any instrument"
//! key and serve as a fallback). Outputs are kept per `(family, instrument)` in event-time
//! order, bounded, and every query is point-in-time with a recency window: the latest output
//! with `as_of ≤ at_ms` and `as_of ≥ at_ms − max_age` wins.
//!
//! | primitive | source |
//! |---|---|
//! | `fc_quantile(q, k)` | linear interpolation of the latest `rnn_prior` quantile vector at level `q` (levels are the decoder defaults `0.05 0.25 0.5 0.75 0.95` for five values, else `(i+1)/(n+1)`); `k` is informational: the output is for the decoder's own horizon |
//! | `fc_dist(k)` | mean of that quantile vector (expected forecast return) |
//! | `analog_outcome(k)` | similarity-weighted mean realised `k`-bar log return after each analog's `window_end`, from the caller's price history ([`PriceLookup`]) |
//! | `analog_age` | mean `time_distance_days` of the analogs |
//! | `score(family)` | freshness of the latest output of `family`: `1 − age / max_age` |
//! | `text_score(topic)`, `visual_score(kind)` | **no producer yet**: no `text.*` / `visual.*` family is published, so they answer 0 (reported as fallback) |
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
    pub quantiles: Vec<f64>,
    pub analogs: Vec<Analog>,
}

impl Output {
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

    /// Latest output of `family` for `inst` as of `at_ms` within `max_age_ms`; falls back to
    /// outputs without an instrument reference. `family` may be a prefix (`text.` matches `text.news`).
    pub fn latest(&self, family: &str, inst: InstrumentId, at_ms: i64, max_age_ms: i64) -> Option<&Output> {
        let pick = |key_inst: Option<InstrumentId>| {
            self.by_key
                .iter()
                .filter(|((f, i), _)| *i == key_inst && (f == family || f.starts_with(family)))
                .filter_map(|(_, q)| q.iter().rev().find(|o| o.as_of_ms <= at_ms && at_ms - o.as_of_ms <= max_age_ms))
                .max_by_key(|o| o.as_of_ms)
        };
        pick(Some(inst)).or_else(|| pick(None))
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

    /// `fc_quantile(q, k)`.
    pub fn fc_quantile(&self, q: f64) -> Option<f64> {
        self.latest(FAMILY_FORECAST)?.quantile(q)
    }

    /// `fc_dist(k)` as its expected value.
    pub fn fc_dist(&self) -> Option<f64> {
        self.latest(FAMILY_FORECAST)?.mean()
    }

    /// `analog_outcome(k)`: similarity-weighted mean of `ln(p(window_end + k·bar) / p(window_end))`
    /// over analogs whose prices are available; `None` when none is.
    pub fn analog_outcome(&self, k: u32, bar: Timescale, prices: &dyn PriceLookup) -> Option<f64> {
        let out = self.latest(FAMILY_RETRIEVAL)?;
        let step = bar.duration().as_millis() as i64 * i64::from(k.max(1));
        let mut num = 0.0;
        let mut den = 0.0;
        for a in &out.analogs {
            let Some(end) = a.window_end_ms else { continue };
            if end + step > self.at_ms {
                continue; // outcome not realised yet (point-in-time)
            }
            let (Some(p0), Some(p1)) = (prices.price_at(end), prices.price_at(end + step)) else { continue };
            if p0 <= 0.0 || p1 <= 0.0 {
                continue;
            }
            let w = a.similarity.max(0.0);
            num += w * (p1 / p0).ln();
            den += w;
        }
        (den > 0.0).then(|| num / den)
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

    /// `text_score(topic)`: `text.*` outputs are not produced yet; `None` until a producer exists.
    pub fn text_score(&self, topic: &str) -> Option<f64> {
        self.latest(&format!("text.{topic}")).or_else(|| self.latest("text.")).map(|_| 0.0)
    }

    /// `visual_score(kind)`: `visual.*` outputs are not produced yet; `None` until a producer exists.
    pub fn visual_score(&self, kind: &str) -> Option<f64> {
        self.latest(&format!("visual.{kind}")).or_else(|| self.latest("visual.")).map(|_| 0.0)
    }
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
        assert_eq!(v.fc_quantile(0.5), Some(0.01));
        assert_eq!(v.fc_quantile(0.0), Some(-0.01));
        assert_eq!(v.fc_quantile(1.0), Some(0.03));
        let q = v.fc_quantile(0.375).unwrap(); // halfway between 0.25 and 0.5 levels
        assert!((q - 0.005).abs() < 1e-12, "{q}");
        assert_eq!(v.fc_dist(), Some(0.01));
        assert!(v.score(FAMILY_FORECAST).unwrap() > 0.99);
        // point in time: at 12 s only the first output is visible
        assert_eq!(d.at(inst, 12_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5), Some(0.0));
        // recency window: nothing within 1 s of 30 s
        assert_eq!(d.at(inst, 30_000, 1_000).fc_quantile(0.5), None);
        // another instrument falls back to the instrument-less output (levels (i+1)/(n+1) → median 0.2)
        assert_eq!(d.at(other, 25_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5), Some(0.2));
        assert_eq!(d.at(other, 14_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5), None);
        // nothing published for text/visual
        assert_eq!(v.text_score("news"), None);
        assert_eq!(v.visual_score("future"), None);
        assert_eq!(v.analog_age(), None);
        // late arrival keeps order
        assert!(d.on_output(&forecast(Some(inst), 5_000, &[0.0; 5]), 30_000));
        assert_eq!(d.at(inst, 6_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5), Some(0.0));
        assert_eq!(d.at(inst, 25_000, DEFAULT_MAX_AGE_MS).fc_quantile(0.5), Some(0.01));
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
        let o15 = v.analog_outcome(15, Timescale::Second, &prices).unwrap();
        assert!((o15 - (125.0f64 / 110.0).ln()).abs() < 1e-12);
        // no prices → None
        assert_eq!(v.analog_outcome(1, Timescale::Second, &|_: i64| None), None);
        assert_eq!(DecoderOutputs::instrument_of(&retrieval(inst, 0, &[])), Some(inst));
    }
}
