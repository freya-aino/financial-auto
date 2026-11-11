//! Per-source knowledge delays (spec 41 §6.2, §7 `backfill.knowledge_delay.<source>`): an event
//! stored at its **event time** (VictoriaMetrics trade samples, backfilled bars) became known to
//! the live system only when it reached the bus. Polled feeds publish late — measured on the
//! retained `MD` stream on 2026-11-11 (`examples/knowledge_delay.rs`, 100 min, publish time −
//! `event_time`): Kraken trades p50 43 s, p99 87 s, max 117 s; Yahoo p50 51 s, p99 94 s; the
//! synthetic `sim` feed 0 s. An event-time replay of such a feed sees prices the live venue and
//! strategies did not know yet (look-ahead). Replays therefore shift every series of a source by
//! its catalog delay before use: knowledge time = event time + delay.
//!
//! The delays come from `config/sources.yaml` (`sources[].backfill.knowledge_delay_secs`),
//! located through `FA_SOURCES` (the runtime image sets it) or `config/sources.yaml` relative to
//! the working directory; [`KnowledgeDelays::builtin`] carries the same numbers for processes
//! that have neither (a test pins them to the repo catalog).

use std::{collections::BTreeMap, sync::OnceLock};

use serde::{Deserialize, Serialize};

use crate::replay::BarSeries;

/// Source → knowledge delay (ms) after the event time.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct KnowledgeDelays {
    pub by_source: BTreeMap<String, i64>,
    /// Delay of sources without an entry.
    pub default_ms: i64,
}

#[derive(Deserialize)]
struct CatalogFile {
    #[serde(default)]
    sources: Vec<CatalogSource>,
}

#[derive(Deserialize)]
struct CatalogSource {
    id: String,
    #[serde(default)]
    backfill: Option<Backfill>,
}

#[derive(Deserialize)]
struct Backfill {
    #[serde(default)]
    knowledge_delay_secs: Option<f64>,
}

impl KnowledgeDelays {
    /// No delays (event time = knowledge time): recordings already stamped at receive time.
    pub fn none() -> Self {
        Self::default()
    }

    /// The measured delays of the repo catalog (kept equal to `config/sources.yaml` by a test).
    pub fn builtin() -> Self {
        Self { by_source: [("kraken".to_owned(), 120_000), ("yahoo".to_owned(), 100_000)].into(), default_ms: 0 }
    }

    /// Reads `sources[].backfill.knowledge_delay_secs` from a `sources.yaml`.
    pub fn from_sources_yaml(src: &str) -> Result<Self, serde_yaml::Error> {
        let f: CatalogFile = serde_yaml::from_str(src)?;
        let by_source = f
            .sources
            .into_iter()
            .filter_map(|s| {
                let secs = s.backfill?.knowledge_delay_secs?;
                (secs.is_finite() && secs > 0.0).then(|| (s.id, (secs * 1000.0).round() as i64))
            })
            .collect();
        Ok(Self { by_source, default_ms: 0 })
    }

    /// The process-wide catalog: `FA_SOURCES`, else `config/sources.yaml`, else [`Self::builtin`].
    pub fn catalog() -> &'static Self {
        static CATALOG: OnceLock<KnowledgeDelays> = OnceLock::new();
        CATALOG.get_or_init(|| {
            let path = std::env::var("FA_SOURCES").unwrap_or_else(|_| "config/sources.yaml".into());
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| Self::from_sources_yaml(&s).ok())
                .filter(|d| !d.by_source.is_empty())
                .unwrap_or_else(Self::builtin)
        })
    }

    /// Delay of a source (a market-data venue token is the source id for the exchange feeds).
    pub fn delay_ms(&self, source: &str) -> i64 {
        self.by_source.get(source).copied().unwrap_or(self.default_ms).max(0)
    }

    /// The series as the live system knew it: every point moved to event time + delay of its
    /// venue / source. Series of zero-delay sources come back unchanged.
    pub fn apply(&self, bars: &[BarSeries]) -> Vec<BarSeries> {
        bars.iter()
            .map(|b| {
                let d = self.delay_ms(&b.venue);
                if d == 0 {
                    b.clone()
                } else {
                    BarSeries { ts_ms: b.ts_ms.iter().map(|t| t + d).collect(), ..b.clone() }
                }
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.default_ms == 0 && self.by_source.values().all(|d| *d == 0)
    }
}

/// Serde / constructor default: a copy of [`KnowledgeDelays::catalog`].
pub fn catalog_delays() -> KnowledgeDelays {
    KnowledgeDelays::catalog().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_repo_catalog_carries_the_measured_delays() {
        let d = KnowledgeDelays::from_sources_yaml(include_str!("../../../config/sources.yaml")).unwrap();
        assert_eq!(d, KnowledgeDelays::builtin(), "keep builtin() equal to config/sources.yaml");
        assert_eq!(d.delay_ms("kraken"), 120_000);
        assert_eq!(d.delay_ms("sim"), 0);
    }

    #[test]
    fn series_move_to_knowledge_time() {
        let k = BarSeries::from_points("a", "kraken", &[(0, 1.0), (1_000, 2.0)], 1_000);
        let s = BarSeries::from_points("b", "sim", &[(0, 1.0)], 1_000);
        let out = KnowledgeDelays::builtin().apply(&[k.clone(), s.clone()]);
        assert_eq!(out[0].ts_ms, vec![121_000, 122_000]);
        assert_eq!(out[1], s);
        assert!(KnowledgeDelays::none().is_empty() && !KnowledgeDelays::builtin().is_empty());
    }
}
