//! Shadow serving (spec 36 §6.4: "the runtime can serve multiple snapshots simultaneously"):
//! a candidate snapshot runs alongside the active one on the same bars.
//!
//! `ctl.model.shadow {timescale, snapshot_id | none}` loads the candidate through the same
//! resolver as a reload and wraps it in a [`Shadow`]: its own [`TimescalePipeline`] (region sets
//! seeded from the candidate's `region_map`, transitions observed in memory). Every step of the
//! active pipeline is repeated on the shadow; its `LatentState` goes to
//! `latent.shadow.<ts>.<instrument token>` (inside the `LATENT` stream through `latent.>`), while
//! its region events and transition rows stay in memory (never published, never written to
//! Helix). The shadow's agreement with the active snapshot is tracked per expert step and
//! reported as heartbeat gauges (see [`Shadow::gauges`]) that control-api's promotion check reads.

use std::{collections::HashMap, time::Instant};

use fa_core::Timescale;
use fa_patterns::RegionMap;
use fa_proto::v1::LatentState;

use crate::pipeline::{StepInput, TimescalePipeline};

/// Subject of a shadow state: `latent.shadow.<ts>.<instrument token>`.
pub fn shadow_subject(ts: Timescale, token: &str) -> String {
    format!("latent.shadow.{ts}.{token}")
}

/// Region agreement of one step: `(matching experts, compared experts)`.
///
/// An expert agrees when the shadow's region continues the active region's lineage: equal
/// non-empty `region_lineage_id`s (a matched region inherits its parent's lineage, D-007), or the
/// shadow region's `parent_region_id` in the candidate's `region_map` is the active region.
pub fn agreement(active: &LatentState, shadow: &LatentState, parent_of: &HashMap<String, String>) -> (u64, u64) {
    let mut matches = 0;
    let mut compared = 0;
    for a in &active.experts {
        let Some(s) = shadow.experts.iter().find(|s| s.expert_index == a.expert_index) else {
            continue;
        };
        compared += 1;
        let same_lineage = !a.region_lineage_id.is_empty() && a.region_lineage_id == s.region_lineage_id;
        let mapped = parent_of.get(&s.region_id).is_some_and(|p| *p == a.region_id);
        if same_lineage || mapped {
            matches += 1;
        }
    }
    (matches, compared)
}

/// `region_id → parent_region_id` of every matched region in a map.
pub fn parent_index(map: Option<&RegionMap>) -> HashMap<String, String> {
    map.map(|m| {
        m.experts
            .iter()
            .flat_map(|e| e.regions.iter())
            .filter_map(|r| r.parent_region_id.clone().map(|p| (r.region_id.clone(), p)))
            .collect()
    })
    .unwrap_or_default()
}

/// A candidate snapshot served in shadow on one timescale.
pub struct Shadow {
    pub pipeline: TimescalePipeline,
    pub snapshot_id: String,
    pub generation: u32,
    /// `region_map.carried_mass()` of the candidate (spec 36 §6.4 item 3).
    pub carried_mass: f64,
    pub started: Instant,
    parent_of: HashMap<String, String>,
    /// Shadow states produced (closed bars and previews).
    pub states: u64,
    pub matches: u64,
    pub compared: u64,
}

impl Shadow {
    pub fn new(pipeline: TimescalePipeline, generation: u32, region_map: Option<&RegionMap>) -> Self {
        Self {
            snapshot_id: pipeline.snapshot_id().to_owned(),
            generation,
            carried_mass: region_map.map_or(0.0, RegionMap::carried_mass),
            parent_of: parent_index(region_map),
            pipeline,
            started: Instant::now(),
            states: 0,
            matches: 0,
            compared: 0,
        }
    }

    /// Runs the step the active pipeline just ran; returns the shadow state to publish.
    pub fn step(&mut self, input: &StepInput<'_>, active: Option<&LatentState>) -> Option<LatentState> {
        let out = match self.pipeline.step(input) {
            Ok(o) => o,
            Err(e) => {
                tracing::debug!(snapshot = %self.snapshot_id, error = %e, "shadow encode failed");
                return None;
            }
        };
        let state = out.state?;
        self.states += 1;
        if let Some(a) = active {
            let (m, c) = agreement(a, &state, &self.parent_of);
            self.matches += m;
            self.compared += c;
        }
        Some(state)
    }

    /// Share of compared expert steps whose regions agree (0 before any comparison).
    pub fn region_agreement(&self) -> f64 {
        if self.compared == 0 { 0.0 } else { self.matches as f64 / self.compared as f64 }
    }

    pub fn minutes(&self) -> f64 {
        self.started.elapsed().as_secs_f64() / 60.0
    }

    /// Heartbeat gauges: `shadow.<ts>.snapshot` (the candidate's generation; the counters map
    /// only carries numbers, so the id itself travels as `shadow.<ts>.snapshot_id.<id> = 1`),
    /// `shadow.<ts>.states`, `shadow.<ts>.minutes`, `shadow.<ts>.region_agreement`,
    /// `shadow.<ts>.carried_mass`.
    pub fn gauges(&self, ts: Timescale) -> Vec<(String, f64)> {
        vec![
            (format!("shadow.{ts}.snapshot"), f64::from(self.generation)),
            (format!("shadow.{ts}.snapshot_id.{}", self.snapshot_id), 1.0),
            (format!("shadow.{ts}.states"), self.states as f64),
            (format!("shadow.{ts}.minutes"), self.minutes()),
            (format!("shadow.{ts}.region_agreement"), self.region_agreement()),
            (format!("shadow.{ts}.carried_mass"), self.carried_mass),
        ]
    }
}

/// Prefix of every shadow gauge of a timescale (to clear them when shadowing stops).
pub fn gauge_prefix(ts: Timescale) -> String {
    format!("shadow.{ts}.")
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::ExpertLatent;

    use super::*;
    use crate::pipeline::PipelineConfig;

    fn state(regions: &[(&str, &str)]) -> LatentState {
        LatentState {
            experts: regions
                .iter()
                .enumerate()
                .map(|(k, (r, l))| ExpertLatent {
                    expert_index: k as u32,
                    region_id: (*r).into(),
                    region_lineage_id: (*l).into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn agreement_by_lineage_or_region_map() {
        let active = state(&[("1m-e0-r000", "1m-e0-r000"), ("1m-e1-r002", "1m-e1-r002"), ("1m-e2-r001", "1m-e2-r001")]);
        // e0: inherited lineage; e1: parent via map only; e2: different region
        let shadow = state(&[("c-e0-r000", "1m-e0-r000"), ("c-e1-r004", "c-e1-r004"), ("c-e2-r000", "c-e2-r000")]);
        let parents = HashMap::from([("c-e1-r004".to_owned(), "1m-e1-r002".to_owned())]);
        assert_eq!(agreement(&active, &shadow, &parents), (2, 3));
        assert_eq!(agreement(&active, &shadow, &HashMap::new()), (1, 3));
        // empty lineages never count as equal
        assert_eq!(agreement(&state(&[("a", "")]), &state(&[("b", "")]), &HashMap::new()), (0, 1));
        // missing experts are not compared
        assert_eq!(agreement(&active, &state(&[]), &parents), (0, 0));
    }

    #[test]
    fn identical_shadow_agrees_everywhere_and_reports_gauges() {
        let cfg = PipelineConfig::default();
        let mut active = TimescalePipeline::baseline(Timescale::Minute, &cfg);
        let mut sh = Shadow::new(TimescalePipeline::baseline(Timescale::Minute, &cfg), 0, None);
        let mut rng = fa_core::rng::Rng::seed(4);
        let mut p = 100.0;
        let bars: Vec<fa_encoders::MarketBar> = (0..30)
            .map(|_| {
                p *= 1.0 + 0.003 * rng.normal();
                fa_encoders::MarketBar::from_trade(p, 1.0)
            })
            .collect();
        for end in 2..=bars.len() {
            let input =
                StepInput { instrument_id: "i", as_of_ms: end as i64 * 60_000, bars: &bars[..end], bar_closed: true };
            let a = active.step(&input).unwrap().state;
            assert!(sh.step(&input, a.as_ref()).is_some());
        }
        assert_eq!(sh.states, 29);
        assert_eq!(sh.compared, 29 * 3);
        assert_eq!(sh.region_agreement(), 1.0, "the same encoder walks the same regions");
        let g: HashMap<String, f64> = sh.gauges(Timescale::Minute).into_iter().collect();
        assert_eq!(g["shadow.1m.snapshot"], 0.0);
        assert_eq!(g["shadow.1m.snapshot_id.stat-v1-1m"], 1.0);
        assert_eq!(g["shadow.1m.states"], 29.0);
        assert_eq!(g["shadow.1m.region_agreement"], 1.0);
        assert_eq!(g["shadow.1m.carried_mass"], 0.0);
        assert!(g["shadow.1m.minutes"] >= 0.0);
        assert!(g.keys().all(|k| k.starts_with(&gauge_prefix(Timescale::Minute))));
        assert_eq!(shadow_subject(Timescale::Minute, "ab"), "latent.shadow.1m.ab");
    }
}
