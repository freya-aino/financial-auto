//! The decoder-facing expert-space contract on the bus (spec 32 §4.2–§4.3, §6.6, §8):
//!
//! * `latent.expert.<timescale>.<instrument token>` — `fa.v1.ExpertView` per step: the expert
//!   latents of `LatentState` joined with their soft region membership, router gate, the
//!   expert's own next-step prediction and FSQ code (learned encoders), the family prior's
//!   novelty score (families 2 and 4) and a reference to the MCMC kernel subjects.
//! * `latent.expert.health` — `fa.v1.ExpertHealth` from **live** residuals: every closed bar
//!   scores the previous closed window's per-expert next-step prediction against the realised
//!   targets; every `every_ms` (≥ 30 rows) the spec 32 §6.4 measures (gated residual
//!   correlation, `K_eff`, per-expert skill, usage) over the last `window` rows are published
//!   and mirrored as heartbeat gauges `expert.<ts>.k_eff|resid_corr_max|skill_min`. A
//!   residual-correlation spike is the §8 health signal.

use std::collections::{BTreeMap, HashMap, VecDeque};

use fa_core::Timescale;
use fa_encoders::ExpertExtras;
use fa_expert_space::health::HealthReport;
use fa_proto::v1::{
    ExpertHealth, ExpertView, ExpertViewLatent, LatentState, McmcStateRef, PredictiveDist, RegionMembership,
};

/// `latent.expert.<ts>.<asset set>`.
pub fn subject(ts: Timescale, asset_set: &str) -> String {
    format!("latent.expert.{ts}.{asset_set}")
}

/// `latent.expert.health`.
pub const HEALTH_SUBJECT: &str = "latent.expert.health";

/// Builds the `ExpertView` of one step (all experts selected).
pub fn build(
    ts: Timescale,
    asset_set: &str,
    family: &str,
    state: &LatentState,
    membership: Option<&RegionMembership>,
    extras: Option<&ExpertExtras>,
) -> ExpertView {
    let experts: Vec<ExpertViewLatent> = state
        .experts
        .iter()
        .enumerate()
        .map(|(pos, e)| ExpertViewLatent {
            expert_id: e.expert_index,
            mu: e.mean.clone(),
            logvar: e.log_var.clone(),
            gate: state.router_weights.get(pos).copied().unwrap_or(0.0),
            region: membership.and_then(|m| m.experts.iter().find(|x| x.expert_index == e.expert_index).cloned()),
            next_step: extras.and_then(|x| x.next_step.get(pos)).map(|p| PredictiveDist {
                targets: extras.map(|x| x.targets.clone()).unwrap_or_default(),
                mean: p.clone(),
                log_var: Vec::new(),
            }),
            codes: extras.and_then(|x| x.codes.as_ref()).and_then(|c| c.get(pos)).map(|c| vec![*c]).unwrap_or_default(),
        })
        .collect();
    let prior = extras.and_then(|x| x.prior_score).filter(|v| v.is_finite());
    ExpertView {
        snapshot_id: state.snapshot_id.clone(),
        timescale: ts.to_string(),
        t: state.as_of,
        instrument_set: asset_set.to_owned(),
        expert_mask: vec![true; experts.len()],
        mcmc_state: Some(McmcStateRef {
            region_ids: state.experts.iter().map(|e| e.region_id.clone()).collect(),
            kernel_subject: format!("latent.mcmc.{ts}"),
            sampler_id: String::new(),
        }),
        experts,
        encoder_family: family.to_owned(),
        prior_score: prior.unwrap_or(0.0),
        has_prior_score: prior.is_some(),
    }
}

/// One scored step: scalarised residual / squared error per expert, naive squared error, gates.
#[derive(Debug, Clone)]
struct Row {
    resid: Vec<f64>,
    sq: Vec<f64>,
    y2: f64,
    gates: Vec<f32>,
    probs: Vec<f32>,
}

/// A closed window's `(next_step [expert][target], gates, router probabilities)`.
type Pending = (Vec<Vec<f32>>, Vec<f32>, Vec<f32>);

/// Rolling live health of one timescale's serving snapshot.
#[derive(Debug, Default)]
pub struct LiveHealth {
    snapshot_id: String,
    n_experts: usize,
    /// Previous closed window per instrument: `(next_step [expert][target], gates, probs)`.
    last: HashMap<String, Pending>,
    rows: VecDeque<Row>,
    last_publish_ms: i64,
}

/// Live-health settings.
#[derive(Debug, Clone, Copy)]
pub struct HealthCfg {
    pub window: usize,
    pub every_ms: i64,
    pub min_rows: usize,
}

impl Default for HealthCfg {
    fn default() -> Self {
        Self { window: 2000, every_ms: 300_000, min_rows: 30 }
    }
}

impl LiveHealth {
    /// Scores a closed step; returns a report when one is due.
    pub fn observe(
        &mut self,
        cfg: &HealthCfg,
        ts: Timescale,
        instrument: &str,
        state: &LatentState,
        extras: &ExpertExtras,
        as_of_ms: i64,
    ) -> Option<ExpertHealth> {
        let k = extras.next_step.len();
        if k == 0 {
            return None;
        }
        if self.snapshot_id != state.snapshot_id || self.n_experts != k {
            *self = Self { snapshot_id: state.snapshot_id.clone(), n_experts: k, ..Default::default() };
        }
        if let (Some((pred, gates, probs)), Some(y)) = (self.last.get(instrument), &extras.realised_prev) {
            let mut resid = Vec::with_capacity(k);
            let mut sq = Vec::with_capacity(k);
            for p in pred {
                let r: Vec<f64> = p.iter().zip(y).map(|(a, b)| f64::from(a - b)).collect();
                let n = r.len().max(1) as f64;
                resid.push(r.iter().sum::<f64>() / n);
                sq.push(r.iter().map(|v| v * v).sum::<f64>() / n);
            }
            let y2 = y.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / y.len().max(1) as f64;
            self.rows.push_back(Row { resid, sq, y2, gates: gates.clone(), probs: probs.clone() });
            while self.rows.len() > cfg.window {
                self.rows.pop_front();
            }
        }
        self.last.insert(
            instrument.to_owned(),
            (extras.next_step.clone(), state.router_weights.clone(), extras.probs.clone()),
        );
        if self.rows.len() < cfg.min_rows || as_of_ms - self.last_publish_ms < cfg.every_ms {
            return None;
        }
        self.last_publish_ms = as_of_ms;
        let n = self.rows.len();
        let resid: Vec<Vec<f64>> = (0..k).map(|e| self.rows.iter().map(|r| r.resid[e]).collect()).collect();
        let sq: Vec<Vec<f64>> = (0..k).map(|e| self.rows.iter().map(|r| r.sq[e]).collect()).collect();
        let y2: Vec<f64> = self.rows.iter().map(|r| r.y2).collect();
        let gates: Vec<Vec<f32>> = self.rows.iter().map(|r| r.gates.clone()).collect();
        let usage: Vec<f64> = (0..k)
            .map(|e| {
                self.rows.iter().map(|r| f64::from(r.probs.get(e).copied().unwrap_or(0.0))).sum::<f64>() / n as f64
            })
            .collect();
        let h = HealthReport::compute_gated(&resid, &sq, &y2, &gates, usage, cfg.min_rows);
        Some(ExpertHealth {
            snapshot_id: self.snapshot_id.clone(),
            timescale: ts.to_string(),
            as_of: Some(crate::pipeline::ms_to_ts(as_of_ms)),
            source: "live".into(),
            expert_ids: state.experts.iter().map(|e| e.expert_index).collect(),
            usage: h.usage.clone(),
            skill: h.skill.clone(),
            resid_corr: h.resid_corr.iter().flatten().copied().collect(),
            k_eff: h.k_eff,
            resid_corr_max: h.resid_corr_max,
            n_windows: n as u64,
        })
    }
}

/// Live health per timescale.
#[derive(Debug, Default)]
pub struct ExpertPublisher {
    pub cfg: HealthCfg,
    live: BTreeMap<Timescale, LiveHealth>,
}

impl ExpertPublisher {
    /// The view of a step and, on closed bars, a due health report.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        ts: Timescale,
        token: &str,
        family: &str,
        closed: bool,
        state: &LatentState,
        membership: Option<&RegionMembership>,
        extras: Option<&ExpertExtras>,
        as_of_ms: i64,
    ) -> (ExpertView, Option<ExpertHealth>) {
        let view = build(ts, token, family, state, membership, extras);
        let health = match (closed, extras) {
            (true, Some(x)) => {
                let cfg = self.cfg;
                self.live.entry(ts).or_default().observe(&cfg, ts, token, state, x, as_of_ms)
            }
            _ => None,
        };
        (view, health)
    }
}

/// Publishes a step's view (and a due health report) and counts them.
pub async fn publish(
    bus: &fa_bus::Bus,
    counters: &tokio::sync::Mutex<BTreeMap<String, f64>>,
    ts: Timescale,
    token: &str,
    view: ExpertView,
    health: Option<ExpertHealth>,
) {
    let mut published = vec![];
    match bus.publish(subject(ts, token), &view).await {
        Ok(()) => published.push(format!("expert_views.{ts}")),
        Err(e) => tracing::warn!(error = %e, "expert view publish failed"),
    }
    let mut g = Vec::new();
    if let Some(h) = health {
        match bus.publish(HEALTH_SUBJECT.to_owned(), &h).await {
            Ok(()) => {
                published.push(format!("expert_health.{ts}"));
                g = gauges(&h);
            }
            Err(e) => tracing::warn!(error = %e, "expert health publish failed"),
        }
    }
    let mut c = counters.lock().await;
    for k in published {
        *c.entry(k).or_default() += 1.0;
    }
    c.extend(g);
}

/// Heartbeat gauges of a health report.
pub fn gauges(h: &ExpertHealth) -> Vec<(String, f64)> {
    let ts = &h.timescale;
    let mut g = vec![
        (format!("expert.{ts}.k_eff"), h.k_eff),
        (format!("expert.{ts}.resid_corr_max"), h.resid_corr_max),
        (format!("expert.{ts}.health_windows"), h.n_windows as f64),
    ];
    if let Some(m) = h.skill.iter().copied().reduce(f64::min) {
        g.push((format!("expert.{ts}.skill_min"), m));
    }
    g
}

#[cfg(test)]
mod tests {
    use fa_proto::Message;
    use fa_proto::v1::{ExpertLatent, ExpertMembership, RegionWeight};

    use super::*;

    fn state(snapshot: &str, gates: Vec<f32>) -> LatentState {
        LatentState {
            snapshot_id: snapshot.into(),
            timescale: "1m".into(),
            instrument_id: "i".into(),
            as_of: Some(crate::pipeline::ms_to_ts(60_000)),
            experts: (0..2)
                .map(|e| ExpertLatent {
                    expert_index: e,
                    mean: vec![0.1 * e as f32; 3],
                    log_var: vec![-2.0; 3],
                    region_id: format!("r{e}"),
                    region_lineage_id: String::new(),
                })
                .collect(),
            router_weights: gates,
        }
    }

    #[test]
    fn view_carries_every_spec_32_field_and_round_trips() {
        let st = state("s", vec![0.7, 0.3]);
        let mem = RegionMembership {
            experts: vec![ExpertMembership {
                expert_index: 1,
                weights: vec![RegionWeight { region_id: "r1".into(), weight: 1.0 }],
                novelty: 0.2,
                out_of_region: false,
            }],
            ..Default::default()
        };
        let x = ExpertExtras {
            targets: vec!["next_ret".into(), "next_abs_ret".into(), "next_range".into()],
            next_step: vec![vec![0.1, 0.2, 0.3], vec![-0.1, 0.0, 0.1]],
            codes: Some(vec![5, 9]),
            prior_score: Some(1.5),
            realised_prev: None,
            probs: vec![0.7, 0.3],
        };
        let v = build(Timescale::Minute, "tok", "hier_vae_flow", &st, Some(&mem), Some(&x));
        assert_eq!(v.experts.len(), 2);
        assert_eq!(v.experts[0].gate, 0.7);
        assert_eq!(v.experts[1].region.as_ref().unwrap().weights[0].region_id, "r1");
        assert!(v.experts[0].region.is_none());
        assert_eq!(v.experts[1].next_step.as_ref().unwrap().mean, vec![-0.1, 0.0, 0.1]);
        assert_eq!(v.experts[1].codes, vec![9]);
        assert_eq!(v.mcmc_state.as_ref().unwrap().region_ids, vec!["r0", "r1"]);
        assert!(v.has_prior_score && v.prior_score == 1.5);
        assert_eq!(v.expert_mask, vec![true, true]);
        let back = ExpertView::decode(v.encode_to_vec().as_slice()).unwrap();
        assert_eq!(back, v);
        // spec 32 §9.3: the same bytes decode in Go (go/pkg/bus/expert_view_test.go)
        let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/../../go/pkg/bus/testdata/expert_view.bin");
        if std::env::var("FA_UPDATE_FIXTURES").is_ok() {
            std::fs::create_dir_all(std::path::Path::new(fixture).parent().unwrap()).unwrap();
            std::fs::write(fixture, v.encode_to_vec()).unwrap();
        }
        let golden = std::fs::read(fixture).expect("fixture (FA_UPDATE_FIXTURES=1 writes it)");
        assert_eq!(ExpertView::decode(golden.as_slice()).unwrap(), v);
        // the statistical family has no extras: still a valid view
        let plain = build(Timescale::Minute, "tok", "stat_baseline", &st, None, None);
        assert!(plain.experts.iter().all(|e| e.next_step.is_none()) && !plain.has_prior_score);
        assert_eq!(subject(Timescale::Minute, "tok"), "latent.expert.1m.tok");
    }

    #[test]
    fn live_health_scores_previous_predictions() {
        let mut p = ExpertPublisher { cfg: HealthCfg { window: 500, every_ms: 0, min_rows: 30 }, ..Default::default() };
        let mut rng = fa_core::rng::Rng::seed(1);
        let ys: Vec<Vec<f32>> = (0..82).map(|_| (0..3).map(|_| rng.normal() as f32).collect()).collect();
        for i in 0..80i64 {
            // expert 0 forecasts the next realised targets closely, expert 1 is unrelated noise
            let next0: Vec<f32> = ys[i as usize + 1].iter().map(|v| v + 0.1 * rng.normal() as f32).collect();
            let x = ExpertExtras {
                targets: vec![],
                next_step: vec![next0, (0..3).map(|_| rng.normal() as f32).collect()],
                codes: None,
                prior_score: None,
                realised_prev: Some(ys[i as usize].clone()),
                probs: vec![0.5, 0.5],
            };
            let (_, h) = p.step(
                Timescale::Minute,
                "tok",
                "jepa_vae",
                true,
                &state("s", vec![0.5, 0.5]),
                None,
                Some(&x),
                i * 60_000,
            );
            if let Some(h) = h {
                assert_eq!(h.source, "live");
                assert_eq!(h.resid_corr.len(), 4);
                assert!(h.skill[0] > 0.8, "{:?}", h.skill);
                assert!(h.skill[1] < h.skill[0]);
                assert!(h.k_eff > 1.0 && h.k_eff <= 2.0);
                assert!(gauges(&h).iter().any(|(k, _)| k == "expert.1m.k_eff"));
                return;
            }
        }
        panic!("no health report after 80 closed steps");
    }
}
