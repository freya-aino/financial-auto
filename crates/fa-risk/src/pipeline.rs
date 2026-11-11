//! Automatic strategy lifecycle (spec 35 §6.4, spec 40 §6.6, spec 41 §6.4–§6.6): the pure
//! decisions behind the closed loop evolution → holdout → shadow → paper → retirement.
//!
//! 1. **Selection** ([`select_candidates`]): members of a strategy generation's final front
//!    become `candidate`s when the generation's PBO is known and at most `pbo_max` (spec 35
//!    §6.2: an overfit generation promotes no one) and their selection-window deflated Sharpe
//!    (probability) is at least `dsr_min_candidate`; at most `max_candidates_per_generation`,
//!    best score first.
//! 2. **Holdout** ([`decide`] on `candidate`): exactly one evaluation on the locked holdout of
//!    the candidate's timescale (spec 41 §6.4, "used once per strategy"); retried only when the
//!    job produced no trial row, up to `holdout_max_attempts`. The verdict ([`holdout_passes`])
//!    needs `holdout_min_dsr`, a positive Sharpe and `holdout_min_obs` observations; failures
//!    are retired.
//! 3. **Admission**: `holdout_passed` → `shadow` while the live set has room
//!    (`max_live_evolved`); the I/O layer then rewrites the managed population and reloads
//!    strategy-runtime.
//! 4. **Promotion**: `shadow` → `active` when the spec 40 §6.6 gate is green and
//!    `auto_promote` is on (paper only, D-029); otherwise the operator promotes.
//! 5. **Demotion / retirement**: a shadow strategy is retired on a shadow drawdown breach, on a
//!    definitive gate failure once its minimum shadow time is over, when it exceeds
//!    `max_shadow_days`, or when strategy-runtime auto-paused it (NaN / step budget, spec 35 §8);
//!    an active one is demoted to shadow on a drawdown breach, gate decay (shadow-vs-backtest
//!    consistency failing on its live book) or a runtime auto-pause, and retired after
//!    `max_demotions` demotions.
//!
//! Everything here is deterministic and free of I/O; `bins/risk-engine/src/pipeline.rs` feeds
//! it from Garage, Postgres and the bus and applies the transitions.

use serde::{Deserialize, Serialize};

use crate::strategy::{Checklist, StrategyState, StrategyStatus};

/// `[pipeline]` in `risk.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelinePolicy {
    pub enabled: bool,
    /// Seconds between pipeline rounds.
    pub interval_secs: u64,
    /// Newest strategy generations (Garage `fa-artifacts/strategies/*.toml`) scanned per round.
    pub generations_scanned: usize,
    /// Spec 35 §7 `dsr_min_candidate`: selection deflated Sharpe probability for holdout eligibility.
    pub dsr_min_candidate: f64,
    /// Spec 35 §7 `pbo_max`: a generation above it promotes no one; unknown PBO blocks too.
    pub pbo_max: f64,
    pub max_candidates_per_generation: usize,
    /// Spec 41 §7 `dsr.min_for_promotion` applied to the holdout trial.
    pub holdout_min_dsr: f64,
    pub holdout_min_obs: u64,
    /// A requested holdout evaluation without a trial row after this long is retried.
    pub holdout_timeout_secs: u64,
    pub holdout_max_attempts: u32,
    /// Evolved strategies allowed in the live set at once (shadow + active + paused).
    pub max_live_evolved: usize,
    /// Promote automatically when the gate is green (paper only).
    pub auto_promote: bool,
    /// A shadow strategy that has not been promoted after this long is retired.
    pub max_shadow_days: f64,
    /// Demotions (active → shadow) before an evolved strategy is retired.
    pub max_demotions: u32,
    /// Hourly points of an active strategy's own book behind the decay check.
    pub decay_min_hours: usize,
}

impl Default for PipelinePolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 60,
            generations_scanned: 20,
            dsr_min_candidate: 0.95,
            pbo_max: 0.2,
            max_candidates_per_generation: 3,
            holdout_min_dsr: 0.95,
            holdout_min_obs: 100,
            holdout_timeout_secs: 1_800,
            holdout_max_attempts: 2,
            max_live_evolved: 8,
            auto_promote: true,
            max_shadow_days: 90.0,
            max_demotions: 2,
            decay_min_hours: 24,
        }
    }
}

/// Selection-window evidence of a candidate (its trial row and its generation's statistics).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Selection {
    /// Deflated Sharpe probability of the candidate's own selection trial.
    pub deflated_sharpe: Option<f64>,
    /// Per-step Sharpe on the selection window (the backtest side of the consistency check).
    pub sharpe: Option<f64>,
    pub n_obs: Option<u64>,
    /// PBO of the candidate's generation (latest loop).
    pub pbo: Option<f64>,
    /// Union of the candidate's selection-trial windows (ms): a holdout that intersects it was
    /// seen during selection and cannot test the candidate.
    #[serde(default)]
    pub window_ms: Option<(i64, i64)>,
}

impl Selection {
    /// Whether a holdout `[start, end]` is clean for this candidate (no overlap with any
    /// selection window; unknown windows are treated as clean).
    pub fn clean_for(&self, start_ms: i64, end_ms: i64) -> bool {
        self.window_ms.is_none_or(|(s, e)| e < start_ms || s > end_ms)
    }
}

/// One holdout trial (`validation.trial`, `kind = holdout`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HoldoutOutcome {
    pub holdout_id: String,
    pub trial_id: String,
    pub deflated_sharpe: Option<f64>,
    pub sharpe: Option<f64>,
    pub n_obs: u64,
    pub at_ms: i64,
}

/// Pipeline bookkeeping of an evolved strategy (`ledger.strategy_state.pipeline`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineRecord {
    /// Genome text (one-line inline TOML table, `fa_strategies::genome`).
    pub genome: String,
    pub generation: String,
    pub timescale: String,
    /// Decision cadence of the strategy (for scaling the per-step backtest Sharpe).
    pub step_secs: f64,
    pub score: Option<f64>,
    pub selection: Selection,
    pub holdout: Option<HoldoutOutcome>,
    /// Holdout the evaluation was requested on, its job and when.
    pub holdout_id: Option<String>,
    pub holdout_job_id: Option<String>,
    pub holdout_requested_ms: Option<i64>,
    pub holdout_attempts: u32,
    pub demotions: u32,
    pub admitted_ms: Option<i64>,
    /// Last thing the pipeline waited on or retired for.
    pub note: String,
}

/// A member of a generation's promoted front with its evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct FrontMember {
    pub strategy_id: String,
    pub genome: String,
    pub generation: String,
    pub timescale: String,
    pub step_secs: f64,
    pub score: Option<f64>,
    pub selection: Selection,
}

impl FrontMember {
    pub fn record(&self) -> PipelineRecord {
        PipelineRecord {
            genome: self.genome.clone(),
            generation: self.generation.clone(),
            timescale: self.timescale.clone(),
            step_secs: self.step_secs,
            score: self.score,
            selection: self.selection.clone(),
            ..PipelineRecord::default()
        }
    }
}

/// Chooses the candidates of one generation. Members already known (any status) are skipped;
/// the second value lists `(strategy_id, reason)` for every rejected member.
pub fn select_candidates(
    policy: &PipelinePolicy,
    members: &[FrontMember],
    generation_pbo: Option<f64>,
    known: impl Fn(&str) -> bool,
) -> (Vec<FrontMember>, Vec<(String, String)>) {
    let mut rejected = Vec::new();
    let fresh: Vec<&FrontMember> = members.iter().filter(|m| !known(&m.strategy_id)).collect();
    match generation_pbo {
        Some(p) if p.is_finite() && p <= policy.pbo_max => {}
        Some(p) => {
            for m in fresh {
                rejected.push((m.strategy_id.clone(), format!("generation PBO {p:.3} > pbo_max {}", policy.pbo_max)));
            }
            return (Vec::new(), rejected);
        }
        None => {
            for m in fresh {
                rejected.push((m.strategy_id.clone(), "generation PBO unknown".into()));
            }
            return (Vec::new(), rejected);
        }
    }
    let mut ok: Vec<&FrontMember> = Vec::new();
    for m in fresh {
        match m.selection.deflated_sharpe {
            Some(d) if d.is_finite() && d >= policy.dsr_min_candidate => ok.push(m),
            Some(d) => rejected.push((
                m.strategy_id.clone(),
                format!("deflated Sharpe {d:.3} < dsr_min_candidate {}", policy.dsr_min_candidate),
            )),
            None => rejected.push((m.strategy_id.clone(), "no selection trial".into())),
        }
    }
    ok.sort_by(|a, b| {
        let key = |m: &FrontMember| m.score.or(m.selection.deflated_sharpe).unwrap_or(f64::NEG_INFINITY);
        key(b).total_cmp(&key(a)).then_with(|| a.strategy_id.cmp(&b.strategy_id))
    });
    let chosen: Vec<FrontMember> = ok.iter().take(policy.max_candidates_per_generation).map(|m| (*m).clone()).collect();
    for m in ok.iter().skip(policy.max_candidates_per_generation) {
        rejected.push((m.strategy_id.clone(), "beyond max_candidates_per_generation".into()));
    }
    (chosen, rejected)
}

/// Holdout verdict: `Ok` passes, `Err(reason)` retires.
pub fn holdout_passes(policy: &PipelinePolicy, h: &HoldoutOutcome) -> Result<(), String> {
    if h.n_obs < policy.holdout_min_obs {
        return Err(format!("holdout: {} observations < {}", h.n_obs, policy.holdout_min_obs));
    }
    match (h.deflated_sharpe, h.sharpe) {
        (Some(d), Some(s)) if d.is_finite() && s.is_finite() => {
            if d < policy.holdout_min_dsr {
                Err(format!("holdout: deflated Sharpe {d:.3} < {}", policy.holdout_min_dsr))
            } else if s <= 0.0 {
                Err(format!("holdout: Sharpe {s:.4} ≤ 0"))
            } else {
                Ok(())
            }
        }
        _ => Err("holdout: statistics missing".into()),
    }
}

/// What the runtime and the risk-engine know about one strategy this round.
#[derive(Debug, Clone, Default)]
pub struct Observation<'a> {
    pub now_ms: i64,
    /// Locked holdout of the strategy's timescale, if any.
    pub holdout_id: Option<&'a str>,
    /// The holdout trial of this strategy on that holdout, if one exists.
    pub holdout_result: Option<HoldoutOutcome>,
    /// The live set has room for one more evolved strategy.
    pub live_slot_free: bool,
    /// Current gate checklist (shadow strategies).
    pub gate: Option<&'a Checklist>,
    /// strategy-runtime auto-paused the strategy (`nan` / `budget`), spec 35 §8.
    pub runtime_auto_paused: Option<&'a str>,
    /// Shadow-vs-backtest consistency of an **active** strategy's own book (gate decay).
    pub live_consistent: Option<bool>,
}

/// One pipeline step for one strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Nothing,
    Wait(String),
    RequestHoldout { holdout_id: String },
    HoldoutPassed(HoldoutOutcome),
    Admit,
    Promote,
    Demote(String),
    Retire(String),
}

/// The next step for an evolved strategy (baselines are never touched by the pipeline).
pub fn decide(policy: &PipelinePolicy, st: &StrategyState, obs: &Observation<'_>) -> Decision {
    use StrategyStatus as S;
    let Some(p) = st.pipeline.as_ref() else {
        return Decision::Nothing;
    };
    let days = |since: Option<i64>| since.map_or(0.0, |t| (obs.now_ms - t) as f64 / 86_400_000.0);
    match st.status {
        S::Retired => Decision::Nothing,
        S::Candidate => {
            if let Some(h) = &obs.holdout_result {
                return match holdout_passes(policy, h) {
                    Ok(()) => Decision::HoldoutPassed(h.clone()),
                    Err(e) => Decision::Retire(e),
                };
            }
            let timeout = policy.holdout_timeout_secs as i64 * 1000;
            if let Some(t) = p.holdout_requested_ms
                && obs.now_ms - t < timeout
            {
                return Decision::Wait("holdout evaluation running".into());
            }
            if p.holdout_attempts >= policy.holdout_max_attempts.max(1) {
                let why = if p.note.is_empty() { "no trial row".to_owned() } else { p.note.clone() };
                return Decision::Retire(format!("holdout evaluation failed {} times ({why})", p.holdout_attempts));
            }
            match obs.holdout_id {
                Some(id) => Decision::RequestHoldout { holdout_id: id.to_owned() },
                None => Decision::Wait(format!("no locked {} holdout", p.timescale)),
            }
        }
        S::HoldoutPassed => {
            if obs.live_slot_free {
                Decision::Admit
            } else {
                Decision::Wait(format!("live set full (max_live_evolved {})", policy.max_live_evolved))
            }
        }
        S::Shadow => {
            if let Some(r) = obs.runtime_auto_paused {
                return Decision::Retire(format!("strategy-runtime auto-paused it ({r})"));
            }
            let d = days(st.shadow_since_ms);
            if let Some(g) = obs.gate {
                if g.green {
                    return if policy.auto_promote { Decision::Promote } else { Decision::Wait("gate green".into()) };
                }
                let time_done = g.items.iter().any(|i| i.name == "min_shadow_days" && i.pass == Some(true))
                    && g.items.iter().any(|i| i.name == "min_horizons" && i.pass == Some(true));
                let failed: Vec<&str> = g
                    .items
                    .iter()
                    .filter(|i| {
                        i.pass == Some(false)
                            && matches!(i.name.as_str(), "deflated_sharpe" | "pbo" | "shadow_vs_backtest")
                    })
                    .map(|i| i.name.as_str())
                    .collect();
                if time_done && !failed.is_empty() {
                    return Decision::Retire(format!("gate failed after the shadow period: {}", failed.join(", ")));
                }
            }
            if d > policy.max_shadow_days {
                return Decision::Retire(format!("{d:.1} days in shadow > max_shadow_days {}", policy.max_shadow_days));
            }
            Decision::Wait(obs.gate.map_or_else(|| "shadow".into(), |g| format!("gate: {}", g.missing.join(", "))))
        }
        S::Paused => {
            // automatic drawdown pause (spec 40 §6.5) of an evolved strategy
            if !st.last_reason.starts_with("strategy drawdown") {
                return Decision::Nothing; // operator pause: leave it to the operator
            }
            match st.prev_status {
                Some(S::Shadow) => Decision::Retire(format!("shadow {}", st.last_reason)),
                _ if p.demotions + 1 >= policy.max_demotions.max(1) => {
                    Decision::Retire(format!("{} after {} demotion(s)", st.last_reason, p.demotions))
                }
                _ => Decision::Demote(st.last_reason.clone()),
            }
        }
        S::Active => {
            let reason = if let Some(r) = obs.runtime_auto_paused {
                Some(format!("strategy-runtime auto-paused it ({r})"))
            } else if obs.live_consistent == Some(false) {
                Some("gate decay: live Sharpe below the backtest band".to_owned())
            } else {
                None
            };
            match reason {
                Some(r) if p.demotions + 1 >= policy.max_demotions.max(1) => {
                    Decision::Retire(format!("{r} after {} demotion(s)", p.demotions))
                }
                Some(r) => Decision::Demote(r),
                None => Decision::Nothing,
            }
        }
        S::Flattening => Decision::Nothing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::{GateEvidence, PromotionGate};

    const DAY: i64 = 86_400_000;

    fn member(id: &str, dsr: Option<f64>, score: f64) -> FrontMember {
        FrontMember {
            strategy_id: id.into(),
            genome: format!("{{ id = \"{id}\" }}"),
            generation: "g1".into(),
            timescale: "1s".into(),
            step_secs: 1.0,
            score: Some(score),
            selection: Selection {
                deflated_sharpe: dsr,
                sharpe: Some(0.01),
                n_obs: Some(1000),
                pbo: Some(0.1),
                window_ms: Some((0, 10)),
            },
        }
    }

    fn evolved(status: StrategyStatus) -> StrategyState {
        let mut s = StrategyState::new("gp-1", status, 0);
        s.pipeline = Some(PipelineRecord { timescale: "1s".into(), ..Default::default() });
        s
    }

    #[test]
    fn selection_needs_a_sane_generation_and_a_deflated_sharpe() {
        let p = PipelinePolicy { max_candidates_per_generation: 2, ..Default::default() };
        let ms = vec![
            member("a", Some(0.99), 1.0),
            member("b", Some(0.5), 3.0),
            member("c", Some(0.97), 2.0),
            member("d", None, 0.0),
            member("e", Some(0.96), 0.5),
        ];
        let (chosen, rejected) = select_candidates(&p, &ms, Some(0.1), |_| false);
        assert_eq!(
            chosen.iter().map(|m| m.strategy_id.as_str()).collect::<Vec<_>>(),
            vec!["c", "a"],
            "best score first"
        );
        assert_eq!(rejected.len(), 3);
        assert!(rejected.iter().any(|(id, r)| id == "b" && r.contains("0.500")));
        assert!(rejected.iter().any(|(id, r)| id == "e" && r.contains("max_candidates")));
        // overfit or unknown generation promotes no one; known ids are skipped
        let (c, r) = select_candidates(&p, &ms, Some(0.5), |_| false);
        assert!(c.is_empty() && r.len() == 5 && r[0].1.contains("PBO 0.500"));
        assert!(select_candidates(&p, &ms, None, |_| false).0.is_empty());
        let (c, r) = select_candidates(&p, &ms, Some(0.1), |id| id != "a");
        assert_eq!((c.len(), r.len()), (1, 0));
        assert_eq!(c[0].record().genome, "{ id = \"a\" }");
        // holdout hygiene: a holdout overlapping the selection window is not clean
        assert!(
            c[0].selection.clean_for(11, 20) && !c[0].selection.clean_for(5, 20) && !c[0].selection.clean_for(-5, 0)
        );
        assert!(Selection::default().clean_for(0, 1));
    }

    #[test]
    fn candidate_requests_one_holdout_then_passes_or_retires() {
        let p = PipelinePolicy { holdout_timeout_secs: 60, holdout_max_attempts: 2, ..Default::default() };
        let mut c = evolved(StrategyStatus::Candidate);
        let obs = Observation { now_ms: 1_000, ..Default::default() };
        assert_eq!(decide(&p, &c, &obs), Decision::Wait("no locked 1s holdout".into()));
        let obs = Observation { now_ms: 1_000, holdout_id: Some("h1"), ..Default::default() };
        assert_eq!(decide(&p, &c, &obs), Decision::RequestHoldout { holdout_id: "h1".into() });
        let rec = c.pipeline.as_mut().unwrap();
        rec.holdout_requested_ms = Some(1_000);
        rec.holdout_attempts = 1;
        assert!(matches!(
            decide(&p, &c, &Observation { now_ms: 30_000, holdout_id: Some("h1"), ..Default::default() }),
            Decision::Wait(_)
        ));
        // timed out without a trial row → one retry, then retirement
        assert!(matches!(
            decide(&p, &c, &Observation { now_ms: 70_000, holdout_id: Some("h1"), ..Default::default() }),
            Decision::RequestHoldout { .. }
        ));
        c.pipeline.as_mut().unwrap().holdout_attempts = 2;
        c.pipeline.as_mut().unwrap().note = "insufficient market data".into();
        let d = decide(&p, &c, &Observation { now_ms: 70_000, holdout_id: Some("h1"), ..Default::default() });
        assert!(matches!(&d, Decision::Retire(r) if r.contains("insufficient market data")), "{d:?}");
        // a trial row decides, whatever the attempt count
        let good = HoldoutOutcome { deflated_sharpe: Some(0.99), sharpe: Some(0.02), n_obs: 500, ..Default::default() };
        let obs = Observation { now_ms: 70_000, holdout_result: Some(good.clone()), ..Default::default() };
        assert_eq!(decide(&p, &c, &obs), Decision::HoldoutPassed(good.clone()));
        for bad in [
            HoldoutOutcome { deflated_sharpe: Some(0.5), ..good.clone() },
            HoldoutOutcome { sharpe: Some(-0.01), ..good.clone() },
            HoldoutOutcome { n_obs: 10, ..good.clone() },
            HoldoutOutcome { deflated_sharpe: None, ..good.clone() },
        ] {
            let obs = Observation { holdout_result: Some(bad), ..Default::default() };
            assert!(matches!(decide(&p, &c, &obs), Decision::Retire(_)));
        }
    }

    #[test]
    fn shadow_is_admitted_promoted_on_green_and_retired_on_failure() {
        let p = PipelinePolicy::default();
        let hp = evolved(StrategyStatus::HoldoutPassed);
        assert_eq!(decide(&p, &hp, &Observation { live_slot_free: true, ..Default::default() }), Decision::Admit);
        assert!(matches!(decide(&p, &hp, &Observation::default()), Decision::Wait(_)));

        let gate = PromotionGate { min_shadow_days: 1.0, min_horizons: 1, ..Default::default() };
        let mut sh = evolved(StrategyStatus::Shadow);
        sh.horizons_completed = 5;
        let green = GateEvidence {
            deflated_sharpe: Some(0.99),
            pbo: Some(0.1),
            consistent_with_backtest: Some(true),
            ..Default::default()
        };
        let cl = gate.check(&sh, &green, 2 * DAY);
        assert!(cl.green);
        assert_eq!(
            decide(&p, &sh, &Observation { now_ms: 2 * DAY, gate: Some(&cl), ..Default::default() }),
            Decision::Promote
        );
        let manual = PipelinePolicy { auto_promote: false, ..p.clone() };
        assert!(matches!(
            decide(&manual, &sh, &Observation { gate: Some(&cl), ..Default::default() }),
            Decision::Wait(_)
        ));
        // inconsistent after the shadow period → retired; before it → keeps waiting
        let bad = GateEvidence { consistent_with_backtest: Some(false), ..green };
        let cl = gate.check(&sh, &bad, 2 * DAY);
        assert!(matches!(
            decide(&p, &sh, &Observation { now_ms: 2 * DAY, gate: Some(&cl), ..Default::default() }),
            Decision::Retire(r) if r.contains("shadow_vs_backtest")
        ));
        let cl = gate.check(&sh, &bad, DAY / 2);
        assert!(matches!(
            decide(&p, &sh, &Observation { now_ms: DAY / 2, gate: Some(&cl), ..Default::default() }),
            Decision::Wait(_)
        ));
        // expiry and runtime auto-pause
        assert!(matches!(
            decide(&p, &sh, &Observation { now_ms: 91 * DAY, ..Default::default() }),
            Decision::Retire(_)
        ));
        assert!(matches!(
            decide(&p, &sh, &Observation { runtime_auto_paused: Some("nan"), ..Default::default() }),
            Decision::Retire(r) if r.contains("nan")
        ));
        // baselines (no pipeline record) are never touched
        let base = StrategyState::new("baseline-x", StrategyStatus::Shadow, 0);
        assert_eq!(decide(&p, &base, &Observation { now_ms: 500 * DAY, ..Default::default() }), Decision::Nothing);
    }

    #[test]
    fn active_is_demoted_then_retired_and_drawdown_pauses_resolve() {
        let p = PipelinePolicy { max_demotions: 2, ..Default::default() };
        let mut a = evolved(StrategyStatus::Active);
        assert_eq!(decide(&p, &a, &Observation::default()), Decision::Nothing);
        let decay = Observation { live_consistent: Some(false), ..Default::default() };
        assert!(matches!(decide(&p, &a, &decay), Decision::Demote(r) if r.contains("decay")));
        a.pipeline.as_mut().unwrap().demotions = 1;
        assert!(matches!(decide(&p, &a, &decay), Decision::Retire(_)));
        // drawdown pause: from active → demote, from shadow → retire, operator pause → untouched
        let mut paused = evolved(StrategyStatus::Active);
        paused.breach(0.06, 0.05, 1);
        assert!(matches!(decide(&p, &paused, &Observation::default()), Decision::Demote(_)));
        let mut sp = evolved(StrategyStatus::Shadow);
        sp.breach(0.06, 0.05, 1);
        assert!(
            matches!(decide(&p, &sp, &Observation::default()), Decision::Retire(r) if r.starts_with("shadow strategy drawdown"))
        );
        let mut op = evolved(StrategyStatus::Active);
        op.apply(crate::strategy::StrategyAction::Pause, 1, None).unwrap();
        assert_eq!(decide(&p, &op, &Observation::default()), Decision::Nothing);
    }
}
