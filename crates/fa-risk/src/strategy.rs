//! Strategy lifecycle, per-strategy drawdown and the promotion gate (spec 40 §3, §6.5, §6.6).
//!
//! A strategy is `active` (its signals are allocated real paper capital), `paused` (ignored by
//! the allocator; its attributed position is flattened), `flattening` (transitional: targets
//! dropped this tick, becomes `paused` once the closing intents are out) or `shadow` (signals
//! tracked in a virtual book only, the clock counts toward promotion). `promote` is the only
//! transition guarded by the gate ([`PromotionGate::check`]).
//!
//! Evolved strategies enter earlier (spec 35 §6.4): `candidate` (selected from a generation's
//! front, waiting for its single locked-holdout evaluation) → `holdout_passed` → `shadow`
//! (admitted into the live strategy-runtime set) → `active`; `retired` is terminal (holdout
//! failure, shadow drawdown, repeated demotion, operator `retire`). The automatic driver of
//! these transitions is [`crate::pipeline`]; this module only enforces which transitions exist.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyStatus {
    Active,
    Paused,
    Flattening,
    Shadow,
    /// Evolved, selected for a holdout evaluation; not running anywhere yet.
    Candidate,
    /// Passed the locked holdout; waits for a free slot in the live set.
    HoldoutPassed,
    /// Terminal: never runs again (a new genome id is a new strategy).
    Retired,
}

impl StrategyStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Flattening => "flattening",
            Self::Shadow => "shadow",
            Self::Candidate => "candidate",
            Self::HoldoutPassed => "holdout_passed",
            Self::Retired => "retired",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "active" => Self::Active,
            "paused" => Self::Paused,
            "flattening" => Self::Flattening,
            "shadow" => Self::Shadow,
            "candidate" => Self::Candidate,
            "holdout_passed" => Self::HoldoutPassed,
            "retired" => Self::Retired,
            _ => return None,
        })
    }

    /// Whether the strategy's signals receive real (paper) exposure.
    pub fn allocates(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Whether an evolved strategy in this status belongs in the live strategy-runtime set.
    pub fn in_live_set(self) -> bool {
        matches!(self, Self::Active | Self::Paused | Self::Flattening | Self::Shadow)
    }

    /// Statuses before admission (no signals expected).
    pub fn pre_live(self) -> bool {
        matches!(self, Self::Candidate | Self::HoldoutPassed)
    }
}

/// Operator / automatic lifecycle actions (`ctl.risk.strategy` `action`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyAction {
    Pause,
    Resume,
    Flatten,
    Promote,
    /// Extension beyond spec 40 §4.4: send an `active` strategy back to shadow (demotion).
    Shadow,
    /// Terminal retirement (spec 35 §6.4): removed from the live set for good.
    Retire,
    /// `holdout_passed` → `shadow`: admission into the live set (the pipeline's step; also
    /// accepted from an operator).
    Admit,
}

impl StrategyAction {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "pause" => Self::Pause,
            "resume" => Self::Resume,
            "flatten" => Self::Flatten,
            "promote" => Self::Promote,
            "shadow" => Self::Shadow,
            "retire" => Self::Retire,
            "admit" => Self::Admit,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pause => "pause",
            Self::Resume => "resume",
            Self::Flatten => "flatten",
            Self::Promote => "promote",
            Self::Shadow => "shadow",
            Self::Retire => "retire",
            Self::Admit => "admit",
        }
    }
}

/// Lifecycle + shadow progress + drawdown tracking for one strategy (persisted in
/// `ledger.strategy_state`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyState {
    pub strategy_id: String,
    pub status: StrategyStatus,
    /// Status `resume` returns to (set by `pause`/`flatten`).
    pub prev_status: Option<StrategyStatus>,
    pub since_ms: i64,
    pub first_seen_ms: i64,
    pub shadow_since_ms: Option<i64>,
    /// Horizons completed while in shadow (spec 41 §6.6).
    pub horizons_completed: u64,
    /// Capital the strategy was last allocated (active) or tracks virtually (shadow).
    pub sleeve_eur: f64,
    /// Attributed (active) or virtual (shadow) P&L since the current status began.
    pub pnl_eur: f64,
    pub peak_pnl_eur: f64,
    /// Worst observed drawdown as a fraction of the sleeve.
    pub max_drawdown: f64,
    /// Why the last automatic transition happened (empty for manual ones).
    pub last_reason: String,
    /// Evolved strategies: genome text (inline TOML table) the live set is built from, its
    /// generation and the pipeline's evidence (`None` for baselines and unknown strategies).
    #[serde(default)]
    pub pipeline: Option<crate::pipeline::PipelineRecord>,
}

impl StrategyState {
    pub fn new(strategy_id: &str, status: StrategyStatus, now_ms: i64) -> Self {
        Self {
            strategy_id: strategy_id.into(),
            status,
            prev_status: None,
            since_ms: now_ms,
            first_seen_ms: now_ms,
            shadow_since_ms: if status == StrategyStatus::Shadow { Some(now_ms) } else { None },
            horizons_completed: 0,
            sleeve_eur: 0.0,
            pnl_eur: 0.0,
            peak_pnl_eur: 0.0,
            max_drawdown: 0.0,
            last_reason: String::new(),
            pipeline: None,
        }
    }

    /// Whether the strategy came out of evolution (managed by the pipeline).
    pub fn evolved(&self) -> bool {
        self.pipeline.is_some()
    }

    /// Current drawdown of the tracked P&L as a fraction of the sleeve (0 without a sleeve).
    pub fn drawdown(&self) -> f64 {
        if self.sleeve_eur > 0.0 { ((self.peak_pnl_eur - self.pnl_eur) / self.sleeve_eur).max(0.0) } else { 0.0 }
    }

    /// Adds a P&L increment and records the sleeve it was earned on; returns the drawdown.
    pub fn observe_pnl(&mut self, delta_eur: f64, sleeve_eur: f64) -> f64 {
        if delta_eur.is_finite() {
            self.pnl_eur += delta_eur;
        }
        if sleeve_eur.is_finite() && sleeve_eur > 0.0 {
            self.sleeve_eur = sleeve_eur;
        }
        self.peak_pnl_eur = self.peak_pnl_eur.max(self.pnl_eur);
        let dd = self.drawdown();
        self.max_drawdown = self.max_drawdown.max(dd);
        dd
    }

    fn enter(&mut self, status: StrategyStatus, now_ms: i64, reason: &str) {
        if status == StrategyStatus::Shadow && self.status != StrategyStatus::Shadow {
            self.shadow_since_ms = Some(now_ms);
            self.horizons_completed = 0;
        }
        if status != StrategyStatus::Shadow {
            self.shadow_since_ms = None;
        }
        if status.allocates() != self.status.allocates() || status == StrategyStatus::Shadow {
            // a new book starts: P&L and peak restart, the worst drawdown is kept for the record
            self.pnl_eur = 0.0;
            self.peak_pnl_eur = 0.0;
        }
        self.status = status;
        self.since_ms = now_ms;
        self.last_reason = reason.into();
    }

    /// Applies an operator action. `gate` is the checklist for `promote` (ignored otherwise).
    /// Errors use the control-api prefix convention (`conflict:` / `invalid:`).
    pub fn apply(&mut self, action: StrategyAction, now_ms: i64, gate: Option<&Checklist>) -> Result<(), String> {
        use StrategyStatus as S;
        match (action, self.status) {
            (StrategyAction::Retire, S::Retired) => Err(format!("conflict: {} is already retired", self.strategy_id)),
            (StrategyAction::Retire, _) => {
                self.prev_status = None;
                self.enter(S::Retired, now_ms, "retired");
                Ok(())
            }
            (_, S::Retired) => Err(format!("conflict: {} is retired", self.strategy_id)),
            (StrategyAction::Admit, S::HoldoutPassed) => {
                self.prev_status = None;
                self.enter(S::Shadow, now_ms, "admitted: holdout passed");
                Ok(())
            }
            (StrategyAction::Admit, s) => Err(format!(
                "conflict: {} is {}; only holdout_passed strategies are admitted",
                self.strategy_id,
                s.as_str()
            )),
            (_, s) if s.pre_live() => Err(format!(
                "conflict: {} is {}; it runs only after passing the locked holdout (spec 35 §6.4)",
                self.strategy_id,
                s.as_str()
            )),
            (StrategyAction::Pause, S::Paused) => Err(format!("conflict: {} is already paused", self.strategy_id)),
            (StrategyAction::Pause, s) => {
                self.prev_status = Some(if s == S::Flattening { S::Active } else { s });
                self.enter(S::Paused, now_ms, "");
                Ok(())
            }
            (StrategyAction::Resume, S::Paused) => {
                let to = self.prev_status.take().unwrap_or(S::Active);
                self.enter(to, now_ms, "");
                Ok(())
            }
            (StrategyAction::Resume, s) => Err(format!("conflict: {} is {}, not paused", self.strategy_id, s.as_str())),
            (StrategyAction::Flatten, S::Active) => {
                self.prev_status = Some(S::Active);
                self.enter(S::Flattening, now_ms, "");
                Ok(())
            }
            (StrategyAction::Flatten, s) => {
                Err(format!("conflict: {} is {}; only active strategies hold exposure", self.strategy_id, s.as_str()))
            }
            (StrategyAction::Promote, S::Shadow) => match gate {
                Some(g) if g.green => {
                    self.prev_status = None;
                    self.enter(S::Active, now_ms, "promoted: gate green");
                    Ok(())
                }
                Some(g) => Err(format!("conflict: promotion gate not green for {}: {}", self.strategy_id, g.summary())),
                None => Err("invalid: promotion needs a gate checklist".into()),
            },
            (StrategyAction::Promote, s) => {
                Err(format!("conflict: {} is {}; only shadow strategies can be promoted", self.strategy_id, s.as_str()))
            }
            (StrategyAction::Shadow, S::Shadow) => Err(format!("conflict: {} is already shadow", self.strategy_id)),
            (StrategyAction::Shadow, _) => {
                self.prev_status = None;
                self.enter(S::Shadow, now_ms, "");
                Ok(())
            }
        }
    }

    /// Automatic transition on a per-strategy drawdown breach (spec 40 §6.5): flatten + pause.
    pub fn breach(&mut self, drawdown: f64, limit: f64, now_ms: i64) {
        self.prev_status =
            Some(if self.status == StrategyStatus::Shadow { StrategyStatus::Shadow } else { StrategyStatus::Active });
        self.enter(
            StrategyStatus::Paused,
            now_ms,
            &format!("strategy drawdown {:.2}% ≥ limit {:.2}%", drawdown * 100.0, limit * 100.0),
        );
    }

    /// Pipeline transition with a reason (`candidate` → `holdout_passed`, any → `retired`,
    /// demotion to `shadow`). Unlike [`Self::apply`] it records `reason` and counts demotions.
    pub fn transition(&mut self, to: StrategyStatus, now_ms: i64, reason: &str) {
        if to == StrategyStatus::Shadow
            && matches!(self.status, StrategyStatus::Active | StrategyStatus::Paused)
            && let Some(p) = self.pipeline.as_mut()
        {
            p.demotions += 1;
        }
        self.prev_status = None;
        self.enter(to, now_ms, reason);
    }

    /// `flattening` → `paused` once the closing intents have been issued.
    pub fn flattened(&mut self, now_ms: i64) {
        if self.status == StrategyStatus::Flattening {
            self.enter(StrategyStatus::Paused, now_ms, "");
        }
    }
}

/// Promotion gate thresholds (spec 40 §7 `promotion.*`, spec 41 §7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PromotionGate {
    pub min_shadow_days: f64,
    pub min_horizons: u64,
    /// Deflated Sharpe on the locked holdout must exceed this (spec 41 §6.5).
    pub min_deflated_sharpe: f64,
    /// PBO of the strategy's generation must stay below this (spec 41 §7 `pbo.max_for_promotion`).
    pub max_pbo: f64,
    /// Shadow-vs-backtest band: the shadow Sharpe may fall at most `consistency_z` standard
    /// errors below the backtest Sharpe ([`consistency`]).
    pub consistency_z: f64,
}

impl Default for PromotionGate {
    fn default() -> Self {
        Self { min_shadow_days: 30.0, min_horizons: 20, min_deflated_sharpe: 0.0, max_pbo: 0.2, consistency_z: 2.0 }
    }
}

/// Evidence available for the gate; `None` = unknown (the criterion cannot be evaluated).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GateEvidence {
    pub deflated_sharpe: Option<f64>,
    pub pbo: Option<f64>,
    /// Shadow Sharpe within the backtest band (spec 40 §6.6 item 4), with its detail.
    pub consistent_with_backtest: Option<bool>,
    pub consistency_detail: String,
    pub unresolved_health_halts: bool,
    /// Hand-written strategy (no evolutionary search behind it): PBO and the backtest
    /// consistency are not applicable and pass as `n/a` when unknown.
    pub selection_free: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateItem {
    pub name: String,
    /// `None` when the criterion could not be evaluated (missing data).
    pub pass: Option<bool>,
    pub detail: String,
    /// Advisory items are reported but do not block promotion. Every spec 40 §6.6 item is
    /// blocking since the statistics are computed (kept for compatibility of stored checklists).
    #[serde(default)]
    pub advisory: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checklist {
    pub green: bool,
    pub items: Vec<GateItem>,
    /// Names of the blocking criteria that failed or are unknown.
    pub missing: Vec<String>,
}

impl Checklist {
    pub fn summary(&self) -> String {
        self.items
            .iter()
            .filter(|i| !i.advisory && i.pass != Some(true))
            .map(|i| format!("{} ({})", i.name, i.detail))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Shadow-vs-backtest consistency (spec 40 §6.6 item 4, spec 41 §6.6): the shadow book's
/// hourly P&L changes give a per-hour Sharpe; the backtest's per-step Sharpe is scaled to an
/// hour (`× sqrt(3600 / step_secs)`, i.i.d. steps). Consistent when the shadow Sharpe is no
/// more than `z` standard errors (`≈ 1/sqrt(n_hours)`) below the backtest Sharpe — only the
/// downside fails, a shadow that beats its backtest is consistent. `None` with fewer than
/// `min_points` hourly changes. Returns `(consistent, shadow_sr_hour, backtest_sr_hour, n)`.
pub fn consistency(
    backtest_sharpe_per_step: f64,
    step_secs: f64,
    hourly_pnl: &[f64],
    z: f64,
    min_points: usize,
) -> Option<(bool, f64, f64, usize)> {
    if !backtest_sharpe_per_step.is_finite() || step_secs <= 0.0 {
        return None;
    }
    let d: Vec<f64> = hourly_pnl.windows(2).map(|w| w[1] - w[0]).filter(|x| x.is_finite()).collect();
    let n = d.len();
    if n < min_points.max(2) {
        return None;
    }
    let m = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    let shadow = if var > 0.0 { m / var.sqrt() } else { 0.0 };
    let bt = backtest_sharpe_per_step * (3600.0 / step_secs).sqrt();
    let se = 1.0 / (n as f64).sqrt();
    Some((shadow >= bt - z.max(0.0) * se, shadow, bt, n))
}

impl PromotionGate {
    /// Evaluates every criterion of spec 40 §6.6; green only when every item passes. An unknown
    /// criterion (no holdout trial for the deflated Sharpe, no generation PBO, too few shadow
    /// hours for the consistency check) is listed as missing and blocks promotion.
    pub fn check(&self, s: &StrategyState, ev: &GateEvidence, now_ms: i64) -> Checklist {
        let mut items = Vec::new();
        let shadow_days = s.shadow_since_ms.map(|t| (now_ms - t) as f64 / 86_400_000.0);
        items.push(match shadow_days {
            Some(d) => GateItem {
                name: "min_shadow_days".into(),
                pass: Some(d >= self.min_shadow_days),
                detail: format!("{d:.2} of {} days", self.min_shadow_days),
                advisory: false,
            },
            None => GateItem {
                name: "min_shadow_days".into(),
                pass: Some(false),
                detail: "not in shadow".into(),
                advisory: false,
            },
        });
        items.push(GateItem {
            name: "min_horizons".into(),
            pass: Some(s.horizons_completed >= self.min_horizons),
            detail: format!("{} of {} horizons", s.horizons_completed, self.min_horizons),
            advisory: false,
        });
        items.push(match ev.deflated_sharpe {
            Some(dsr) => GateItem {
                name: "deflated_sharpe".into(),
                pass: Some(dsr > self.min_deflated_sharpe),
                detail: format!("{dsr:.3} > {}", self.min_deflated_sharpe),
                advisory: false,
            },
            None => GateItem {
                name: "deflated_sharpe".into(),
                pass: None,
                detail: "unknown: no holdout or validation.trial row".into(),
                advisory: false,
            },
        });
        items.push(match ev.pbo {
            Some(p) => GateItem {
                name: "pbo".into(),
                pass: Some(p < self.max_pbo),
                detail: format!("{p:.3} < {}", self.max_pbo),
                advisory: false,
            },
            None if ev.selection_free => GateItem {
                name: "pbo".into(),
                pass: Some(true),
                detail: "n/a: hand-written strategy (no selection to overfit)".into(),
                advisory: false,
            },
            None => GateItem {
                name: "pbo".into(),
                pass: None,
                detail: "unknown: no generation statistics".into(),
                advisory: false,
            },
        });
        items.push(match ev.consistent_with_backtest {
            Some(c) => GateItem {
                name: "shadow_vs_backtest".into(),
                pass: Some(c),
                detail: ev.consistency_detail.clone(),
                advisory: false,
            },
            None if ev.selection_free => GateItem {
                name: "shadow_vs_backtest".into(),
                pass: Some(true),
                detail: "n/a: hand-written strategy (no backtest)".into(),
                advisory: false,
            },
            None => GateItem {
                name: "shadow_vs_backtest".into(),
                pass: None,
                detail: if ev.consistency_detail.is_empty() {
                    "unknown: not enough shadow history or no backtest".into()
                } else {
                    ev.consistency_detail.clone()
                },
                advisory: false,
            },
        });
        items.push(GateItem {
            name: "no_health_halts".into(),
            pass: Some(!ev.unresolved_health_halts),
            detail: if ev.unresolved_health_halts { "health halt active".into() } else { String::new() },
            advisory: false,
        });
        let missing: Vec<String> =
            items.iter().filter(|i| !i.advisory && i.pass != Some(true)).map(|i| i.name.clone()).collect();
        Checklist { green: missing.is_empty(), items, missing }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400_000;

    fn green_evidence() -> GateEvidence {
        GateEvidence {
            deflated_sharpe: Some(0.8),
            pbo: Some(0.1),
            consistent_with_backtest: Some(true),
            consistency_detail: String::new(),
            unresolved_health_halts: false,
            selection_free: false,
        }
    }

    #[test]
    fn status_and_action_strings_round_trip() {
        for s in [StrategyStatus::Active, StrategyStatus::Paused, StrategyStatus::Flattening, StrategyStatus::Shadow] {
            assert_eq!(StrategyStatus::parse(s.as_str()), Some(s));
        }
        for a in [
            StrategyAction::Pause,
            StrategyAction::Resume,
            StrategyAction::Flatten,
            StrategyAction::Promote,
            StrategyAction::Shadow,
        ] {
            assert_eq!(StrategyAction::parse(a.as_str()), Some(a));
        }
        assert_eq!(StrategyAction::parse("kill"), None);
        assert!(StrategyStatus::Active.allocates() && !StrategyStatus::Shadow.allocates());
    }

    #[test]
    fn pause_resume_flatten_transitions() {
        let mut s = StrategyState::new("a", StrategyStatus::Active, 0);
        s.apply(StrategyAction::Pause, 1, None).unwrap();
        assert_eq!(s.status, StrategyStatus::Paused);
        assert!(s.apply(StrategyAction::Pause, 2, None).unwrap_err().starts_with("conflict:"));
        assert!(s.apply(StrategyAction::Flatten, 2, None).unwrap_err().starts_with("conflict:"));
        s.apply(StrategyAction::Resume, 3, None).unwrap();
        assert_eq!(s.status, StrategyStatus::Active);
        assert!(s.apply(StrategyAction::Resume, 4, None).is_err());
        s.apply(StrategyAction::Flatten, 5, None).unwrap();
        assert_eq!(s.status, StrategyStatus::Flattening);
        s.flattened(6);
        assert_eq!((s.status, s.since_ms), (StrategyStatus::Paused, 6));
        // a paused shadow strategy resumes into shadow, never straight into active
        let mut sh = StrategyState::new("b", StrategyStatus::Shadow, 0);
        sh.apply(StrategyAction::Pause, 1, None).unwrap();
        sh.apply(StrategyAction::Resume, 2, None).unwrap();
        assert_eq!(sh.status, StrategyStatus::Shadow);
    }

    #[test]
    fn promotion_requires_green_gate() {
        let gate = PromotionGate { min_shadow_days: 1.0, min_horizons: 2, ..PromotionGate::default() };
        let mut s = StrategyState::new("a", StrategyStatus::Shadow, 0);
        let cl = gate.check(&s, &GateEvidence::default(), DAY / 2);
        assert!(!cl.green);
        assert_eq!(
            cl.missing,
            vec!["min_shadow_days", "min_horizons", "deflated_sharpe", "pbo", "shadow_vs_backtest"],
            "every unknown statistic blocks"
        );
        assert!(cl.items.iter().all(|i| !i.advisory));
        assert!(cl.summary().contains("unknown: no holdout"));
        let err = s.apply(StrategyAction::Promote, DAY / 2, Some(&cl)).unwrap_err();
        assert!(err.starts_with("conflict: promotion gate not green"), "{err}");
        assert_eq!(s.status, StrategyStatus::Shadow);
        s.horizons_completed = 2;
        let cl = gate.check(&s, &green_evidence(), 2 * DAY);
        assert!(cl.green, "{cl:?}");
        let only_dsr = GateEvidence { deflated_sharpe: Some(0.1), ..Default::default() };
        assert_eq!(gate.check(&s, &only_dsr, 2 * DAY).missing, vec!["pbo", "shadow_vs_backtest"]);
        let overfit = GateEvidence { pbo: Some(0.5), ..green_evidence() };
        assert_eq!(gate.check(&s, &overfit, 2 * DAY).missing, vec!["pbo"]);
        let drifted = GateEvidence { consistent_with_backtest: Some(false), ..green_evidence() };
        assert_eq!(gate.check(&s, &drifted, 2 * DAY).missing, vec!["shadow_vs_backtest"]);
        // hand-written strategies: PBO / backtest consistency are n/a, the DSR still blocks
        let hand = GateEvidence { selection_free: true, ..Default::default() };
        assert_eq!(gate.check(&s, &hand, 2 * DAY).missing, vec!["deflated_sharpe"]);
        let hand_bad = GateEvidence { selection_free: true, consistent_with_backtest: Some(false), ..green_evidence() };
        assert_eq!(
            gate.check(&s, &hand_bad, 2 * DAY).missing,
            vec!["shadow_vs_backtest"],
            "known statistics still count"
        );
        s.apply(StrategyAction::Promote, 2 * DAY, Some(&cl)).unwrap();
        assert_eq!(s.status, StrategyStatus::Active);
        assert_eq!(s.last_reason, "promoted: gate green");
        assert!(s.apply(StrategyAction::Promote, 3 * DAY, Some(&cl)).unwrap_err().contains("only shadow"));
        // health halt blocks even with statistics in place
        let mut sh = StrategyState::new("c", StrategyStatus::Shadow, 0);
        sh.horizons_completed = 5;
        let ev = GateEvidence { unresolved_health_halts: true, ..green_evidence() };
        assert_eq!(gate.check(&sh, &ev, 2 * DAY).missing, vec!["no_health_halts"]);
        // demotion to shadow restarts the shadow clock
        sh.apply(StrategyAction::Shadow, 0, None).unwrap_err();
        s.apply(StrategyAction::Shadow, 5 * DAY, None).unwrap();
        assert_eq!((s.status, s.shadow_since_ms, s.horizons_completed), (StrategyStatus::Shadow, Some(5 * DAY), 0));
    }

    #[test]
    fn evolved_lifecycle_runs_through_the_holdout_and_retirement_is_terminal() {
        let mut c = StrategyState::new("gp-1", StrategyStatus::Candidate, 0);
        for a in [StrategyAction::Pause, StrategyAction::Resume, StrategyAction::Promote, StrategyAction::Shadow] {
            assert!(c.apply(a, 1, None).unwrap_err().contains("locked holdout"), "{a:?}");
        }
        assert!(c.apply(StrategyAction::Admit, 1, None).unwrap_err().contains("only holdout_passed"));
        c.transition(StrategyStatus::HoldoutPassed, 2, "holdout passed");
        c.apply(StrategyAction::Admit, 3, None).unwrap();
        assert_eq!((c.status, c.shadow_since_ms), (StrategyStatus::Shadow, Some(3)));
        assert!(c.status.in_live_set() && !c.status.allocates());
        c.apply(StrategyAction::Retire, 4, None).unwrap();
        assert_eq!(c.status, StrategyStatus::Retired);
        assert!(!c.status.in_live_set());
        for a in [StrategyAction::Resume, StrategyAction::Admit, StrategyAction::Retire, StrategyAction::Shadow] {
            assert!(c.apply(a, 5, None).unwrap_err().starts_with("conflict:"), "{a:?}");
        }
        for st in [StrategyStatus::Candidate, StrategyStatus::HoldoutPassed, StrategyStatus::Retired] {
            assert_eq!(StrategyStatus::parse(st.as_str()), Some(st));
        }
        // demotion counts on the pipeline record
        let mut a = StrategyState::new("gp-2", StrategyStatus::Active, 0);
        a.pipeline = Some(crate::pipeline::PipelineRecord::default());
        a.transition(StrategyStatus::Shadow, 9, "demoted: gate decay");
        assert_eq!(a.pipeline.as_ref().unwrap().demotions, 1);
        assert_eq!((a.status, a.last_reason.as_str()), (StrategyStatus::Shadow, "demoted: gate decay"));
    }

    #[test]
    fn consistency_compares_hourly_shadow_sharpe_with_the_scaled_backtest() {
        // steady gains: shadow Sharpe is high → consistent with a modest backtest
        let up: Vec<f64> = (0..30).map(|i| i as f64 * 10.0 + if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let (ok, sh, bt, n) = consistency(0.001, 1.0, &up, 2.0, 20).unwrap();
        assert!(ok && sh > bt && n == 29, "{sh} {bt}");
        // steady losses against a strong backtest → inconsistent
        let down: Vec<f64> = up.iter().map(|x| -x).collect();
        let (ok, ..) = consistency(0.01, 1.0, &down, 2.0, 20).unwrap();
        assert!(!ok);
        assert!(consistency(0.01, 1.0, &down[..10], 2.0, 20).is_none(), "too few hours");
        assert!(consistency(f64::NAN, 1.0, &down, 2.0, 20).is_none());
    }

    #[test]
    fn drawdown_tracks_peak_relative_to_sleeve() {
        let mut s = StrategyState::new("a", StrategyStatus::Active, 0);
        assert_eq!(s.observe_pnl(100.0, 10_000.0), 0.0);
        assert!((s.observe_pnl(-600.0, 10_000.0) - 0.06).abs() < 1e-12);
        assert!((s.max_drawdown - 0.06).abs() < 1e-12);
        assert_eq!(s.observe_pnl(f64::NAN, 0.0), 0.06, "bad inputs are ignored");
        s.breach(0.06, 0.05, 7);
        assert_eq!((s.status, s.prev_status), (StrategyStatus::Paused, Some(StrategyStatus::Active)));
        assert!(s.last_reason.contains("6.00% ≥ limit 5.00%"));
        assert_eq!((s.pnl_eur, s.peak_pnl_eur), (0.0, 0.0), "book restarts");
        assert!((s.max_drawdown - 0.06).abs() < 1e-12, "worst drawdown is kept");
        assert_eq!(StrategyState::new("z", StrategyStatus::Active, 0).drawdown(), 0.0);
    }
}
