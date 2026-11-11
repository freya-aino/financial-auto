//! SGD fine-tune of neural strategy children (spec 35 §4.3, §6.3; the "weight warm-start + SGD
//! fine-tune" operator of the neural breeding pool, run inside `evo.vary.strategy`).
//!
//! **Training window** (no leak into selection): the `train_hours` (default = the selection
//! window's length, at most 6 h) ending one embargo (`max(2 × horizon, 60 s)`) before the start
//! of the now-relative selection window `[now − data_window_hours, now]` the child will be
//! scored on; a window that would intersect a locked holdout of the timescale is skipped
//! (`holdout`), like a selection replay. The child's policy is replayed over it through the same
//! `fa_backtest::replay` engine as fitness (recorded latents and decoder outputs, the cost
//! model, its own book), recording every policy input; each input is paired with the asset's
//! realised log return over the strategy horizon after the step (bars of the window), and
//! [`fa_strategies::neural::train::fine_tune`] maximises the cost-adjusted log utility on them,
//! keeping the weights with the best held-out (latest 25 %) utility. Fitness is always the
//! non-differentiable replay of the selection window (the ES half of the hybrid).

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use fa_backtest::replay::{BarSeries, ReplayControls, ReplayPurpose, ReplaySpec, check_holdout, instrument_id, replay_strategy};
use fa_core::InstrumentId;
use fa_strategies::{
    genome::StrategyGenome,
    neural::{
        NeuralStrategy,
        train::{TrainConfig, TrainReport, TrainSample, fine_tune},
    },
};
use serde_json::{Value, json};

use crate::{
    data::{self, DataWindow},
    job::Worker,
    market::now_ms,
};

/// Training rows from a replay's recorded policy inputs: each paired with the log return over
/// `horizon_bars` bars after its step (rows without a complete horizon in `bars` are dropped).
pub fn samples(rows: &[fa_strategies::neural::FeatureRow], bars: &[BarSeries], horizon_bars: usize) -> Vec<TrainSample> {
    let series: BTreeMap<InstrumentId, &BarSeries> = bars.iter().map(|b| (instrument_id(&b.instrument_id), b)).collect();
    rows.iter()
        .filter_map(|r| {
            let s = series.get(&r.instrument)?;
            let i = s.ts_ms.partition_point(|t| *t <= r.as_of_ms).checked_sub(1)?;
            let j = i + horizon_bars.max(1);
            let (p0, p1) = (*s.close.get(i)?, *s.close.get(j)?);
            (p0 > 0.0 && p1 > 0.0).then(|| TrainSample {
                t_ms: r.as_of_ms,
                instrument: r.instrument,
                x: r.x.clone(),
                fwd_ret: (p1 / p0).ln() as f32,
            })
        })
        .collect()
}

/// `[start, end)` of the training window before a selection window starting at `sel_start`.
pub fn training_window(sel_start: i64, train_ms: i64, horizon_ms: i64) -> (i64, i64) {
    let embargo = (2 * horizon_ms).max(60_000);
    let end = sel_start - embargo;
    (end - train_ms.max(60_000), end)
}

impl Worker {
    /// Fine-tunes the policy of `g` (see module docs). `Err` = skipped (reason), never fatal.
    pub(crate) async fn fine_tune_policy(&self, g: &StrategyGenome, snapshot: &str) -> Result<(StrategyGenome, Value), String> {
        let Some(net) = g.policy.clone() else { return Err("not a neural genome".into()) };
        let step_ms = i64::try_from(g.timescale.duration().as_millis()).unwrap_or(1000).max(1);
        let horizon_ms = step_ms * i64::from(g.steps.max(1));
        let now = now_ms();
        let sel_hours = self.data_window_hours.max(0.05);
        let sel_start = now - (sel_hours * 3_600_000.0) as i64;
        let train_ms = ((sel_hours.min(6.0)) * 3_600_000.0) as i64;
        let (w0, w1) = training_window(sel_start, train_ms, horizon_ms);
        let spec = ReplaySpec {
            window_ms: (w0, w1),
            purpose: ReplayPurpose::Selection,
            snapshot: snapshot.to_owned(),
            cost_model: self.cost.id.clone(),
            capital: crate::strategy::CAPITAL,
            instruments: g.assets.clone(),
            trade_rule: self.trade_rule.clone(),
            ..ReplaySpec::selection(g.timescale, w0, w1)
        };
        if let Some(t) = &self.trials {
            let hs = t.holdouts(g.timescale).await.unwrap_or_default();
            check_holdout(&spec, &hs).map_err(|e| format!("holdout: {e}"))?;
        }
        let need = g.lookback_bars + g.steps as usize + 8;
        let window = DataWindow { hours: None, start_ms: Some(w0), end_ms: Some(w1) };
        let loaded = data::load_bars(self.tsdb.as_ref(), &self.market, g.timescale, None, &window, sel_hours, need).await;
        if loaded.bars.is_empty() {
            return Err(format!("no bars in the training window ({})", loaded.source.vm_error.as_deref().unwrap_or("ok")));
        }
        let latents = self.recorded.load(&self.bus.jetstream(), &spec).await;
        let rec = latents.latents.clone();
        let cost = self.cost.clone();
        let def = g.def();
        let horizon_bars = g.steps as usize;
        let lookback = g.lookback_bars;
        let task = tokio::task::spawn_blocking(move || -> Result<(fa_strategies::neural::PolicyNet, TrainReport, usize), String> {
            let recorder = Arc::new(Mutex::new(Vec::new()));
            let strategy = NeuralStrategy {
                id: def.id.clone(),
                genome_id: def.genome_id.clone(),
                horizon: def.horizon(),
                net: net.clone(),
                recorder: Some(recorder.clone()),
            };
            replay_strategy(Box::new(strategy), lookback, &loaded.bars, &spec, &cost, &ReplayControls::default(), rec.as_deref())
                .map_err(|e| format!("training replay: {e}"))?;
            let rows = recorder.lock().map(|r| r.clone()).unwrap_or_default();
            let samples = samples(&rows, &loaded.bars, horizon_bars);
            // round-trip cost of a unit exposure change ≈ commission + spread of the dearest venue
            let cfg = TrainConfig { turnover_cost: 0.001, ..TrainConfig::default() };
            let (tuned, report) = fine_tune(&net, &samples, &cfg).map_err(|e| format!("fine-tune: {e}"))?;
            Ok((tuned, report, rows.len()))
        });
        let (tuned, report, rows) = task.await.map_err(|e| format!("fine-tune task: {e}"))??;
        if report.best_step == 0 {
            return Err(format!("no held-out improvement over {} steps ({} rows)", report.steps, report.train_rows));
        }
        let mut out = g.clone();
        out.policy = Some(tuned);
        Ok((out, json!({"window": [w0, w1], "rows": rows, "report": report})))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_strategies::neural::FeatureRow;

    #[test]
    fn samples_pair_inputs_with_forward_returns_and_windows_precede_selection() {
        let bars = vec![BarSeries {
            instrument_id: "a".into(),
            venue: "sim".into(),
            ts_ms: vec![1000, 2000, 3000, 4000],
            close: vec![100.0, 110.0, 121.0, 133.1],
        }];
        let inst = instrument_id("a");
        let rows = vec![
            FeatureRow { as_of_ms: 1000, instrument: inst, x: vec![1.0], mean: 0.0 },
            FeatureRow { as_of_ms: 3000, instrument: inst, x: vec![2.0], mean: 0.0 },
            FeatureRow { as_of_ms: 2500, instrument: InstrumentId::new(), x: vec![3.0], mean: 0.0 },
        ];
        let s = samples(&rows, &bars, 2);
        assert_eq!(s.len(), 1, "the 3000 row has no 2-bar horizon left; unknown instrument dropped");
        assert!((f64::from(s[0].fwd_ret) - (121.0f64 / 100.0).ln()).abs() < 1e-6);
        let (a, b) = training_window(10_000_000, 3_600_000, 5_000);
        assert_eq!(b, 10_000_000 - 60_000, "embargo max(2h, 60 s)");
        assert_eq!(b - a, 3_600_000);
        let (_, b) = training_window(10_000_000, 3_600_000, 120_000);
        assert_eq!(b, 10_000_000 - 240_000);
    }
}
