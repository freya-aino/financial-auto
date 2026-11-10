//! Async side of the analog price cache (D-063): drains the analogs the last steps could not
//! price from memory, reads their outcome prices from VictoriaMetrics through
//! `fa_tsdb::Tsdb::trade_prices_ms` and fills [`crate::runtime::AnalogCache`] so the next step
//! resolves them. Point-in-time: only analogs whose outcome window ended at or before the step
//! were queued, and the price at `window_end + k·bar` is the last bucket at or before it.
use std::future::Future;

use fa_core::{InstrumentId, Timescale};
use fa_strategies::decoders::AnalogRequest;
use tokio::sync::Mutex;

use crate::runtime::{AnalogKey, Runtime};

/// Bars of history before `window_end` searched for its price (sparse trading).
pub const LOOKBACK_BARS: i64 = 30;

/// Reads `(ts_ms, price)` bucket closes of one instrument (VictoriaMetrics in production).
pub trait PriceSource: Send + Sync {
    fn prices(
        &self,
        inst: InstrumentId,
        start_ms: i64,
        end_ms: i64,
        bar: Timescale,
    ) -> impl Future<Output = Result<Vec<(i64, f64)>, String>> + Send;
}

impl PriceSource for fa_tsdb::Tsdb {
    async fn prices(
        &self,
        inst: InstrumentId,
        start_ms: i64,
        end_ms: i64,
        bar: Timescale,
    ) -> Result<Vec<(i64, f64)>, String> {
        self.trade_prices_ms(&inst.to_string(), start_ms, end_ms, bar).await.map_err(|e| e.to_string())
    }
}

/// `(p(window_end), p(window_end + k·bar))` from ascending bucket closes: the last close at or
/// before each time, within [`LOOKBACK_BARS`] bars of `window_end`.
pub fn resolve(points: &[(i64, f64)], req: &AnalogRequest) -> Option<(f64, f64)> {
    let bar_ms = req.bar.duration().as_millis() as i64;
    let at = |t: i64| {
        let n = points.partition_point(|(ts, _)| *ts <= t);
        (n > 0).then(|| points[n - 1]).filter(|(ts, _)| t - ts <= LOOKBACK_BARS * bar_ms)
    };
    let (t0, p0) = at(req.window_end_ms)?;
    let (t1, p1) = at(req.outcome_end_ms())?;
    (t1 >= t0 && p0 > 0.0 && p1 > 0.0).then_some((p0, p1))
}

/// Fetches up to `batch` queued analogs and fills the cache; returns how many were fetched.
pub async fn prefetch_once<P: PriceSource>(rt: &Mutex<Runtime>, src: &P, batch: usize, now_ms: i64) -> usize {
    let keys: Vec<AnalogKey> = rt.lock().await.analogs.take_pending(batch);
    let mut results = Vec::with_capacity(keys.len());
    for (inst, req) in &keys {
        if req.outcome_end_ms() > now_ms {
            results.push(((*inst, *req), Ok(None))); // never ask for the future
            continue;
        }
        let bar_ms = req.bar.duration().as_millis() as i64;
        let r = src
            .prices(*inst, req.window_end_ms - LOOKBACK_BARS * bar_ms, req.outcome_end_ms(), req.bar)
            .await
            .map(|pts| resolve(&pts, req));
        if let Err(e) = &r {
            tracing::debug!(instrument = %inst, error = %e, "analog price fetch failed");
        }
        results.push(((*inst, *req), r.map_err(|_| ())));
    }
    let n = results.len();
    let mut g = rt.lock().await;
    for (key, r) in results {
        g.fill_analog(key, r, now_ms);
    }
    n
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn resolves_last_close_at_or_before_each_end() {
        let req = AnalogRequest { window_end_ms: 10_000, k: 5, bar: Timescale::Second };
        let pts = [(8_000, 99.0), (10_000, 100.0), (14_000, 104.0), (16_000, 106.0)];
        assert_eq!(resolve(&pts, &req), Some((100.0, 104.0)), "15 s → last close at 14 s");
        assert_eq!(resolve(&pts[2..], &req), None, "nothing at or before the window end");
        let far = [(-100_000, 1.0), (15_000, 2.0)];
        assert_eq!(resolve(&far, &req), None, "window-end price older than the lookback");
    }

    /// Mock VictoriaMetrics: a linear price path, counting calls; `fail` makes every call error.
    struct Mock {
        calls: AtomicUsize,
        fail: bool,
    }

    impl PriceSource for Mock {
        async fn prices(
            &self,
            _: InstrumentId,
            start_ms: i64,
            end_ms: i64,
            bar: Timescale,
        ) -> Result<Vec<(i64, f64)>, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err("connection refused".into());
            }
            let step = bar.duration().as_millis() as i64;
            Ok((start_ms.div_euclid(step)..=end_ms.div_euclid(step))
                .map(|b| (b * step, 50.0 + (b * step) as f64 / 1000.0))
                .collect())
        }
    }

    #[tokio::test]
    async fn prefetch_fills_the_cache_and_the_next_step_resolves() {
        use fa_strategies::population::StrategyFile;
        let file = StrategyFile::parse(
            "[[strategy]]\nid='analog'\ntimescale='1s'\nsteps=5\nsignal='(mul 100 (analog_outcome 5))'\n",
        )
        .unwrap();
        let mut runtime = Runtime::new(&file).unwrap();
        let inst = InstrumentId::new();
        // in-memory history only from 100 s on; the analogs ended at 10 s and 20 s
        for t in 100..=130 {
            runtime.on_trade(inst, t * 1_000, 150.0 + t as f64 * 0.01);
        }
        runtime
            .on_decoder_output(&fa_strategies_test_retrieval(inst, 130_000, &[(10_000, 1.0), (20_000, 3.0)]), 130_000);
        let rt = Mutex::new(runtime);
        let d = rt.lock().await.tick(130_000);
        assert_eq!(d.len(), 1);
        {
            let g = rt.lock().await;
            assert_eq!(g.analog_unresolved, 2, "both analogs are older than the in-memory history");
            assert_eq!(g.analog_resolved_tsdb, 0);
            assert_eq!(g.analogs.pending(), 2);
        }
        // a failing VM: nothing cached, the keys come back on the next step
        let bad = Mock { calls: AtomicUsize::new(0), fail: true };
        assert_eq!(prefetch_once(&rt, &bad, 16, 130_500).await, 2);
        assert_eq!(rt.lock().await.analogs.fetch_errors, 2);
        assert_eq!(rt.lock().await.analogs.len(), 0);
        rt.lock().await.tick(131_000);
        assert_eq!(rt.lock().await.analogs.pending(), 2, "re-queued after the failure");
        // the mock VM resolves both
        let vm = Mock { calls: AtomicUsize::new(0), fail: false };
        assert_eq!(prefetch_once(&rt, &vm, 16, 131_500).await, 2);
        assert_eq!(vm.calls.load(Ordering::SeqCst), 2);
        let d = rt.lock().await.tick(132_000);
        let g = rt.lock().await;
        assert_eq!(g.analog_resolved_tsdb, 2, "{}", g.status(132_000));
        assert_eq!(g.analogs.pending(), 0);
        // mock prices 50 + t/1000: outcomes ln(65/60) and ln(75/70), weights 1 and 3
        let want = ((65.0f64 / 60.0).ln() + 3.0 * (75.0f64 / 70.0).ln()) / 4.0;
        assert!((d[0].exposures[0].mean - (100.0 * want).tanh()).abs() < 1e-9, "{}", d[0].exposures[0].mean);
        let hb = g.heartbeat_counters(132_000);
        assert_eq!(hb["analog.resolved_tsdb"], 2.0);
        assert_eq!(hb["analog.unresolved"], 4.0);
        assert_eq!(hb["strategy.analog.analog.resolved_tsdb"], 2.0);
    }

    fn fa_strategies_test_retrieval(
        inst: InstrumentId,
        at_ms: i64,
        analogs: &[(i64, f64)],
    ) -> fa_proto::v1::DecoderOutput {
        let ts = |ms: i64| fa_proto::Timestamp { seconds: ms / 1000, nanos: ((ms % 1000) * 1_000_000) as i32 };
        fa_proto::v1::DecoderOutput {
            decoder_id: "baseline-retrieval-1".into(),
            genome_id: "retrieval:k=2".into(),
            family: "retrieval".into(),
            snapshot_id: "snap".into(),
            timescale: "1s".into(),
            as_of: Some(ts(at_ms)),
            output_refs: vec![format!("inst:{inst}")],
            quantiles: vec![],
            analogs: analogs
                .iter()
                .map(|(end, sim)| fa_proto::v1::Analog {
                    window_id: format!("w{end}"),
                    window_end: Some(ts(*end)),
                    similarity: *sim,
                    time_distance_days: 1.0,
                })
                .collect(),
        }
    }
}
