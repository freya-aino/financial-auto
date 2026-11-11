//! Spec 31 / 32 additions to a neural fit: the asset vocabulary and candidate sets of
//! `asset_attn` genomes (spec 31 §6.4), per-expert decodability (spec 32 §6.4), the collapse
//! verdict (spec 31 §8) and the persistence of a fit's health as VictoriaMetrics series and a
//! Garage matrix per evaluation (spec 32 §5).

use std::collections::BTreeMap;

use fa_core::Timescale;
use fa_encoders::neural::{Dataset, N_FEATURES, NeuralConfig, NeuralTrainer};
use fa_expert_space::health::HealthReport;
use serde_json::{Value, json};

use crate::{data::BarHistory, feedback::ridge};

/// Fills the asset genes of an `asset_attn` genome from the training set: the vocabulary (every
/// instrument of the fit, sorted), the staleness bound (3 steps of the timescale) and — when
/// the vocabulary exceeds `asset_set_size` — each asset's candidate set: its `asset_set_size`
/// most return-correlated peers over the common bars (spec 31 §10 Q4 hand-off rule).
pub fn asset_setup(cfg: &mut NeuralConfig, history: &[BarHistory], ts: Timescale) {
    if !cfg.asset_attn {
        return;
    }
    let mut names: Vec<String> = history.iter().map(|h| h.instrument.clone()).collect();
    names.sort();
    names.dedup();
    cfg.asset_vocab = names;
    cfg.asset_max_age_ms = 3 * ts.duration().as_millis() as i64;
    cfg.asset_candidates = Vec::new();
    if cfg.asset_vocab.len() <= cfg.asset_set_size + 1 {
        return;
    }
    let rets: Vec<BTreeMap<i64, f64>> = cfg
        .asset_vocab
        .iter()
        .map(|n| {
            let h = history.iter().find(|h| &h.instrument == n).expect("vocab from history");
            h.bars
                .windows(2)
                .zip(h.ends_ms.iter().skip(1))
                .filter(|(w, _)| w[0].close > 0.0 && w[1].close > 0.0)
                .map(|(w, t)| (*t, (w[1].close / w[0].close).ln()))
                .collect()
        })
        .collect();
    let corr = |a: &BTreeMap<i64, f64>, b: &BTreeMap<i64, f64>| {
        let pairs: Vec<(f64, f64)> = a.iter().filter_map(|(t, x)| b.get(t).map(|y| (*x, *y))).collect();
        if pairs.len() < 10 {
            return 0.0;
        }
        let (xa, xb): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
        fa_expert_space::pearson(&xa, &xb).abs()
    };
    let n = rets.len();
    cfg.asset_candidates = (0..n)
        .map(|i| {
            let mut peers: Vec<(usize, f64)> =
                (0..n).filter(|&j| j != i).map(|j| (j, corr(&rets[i], &rets[j]))).collect();
            peers.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            peers.into_iter().take(cfg.asset_set_size).map(|(j, _)| j).collect()
        })
        .collect();
}

/// Decoder families probed for per-expert decodability (spec 32 §6.4): each is the linear
/// representative of a spec 34 output family on market data — `forecast` (the next 4 bar
/// returns), `future` (realised absolute return of the next 8 bars), `retrieval` (the last 8
/// bar returns of the window, i.e. recalling the past the latent summarises).
pub const PROBE_FAMILIES: [&str; 3] = ["forecast", "future", "retrieval"];

fn probe_targets(ds: &Dataset, s: usize, e: usize) -> Option<[Vec<f64>; 3]> {
    let f = &ds.series[s].feats;
    if e + 8 > f.len() || e < 8 {
        return None;
    }
    let r = |i: usize| f64::from(f[i][0]);
    let forecast = (e..e + 4).map(r).collect();
    let future = vec![(e..e + 8).map(|i| r(i).abs()).sum::<f64>()];
    let retrieval = (e - 8..e).map(r).collect();
    let _ = N_FEATURES;
    Some([forecast, future, retrieval])
}

/// Per-expert decodability: for every expert and probe family, the validation R² of a ridge
/// readout fitted on the training windows from that expert's latent alone (targets
/// standardised on the training windows). `[expert] → {family → R²}`.
pub fn decodability(
    tr: &NeuralTrainer,
    ds: &Dataset,
    max_train: usize,
    max_val: usize,
) -> anyhow::Result<Vec<BTreeMap<String, f64>>> {
    let pick = |w: &[(usize, usize)], n: usize| -> Vec<(usize, usize)> {
        crate::fit::evenly(w.len(), n)
            .into_iter()
            .map(|i| w[i])
            .filter(|&(s, e)| probe_targets(ds, s, e).is_some())
            .collect()
    };
    let (fit, val) = (pick(&ds.train, max_train), pick(&ds.val, max_val));
    let k = tr.model.moe.n_active();
    if fit.len() < 32 || val.len() < 16 {
        return Ok(vec![BTreeMap::new(); k]);
    }
    let latents = |rows: &[(usize, usize)]| -> anyhow::Result<Vec<Vec<Vec<f64>>>> {
        let mut out = vec![Vec::with_capacity(rows.len()); k];
        for chunk in rows.chunks(256) {
            let b = tr.batch(ds, chunk)?;
            let f = tr.model.forward_ctx(&tr.model.ps, &b.x, b.asset.as_ref(), None, None)?;
            for row in f.out.mu.to_vec3::<f32>()? {
                for (e, mu) in row.into_iter().enumerate() {
                    out[e].push(mu.into_iter().map(f64::from).chain([1.0]).collect());
                }
            }
        }
        Ok(out)
    };
    let (xf, xv) = (latents(&fit)?, latents(&val)?);
    let mut out = vec![BTreeMap::new(); k];
    for (fam, name) in PROBE_FAMILIES.iter().enumerate() {
        let tgt = |rows: &[(usize, usize)]| -> Vec<Vec<f64>> {
            rows.iter().map(|&(s, e)| probe_targets(ds, s, e).expect("filtered")[fam].clone()).collect()
        };
        let (yf, yv) = (tgt(&fit), tgt(&val));
        let t = yf[0].len();
        let mean: Vec<f64> = (0..t).map(|j| yf.iter().map(|y| y[j]).sum::<f64>() / yf.len() as f64).collect();
        let sd: Vec<f64> = (0..t)
            .map(|j| (yf.iter().map(|y| (y[j] - mean[j]).powi(2)).sum::<f64>() / yf.len() as f64).sqrt().max(1e-12))
            .collect();
        let z = |y: &[Vec<f64>]| -> Vec<Vec<f64>> {
            y.iter().map(|r| r.iter().enumerate().map(|(j, v)| (v - mean[j]) / sd[j]).collect()).collect()
        };
        let (zf, zv) = (z(&yf), z(&yv));
        for e in 0..k {
            let w = ridge(&xf[e], &zf, 1.0);
            let (mut sse, mut sst) = (0.0, 0.0);
            for (x, y) in xv[e].iter().zip(&zv) {
                for (j, yt) in y.iter().enumerate() {
                    let p: f64 = x.iter().zip(&w).map(|(a, row)| a * row[j]).sum();
                    sse += (yt - p).powi(2);
                    sst += yt * yt;
                }
            }
            out[e].insert((*name).to_owned(), if sst > 0.0 { 1.0 - sse / sst } else { 0.0 });
        }
    }
    Ok(out)
}

/// Collapse verdict of a fit (spec 31 §8): the collapsed experts and why. A non-empty list
/// marks the candidate **unfit** (`metrics.collapsed` = count; the checklist and the Pareto
/// selection read it) instead of failing the job.
pub fn collapse_reasons(h: &HealthReport) -> Vec<Value> {
    h.collapsed
        .iter()
        .map(|&e| {
            let au = h.active_units.get(e).copied().unwrap_or(0);
            let pp = h.code_perplexity.get(e).copied();
            json!({
                "expert": e,
                "active_units": au,
                "code_perplexity": pp,
                "kind": if au == 0 { "posterior_collapse" } else { "codebook_collapse" },
            })
        })
        .collect()
}

/// Metrics a fit records from the extended health (calibration, collapse, decodability).
pub fn extra_metrics(h: &HealthReport, pos_of: &[usize]) -> Vec<(String, f64)> {
    let mut m = Vec::new();
    if !h.crpss.is_empty() {
        m.push(("crpss_min".into(), h.crpss_min()));
        m.push(("pit_ks_max".into(), h.pit_ks_max()));
    }
    m.push(("collapsed".into(), h.collapsed.len() as f64));
    if let Some(au) = h.active_units.iter().min() {
        m.push(("active_units_min".into(), *au as f64));
    }
    if let Some(pp) = h.code_perplexity.iter().copied().reduce(f64::min) {
        m.push(("code_perplexity_min".into(), pp));
    }
    let dmin: Vec<f64> = h.decodability.iter().map(|d| d.values().copied().fold(f64::NEG_INFINITY, f64::max)).collect();
    if let Some(v) = dmin.iter().copied().filter(|v| v.is_finite()).reduce(f64::min) {
        m.push(("decodability_min".into(), v));
    }
    for (pos, &src) in pos_of.iter().enumerate() {
        if let Some(v) = h.crpss.get(src) {
            m.push((format!("crpss_e{pos}"), *v));
        }
        if let Some(v) = h.pit_ks.get(src) {
            m.push((format!("pit_ks_e{pos}"), *v));
        }
        if let Some(v) = dmin.get(src).filter(|v| v.is_finite()) {
            m.push((format!("decodability_e{pos}"), *v));
        }
    }
    m
}

/// Writes a fit's validation health as the spec 32 §5 VictoriaMetrics series
/// (`fa_expert_resid_corr{tf, expert="i_j", src="validation"}`, `fa_expert_usage`,
/// `fa_expert_skill`, `fa_expert_k_eff`) and the full matrices to Garage
/// (`fa-models/snapshots/<id>/health.json`). Best effort: returns what was written.
pub async fn persist(
    svc: &crate::run::Services<'_>,
    ts: Timescale,
    snapshot_id: &str,
    expert_ids: &[u32],
    h: &HealthReport,
    now_ms: i64,
) -> Value {
    let samples: Vec<fa_tsdb::Sample> = h
        .vm_series(ts.as_str(), "validation", expert_ids)
        .into_iter()
        .map(|(m, labels, v)| labels.into_iter().fold(fa_tsdb::Sample::new(m, v, now_ms), |s, (k, l)| s.label(k, l)))
        .collect();
    let vm = match svc.tsdb.write(&samples).await {
        Ok(()) => json!({ "samples": samples.len() }),
        Err(e) => json!({ "error": e.to_string() }),
    };
    let garage = match &svc.stores.blob {
        Some(blob) => {
            let key = fa_blob::snapshot_file(snapshot_id, "health.json");
            let body = json!({ "snapshot_id": snapshot_id, "timescale": ts, "source": "validation", "as_of_ms": now_ms, "expert_ids": expert_ids, "health": h });
            match fa_blob::BlobStore::put(blob, &key, serde_json::to_vec(&body).unwrap_or_default(), fa_blob::JSON)
                .await
            {
                Ok(_) => json!(key.key),
                Err(e) => json!({ "error": e.to_string() }),
            }
        }
        None => Value::Null,
    };
    json!({ "victoriametrics": vm, "garage": garage })
}

#[cfg(test)]
mod tests {
    use fa_encoders::{
        MarketBar,
        neural::{NeuralFamily, Series},
    };

    use super::*;
    use crate::data::Source;

    fn hist(name: &str, n: usize, seed: u64) -> BarHistory {
        let mut rng = fa_core::rng::Rng::seed(seed);
        let mut p = 100.0;
        let bars: Vec<MarketBar> = (0..n)
            .map(|_| {
                let o = p;
                p *= 1.0 + 0.002 * rng.normal();
                MarketBar {
                    open: o,
                    high: o.max(p) * 1.001,
                    low: o.min(p) * 0.999,
                    close: p,
                    volume: 10.0,
                    trade_count: 3,
                }
            })
            .collect();
        BarHistory {
            instrument: name.into(),
            timescale: Timescale::Minute,
            source: Source::Bars,
            start_ms: 0,
            end_ms: n as i64 * 60_000,
            ends_ms: (1..=n as i64).map(|i| i * 60_000).collect(),
            bars,
        }
    }

    #[test]
    fn asset_vocab_candidates_and_decodability() {
        let h: Vec<BarHistory> = (0..5).map(|i| hist(&format!("i{i}"), 200, i)).collect();
        let mut cfg = NeuralConfig::for_timescale(NeuralFamily::JepaVae, Timescale::Minute);
        asset_setup(&mut cfg, &h, Timescale::Minute);
        assert!(cfg.asset_vocab.is_empty(), "off without the gene");
        cfg.asset_attn = true;
        cfg.asset_set_size = 2;
        asset_setup(&mut cfg, &h, Timescale::Minute);
        assert_eq!(cfg.asset_vocab, vec!["i0", "i1", "i2", "i3", "i4"]);
        assert_eq!(cfg.asset_max_age_ms, 180_000);
        assert_eq!(cfg.asset_candidates.len(), 5);
        assert!(cfg.asset_candidates.iter().enumerate().all(|(i, c)| c.len() == 2 && !c.contains(&i)));

        let small = NeuralConfig {
            context_steps: 16,
            d_model: 8,
            layers: 2,
            batch: 16,
            moe: fa_expert_space::moe::MoeConfig { max_experts: 2, expert_dim: 3, ..Default::default() },
            ..NeuralConfig::new(NeuralFamily::JepaVae)
        };
        let series =
            h.iter().map(|b| Series::with_times(b.instrument.clone(), b.bars.clone(), b.ends_ms.clone())).collect();
        let ds = Dataset::build(series, 16, small.horizon(), 0.25);
        let mut tr = NeuralTrainer::new(small, &ds, 1, &candle_core::Device::Cpu).unwrap();
        tr.train_steps(&ds, 5).unwrap();
        let d = decodability(&tr, &ds, 300, 200).unwrap();
        assert_eq!(d.len(), 2);
        assert!(d.iter().all(|m| PROBE_FAMILIES.iter().all(|f| m.get(*f).is_some_and(|v| v.is_finite()))));
        let mut hr =
            HealthReport { decodability: d, collapsed: vec![1], active_units: vec![3, 0], ..Default::default() };
        hr.crpss = vec![0.1, -0.2];
        hr.pit_ks = vec![0.05, 0.3];
        let m: BTreeMap<String, f64> = extra_metrics(&hr, &[1, 0]).into_iter().collect();
        assert_eq!(m["collapsed"], 1.0);
        assert_eq!(m["crpss_e0"], -0.2, "per final position");
        assert!(m.contains_key("decodability_min"));
        assert_eq!(collapse_reasons(&hr)[0]["kind"], "posterior_collapse");
    }
}
