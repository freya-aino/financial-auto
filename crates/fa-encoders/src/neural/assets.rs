//! Asset conditioning and cross-asset attention (spec 31 §6.4).
//!
//! With the `asset_attn` gene every window is conditioned on a **learned asset embedding** (one
//! row per instrument of the training vocabulary `asset_vocab`, plus an "unknown" row for
//! instruments the snapshot never saw), and the pooled backbone state attends over a
//! **candidate set** of other assets before the expert router:
//!
//! * a candidate's token is its embedding plus a linear map of its last `asset_ctx` normalised
//!   bars that ended **strictly before** the window's last bar (point in time: the same rule
//!   live, where the encoder keeps each instrument's latest window, [`AssetBook`]); candidates
//!   whose latest bar is older than `asset_max_age_ms` are masked;
//! * single-head scaled dot-product attention with a learned **null** token (always available,
//!   so a window with no live peers reads nothing), output projected and added to the state;
//! * the attention weights over the candidates are exported (`asset_attn`, per window, and as
//!   a per-instrument mean in the snapshot manifest), and an **entropy penalty**
//!   (`asset_attn_sparsity`) pushes each window to select few assets.
//!
//! Candidate sets: all other vocabulary assets while the vocabulary fits `asset_set_size`;
//! beyond that the trainer fills `asset_candidates` with each asset's `asset_set_size` most
//! return-correlated peers (the hand-off rule of spec 31 §10 Q4; MCMC cross-asset priors can
//! supply the same list).

use std::collections::{BTreeMap, HashMap, VecDeque};

use candle_core::{D, DType, Device, Result, Tensor};
use fa_expert_space::nn::{Init, Linear, ParamStore, Weights, linear_w, softmax_last};

use super::{
    data::{InputNorm, N_FEATURES, Series},
    model::NeuralConfig,
};

/// Creates the asset parameters (`asset.*`).
pub fn init(ps: &mut ParamStore, cfg: &NeuralConfig) -> Result<()> {
    let d = cfg.d_model;
    ps.get("asset.emb", &[cfg.asset_vocab.len() + 1, d], Init::Normal(0.1))?;
    ps.get("asset.null", &[d], Init::Normal(0.1))?;
    Linear::new(ps, "asset.tok", cfg.asset_ctx.max(1) * N_FEATURES, d, true)?;
    for n in ["asset.q", "asset.k", "asset.v"] {
        Linear::new(ps, n, d, d, true)?;
    }
    // the output projection starts at zero: the asset path begins as the identity
    ps.get("asset.o.w", &[d, d], Init::Zeros)?;
    ps.get("asset.o.b", &[d], Init::Zeros)?;
    Ok(())
}

/// A row's peer lookup: the instrument's normalised token of its latest window strictly before
/// the row's time, or `None`.
pub type PeerLookup<'a> = Box<dyn Fn(&str) -> Option<Vec<f32>> + 'a>;
/// One row of [`build`]: own instrument and its peer lookup.
pub type AssetRow<'a> = (String, PeerLookup<'a>);

/// Asset context of a batch.
#[derive(Debug, Clone)]
pub struct AssetBatch {
    /// Vocabulary slot of each row's own instrument `[B]` (`len(vocab)` = unknown).
    pub slot: Tensor,
    /// Candidate vocabulary slots `[B, M]` (padding repeats the unknown slot, masked).
    pub cand: Tensor,
    /// Candidate tokens `[B, M, asset_ctx · F]` (normalised bars).
    pub tokens: Tensor,
    /// Additive mask `[B, M]` (0 = available, −1e9 = masked).
    pub mask: Tensor,
    /// Instrument of every candidate column per row (`None` = padding / masked).
    pub names: Vec<Vec<Option<String>>>,
}

/// Candidate vocabulary indices of vocabulary entry `v` (all others when no list is given).
pub fn candidates_of(cfg: &NeuralConfig, v: Option<usize>) -> Vec<usize> {
    let n = cfg.asset_vocab.len();
    let m = cfg.asset_set_size.max(1);
    match v {
        Some(v) if cfg.asset_candidates.get(v).is_some_and(|c| !c.is_empty()) => {
            cfg.asset_candidates[v].iter().copied().filter(|c| *c < n && *c != v).take(m).collect()
        }
        Some(v) => (0..n).filter(|c| *c != v).take(m).collect(),
        None => (0..n).take(m).collect(),
    }
}

/// Canonical instrument key (VictoriaMetrics token: dashless, lower case).
pub fn key(instrument: &str) -> String {
    instrument.chars().filter(|c| *c != '-').collect::<String>().to_ascii_lowercase()
}

/// Columns of a batch: the widest candidate list (≤ `asset_set_size`).
fn width(cfg: &NeuralConfig) -> usize {
    cfg.asset_vocab.len().min(cfg.asset_set_size.max(1)).max(1)
}

/// Builds the asset context of rows `(own instrument, window end time, candidate windows)`:
/// `lookup(candidate instrument)` returns that instrument's normalised token features
/// (`asset_ctx · F`) of its latest window strictly before the row's time, or `None`.
pub fn build(
    cfg: &NeuralConfig,
    rows: &[AssetRow<'_>],
    dev: &Device,
) -> Result<AssetBatch> {
    let vocab: HashMap<String, usize> = cfg.asset_vocab.iter().enumerate().map(|(i, s)| (key(s), i)).collect();
    let unknown = cfg.asset_vocab.len();
    let m = width(cfg);
    let tok_w = cfg.asset_ctx.max(1) * N_FEATURES;
    let (mut slots, mut cands, mut toks, mut mask, mut names) = (
        Vec::with_capacity(rows.len()),
        Vec::with_capacity(rows.len() * m),
        Vec::with_capacity(rows.len() * m * tok_w),
        Vec::with_capacity(rows.len() * m),
        Vec::with_capacity(rows.len()),
    );
    for (inst, lookup) in rows {
        let own = vocab.get(&key(inst)).copied();
        slots.push(own.unwrap_or(unknown) as u32);
        let list = candidates_of(cfg, own);
        let mut row_names = Vec::with_capacity(m);
        for col in 0..m {
            let hit = list.get(col).and_then(|&c| lookup(&cfg.asset_vocab[c]).map(|t| (c, t)));
            match hit {
                Some((c, t)) if t.len() == tok_w => {
                    cands.push(c as u32);
                    toks.extend(t);
                    mask.push(0f32);
                    row_names.push(Some(cfg.asset_vocab[c].clone()));
                }
                _ => {
                    cands.push(unknown as u32);
                    toks.extend(std::iter::repeat_n(0f32, tok_w));
                    mask.push(-1e9);
                    row_names.push(None);
                }
            }
        }
        names.push(row_names);
    }
    let b = rows.len();
    Ok(AssetBatch {
        slot: Tensor::from_vec(slots, b, dev)?,
        cand: Tensor::from_vec(cands, (b, m), dev)?,
        tokens: Tensor::from_vec(toks, (b, m, tok_w), dev)?,
        mask: Tensor::from_vec(mask, (b, m), dev)?,
        names,
    })
}

/// Training-side context: rows `(series, end)` of a dataset whose series carry bar end times.
pub fn batch_from_series(
    cfg: &NeuralConfig,
    norm: &InputNorm,
    series: &[Series],
    idx: &[(usize, usize)],
    dev: &Device,
) -> Result<AssetBatch> {
    let by_key: HashMap<String, usize> = series.iter().enumerate().map(|(i, s)| (key(&s.instrument), i)).collect();
    let ctx = cfg.asset_ctx.max(1);
    let rows: Vec<AssetRow<'_>> = idx
        .iter()
        .map(|&(s, e)| {
            let t = series[s].ends_ms.get(e.saturating_sub(1)).copied();
            let by_key = &by_key;
            let f: PeerLookup<'_> = Box::new(move |inst: &str| {
                let t = t?;
                let j = *by_key.get(&key(inst))?;
                let ser = &series[j];
                let n = ser.ends_ms.partition_point(|x| *x < t);
                if n == 0 || t - ser.ends_ms[n - 1] > cfg.asset_max_age_ms {
                    return None;
                }
                Some(norm.window(&ser.feats, n, ctx))
            });
            (series[s].instrument.clone(), f)
        })
        .collect();
    build(cfg, &rows, dev)
}

/// Applies the asset path to the pooled state `h [B, d]`: own embedding, attention over the
/// candidates (+ null token). Returns the conditioned state and the attention `[B, 1 + M]`
/// (column 0 = null).
pub fn forward(w: &dyn Weights, h: &Tensor, a: &AssetBatch) -> Result<(Tensor, Tensor)> {
    let (b, d) = h.dims2()?;
    let emb = w.w("asset.emb")?;
    let own = emb.index_select(&a.slot, 0)?; // [B, d]
    let h1 = (h + own)?;
    let q = linear_w(w, "asset.q", &h1)?; // [B, d]
    let m = a.cand.dim(1)?;
    let cemb = emb.index_select(&a.cand.flatten_all()?, 0)?.reshape((b, m, d))?;
    let tok = (linear_w(w, "asset.tok", &a.tokens)? + cemb)?; // [B, M, d]
    let null = w.w("asset.null")?.reshape((1, 1, d))?.broadcast_as((b, 1, d))?.contiguous()?;
    let all = Tensor::cat(&[&null, &tok], 1)?; // [B, 1+M, d]
    let k = linear_w(w, "asset.k", &all)?;
    let v = linear_w(w, "asset.v", &all)?;
    let logits = k.matmul(&q.unsqueeze(2)?)?.squeeze(2)?.affine(1.0 / (d as f64).sqrt(), 0.0)?; // [B, 1+M]
    let mask = Tensor::cat(&[&Tensor::zeros((b, 1), DType::F32, h.device())?, &a.mask], 1)?;
    let attn = softmax_last(&(logits + mask)?)?;
    let ctx = attn.unsqueeze(1)?.matmul(&v)?.squeeze(1)?; // [B, d]
    Ok(((h1 + linear_w(w, "asset.o", &ctx)?)?, attn))
}

/// Mean entropy of the attention rows (the sparsity penalty: lower = fewer assets selected).
pub fn entropy(attn: &Tensor) -> Result<Tensor> {
    let p = (attn + 1e-9)?;
    (p.log()? * attn)?.sum(D::Minus1)?.neg()?.mean_all()
}

/// Host-side attention of one row over its named candidates (null excluded, renormalised to
/// the attention mass on assets; the null share is returned separately).
pub fn row_weights(attn_row: &[f32], names: &[Option<String>]) -> (Vec<(String, f32)>, f32) {
    let null = attn_row.first().copied().unwrap_or(1.0);
    let out = names
        .iter()
        .enumerate()
        .filter_map(|(c, n)| n.as_ref().map(|n| (n.clone(), attn_row.get(c + 1).copied().unwrap_or(0.0))))
        .collect();
    (out, null)
}

/// Live asset context: each instrument's latest normalised windows by step time, so a window
/// of one instrument can read its peers' most recent windows **strictly before** its own time
/// (the training rule). Two entries per instrument are kept (the current step's and the
/// previous one), so peers encoded earlier in the same step are not read.
#[derive(Debug, Default)]
pub struct AssetBook {
    entries: BTreeMap<String, VecDeque<(i64, Vec<f32>)>>,
}

impl AssetBook {
    /// Records `instrument`'s window at `as_of_ms` (`tokens` = its last `asset_ctx` normalised bars).
    pub fn record(&mut self, instrument: &str, as_of_ms: i64, tokens: Vec<f32>) {
        let q = self.entries.entry(key(instrument)).or_default();
        if q.back().is_some_and(|(t, _)| *t == as_of_ms) {
            q.pop_back();
        }
        q.push_back((as_of_ms, tokens));
        while q.len() > 2 {
            q.pop_front();
        }
    }

    /// `instrument`'s latest window strictly before `t_ms` and no older than `max_age_ms`.
    pub fn before(&self, instrument: &str, t_ms: i64, max_age_ms: i64) -> Option<Vec<f32>> {
        let q = self.entries.get(&key(instrument))?;
        q.iter().rev().find(|(t, _)| *t < t_ms && t_ms - *t <= max_age_ms).map(|(_, v)| v.clone())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        MarketBar,
        neural::{NeuralFamily, data::Dataset},
    };

    fn cfg() -> NeuralConfig {
        NeuralConfig {
            d_model: 8,
            asset_attn: true,
            asset_ctx: 2,
            asset_vocab: vec!["a".into(), "b".into(), "c".into()],
            asset_max_age_ms: 120_000,
            ..NeuralConfig::new(NeuralFamily::JepaVae)
        }
    }

    fn series(name: &str, n: usize, t0: i64) -> Series {
        let bars = (0..n)
            .map(|i| {
                let p = 100.0 + i as f64 * 0.1;
                MarketBar { open: p, high: p + 1.0, low: p - 1.0, close: p + 0.05, volume: 5.0, trade_count: 2 }
            })
            .collect();
        Series::with_times(name, bars, (0..n as i64).map(|i| t0 + (i + 1) * 60_000).collect())
    }

    #[test]
    fn context_is_point_in_time_and_masks_stale_or_unknown_peers() {
        let c = cfg();
        // b is shifted by 30 s, c stops early
        let ds = Dataset::build(vec![series("a", 40, 0), series("b", 40, 30_000), series("c", 10, 0)], 4, 1, 0.0);
        let norm = InputNorm::fit(&ds);
        let rows = [(0usize, 20usize)];
        let ab = batch_from_series(&c, &norm, &ds.series, &rows, &Device::Cpu).unwrap();
        // own slot a = 0, candidates b and c
        assert_eq!(ab.slot.to_vec1::<u32>().unwrap(), vec![0]);
        // columns = min(vocab, asset_set_size); an own-vocab row has one padding column
        assert_eq!(ab.names[0], vec![Some("b".to_owned()), None, None]);
        assert_eq!(ab.mask.to_vec2::<f32>().unwrap()[0], vec![0.0, -1e9, -1e9]);
        // a's window ends at bar 19 (t = 20 min); b's bars end at k min + 30 s: the latest strictly
        // before is bar 18 (19.5 min), so b's token is its window ending at index 19
        let expect = norm.window(&ds.series[1].feats, 19, 2);
        let got = ab.tokens.to_vec3::<f32>().unwrap()[0][0].clone();
        assert_eq!(got, expect);
        // an unknown instrument gets the unknown slot and every vocab entry as candidate
        let ds2 = Dataset::build(vec![series("zz", 30, 0), series("a", 30, 0)], 4, 1, 0.0);
        let ab = batch_from_series(&c, &norm, &ds2.series, &[(0, 10)], &Device::Cpu).unwrap();
        assert_eq!(ab.slot.to_vec1::<u32>().unwrap(), vec![3]);
        assert_eq!(ab.names[0][0].as_deref(), Some("a"));
    }

    #[test]
    fn attention_starts_as_identity_plus_embedding_and_exports_weights() {
        let c = cfg();
        let mut ps = ParamStore::new(3, &Device::Cpu);
        init(&mut ps, &c).unwrap();
        let ds = Dataset::build(vec![series("a", 30, 0), series("b", 30, 0), series("c", 30, 0)], 4, 1, 0.0);
        let norm = InputNorm::fit(&ds);
        let ab = batch_from_series(&c, &norm, &ds.series, &[(0, 10), (1, 12)], &Device::Cpu).unwrap();
        let h = Tensor::ones((2, 8), DType::F32, &Device::Cpu).unwrap();
        let (out, attn) = forward(&ps, &h, &ab).unwrap();
        let own = ps.w("asset.emb").unwrap().index_select(&ab.slot, 0).unwrap();
        let diff = (out - (h + own).unwrap()).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-6, "zero output projection: identity + embedding");
        let a = attn.to_vec2::<f32>().unwrap();
        assert!(a.iter().all(|r| (r.iter().sum::<f32>() - 1.0).abs() < 1e-5 && r.len() == 4 && r[3] < 1e-6));
        let (w, null) = row_weights(&a[0], &ab.names[0]);
        assert_eq!(w.len(), 2);
        assert!(null > 0.0 && null < 1.0);
        assert!(entropy(&attn).unwrap().to_scalar::<f32>().unwrap() > 0.0);
    }

    #[test]
    fn book_reads_peers_strictly_before() {
        let mut b = AssetBook::default();
        b.record("A-1", 60_000, vec![1.0]);
        b.record("A-1", 120_000, vec![2.0]);
        b.record("a1", 180_000, vec![3.0]);
        assert_eq!(b.before("a1", 180_000, 120_000), Some(vec![2.0]), "same-step entry is not read");
        assert_eq!(b.before("a1", 181_000, 120_000), Some(vec![3.0]));
        assert_eq!(b.before("a1", 600_000, 120_000), None, "stale");
        assert_eq!(b.len(), 1);
    }
}
