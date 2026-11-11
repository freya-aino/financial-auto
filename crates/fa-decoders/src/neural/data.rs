//! Inputs and targets of the neural families (spec 34 §6.1), shared by training, evaluation,
//! serving and the encoder feedback path so all four see exactly the same features.
//!
//! **Input row** = `[return window (seq) | extra | latent (latent) | latent-present flag]`:
//! - *return window*: the last `lb` one-step log returns ending at `t`, divided by their own
//!   standard deviation `σ_t` (floored) and clipped to ±5. Absent for `nn_chart` when the decoder
//!   is latent-conditioned (it must reconstruct the chart from the latent alone);
//! - *extra*: `nn_narrative` only, the folded news embedding of the trailing text window;
//! - *latent*: the instrument's latent **as of** `ts[t]` (concatenated expert means, spec 32),
//!   standardised with the training mean / std; zeros + flag 0 when none is known.
//!
//! **Targets** (all known only after `outcome_end_ms`, which training keeps ≤ the train cutoff):
//! `z = r_{t→t+h} / (σ_t·√h)` for `nn_forecast` / `nn_retrieval`; the chart raster of the lookback
//! window for `nn_chart`; the future raster of the next `h` steps (rows ±`band·σ_t·√h`) for
//! `nn_future`; the news embedding of `(t − tw, t]` (`nn_news`) or `(t, t + tw]` (`nn_narrative`).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    baseline::Series,
    genome::{BaselineFamily, DecoderGenome},
    render,
    text_baseline::{LEXICON_NEG, LEXICON_POS, hash_bucket, tokenize},
};

/// Return-window features are clipped to this many σ.
pub const FEATURE_CLIP: f32 = 5.0;
/// Normalised targets `z` are clipped to this many σ.
pub const Z_CLIP: f64 = 10.0;
/// σ floor (log-return units) so flat windows do not explode.
pub const SIGMA_FLOOR: f64 = 1e-7;
/// Standardised latents are clipped to ±this (training, serving and feedback alike).
pub const LATENT_CLIP: f32 = 8.0;
/// Floor of the latent standard deviation (near-constant dimensions must not blow up later).
pub const LATENT_STD_FLOOR: f64 = 1e-3;

/// Very common English tokens dropped from the news embedding.
pub const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "at", "by", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "it", "its", "this", "that", "these", "those", "has", "have", "had", "will", "would", "can",
    "could", "after", "before", "over", "into", "about", "than", "but", "not", "no", "new", "says", "said", "how",
    "what", "why", "who", "you", "your", "we", "our", "they", "their", "he", "she", "his", "her", "up", "out", "more",
    "all", "just", "now", "here", "there", "news", "com", "www", "https", "http",
];

/// Width of each block of an input row and of the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    /// Return-window features (`lb`, or 0).
    pub seq: usize,
    /// Extra context features (news embedding of `nn_narrative`).
    pub extra: usize,
    /// Latent dimension (0 = unconditioned).
    pub latent: usize,
    /// Network output width.
    pub out: usize,
    /// Target width.
    pub target: usize,
}

impl Layout {
    /// Input row width (the flag column exists only with a latent).
    pub fn input(&self) -> usize {
        self.seq + self.extra + self.latent + usize::from(self.latent > 0)
    }

    /// Width of the context part (everything before the latent).
    pub fn context(&self) -> usize {
        self.seq + self.extra
    }

    /// Layout of a genome for a given latent dimension.
    pub fn of(g: &DecoderGenome, latent: usize) -> Self {
        let lb = g.lookback;
        let k = g.levels.len();
        let res = g.nn.res;
        let tdim = g.nn.tdim;
        match g.family {
            BaselineFamily::NnRetrieval => Self { seq: lb, extra: 0, latent, out: g.nn.emb, target: 1 },
            BaselineFamily::NnChart => {
                Self { seq: if latent > 0 { 0 } else { lb }, extra: 0, latent, out: res * res, target: res * res }
            }
            BaselineFamily::NnFuture => Self { seq: lb, extra: 0, latent, out: res * res, target: res * res },
            BaselineFamily::NnNews | BaselineFamily::NnFrame => Self { seq: lb, extra: 0, latent, out: tdim, target: tdim },
            BaselineFamily::NnSummary => {
                let w = crate::summary::width();
                Self { seq: if latent > 0 { 0 } else { lb }, extra: 0, latent, out: w, target: w }
            }
            BaselineFamily::NnNarrative => Self { seq: lb, extra: tdim, latent, out: tdim, target: tdim },
            // nn_forecast (and the baselines, which never train)
            _ => Self { seq: lb, extra: 0, latent, out: k, target: 1 },
        }
    }
}

/// Latent standardisation learnt on the training rows.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Norm {
    pub lat_mean: Vec<f32>,
    pub lat_std: Vec<f32>,
}

impl Norm {
    pub fn fit(latents: &[&[f32]], dim: usize) -> Self {
        if dim == 0 {
            return Self::default();
        }
        let n = latents.len().max(1) as f64;
        let mut mean = vec![0.0f64; dim];
        for l in latents {
            for (m, x) in mean.iter_mut().zip(l.iter()) {
                *m += f64::from(*x) / n;
            }
        }
        let mut var = vec![0.0f64; dim];
        for l in latents {
            for ((v, x), m) in var.iter_mut().zip(l.iter()).zip(&mean) {
                *v += (f64::from(*x) - m).powi(2) / n;
            }
        }
        Self {
            lat_mean: mean.iter().map(|m| *m as f32).collect(),
            lat_std: var.iter().map(|v| (v.sqrt().max(LATENT_STD_FLOOR)) as f32).collect(),
        }
    }

    /// `[standardised latent | 1]`, or `[0… | 0]` when absent / of another dimension.
    pub fn latent_block(&self, latent: Option<&[f32]>, dim: usize) -> Vec<f32> {
        if dim == 0 {
            return Vec::new();
        }
        match latent.filter(|l| l.len() == dim) {
            Some(l) => {
                let mut v: Vec<f32> = l
                    .iter()
                    .zip(&self.lat_mean)
                    .zip(&self.lat_std)
                    .map(|((x, m), s)| ((x - m) / s).clamp(-LATENT_CLIP, LATENT_CLIP))
                    .collect();
                v.push(1.0);
                v
            }
            None => {
                let mut v = vec![0.0; dim];
                v.push(0.0);
                v
            }
        }
    }
}

/// One timeline item: knowledge time, sparse hashed tf vector, matched instrument tokens.
type Doc = (i64, Vec<(u32, f32)>, Vec<String>);

/// News timeline: one sparse hashed term-frequency vector per item, by knowledge time, tagged
/// with the instruments the item was matched to (`text_baseline::InstrumentIndex`, symbol / name
/// / entity). Market-wide reads use every item; genomes with `ninst=1` read only the items of the
/// series' instrument ([`Self::embedding_for`]).
#[derive(Debug, Clone, Default)]
pub struct TextTimeline {
    docs: Vec<Doc>,
    sorted: bool,
    /// A media-frame timeline ([`Self::frames`], `visual.frame`) rather than news.
    frames: bool,
}

impl TextTimeline {
    pub fn new() -> Self {
        Self { docs: Vec::new(), sorted: true, frames: false }
    }

    /// An empty **frame** timeline (items pushed with [`Self::push_features`] from
    /// [`frame_features`]). Only `nn_frame` reads targets from it; the news families read none
    /// from it, and `nn_frame` reads none from a news timeline.
    pub fn frames() -> Self {
        Self { docs: Vec::new(), sorted: true, frames: true }
    }

    pub fn is_frames(&self) -> bool {
        self.frames
    }

    /// Adds one item (title + body) known at `ts_ms`. Items without content tokens are skipped.
    pub fn push(&mut self, ts_ms: i64, title: &str, body: &str) {
        self.push_tagged(ts_ms, title, body, &[]);
    }

    /// [`Self::push`] with the instruments (UUIDs, any case / dashes) the item mentions.
    pub fn push_tagged(&mut self, ts_ms: i64, title: &str, body: &str, instruments: &[String]) {
        let mut tf: HashMap<u32, f32> = HashMap::new();
        for tok in tokenize(&format!("{title} {body}")) {
            if STOPWORDS.contains(&tok.as_str()) || tok.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            *tf.entry(hash_bucket(&tok) as u32).or_default() += 1.0;
        }
        if tf.is_empty() {
            return;
        }
        let mut v: Vec<(u32, f32)> = tf.into_iter().map(|(b, c)| (b, 1.0 + c.ln())).collect();
        v.sort_by_key(|(b, _)| *b);
        if self.docs.last().is_some_and(|(t, _, _)| *t > ts_ms) {
            self.sorted = false;
        }
        let mut tags: Vec<String> = instruments.iter().map(|i| crate::instrument_token(i)).collect();
        tags.sort();
        tags.dedup();
        self.docs.push((ts_ms, v, tags));
    }

    /// Adds one item of precomputed sparse hashed features (`(bucket, weight)`, e.g. the frame
    /// features of [`frame_features`]); items without features are skipped.
    pub fn push_features(&mut self, ts_ms: i64, features: Vec<(u32, f32)>, instruments: &[String]) {
        if features.is_empty() {
            return;
        }
        if self.docs.last().is_some_and(|(t, _, _)| *t > ts_ms) {
            self.sorted = false;
        }
        let mut tags: Vec<String> = instruments.iter().map(|i| crate::instrument_token(i)).collect();
        tags.sort();
        tags.dedup();
        self.docs.push((ts_ms, features, tags));
    }

    /// Items tagged with at least one instrument.
    pub fn matched(&self) -> usize {
        self.docs.iter().filter(|d| !d.2.is_empty()).count()
    }

    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.docs.sort_by_key(|(t, _, _)| *t);
            self.sorted = true;
        }
    }

    /// Sorts the items by time (call after out-of-order pushes, before reading).
    pub fn finish(&mut self) {
        self.ensure_sorted();
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Drops items older than `ms`.
    pub fn prune_before(&mut self, ms: i64) {
        self.ensure_sorted();
        let n = self.docs.partition_point(|(t, _, _)| *t < ms);
        self.docs.drain(..n);
    }

    /// `(first, last)` item times.
    pub fn span(&self) -> Option<(i64, i64)> {
        Some((self.docs.first()?.0, self.docs.last()?.0))
    }

    /// Items with `from_ms < ts ≤ to_ms`.
    pub fn count(&self, from_ms: i64, to_ms: i64) -> usize {
        let a = self.docs.partition_point(|(t, _, _)| *t <= from_ms);
        let b = self.docs.partition_point(|(t, _, _)| *t <= to_ms);
        b.saturating_sub(a)
    }

    /// L2-normalised folded embedding (`bucket mod tdim`) of the items with `from_ms < ts ≤ to_ms`;
    /// `None` when there are none. Requires a sorted timeline ([`Self::finish`]).
    pub fn embedding(&self, from_ms: i64, to_ms: i64, tdim: usize) -> Option<Vec<f32>> {
        self.embedding_for(None, from_ms, to_ms, tdim)
    }

    /// [`Self::embedding`] over the items tagged with `instrument` (`None` = every item).
    pub fn embedding_for(&self, instrument: Option<&str>, from_ms: i64, to_ms: i64, tdim: usize) -> Option<Vec<f32>> {
        debug_assert!(self.sorted, "TextTimeline::finish() before reading");
        let a = self.docs.partition_point(|(t, _, _)| *t <= from_ms);
        let b = self.docs.partition_point(|(t, _, _)| *t <= to_ms);
        if b <= a || tdim == 0 {
            return None;
        }
        let tok = instrument.map(crate::instrument_token);
        let mut v = vec![0.0f32; tdim];
        let mut any = false;
        for (_, doc, tags) in &self.docs[a..b] {
            if let Some(tok) = &tok
                && tags.binary_search(tok).is_err()
            {
                continue;
            }
            any = true;
            for (bucket, w) in doc {
                v[*bucket as usize % tdim] += w;
            }
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        (any && n > 0.0).then(|| v.into_iter().map(|x| x / n).collect())
    }
}

/// Byte n-gram length of [`frame_features`].
pub const FRAME_NGRAM: usize = 4;
/// At most this many n-grams are hashed per frame (evenly strided over the bytes).
pub const FRAME_MAX_GRAMS: usize = 65_536;

/// Frozen front-end of the `visual.frame` family: signed-free feature hashing of the frame's byte
/// 4-grams (FNV-1a buckets, `1 + ln count`), the always-available stand-in of spec 31 §6.3's
/// hashed tower — no pretrained vision weights are fetched (D-010). It carries byte-level, not
/// semantic, similarity; a pretrained tower would replace it behind the same timeline.
pub fn frame_features(bytes: &[u8]) -> Vec<(u32, f32)> {
    if bytes.len() < FRAME_NGRAM {
        return Vec::new();
    }
    let n = bytes.len() - FRAME_NGRAM + 1;
    let stride = n.div_ceil(FRAME_MAX_GRAMS).max(1);
    let mut tf: HashMap<u32, f32> = HashMap::new();
    for i in (0..n).step_by(stride) {
        let mut h: u32 = 0x811c_9dc5;
        for b in &bytes[i..i + FRAME_NGRAM] {
            h ^= u32::from(*b);
            h = h.wrapping_mul(0x0100_0193);
        }
        *tf.entry(h).or_default() += 1.0;
    }
    let mut v: Vec<(u32, f32)> = tf.into_iter().map(|(b, c)| (b, 1.0 + c.ln())).collect();
    v.sort_by_key(|(b, _)| *b);
    v
}

/// Instrument whose news a genome reads (`ninst=1`: the series' own; else market-wide).
pub fn text_scope<'a>(g: &DecoderGenome, s: &'a Series) -> Option<&'a str> {
    g.nn.news_inst.then_some(s.instrument_id.as_str())
}

/// Instrument whose latents condition a genome on series `s` (`src` gene, else `s` itself).
pub fn latent_instrument<'a>(g: &'a DecoderGenome, s: &'a Series) -> &'a str {
    g.source.as_deref().unwrap_or(&s.instrument_id)
}

/// Unit sentiment direction in the folded embedding space (+ lexicon-positive, − negative
/// buckets): `dot(embedding, direction)` is the embedding's lexicon polarity.
pub fn sentiment_direction(tdim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; tdim];
    for w in LEXICON_POS {
        v[hash_bucket(w) % tdim] += 1.0;
    }
    for w in LEXICON_NEG {
        v[hash_bucket(w) % tdim] -= 1.0;
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}

/// σ of the one-step log returns over `(t − lb, t]`, floored at [`SIGMA_FLOOR`].
pub fn sigma_at(s: &Series, t: usize, lb: usize) -> Option<f64> {
    if t < lb || t >= s.len() {
        return None;
    }
    let rets: Vec<f64> = (t + 1 - lb..=t).map(|i| s.log_price[i] - s.log_price[i - 1]).collect();
    let m = rets.iter().sum::<f64>() / lb as f64;
    let var = rets.iter().map(|r| (r - m).powi(2)).sum::<f64>() / lb as f64;
    Some(var.sqrt().max(SIGMA_FLOOR))
}

/// Return window ending at `t` in units of `sigma`, clipped.
pub fn return_window(s: &Series, t: usize, lb: usize, sigma: f64) -> Option<Vec<f32>> {
    if t < lb || t >= s.len() {
        return None;
    }
    Some(
        (t + 1 - lb..=t)
            .map(|i| (((s.log_price[i] - s.log_price[i - 1]) / sigma) as f32).clamp(-FEATURE_CLIP, FEATURE_CLIP))
            .collect(),
    )
}

/// Text window length in ms.
pub fn text_window_ms(g: &DecoderGenome) -> i64 {
    (g.nn.tw_min * 60_000.0).round() as i64
}

/// Input row of a genome at step `t` (see the module docs) and `σ_t`.
pub fn input_at(
    g: &DecoderGenome,
    lay: &Layout,
    norm: &Norm,
    s: &Series,
    t: usize,
    latent: Option<&[f32]>,
    text: &TextTimeline,
) -> Option<(Vec<f32>, f64)> {
    let sigma = sigma_at(s, t, g.lookback)?;
    let mut x = Vec::with_capacity(lay.input());
    if lay.seq > 0 {
        x.extend(return_window(s, t, lay.seq, sigma)?);
    }
    if lay.extra > 0 {
        let now = s.ts_ms[t];
        let e = text
            .embedding_for(text_scope(g, s), now - text_window_ms(g), now, g.nn.tdim)
            .unwrap_or_else(|| vec![0.0; g.nn.tdim]);
        x.extend(e);
    }
    x.extend(norm.latent_block(latent, lay.latent));
    debug_assert_eq!(x.len(), lay.input());
    Some((x, sigma))
}

/// Target of a genome at step `t` and the time it becomes known (`outcome_end_ms`); `None`
/// when the series / timeline cannot provide it yet.
pub fn target_at(g: &DecoderGenome, s: &Series, t: usize, sigma: f64, text: &TextTimeline) -> Option<(Vec<f32>, i64)> {
    let h = g.horizon;
    match g.family {
        BaselineFamily::NnChart => {
            let lb = g.lookback;
            if t < lb || t >= s.len() {
                return None;
            }
            Some((render::chart_raster(&s.log_price[t + 1 - lb..=t], g.nn.res)?, s.ts_ms[t]))
        }
        BaselineFamily::NnFuture => {
            if t + h >= s.len() {
                return None;
            }
            let cum: Vec<f64> = (1..=h).map(|i| s.log_price[t + i] - s.log_price[t]).collect();
            let scale = g.nn.band * sigma * (h as f64).sqrt();
            Some((render::future_raster(&cum, scale, g.nn.res)?, s.ts_ms[t + h]))
        }
        BaselineFamily::NnSummary => {
            // the news slot needs a news timeline (an empty one = no news source: sit out)
            if text.is_frames() || text.is_empty() {
                return None;
            }
            let y = crate::summary::target(s, t, g.lookback, text, text_scope(g, s), text_window_ms(g))?;
            Some((y, s.ts_ms[t]))
        }
        BaselineFamily::NnNews | BaselineFamily::NnFrame => {
            if text.is_frames() != (g.family == BaselineFamily::NnFrame) {
                return None;
            }
            let now = *s.ts_ms.get(t)?;
            Some((text.embedding_for(text_scope(g, s), now - text_window_ms(g), now, g.nn.tdim)?, now))
        }
        BaselineFamily::NnNarrative => {
            if text.is_frames() {
                return None;
            }
            let now = *s.ts_ms.get(t)?;
            let end = now + text_window_ms(g);
            // the leading window must be fully observed
            if text.span().is_none_or(|(_, last)| last < end) {
                return None;
            }
            Some((text.embedding_for(text_scope(g, s), now, end, g.nn.tdim)?, end))
        }
        _ => {
            let y = s.forward_return(t, h)?;
            let z = (y / (sigma * (h as f64).sqrt())).clamp(-Z_CLIP, Z_CLIP);
            Some((vec![z as f32], s.ts_ms[t + h]))
        }
    }
}

/// `z`-target of the horizon return (all families; used by retrieval candidates).
pub fn z_at(s: &Series, t: usize, h: usize, sigma: f64) -> Option<f32> {
    let y = s.forward_return(t, h)?;
    Some((y / (sigma * (h as f64).sqrt())).clamp(-Z_CLIP, Z_CLIP) as f32)
}

/// Most common latent length among the instruments of `series` in `ctx` (0 = none); with a
/// `src` gene the source instrument's latents count instead.
pub fn latent_dim(series: &[Series], ctx: &crate::DecodeCtx, source: Option<&str>) -> usize {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for s in series {
        if let Some(v) = ctx.latents.get(&crate::instrument_token(source.unwrap_or(&s.instrument_id))) {
            for p in v {
                *counts.entry(p.vec.len()).or_default() += 1;
            }
        }
    }
    counts.into_iter().filter(|(d, _)| *d > 0).max_by_key(|(d, c)| (*c, *d)).map_or(0, |(d, _)| d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_timeline_windows_and_sentiment() {
        let mut t = TextTimeline::new();
        t.push(2_000, "Stocks rally to record gains", "");
        t.push(1_000, "The the the", ""); // only stopwords: skipped
        t.push(3_000, "Shares plunge on weak earnings", "");
        t.finish();
        assert_eq!(t.len(), 2);
        assert_eq!(t.count(0, 2_000), 1);
        assert!(t.embedding(3_000, 9_000, 64).is_none());
        let up = t.embedding(0, 2_000, 64).unwrap();
        let down = t.embedding(2_000, 3_000, 64).unwrap();
        assert!((up.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-5);
        let dir = sentiment_direction(64);
        let dot = |a: &[f32]| a.iter().zip(&dir).map(|(x, y)| x * y).sum::<f32>();
        assert!(dot(&up) > dot(&down), "{} vs {}", dot(&up), dot(&down));
        t.prune_before(2_500);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn instrument_scoped_news() {
        let mut t = TextTimeline::new();
        t.push_tagged(1_000, "Apple beats estimates", "", &["0190-AAAA".into()]);
        t.push_tagged(2_000, "Oil slumps", "", &["0190bbbb".into()]);
        t.push(3_000, "Markets drift", "");
        t.finish();
        assert_eq!(t.matched(), 2);
        let all = t.embedding_for(None, 0, 3_000, 64).unwrap();
        let a = t.embedding_for(Some("0190aaaa"), 0, 3_000, 64).unwrap();
        assert_eq!(a, t.embedding(0, 1_000, 64).unwrap(), "only the tagged item");
        assert_ne!(a, all);
        assert!(t.embedding_for(Some("0190cccc"), 0, 3_000, 64).is_none());
        let g = DecoderGenome::parse("nn_news:ninst=1;src=src1").unwrap();
        let s = Series { instrument_id: "0190-AAAA".into(), ..Default::default() };
        assert_eq!(text_scope(&g, &s), Some("0190-AAAA"));
        assert_eq!(latent_instrument(&g, &s), "src1");
        assert_eq!(text_scope(&DecoderGenome::parse("nn_news").unwrap(), &s), None);
    }

    #[test]
    fn layout_and_latent_block() {
        let g = DecoderGenome::parse("nn_narrative:lb=8;tdim=32").unwrap();
        let l = Layout::of(&g, 4);
        assert_eq!((l.seq, l.extra, l.latent, l.input(), l.out), (8, 32, 4, 8 + 32 + 4 + 1, 32));
        let c = Layout::of(&DecoderGenome::parse("nn_chart:lb=8;res=6").unwrap(), 4);
        assert_eq!((c.seq, c.input(), c.out), (0, 5, 36));
        let c0 = Layout::of(&DecoderGenome::parse("nn_chart:lb=8;res=6").unwrap(), 0);
        assert_eq!((c0.seq, c0.input()), (8, 8));
        let n = Norm::fit(&[&[1.0, 2.0], &[3.0, 2.0]], 2);
        assert_eq!(n.latent_block(Some(&[3.0, 2.0]), 2), vec![1.0, 0.0, 1.0]);
        assert_eq!(n.latent_block(None, 2), vec![0.0, 0.0, 0.0]);
        assert_eq!(n.latent_block(Some(&[1.0]), 2), vec![0.0, 0.0, 0.0], "wrong dimension = absent");
    }
}
