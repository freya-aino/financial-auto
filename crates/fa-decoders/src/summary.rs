//! Event-summary targets of the `text.summary` family (spec 34 §6.1 "event summary", Q2).
//!
//! **Where the ground truth comes from (Q2, D-085):** there are no human-written summaries, so
//! the target is a **templated summary generated from the window itself** — the "generated
//! templates from market stats" option of §6.1, extended with the co-occurring news flow. A
//! window `(t − lb, t]` (and the news of `(t − tw, t]`) is described by five slots, each a small
//! categorical choice of phrase; the decoder predicts one distribution per slot and is scored
//! with the mean per-slot cross-entropy (the reconstruction baseline is the training-mean
//! distribution, like the rasters). The argmax phrases form a sentence ([`summary_text`]), so a
//! decoded latent reads as text:
//!
//! | slot | classes | statistic |
//! |---|---|---|
//! | move | fell sharply · fell · was flat · rose · rose sharply | `z = Σr / (σ·√lb)` at ±0.5 / ±1.5 |
//! | volatility | calmed · held steady · turned volatile | σ of the second half / σ of the first half at 0.7 / 1.4 |
//! | close | near the low · mid-range · near the high | close in the window's range at ⅓ / ⅔ |
//! | path | choppy · two-way · one-way | efficiency `|Σr| / Σ|r|` at 0.2 / 0.5 |
//! | news | no news · negative news · neutral news · positive news | lexicon tone of the window's news embedding at ±[`TONE_BAND`] |
//!
//! Everything is known at `ts[t]` (the window and the news up to `t`), so the target is a
//! reconstruction of the present, as the chart raster is.

use crate::{
    baseline::Series,
    neural::data::{TextTimeline, sentiment_direction},
};

/// Phrases per slot, in class order.
pub const SLOTS: [&[&str]; 5] = [
    &["fell sharply", "fell", "was flat", "rose", "rose sharply"],
    &["calmed", "held steady", "turned volatile"],
    &["near the low", "mid-range", "near the high"],
    &["choppy", "two-way", "one-way"],
    &["no news", "negative news", "neutral news", "positive news"],
];
/// Slot names (published as component labels).
pub const SLOT_NAMES: [&str; 5] = ["move", "volatility", "close", "path", "news"];
/// Values of the move classes (the served score is their expectation, in [-1, 1]).
pub const MOVE_VALUES: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];
/// News-tone dead band of `dot(embedding, sentiment direction)`.
pub const TONE_BAND: f64 = 0.01;
/// Folded dimension the tone is measured in (wide, so the ~200 lexicon words rarely collide;
/// independent of the genome's `tdim`).
pub const TONE_DIM: usize = 4096;

/// Class counts of the slots.
pub fn slot_sizes() -> Vec<usize> {
    SLOTS.iter().map(|s| s.len()).collect()
}

/// Width of a summary target (sum of the class counts).
pub fn width() -> usize {
    slot_sizes().iter().sum()
}

/// Class index of every slot for the window ending at `t` (`None` when `t < lb` or `lb < 4`).
/// `news` = the [`TONE_DIM`]-folded embedding of the news in `(t − tw, t]` (`None` = no item).
pub fn classes(s: &Series, t: usize, lb: usize, news: Option<&[f32]>) -> Option<[usize; 5]> {
    if lb < 4 || t < lb || t >= s.len() {
        return None;
    }
    let r: Vec<f64> = (t + 1 - lb..=t).map(|i| s.log_price[i] - s.log_price[i - 1]).collect();
    let n = r.len() as f64;
    let mean = r.iter().sum::<f64>() / n;
    let sd = (r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt().max(1e-9);
    let net: f64 = r.iter().sum();
    let z = net / (sd * n.sqrt());
    let mv = if z < -1.5 {
        0
    } else if z < -0.5 {
        1
    } else if z <= 0.5 {
        2
    } else if z <= 1.5 {
        3
    } else {
        4
    };
    let half = r.len() / 2;
    let sd_of = |x: &[f64]| {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
    };
    let (a, b) = (sd_of(&r[..half]), sd_of(&r[half..]));
    let ratio = if a > 1e-12 {
        b / a
    } else if b > 1e-12 {
        f64::INFINITY
    } else {
        1.0
    };
    let vol = if ratio < 0.7 {
        0
    } else if ratio <= 1.4 {
        1
    } else {
        2
    };
    let w = &s.log_price[t + 1 - lb..=t];
    let (lo, hi) = w.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), x| (l.min(*x), h.max(*x)));
    let pos = if hi > lo { (s.log_price[t] - lo) / (hi - lo) } else { 0.5 };
    let close = if pos < 1.0 / 3.0 {
        0
    } else if pos <= 2.0 / 3.0 {
        1
    } else {
        2
    };
    let gross: f64 = r.iter().map(|x| x.abs()).sum();
    let eff = if gross > 0.0 { net.abs() / gross } else { 0.0 };
    let path = if eff < 0.2 {
        0
    } else if eff < 0.5 {
        1
    } else {
        2
    };
    let tone = match news {
        None => 0,
        Some(e) => {
            let dir = sentiment_direction(TONE_DIM);
            let d: f64 = e.iter().zip(&dir).map(|(a, b)| f64::from(a * b)).sum();
            if d < -TONE_BAND {
                1
            } else if d > TONE_BAND {
                3
            } else {
                2
            }
        }
    };
    Some([mv, vol, close, path, tone])
}

/// One-hot target (concatenated slots) of the window ending at `t`.
pub fn target(s: &Series, t: usize, lb: usize, text: &TextTimeline, scope: Option<&str>, tw_ms: i64) -> Option<Vec<f32>> {
    let now = *s.ts_ms.get(t)?;
    let news = text.embedding_for(scope, now - tw_ms, now, TONE_DIM);
    let c = classes(s, t, lb, news.as_deref())?;
    let mut y = vec![0.0f32; width()];
    let mut at = 0;
    for (slot, k) in c.iter().enumerate() {
        y[at + k] = 1.0;
        at += SLOTS[slot].len();
    }
    Some(y)
}

/// Per-slot argmax of a prediction (concatenated slot distributions).
pub fn argmax(p: &[f32]) -> Vec<usize> {
    let mut out = Vec::with_capacity(SLOTS.len());
    let mut at = 0;
    for s in SLOTS {
        let seg = &p[at..at + s.len()];
        let best = seg.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        out.push(best);
        at += s.len();
    }
    out
}

/// Expected event intensity `Σ p_k·|v_k|` of the move slot in [0, 1] — the served
/// `text.summary` score, oriented like the baseline's vocabulary novelty (0 = nothing happened),
/// so the neural and baseline outputs of the family share one scale.
pub fn intensity_score(p: &[f32]) -> f64 {
    p.iter().take(MOVE_VALUES.len()).zip(MOVE_VALUES).map(|(q, v)| f64::from(*q) * v.abs()).sum::<f64>().clamp(0.0, 1.0)
}

/// Expected signed move in [-1, 1] (published as a component).
pub fn move_score(p: &[f32]) -> f64 {
    p.iter().take(MOVE_VALUES.len()).zip(MOVE_VALUES).map(|(q, v)| f64::from(*q) * v).sum::<f64>().clamp(-1.0, 1.0)
}

/// The summary sentence of a prediction (or a one-hot target): "rose sharply and turned
/// volatile, closing near the high on one-way trade; positive news".
pub fn summary_text(p: &[f32]) -> String {
    let c = argmax(p);
    format!(
        "{} and {}, closing {} on {} trade; {}",
        SLOTS[0][c[0]], SLOTS[1][c[1]], SLOTS[2][c[2]], SLOTS[3][c[3]], SLOTS[4][c[4]]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(rets: &[f64]) -> Series {
        let mut lp = vec![0.0];
        for r in rets {
            lp.push(lp.last().unwrap() + r);
        }
        Series {
            instrument_id: "x".into(),
            ts_ms: (0..lp.len() as i64).map(|i| i * 1000).collect(),
            log_price: lp,
            step_ms: 1000,
        }
    }

    #[test]
    fn slots_describe_shapes() {
        // a steady climb: rose sharply, one-way, near the high
        let up: Vec<f64> = (0..16).map(|i| 0.01 + 0.001 * f64::from(i % 3)).collect();
        let s = series(&up);
        let c = classes(&s, 16, 16, None).unwrap();
        assert_eq!((c[0], c[2], c[3], c[4]), (4, 2, 2, 0), "{c:?}");
        // a calm first half then a volatile, flat second half
        let mut r: Vec<f64> = (0..8).map(|i| if i % 2 == 0 { 0.001 } else { -0.001 }).collect();
        r.extend((0..8).map(|i| if i % 2 == 0 { 0.02 } else { -0.02 }));
        let c = classes(&series(&r), 16, 16, None).unwrap();
        assert_eq!((c[0], c[1], c[3]), (2, 2, 0), "{c:?}");
        // the mirror image of the climb falls sharply, closes near the low
        let down: Vec<f64> = up.iter().map(|x| -x).collect();
        let c = classes(&series(&down), 16, 16, None).unwrap();
        assert_eq!((c[0], c[2]), (0, 0));
        assert!(classes(&s, 3, 16, None).is_none());
    }

    #[test]
    fn target_text_and_score() {
        let up: Vec<f64> = (0..16).map(|_| 0.01).collect();
        let s = series(&up);
        let mut text = TextTimeline::new();
        text.push(15_000, "Stocks rally to record gains on strong surge", "");
        text.finish();
        let y = target(&s, 16, 16, &text, None, 5_000).unwrap();
        assert_eq!(y.len(), width());
        assert_eq!(y.iter().sum::<f32>(), 5.0);
        let sentence = summary_text(&y);
        assert!(sentence.starts_with("rose sharply"), "{sentence}");
        assert!(sentence.ends_with("positive news"), "{sentence}");
        assert_eq!(move_score(&y), 1.0);
        assert_eq!(intensity_score(&y), 1.0);
        // without news in the window the news slot says so
        let y = target(&s, 16, 16, &text, None, 500).unwrap();
        assert!(summary_text(&y).ends_with("no news"));
    }
}
