//! The text expert of the statistical family (D-119): eight bounded features from the news
//! embeddings of a window.
//!
//! The embeddings themselves come from `llm-suite` (D-117), which embeds every news item once on
//! arrival and stores it with its knowledge time; the encoder is handed the items whose knowledge
//! time falls inside the window it is encoding ([`crate::EncoderInput::media`], `Modality::Text`).
//! Nothing here calls an API, which is what keeps a re-encode of the same window reproducible.
//!
//! Why these eight features. A window's news is a *set* of vectors, and three things about that set
//! matter to anything downstream: **how much** news there was, **how much it agreed with itself**,
//! and **what it was about**. The first two are one feature each; the remaining six carry content
//! as projections of the window's mean embedding onto fixed directions. That is deliberate: a
//! decoder reconstructing a news embedding can only beat its training mean if the latent carries
//! some of the embedding's own geometry, and six projections is what fits in `EXPERT_DIM` next to
//! the two summary features. The alternative — summary statistics only — is what the latent had
//! before, and the live text decoders sat exactly at naive on it.
//!
//! The projection directions are pseudo-random but **fixed**: derived from the embedding width with
//! a constant seed, so every process, replay and re-fit uses the same basis. A random basis is
//! enough because the only requirement is that distances survive the projection (Johnson–
//! Lindenstrauss), and a fixed one costs nothing to store.
use crate::baseline::{EXPERT_DIM, squash};

/// Projections of the mean embedding carried in the expert's dims (the rest are the two summary
/// features).
pub const PROJECTIONS: usize = EXPERT_DIM - 2;

/// Seed of the projection basis. Changing it changes every text latent, so it is versioned with
/// the encoder (`StatisticalEncoder::VERSION`) rather than tuned.
const BASIS_SEED: u64 = 0x7ADE_C0DE_0000_0001;

/// Most embeddings read for one window. A window with more news than this uses the newest, which
/// is the bound that keeps a quiet instrument and a news storm the same cost to encode.
pub const MAX_ITEMS: usize = 256;

/// One fixed pseudo-random unit direction of `dim`, for projection `index`.
///
/// SplitMix64 over `(seed, index, position)` — a counter-based generator, so a direction is
/// computed from its coordinates rather than from an iteration order, and the basis is identical
/// on every platform and in every process.
fn direction(dim: usize, index: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut z = BASIS_SEED
            .wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Uniform in [-1, 1): the sum of dim such coordinates is near-Gaussian, which is all a
        // random projection needs.
        v.push(((z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// `PROJECTIONS` unit directions of one width, shared by every encode at that width.
type Basis = std::sync::Arc<Vec<Vec<f32>>>;

/// The projection basis for `dim`, cached per width.
///
/// Cached because a basis of 1 536 × 6 floats is rebuilt on every encode otherwise, and the
/// encoder runs per instrument per step.
fn basis(dim: usize) -> Basis {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<Vec<(usize, Basis)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, b)) = guard.iter().find(|(d, _)| *d == dim) {
        return b.clone();
    }
    let built: Basis = std::sync::Arc::new((0..PROJECTIONS).map(|i| direction(dim, i)).collect::<Vec<_>>());
    guard.push((dim, built.clone()));
    built
}

/// The text expert's features for one window: `[volume, coherence, p0 … p5]`, each in (−1, 1).
///
/// `embeddings` are the window's news vectors, all of the same width; vectors of another width are
/// skipped rather than mixed in, because they come from another model and share no geometry.
/// An empty window is all zeros — "no news", not "news saying nothing" — and the caller pairs that
/// with a wide posterior through [`log_var`].
pub fn features(embeddings: &[Vec<f32>]) -> Vec<f32> {
    let mut out = vec![0f32; EXPERT_DIM];
    // The width to use is the most common one, not the largest: a stray vector from another model
    // (a leftover of an endpoint change, or a hashed fallback of a different width) must be the
    // one that is dropped, not the window.
    let Some(dim) = modal_width(embeddings) else { return out };
    let used: Vec<&Vec<f32>> = embeddings.iter().filter(|v| v.len() == dim).take(MAX_ITEMS).collect();
    let k = used.len();
    if k == 0 {
        return out;
    }

    // How much news: monotone in the count, saturating, so one item and two items differ more than
    // fifty and fifty-one do.
    out[0] = squash((k as f64).ln() / 2.0) as f32;

    // Mean vector and the norms the two summary features need.
    let mut sum = vec![0f64; dim];
    let mut norms = Vec::with_capacity(k);
    for v in &used {
        let n = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
        norms.push(n);
        // Normalise per item, so a provider that does not return unit vectors cannot let one long
        // vector dominate the window's mean.
        let scale = if n > 0.0 { 1.0 / n } else { 0.0 };
        for (s, x) in sum.iter_mut().zip(v.iter()) {
            *s += f64::from(*x) * scale;
        }
    }
    let mean_norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt() / k as f64;
    // Coherence: the mean pairwise cosine of unit vectors is ‖Σv‖²/k² rescaled off the diagonal.
    // 1 = every item points the same way, 0 = spread, so it maps to (−1, 1) directly.
    out[1] = if k > 1 {
        let s2 = sum.iter().map(|x| x * x).sum::<f64>();
        (((s2 - k as f64) / (k as f64 * (k as f64 - 1.0))).clamp(-1.0, 1.0)) as f32
    } else {
        1.0 // a single item agrees with itself
    };

    // Content: the mean direction projected onto the fixed basis. Projecting the *direction*
    // (not the sum) keeps the features independent of how much news arrived, which `out[0]`
    // already carries.
    if mean_norm > 0.0 {
        let b = basis(dim);
        let inv = 1.0 / (mean_norm * k as f64);
        // A unit vector projected on a random unit direction has E[(v·d)²] = 1/dim, so the raw
        // dot product in 1 536 dimensions is ~0.026 — an order of magnitude below everything else
        // in the latent, which region discovery and the decoders would read as "no signal". √dim
        // rescales it to unit variance, putting the text features on the same footing as the
        // market experts' z-scores; `squash` then bounds them like every other feature.
        let scale = (dim as f64).sqrt();
        for (j, d) in b.iter().enumerate() {
            let dot: f64 = d.iter().zip(sum.iter()).map(|(w, s)| f64::from(*w) * *s * inv).sum();
            out[2 + j] = squash(dot * scale) as f32;
        }
    }
    out
}

/// The width shared by most of `embeddings`, ties going to the one seen first; `None` when they
/// are all empty.
fn modal_width(embeddings: &[Vec<f32>]) -> Option<usize> {
    let mut counts: Vec<(usize, usize)> = Vec::new();
    for v in embeddings.iter().filter(|v| !v.is_empty()) {
        match counts.iter_mut().find(|(d, _)| *d == v.len()) {
            Some((_, n)) => *n += 1,
            None => counts.push((v.len(), 1)),
        }
    }
    // `max_by_key` would return the *last* maximum on a tie; the first-seen width wins instead,
    // so one stray vector never redefines a window of one.
    counts.into_iter().reduce(|best, c| if c.1 > best.1 { c } else { best }).map(|(d, _)| d)
}

/// Posterior log-variance of the text expert for `k` items in the window.
///
/// `k = 0` → 0, i.e. unit variance: the expert says nothing and says so. Otherwise the same
/// `−ln(n)` shape the market experts use, so "two headlines" is visibly less certain than "forty".
pub fn log_var(k: usize) -> f32 {
    if k == 0 { 0.0 } else { -((k as f32).ln() + 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dim: usize, seed: u64) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim)
            .map(|i| {
                let mut z = seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                z = (z ^ (z >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                ((z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
            })
            .collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= n;
        }
        v
    }

    #[test]
    fn an_empty_window_is_zeros_with_unit_variance() {
        let f = features(&[]);
        assert_eq!(f.len(), EXPERT_DIM);
        assert!(f.iter().all(|x| *x == 0.0), "no news is zeros, not noise: {f:?}");
        assert_eq!(log_var(0), 0.0, "and it says it knows nothing");
        assert!(log_var(1) < 0.0);
        assert!(log_var(40) < log_var(2), "more items, tighter posterior");
    }

    #[test]
    fn features_are_deterministic_and_bounded() {
        let a = vec![unit(64, 1), unit(64, 2), unit(64, 3)];
        let f1 = features(&a);
        let f2 = features(&a);
        assert_eq!(f1, f2, "a replay of the same window must give the same latent");
        assert!(f1.iter().all(|x| x.is_finite() && x.abs() <= 1.0), "{f1:?}");
        assert_eq!(f1.len(), EXPERT_DIM);
    }

    #[test]
    fn the_basis_is_fixed_for_a_width_and_orthogonal_enough_to_carry_content() {
        let b1 = basis(1536);
        let b2 = basis(1536);
        assert_eq!(*b1, *b2, "the basis is the same object for a width");
        assert_eq!(b1.len(), PROJECTIONS);
        for d in b1.iter() {
            let n = d.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-5, "directions are unit vectors: {n}");
        }
        // Random directions in 1 536 dimensions are nearly orthogonal; anything above ~0.1 would
        // mean the projections carry the same number twice.
        for i in 0..b1.len() {
            for j in i + 1..b1.len() {
                let dot: f32 = b1[i].iter().zip(b1[j].iter()).map(|(a, b)| a * b).sum();
                assert!(dot.abs() < 0.1, "directions {i} and {j} overlap: {dot}");
            }
        }
    }

    #[test]
    fn volume_rises_with_the_count_and_saturates() {
        let one = features(&[unit(32, 7)]);
        let few = features(&(0..5).map(|i| unit(32, i)).collect::<Vec<_>>());
        let many = features(&(0..50).map(|i| unit(32, i)).collect::<Vec<_>>());
        assert!(one[0] < few[0] && few[0] < many[0], "{} {} {}", one[0], few[0], many[0]);
        assert!(many[0] < 1.0, "bounded");
        let more = features(&(0..51).map(|i| unit(32, i)).collect::<Vec<_>>());
        assert!((more[0] - many[0]).abs() < (few[0] - one[0]).abs(), "the scale saturates");
    }

    #[test]
    fn coherence_separates_agreement_from_spread() {
        let v = unit(64, 11);
        let same = features(&[v.clone(), v.clone(), v.clone()]);
        let spread = features(&(0..3).map(|i| unit(64, 100 + i)).collect::<Vec<_>>());
        assert!((same[1] - 1.0).abs() < 1e-5, "identical items are fully coherent: {}", same[1]);
        assert!(spread[1].abs() < 0.5, "unrelated items are not: {}", spread[1]);
        let opposite = features(&[v.clone(), v.iter().map(|x| -x).collect()]);
        assert!(opposite[1] < -0.9, "opposed items are negatively coherent: {}", opposite[1]);
    }

    #[test]
    fn content_reaches_the_latent() {
        // Two different windows must differ in the projection dims, or a decoder has nothing to
        // reconstruct — which is exactly the state the text decoders were stuck in before.
        let a = features(&[unit(128, 21)]);
        let b = features(&[unit(128, 22)]);
        let differing = (2..EXPERT_DIM).filter(|i| (a[*i] - b[*i]).abs() > 1e-6).count();
        assert!(differing >= PROJECTIONS - 1, "only {differing} of {PROJECTIONS} projections differ");
        // The same content gives the same projections regardless of how many copies arrived.
        let v = unit(128, 33);
        let one = features(std::slice::from_ref(&v));
        let three = features(&[v.clone(), v.clone(), v.clone()]);
        for i in 2..EXPERT_DIM {
            assert!((one[i] - three[i]).abs() < 1e-5, "projection {i} moved with the count");
        }
    }

    #[test]
    fn projections_are_on_the_same_scale_as_the_other_experts() {
        // Without the √dim rescaling these come out at ~0.02 in 1 536 dimensions, which reads as
        // "no signal" next to features built from z-scores. Across many windows the projections
        // should span a usable range instead.
        let mut max_abs = 0f32;
        for seed in 0..64u64 {
            let f = features(std::slice::from_ref(&unit(1536, seed)));
            for x in &f[2..] {
                max_abs = max_abs.max(x.abs());
                assert!(x.abs() <= 1.0, "still bounded: {x}");
            }
        }
        assert!(max_abs > 0.2, "projections stay near zero ({max_abs}): the text expert carries nothing");
    }

    #[test]
    fn a_vector_of_another_width_is_skipped_not_mixed() {
        let good = unit(64, 5);
        let other_model = unit(1536, 5);
        let mixed = features(&[good.clone(), good.clone(), other_model]);
        let clean = features(&[good.clone(), good.clone()]);
        assert_eq!(mixed, clean, "a vector from another model shares no geometry and must be dropped");
        // A tie goes to the width seen first, so a single stray vector never redefines the window.
        assert_eq!(features(&[good.clone(), unit(1536, 9)]), features(std::slice::from_ref(&good)));
        // Unnormalised input cannot dominate: each item is scaled to unit length first, so a
        // ten-times-longer copy of the same text contributes the same direction. Compared
        // approximately because scaling and re-normalising in f32 is not bit-identical.
        let long: Vec<f32> = good.iter().map(|x| x * 10.0).collect();
        let scaled = features(&[good.clone(), long]);
        let plain = features(&[good.clone(), good.clone()]);
        for (a, b) in scaled.iter().zip(&plain) {
            assert!((a - b).abs() < 1e-6, "{scaled:?} vs {plain:?}");
        }
    }

    #[test]
    fn a_news_storm_is_bounded_work() {
        let many: Vec<Vec<f32>> = (0..MAX_ITEMS + 500).map(|i| unit(64, i as u64)).collect();
        let f = features(&many);
        assert!(f.iter().all(|x| x.is_finite()));
        // The newest MAX_ITEMS are used; the rest cost nothing.
        assert_eq!(features(&many[..MAX_ITEMS]), f);
    }
}
