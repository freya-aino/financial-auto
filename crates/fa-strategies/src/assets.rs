//! Asset-set genes (spec 35 §3, §6.3): which instruments a strategy trades, mutated by
//! add / remove / swap with the MCMC cross-asset pair proposals (spec 33 §6.5,
//! `latent.crossprior.<timescale>`) as the mutation distribution.
//!
//! A genome's `assets` is a sorted set of instrument UUIDs; empty means "every instrument it
//! sees" (the pre-gene behaviour, and what random immigrants start with). Proposals are ranked
//! pairs `(source, target, score)`: a pair whose source leads or predicts its target. Mutation:
//!
//! - **unrestricted → pair**: an empty set becomes a proposed pair, drawn ∝ score;
//! - **add**: a partner of a current member, drawn ∝ the score of the pairs linking it to the
//!   set (uniform over the universe when no pair links it);
//! - **remove**: the member with the least pair support in the set is dropped (ties: random);
//!   removing the last member makes the set unrestricted again;
//! - **swap**: a member is replaced by one of its proposed partners outside the set.
//!
//! Deterministic in the `Rng`.

use std::collections::{BTreeMap, BTreeSet};

use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::genome::StrategyGenome;

/// Ranked cross-asset proposals of one timescale (from `fa.v1.CrossAssetPriors`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AssetProposals {
    /// `(source, target, score > 0)`.
    pub pairs: Vec<(String, String, f64)>,
    /// Extra instruments that may be added without a proposal (uniform fallback).
    pub universe: Vec<String>,
}

/// Maximum asset-set size.
pub const MAX_ASSETS: usize = 8;

/// Which asset-set mutation fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetOp {
    Pair,
    Add,
    Remove,
    Swap,
    /// Nothing applicable (no proposals and no universe).
    None,
}

impl AssetOp {
    pub fn as_str(self) -> &'static str {
        match self {
            AssetOp::Pair => "asset_pair",
            AssetOp::Add => "asset_add",
            AssetOp::Remove => "asset_remove",
            AssetOp::Swap => "asset_swap",
            AssetOp::None => "asset_none",
        }
    }
}

impl AssetProposals {
    /// From a `CrossAssetPriors` message (positive, finite scores only).
    pub fn from_proto(p: &fa_proto::v1::CrossAssetPriors) -> Self {
        Self {
            pairs: p
                .proposals
                .iter()
                .filter(|x| x.score.is_finite() && x.score > 0.0)
                .map(|x| (x.source_instrument_id.clone(), x.target_instrument_id.clone(), x.score))
                .collect(),
            universe: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty() && self.universe.is_empty()
    }

    /// Every instrument named by a pair or the universe.
    pub fn instruments(&self) -> BTreeSet<String> {
        self.pairs.iter().flat_map(|(a, b, _)| [a.clone(), b.clone()]).chain(self.universe.iter().cloned()).collect()
    }

    /// Summed pair score between `x` and each member of `set`.
    fn support(&self, x: &str, set: &BTreeSet<String>) -> f64 {
        self.pairs
            .iter()
            .filter(|(a, b, _)| (a == x && set.contains(b)) || (b == x && set.contains(a)))
            .map(|p| p.2)
            .sum()
    }

    /// Candidates outside `set` linked to it, with their support.
    fn partners(&self, set: &BTreeSet<String>) -> BTreeMap<String, f64> {
        let mut out: BTreeMap<String, f64> = BTreeMap::new();
        for (a, b, s) in &self.pairs {
            if set.contains(a) && !set.contains(b) {
                *out.entry(b.clone()).or_default() += s;
            }
            if set.contains(b) && !set.contains(a) {
                *out.entry(a.clone()).or_default() += s;
            }
        }
        out
    }
}

fn draw<'a>(rng: &mut Rng, items: &'a [(String, f64)]) -> Option<&'a str> {
    let total: f64 = items.iter().map(|x| x.1.max(0.0)).sum();
    if items.is_empty() {
        return None;
    }
    if total <= 0.0 {
        return Some(&items[rng.below(items.len())].0);
    }
    let mut u = rng.uniform() * total;
    for (k, w) in items {
        u -= w.max(0.0);
        if u <= 0.0 {
            return Some(k);
        }
    }
    items.last().map(|x| x.0.as_str())
}

/// Mutates the asset set of `g` in place (module docs); returns which operation fired.
pub fn mutate_assets(g: &mut StrategyGenome, rng: &mut Rng, props: &AssetProposals) -> AssetOp {
    let set: BTreeSet<String> = g.assets.iter().cloned().collect();
    if set.is_empty() {
        let pairs: Vec<(String, f64)> =
            props.pairs.iter().enumerate().map(|(i, p)| (i.to_string(), p.2)).collect();
        let Some(i) = draw(rng, &pairs).and_then(|i| i.parse::<usize>().ok()) else {
            // no proposals: two random instruments of the universe
            let uni: Vec<String> = props.instruments().into_iter().collect();
            if uni.len() < 2 {
                return AssetOp::None;
            }
            let a = uni[rng.below(uni.len())].clone();
            let mut b = uni[rng.below(uni.len())].clone();
            if b == a {
                b = uni[(uni.iter().position(|x| *x == a).unwrap_or(0) + 1) % uni.len()].clone();
            }
            g.set_assets([a, b]);
            return AssetOp::Pair;
        };
        let (a, b, _) = &props.pairs[i];
        g.set_assets([a.clone(), b.clone()]);
        return AssetOp::Pair;
    }
    let partners: Vec<(String, f64)> = props.partners(&set).into_iter().collect();
    let outside: Vec<(String, f64)> =
        props.instruments().into_iter().filter(|x| !set.contains(x)).map(|x| (x, 1.0)).collect();
    let can_add = set.len() < MAX_ASSETS && (!partners.is_empty() || !outside.is_empty());
    let u = rng.uniform();
    if can_add && (u < 0.4 || set.len() == 1) {
        let pick = draw(rng, if partners.is_empty() { &outside } else { &partners }).map(str::to_owned);
        if let Some(x) = pick {
            g.set_assets(set.iter().cloned().chain([x]));
            return AssetOp::Add;
        }
    }
    if u < 0.7 || (partners.is_empty() && outside.is_empty()) {
        // remove the least-supported member
        let mut members: Vec<(String, f64)> = set
            .iter()
            .map(|m| {
                let others: BTreeSet<String> = set.iter().filter(|x| *x != m).cloned().collect();
                (m.clone(), props.support(m, &others))
            })
            .collect();
        members.sort_by(|a, b| a.1.total_cmp(&b.1));
        let low = members[0].1;
        let ties: Vec<&(String, f64)> = members.iter().filter(|m| m.1 <= low + 1e-12).collect();
        let drop = ties[rng.below(ties.len())].0.clone();
        g.set_assets(set.into_iter().filter(|x| *x != drop));
        return AssetOp::Remove;
    }
    // swap: a member for a partner (or any outside instrument)
    let members: Vec<String> = set.iter().cloned().collect();
    let out_m = members[rng.below(members.len())].clone();
    let pool = if partners.is_empty() { &outside } else { &partners };
    match draw(rng, pool).map(str::to_owned) {
        Some(x) => {
            g.set_assets(set.into_iter().filter(|m| *m != out_m).chain([x]));
            AssetOp::Swap
        }
        None => AssetOp::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomSpec, random_genome};
    use fa_core::Timescale;

    fn props() -> AssetProposals {
        AssetProposals {
            pairs: vec![("a".into(), "b".into(), 10.0), ("b".into(), "c".into(), 1.0), ("c".into(), "d".into(), 0.5)],
            universe: vec!["e".into()],
        }
    }

    #[test]
    fn mutations_follow_the_pair_proposals() {
        let mut rng = Rng::seed(1);
        let base = random_genome(&mut rng, Timescale::Minute, &RandomSpec::default());
        assert!(base.assets.is_empty());
        // unrestricted → a proposed pair, mostly the strong one
        let mut strong = 0;
        for _ in 0..200 {
            let mut g = base.clone();
            assert_eq!(mutate_assets(&mut g, &mut rng, &props()), AssetOp::Pair);
            assert_eq!(g.assets.len(), 2);
            strong += usize::from(g.assets == vec!["a", "b"]);
        }
        assert!(strong > 150, "∝ score: {strong}/200");
        // a set grows by partners, shrinks by its least-supported member, swaps, stays bounded
        let mut seen = BTreeSet::new();
        let mut g = base.clone();
        g.set_assets(["a".to_owned(), "b".to_owned()]);
        for _ in 0..300 {
            let mut c = g.clone();
            let op = mutate_assets(&mut c, &mut rng, &props());
            seen.insert(op);
            assert!(c.assets.len() <= MAX_ASSETS);
            assert!(c.assets.windows(2).all(|w| w[0] < w[1]), "sorted, distinct");
            if op == AssetOp::Add {
                assert_eq!(c.assets, vec!["a", "b", "c"], "the only partner of {{a, b}} is c");
            }
            if op == AssetOp::Remove {
                assert_eq!(c.assets.len(), 1);
            }
            assert_ne!(c.id(), g.id());
        }
        assert!(seen.contains(&AssetOp::Add) && seen.contains(&AssetOp::Remove) && seen.contains(&AssetOp::Swap), "{seen:?}");
        // removing the last member makes the set unrestricted again
        let mut one = base.clone();
        one.set_assets(["d".to_owned()]);
        let empty = AssetProposals::default();
        assert_eq!(mutate_assets(&mut one, &mut rng, &empty), AssetOp::Remove);
        assert!(one.assets.is_empty());
        assert_eq!(mutate_assets(&mut one, &mut rng, &empty), AssetOp::None);
        // deterministic
        let (mut x, mut y) = (base.clone(), base.clone());
        mutate_assets(&mut x, &mut Rng::seed(7), &props());
        mutate_assets(&mut y, &mut Rng::seed(7), &props());
        assert_eq!(x.assets, y.assets);
        let proto = fa_proto::v1::CrossAssetPriors {
            proposals: vec![
                fa_proto::v1::CrossAssetPair { source_instrument_id: "a".into(), target_instrument_id: "b".into(), score: 2.0, ..Default::default() },
                fa_proto::v1::CrossAssetPair { source_instrument_id: "a".into(), target_instrument_id: "c".into(), score: -1.0, ..Default::default() },
            ],
            ..Default::default()
        };
        assert_eq!(AssetProposals::from_proto(&proto).pairs.len(), 1);
    }
}
