//! Region remapping on an encoder snapshot change (spec 35 §6.5, §8).
//!
//! A program may name a fixed region with the literal `(region expert ordinal)`; the literal
//! means region `<prefix>-e<expert>-r<nnn>` of the snapshot the strategy is bound to
//! ([`crate::population::StrategyDef::snapshot`]). When model-runtime swaps the encoder it
//! aligns the new regions with the old ones (D-069, `fa_patterns::lineage`) and publishes the
//! result as `RegionLifecycleEvent`s on `latent.lifecycle.<ts>` with `reason = "snapshot"`: every
//! region of the new snapshot lists its ancestors in the parent snapshot (`carry | split |
//! merge | partial` with the Jaccard probe overlap) and every unmatched parent region dies
//! (`reason = "unmatched"`). [`LineageBook`] collects those events; [`remap_def`] rewrites a
//! bound strategy's literals into the new snapshot:
//!
//! | old region's descendants | literal becomes |
//! |---|---|
//! | a `carry` successor | that successor |
//! | `split` / `merge` / `partial` successors only | the successor with the largest overlap (recorded) |
//! | none (died / unmatched / regime break without match) | **unmappable** |
//!
//! A remapped strategy is a different phenotype on a new snapshot: the runtime swaps it in and
//! the risk-engine sends it back to shadow until it is revalidated; an unmappable one is
//! suspended (spec 35 §8 "Snapshot or region lineage breaks").

use std::collections::{BTreeMap, BTreeSet};

use fa_proto::v1::{RegionLifecycleEvent, region_lifecycle_event};
use serde::Serialize;

use crate::{
    Expr, GraphOp, RegionRef, SetRef,
    graph::{literal_suffix, parse_region_id},
    population::StrategyDef,
    sexpr,
};

/// One descendant of an old region in the new snapshot.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Successor {
    pub region_id: String,
    pub overlap: f64,
    /// `carry | split | merge | partial`
    pub kind: String,
}

/// Lineage of one new snapshot back into its parent.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Lineage {
    /// Old region id → its descendants in the new snapshot.
    pub successors: BTreeMap<String, Vec<Successor>>,
    /// New regions seen (any kind, including births without ancestors).
    pub regions: BTreeSet<String>,
    /// Any expert's alignment was a regime break (< 50 % of the probe mass carried).
    pub regime_break: bool,
    pub as_of_ms: i64,
}

/// Snapshot lineages collected from `latent.lifecycle.>`.
#[derive(Debug, Clone, Default)]
pub struct LineageBook {
    /// New snapshot id → lineage into its parent.
    by_snapshot: BTreeMap<String, Lineage>,
    /// Parent snapshot id → its regions that died unmatched at a swap.
    died: BTreeMap<String, BTreeSet<String>>,
    pub events: u64,
}

const MAX_SNAPSHOTS: usize = 16;

impl LineageBook {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingests one lifecycle event; only snapshot-alignment events (`reason = snapshot` for the
    /// seeded regions, `unmatched` deaths) carry cross-snapshot lineage. Returns whether it was used.
    pub fn on_lifecycle(&mut self, ev: &RegionLifecycleEvent) -> bool {
        use region_lifecycle_event::Kind;
        let Some(space) = ev.space.as_ref() else { return false };
        let kind = Kind::try_from(ev.kind).unwrap_or_default();
        let at = ev.as_of.as_ref().map_or(0, |t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000);
        match (kind, ev.reason.as_str()) {
            (Kind::Died, "unmatched") => {
                self.died.entry(space.snapshot_id.clone()).or_default().insert(ev.region_id.clone());
            }
            (Kind::Born | Kind::Carried | Kind::Split | Kind::Merged, "snapshot") => {
                let l = self.by_snapshot.entry(space.snapshot_id.clone()).or_default();
                l.regions.insert(ev.region_id.clone());
                l.regime_break |= ev.regime_break;
                l.as_of_ms = l.as_of_ms.max(at);
                for a in &ev.ancestors {
                    let succ = l.successors.entry(a.region_id.clone()).or_default();
                    if !succ.iter().any(|s| s.region_id == ev.region_id) {
                        succ.push(Successor {
                            region_id: ev.region_id.clone(),
                            overlap: a.overlap,
                            kind: a.kind.clone(),
                        });
                    }
                }
            }
            _ => return false,
        }
        self.events += 1;
        while self.by_snapshot.len() > MAX_SNAPSHOTS {
            let oldest = self.by_snapshot.iter().min_by_key(|(_, l)| l.as_of_ms).map(|(k, _)| k.clone());
            match oldest {
                Some(k) => self.by_snapshot.remove(&k),
                None => break,
            };
        }
        true
    }

    /// Lineage of `snapshot` into its parent, when its swap was observed.
    pub fn lineage(&self, snapshot: &str) -> Option<&Lineage> {
        self.by_snapshot.get(snapshot)
    }

    pub fn snapshots(&self) -> usize {
        self.by_snapshot.len()
    }

    /// Whether `region_id` of `snapshot` died unmatched at a swap.
    pub fn died(&self, snapshot: &str, region_id: &str) -> bool {
        self.died.get(snapshot).is_some_and(|d| d.contains(region_id))
    }
}

/// How one literal was mapped.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiteralMap {
    pub from: (u32, u32),
    pub to: Option<(u32, u32)>,
    pub from_region: String,
    pub to_region: Option<String>,
    /// `carry | split | merge | partial`, or why it is unmappable (`died | no_descendant | bad_id`).
    pub kind: String,
    pub overlap: f64,
}

/// Outcome of remapping one strategy.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RemapReport {
    pub strategy_id: String,
    pub from_snapshot: String,
    pub to_snapshot: String,
    pub literals: Vec<LiteralMap>,
    /// Every literal found a descendant (vacuously true without literals).
    pub ok: bool,
    /// A literal was matched on its `-e<k>-r<nnn>` suffix only (the old region id did not carry
    /// the bound snapshot as prefix, e.g. the in-process baseline's `<ts>-e<k>-r<nnn>` ids).
    pub prefix_unverified: bool,
    pub regime_break: bool,
}

/// Region literals `(expert, ordinal)` used by a program, in pre-order, deduplicated.
pub fn literals(e: &Expr) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut add = |r: RegionRef| -> RegionRef {
        if let RegionRef::Lit(a, b) = r
            && !out.contains(&(a, b))
        {
            out.push((a, b));
        }
        r
    };
    let _ = map_regions(e, &mut add);
    out
}

fn map_set(s: SetRef, f: &mut impl FnMut(RegionRef) -> RegionRef) -> SetRef {
    match s {
        SetRef::Neighbors(r, n) => SetRef::Neighbors(f(r), n),
    }
}

/// Rebuilds `e` with every region reference passed through `f`.
pub fn map_regions(e: &Expr, f: &mut impl FnMut(RegionRef) -> RegionRef) -> Expr {
    use Expr::*;
    let mut m = |x: &Expr| Box::new(map_regions(x, f));
    match e {
        Graph(g) => Graph(match *g {
            GraphOp::PTrans(a, b, k) => {
                let a = f(a);
                GraphOp::PTrans(a, f(b), k)
            }
            GraphOp::PToSet(r, s, k) => {
                let r = f(r);
                GraphOp::PToSet(r, map_set(s, f), k)
            }
            GraphOp::HittingTime(r, s) => {
                let r = f(r);
                GraphOp::HittingTime(r, map_set(s, f))
            }
            GraphOp::Dwell(r) => GraphOp::Dwell(f(r)),
            GraphOp::RegionFeature(r, x) => GraphOp::RegionFeature(f(r), x),
            GraphOp::PathValue(r, k, n, p) => GraphOp::PathValue(f(r), k, n, p),
            other => other,
        }),
        Const(_) | Decoder(_) | Portfolio(_) => e.clone(),
        Add(a, b) => {
            let a = m(a);
            Add(a, m(b))
        }
        Sub(a, b) => {
            let a = m(a);
            Sub(a, m(b))
        }
        Mul(a, b) => {
            let a = m(a);
            Mul(a, m(b))
        }
        Div(a, b) => {
            let a = m(a);
            Div(a, m(b))
        }
        Min(a, b) => {
            let a = m(a);
            Min(a, m(b))
        }
        Max(a, b) => {
            let a = m(a);
            Max(a, m(b))
        }
        Neg(a) => Neg(m(a)),
        Tanh(a) => Tanh(m(a)),
        IfGt(a, b, c, d) => {
            let (a, b, c) = (m(a), m(b), m(c));
            IfGt(a, b, c, m(d))
        }
    }
}

/// Maps one literal of `from` through `lineage`.
fn map_literal(lit: (u32, u32), from: &str, lineage: &Lineage, book: &LineageBook) -> (LiteralMap, bool) {
    let suffix = literal_suffix(lit.0, lit.1);
    let exact = format!("{from}{suffix}");
    let (old, succ, unverified) = match lineage.successors.get(&exact) {
        Some(s) => (exact.clone(), Some(s), false),
        None => match lineage.successors.iter().find(|(id, _)| id.ends_with(&suffix)) {
            Some((id, s)) => (id.clone(), Some(s), true),
            None => (exact.clone(), None, false),
        },
    };
    let none = |kind: &str| LiteralMap {
        from: lit,
        to: None,
        from_region: old.clone(),
        to_region: None,
        kind: kind.into(),
        overlap: 0.0,
    };
    let Some(succ) = succ.filter(|s| !s.is_empty()) else {
        let why = if book.died(from, &old) { "died" } else { "no_descendant" };
        return (none(why), unverified);
    };
    let best = succ.iter().find(|s| s.kind == "carry").or_else(|| {
        succ.iter().max_by(|a, b| a.overlap.total_cmp(&b.overlap).then_with(|| b.region_id.cmp(&a.region_id)))
    });
    let Some(best) = best else { return (none("no_descendant"), unverified) };
    match parse_region_id(&best.region_id) {
        Some(to) => (
            LiteralMap {
                from: lit,
                to: Some(to),
                from_region: old,
                to_region: Some(best.region_id.clone()),
                kind: best.kind.clone(),
                overlap: best.overlap,
            },
            unverified,
        ),
        None => (none("bad_id"), unverified),
    }
}

/// Remaps a strategy bound to `def.snapshot` onto `to_snapshot`. Returns the rewritten
/// definition (programs with mapped literals, `snapshot = to_snapshot`) when every literal
/// mapped, plus the report either way. `None` definition = unmappable (or unbound).
pub fn remap_def(
    def: &StrategyDef,
    to_snapshot: &str,
    book: &LineageBook,
) -> (Option<StrategyDef>, Option<RemapReport>) {
    let Some(from) = def.snapshot.as_deref() else { return (None, None) };
    let (Ok(signal), Ok(uncertainty)) = (sexpr::parse(&def.signal), sexpr::parse(&def.uncertainty)) else {
        return (None, None);
    };
    let mut report = RemapReport {
        strategy_id: def.id.clone(),
        from_snapshot: from.to_owned(),
        to_snapshot: to_snapshot.to_owned(),
        literals: Vec::new(),
        ok: true,
        prefix_unverified: false,
        regime_break: false,
    };
    let mut lits = literals(&signal);
    for l in literals(&uncertainty) {
        if !lits.contains(&l) {
            lits.push(l);
        }
    }
    let lineage = book.lineage(to_snapshot);
    if !lits.is_empty() && lineage.is_none() {
        // the swap's lineage was never observed: nothing can be mapped
        report.ok = false;
        report.literals = lits
            .iter()
            .map(|l| LiteralMap {
                from: *l,
                to: None,
                from_region: format!("{from}{}", literal_suffix(l.0, l.1)),
                to_region: None,
                kind: "no_lineage".into(),
                overlap: 0.0,
            })
            .collect();
        return (None, Some(report));
    }
    let mut table: BTreeMap<(u32, u32), (u32, u32)> = BTreeMap::new();
    if let Some(lineage) = lineage {
        report.regime_break = lineage.regime_break;
        for l in lits {
            let (m, unverified) = map_literal(l, from, lineage, book);
            report.prefix_unverified |= unverified;
            match m.to {
                Some(to) => {
                    table.insert(l, to);
                }
                None => report.ok = false,
            }
            report.literals.push(m);
        }
    }
    if !report.ok {
        return (None, Some(report));
    }
    let mut f = |r: RegionRef| match r {
        RegionRef::Lit(e, k) => table.get(&(e, k)).map_or(r, |(e2, k2)| RegionRef::Lit(*e2, *k2)),
        other => other,
    };
    let signal = map_regions(&signal, &mut f);
    let uncertainty = map_regions(&uncertainty, &mut f);
    let out = StrategyDef {
        signal: signal.to_string(),
        uncertainty: uncertainty.to_string(),
        snapshot: Some(to_snapshot.to_owned()),
        ..def.clone()
    };
    (Some(out), Some(report))
}

#[cfg(test)]
mod tests {
    use fa_core::Timescale;
    use fa_proto::v1::{ExpertSpaceRef, RegionAncestor};

    use super::*;

    fn ev(
        kind: region_lifecycle_event::Kind,
        snap: &str,
        id: &str,
        reason: &str,
        anc: &[(&str, f64, &str)],
    ) -> RegionLifecycleEvent {
        RegionLifecycleEvent {
            kind: kind as i32,
            space: Some(ExpertSpaceRef { snapshot_id: snap.into(), timescale: "1m".into(), expert_index: 0 }),
            region_id: id.into(),
            reason: reason.into(),
            ancestors: anc
                .iter()
                .map(|(r, o, k)| RegionAncestor {
                    region_id: (*r).into(),
                    lineage_id: String::new(),
                    overlap: *o,
                    kind: (*k).into(),
                })
                .collect(),
            ..Default::default()
        }
    }

    fn book() -> LineageBook {
        use region_lifecycle_event::Kind as K;
        let mut b = LineageBook::new();
        for e in [
            ev(K::Carried, "s2", "s2-e0-r000", "snapshot", &[("s1-e0-r002", 0.8, "carry")]),
            ev(K::Split, "s2", "s2-e0-r001", "snapshot", &[("s1-e0-r001", 0.4, "split")]),
            ev(K::Split, "s2", "s2-e0-r002", "snapshot", &[("s1-e0-r001", 0.3, "split")]),
            ev(
                K::Merged,
                "s2",
                "s2-e1-r000",
                "snapshot",
                &[("s1-e1-r000", 0.5, "merge"), ("s1-e1-r001", 0.35, "merge")],
            ),
            ev(K::Born, "s2", "s2-e1-r001", "snapshot", &[]),
            ev(K::Died, "s1", "s1-e0-r003", "unmatched", &[]),
            // online events of the new snapshot are not lineage of the swap
            ev(K::Split, "s2", "s2-e0-r009", "online", &[("s2-e0-r000", 0.5, "split")]),
        ] {
            b.on_lifecycle(&e);
        }
        b
    }

    fn def(signal: &str, snapshot: Option<&str>) -> StrategyDef {
        StrategyDef {
            id: "gp-x".into(),
            genome_id: "gp-x".into(),
            timescale: Timescale::Minute,
            steps: 3,
            lookback_bars: 16,
            step_secs: None,
            signal: signal.into(),
            uncertainty: "region_entropy".into(),
            snapshot: snapshot.map(Into::into),
        }
    }

    #[test]
    fn carry_split_and_merge_map_and_programs_are_rewritten() {
        let b = book();
        assert_eq!(b.events, 6, "the online split is ignored");
        let d = def(
            "(add (p_trans (cur_region 0) (region 0 2) 3) (mul (dwell (region 0 1)) (region_feature (region 1 1) mass)))",
            Some("s1"),
        );
        let (out, rep) = remap_def(&d, "s2", &b);
        let rep = rep.unwrap();
        assert!(rep.ok, "{rep:?}");
        assert!(!rep.prefix_unverified);
        let out = out.unwrap();
        assert_eq!(
            out.signal,
            "(add (p_trans (cur_region 0) (region 0 0) 3) (mul (dwell (region 0 1)) (region_feature (region 1 0) mass)))"
        );
        assert_eq!(out.snapshot.as_deref(), Some("s2"));
        let kinds: Vec<(&str, f64)> = rep.literals.iter().map(|l| (l.kind.as_str(), l.overlap)).collect();
        // carry; split → the larger child (0.4); merge → the survivor
        assert_eq!(kinds, [("carry", 0.8), ("split", 0.4), ("merge", 0.35)]);
        assert!(out.build().is_ok());
    }

    #[test]
    fn died_or_missing_lineage_is_unmappable_and_unbound_is_untouched() {
        let b = book();
        let (out, rep) = remap_def(&def("(dwell (region 0 3))", Some("s1")), "s2", &b);
        let rep = rep.unwrap();
        assert!(out.is_none() && !rep.ok);
        assert_eq!(rep.literals[0].kind, "died");
        let (out, rep) = remap_def(&def("(dwell (region 0 7))", Some("s1")), "s2", &b);
        assert!(out.is_none());
        assert_eq!(rep.unwrap().literals[0].kind, "no_descendant");
        // the lineage of s3 was never observed
        let (out, rep) = remap_def(&def("(dwell (region 0 2))", Some("s1")), "s3", &b);
        assert!(out.is_none());
        assert_eq!(rep.unwrap().literals[0].kind, "no_lineage");
        // no literals: rebinds trivially even without lineage
        let (out, rep) = remap_def(&def("(dwell (cur_region 0))", Some("s1")), "s3", &b);
        assert_eq!(out.unwrap().snapshot.as_deref(), Some("s3"));
        assert!(rep.unwrap().ok);
        // unbound strategies are never remapped
        assert!(matches!(remap_def(&def("(dwell (region 0 2))", None), "s2", &b), (None, None)));
    }

    #[test]
    fn suffix_match_for_unprefixed_ids_is_flagged() {
        use region_lifecycle_event::Kind as K;
        let mut b = LineageBook::new();
        b.on_lifecycle(&ev(K::Carried, "s2", "s2-e0-r005", "snapshot", &[("1m-e0-r002", 0.9, "carry")]));
        let (out, rep) = remap_def(&def("(dwell (region 0 2))", Some("stat-v1-1m")), "s2", &b);
        assert_eq!(out.unwrap().signal, "(dwell (region 0 5))");
        assert!(rep.unwrap().prefix_unverified);
    }

    #[test]
    fn literals_are_collected_once_in_preorder() {
        let e =
            sexpr::parse("(add (dwell (region 1 4)) (p_to_set (region 0 2) (neighbors (region 1 4) 3) 2))").unwrap();
        assert_eq!(literals(&e), [(1, 4), (0, 2)]);
        let back = map_regions(&e, &mut |r| r);
        assert_eq!(back, e);
    }
}
