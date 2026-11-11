//! Garbage collection of non-elite genome weights (spec 20 §6.5: "weights of non-elite genomes
//! GC'd after N generations").
//!
//! **Eligible objects** are decoder weight bundles only:
//! `fa-models/decoders/<generation>/<family>-<hash>-<job>.safetensors` ([`is_weight_key`]).
//! Cards (`*.card.json`, the lineage/fitness record), snapshots, adapters (LoRA weights belong to
//! their snapshot), checkpoints, eval reports and live sets are never touched — [`plan`] refuses
//! every other key, whatever the caller passes in.
//!
//! **Generations** are the `<generation>` path segment (the evo generation's job id, a ULID, or
//! `adhoc`). They are ordered by the newest `last_modified` of their objects (ties by name), so the
//! rule does not depend on how generations are named; the newest [`GcPolicy::keep_generations`]
//! generations are kept whole.
//!
//! **Protected** (never deleted, whatever their age):
//! - every key in [`Protection::keys`] — the caller fills it from the live serving state
//!   (serving / shadow / pinned decoders, `GET /api/decoders/serving`) and from every snapshot
//!   manifest that references a bundle (the D-009 feedback set, `manifest.feedback`), see
//!   [`collect_bundle_refs`];
//! - the card elites: per `(output_family, step_ms)` the top [`GcPolicy::elite_per_family`] cards
//!   by `meta.score` among those with `score > 0` (what model-runtime ranks into serving slots and
//!   what `feedback::select_top_k` draws from), computed by [`card_elites`];
//! - anything younger than [`GcPolicy::min_age`] (a guard against clock / ordering surprises and
//!   in-flight writes).
//!
//! A bundle without a card is unloadable (model-runtime and the trainer discover bundles through
//! cards) and is collected like any other non-elite once its generation is old enough.
use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Duration, Utc};
use serde::Deserialize;

use crate::{BlobError, BlobStore, ObjectKey, ObjectMeta, buckets};

/// Key prefix of decoder bundles and cards in `fa-models`.
pub const DECODER_PREFIX: &str = "decoders/";
/// Suffix of a weight bundle.
pub const WEIGHT_SUFFIX: &str = ".safetensors";
/// Suffix of the JSON card next to a bundle.
pub const CARD_SUFFIX: &str = ".card.json";

/// Tunables (spec 20 §7 `retention.genome_weights.*`).
#[derive(Debug, Clone, PartialEq)]
pub struct GcPolicy {
    /// Newest generations kept whole (N of spec 20 §6.5). Default 5.
    pub keep_generations: usize,
    /// Card elites protected per `(output_family, step_ms)`. Default 8.
    pub elite_per_family: usize,
    /// Objects younger than this are never deleted. Default 7 days.
    pub min_age: Duration,
}

impl Default for GcPolicy {
    fn default() -> Self {
        Self { keep_generations: 5, elite_per_family: 8, min_age: Duration::days(7) }
    }
}

/// Keys (in `fa-models`, without bucket prefix) that must survive.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Protection {
    pub keys: BTreeSet<String>,
}

impl Protection {
    pub fn insert(&mut self, key: &str) {
        self.keys.insert(normalize_key(key));
    }

    pub fn extend<I: IntoIterator<Item = String>>(&mut self, keys: I) {
        for k in keys {
            self.insert(&k);
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.keys.contains(key)
    }
}

/// `fa-models/decoders/x` and `decoders/x` → `decoders/x`.
pub fn normalize_key(key: &str) -> String {
    let k = key.trim().trim_start_matches('/');
    k.strip_prefix(&format!("{}/", buckets::MODELS)).unwrap_or(k).to_owned()
}

/// True only for decoder weight bundles `decoders/<generation>/<file>.safetensors`.
pub fn is_weight_key(key: &str) -> bool {
    let Some(rest) = key.strip_prefix(DECODER_PREFIX) else { return false };
    let mut parts = rest.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(g), Some(f), None)
        if !g.is_empty() && f.len() > WEIGHT_SUFFIX.len() && f.ends_with(WEIGHT_SUFFIX))
}

/// Generation segment of a decoder key.
pub fn generation_of(key: &str) -> Option<&str> {
    key.strip_prefix(DECODER_PREFIX)?.split('/').next().filter(|g| !g.is_empty())
}

/// Card key of a bundle key.
pub fn card_key(bundle_key: &str) -> String {
    format!("{}{CARD_SUFFIX}", bundle_key.trim_end_matches(WEIGHT_SUFFIX))
}

/// Why an object is kept or deleted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reason {
    /// In one of the newest `keep_generations` generations.
    RecentGeneration,
    /// Listed in [`Protection`] (serving / shadow / pinned / snapshot feedback set).
    Protected,
    /// A card elite of its output family.
    Elite,
    /// Younger than `min_age` (or without a parseable timestamp).
    TooYoung,
    /// Deleted: old generation, not protected, not elite.
    NonEliteOld,
}

impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RecentGeneration => "recent_generation",
            Self::Protected => "protected",
            Self::Elite => "elite",
            Self::TooYoung => "too_young",
            Self::NonEliteOld => "non_elite_old",
        }
    }
}

/// One planned decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub key: String,
    pub size: u64,
    pub generation: String,
    pub reason: Reason,
}

/// What [`plan`] decided; `delete` is the only list [`execute`] acts on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GcPlan {
    pub delete: Vec<Decision>,
    pub keep: Vec<Decision>,
    /// Generations ordered newest first.
    pub generations: Vec<String>,
}

impl GcPlan {
    pub fn delete_bytes(&self) -> u64 {
        self.delete.iter().map(|d| d.size).sum()
    }

    pub fn keep_bytes(&self) -> u64 {
        self.keep.iter().map(|d| d.size).sum()
    }

    /// Count and bytes per reason (kept and deleted).
    pub fn by_reason(&self) -> BTreeMap<&'static str, (usize, u64)> {
        let mut out: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();
        for d in self.delete.iter().chain(&self.keep) {
            let e = out.entry(d.reason.as_str()).or_default();
            e.0 += 1;
            e.1 += d.size;
        }
        out
    }
}

fn modified(o: &ObjectMeta) -> Option<DateTime<Utc>> {
    o.last_modified.as_deref().and_then(|t| DateTime::parse_from_rfc3339(t).ok()).map(|t| t.with_timezone(&Utc))
}

/// Plans the collection. `objects` is a listing of `fa-models` (any prefix; non-weight keys are
/// ignored and only used to date generations), `elites` the card-elite bundle keys
/// ([`card_elites`]), `now` the reference time. Pure: no I/O.
pub fn plan(
    objects: &[ObjectMeta],
    policy: &GcPolicy,
    protection: &Protection,
    elites: &BTreeSet<String>,
    now: DateTime<Utc>,
) -> GcPlan {
    // Date each generation by the newest object in it (cards included).
    let mut newest: BTreeMap<&str, Option<DateTime<Utc>>> = BTreeMap::new();
    for o in objects {
        if let Some(g) = generation_of(&o.key) {
            let t = modified(o);
            let e = newest.entry(g).or_insert(t);
            *e = (*e).max(t);
        }
    }
    let mut generations: Vec<(&str, Option<DateTime<Utc>>)> = newest.into_iter().collect();
    // newest first; undated generations count as newest (never collected on a guess)
    generations.sort_by(|a, b| match (a.1, b.1) {
        (None, None) => b.0.cmp(a.0),
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some(x), Some(y)) => y.cmp(&x).then_with(|| b.0.cmp(a.0)),
    });
    let recent: BTreeSet<&str> = generations.iter().take(policy.keep_generations).map(|g| g.0).collect();

    let mut out =
        GcPlan { generations: generations.iter().map(|g| g.0.to_owned()).collect(), ..Default::default() };
    for o in objects.iter().filter(|o| is_weight_key(&o.key)) {
        let generation = generation_of(&o.key).unwrap_or_default().to_owned();
        let reason = if recent.contains(generation.as_str()) {
            Reason::RecentGeneration
        } else if protection.contains(&o.key) {
            Reason::Protected
        } else if elites.contains(&o.key) {
            Reason::Elite
        } else if modified(o).is_none_or(|t| now - t < policy.min_age) {
            Reason::TooYoung
        } else {
            Reason::NonEliteOld
        };
        let d = Decision { key: o.key.clone(), size: o.size, generation, reason };
        if reason == Reason::NonEliteOld { out.delete.push(d) } else { out.keep.push(d) }
    }
    out
}

/// The fields of a decoder card the elite rule reads (`fa_decoders::neural::feedback::EliteCard`).
#[derive(Debug, Clone, Deserialize)]
pub struct CardView {
    pub key: String,
    #[serde(default)]
    pub created_ms: i64,
    pub meta: CardMeta,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CardMeta {
    #[serde(default)]
    pub output_family: String,
    #[serde(default)]
    pub step_ms: i64,
    #[serde(default = "nan")]
    pub score: f64,
}

fn nan() -> f64 {
    f64::NAN
}

/// Bundle keys of the card elites: per `(output_family, step_ms)` the `per_family` best cards by
/// `score` (newest first on ties) among those with a finite `score > 0`.
pub fn card_elites(cards: &[CardView], per_family: usize) -> BTreeSet<String> {
    let mut groups: BTreeMap<(&str, i64), Vec<&CardView>> = BTreeMap::new();
    for c in cards.iter().filter(|c| c.meta.score.is_finite() && c.meta.score > 0.0) {
        groups.entry((c.meta.output_family.as_str(), c.meta.step_ms)).or_default().push(c);
    }
    let mut out = BTreeSet::new();
    for (_, mut g) in groups {
        g.sort_by(|a, b| b.meta.score.total_cmp(&a.meta.score).then_with(|| b.created_ms.cmp(&a.created_ms)));
        out.extend(g.into_iter().take(per_family).map(|c| normalize_key(&c.key)));
    }
    out
}

/// Every decoder bundle key referenced anywhere in a JSON document (a snapshot manifest, the
/// `GET /api/decoders/serving` payload): strings that normalise to a [`is_weight_key`] key.
pub fn collect_bundle_refs(v: &serde_json::Value, out: &mut BTreeSet<String>) {
    match v {
        serde_json::Value::String(s) => {
            let k = normalize_key(s);
            if is_weight_key(&k) {
                out.insert(k);
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| collect_bundle_refs(x, out)),
        serde_json::Value::Object(m) => m.values().for_each(|x| collect_bundle_refs(x, out)),
        _ => {}
    }
}

/// Result of [`execute`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GcReport {
    pub deleted: usize,
    pub deleted_bytes: u64,
    /// `(key, error)` of failed deletes; the run continues past them.
    pub failed: Vec<(String, String)>,
    pub dry_run: bool,
}

/// Deletes `plan.delete` from `fa-models` (nothing with `dry_run`). Re-checks every key with
/// [`is_weight_key`] so a hand-edited plan cannot reach other objects.
pub async fn execute<S: BlobStore>(store: &S, plan: &GcPlan, dry_run: bool) -> Result<GcReport, BlobError> {
    let mut report = GcReport { dry_run, ..Default::default() };
    for d in &plan.delete {
        if !is_weight_key(&d.key) {
            return Err(BlobError::Backend(format!("refusing to delete non-weight key {}", d.key)));
        }
    }
    for d in &plan.delete {
        if !dry_run {
            let key = ObjectKey { bucket: buckets::MODELS, key: d.key.clone() };
            if let Err(e) = store.delete(&key).await {
                report.failed.push((d.key.clone(), e.to_string()));
                continue;
            }
        }
        report.deleted += 1;
        report.deleted_bytes += d.size;
    }
    Ok(report)
}

/// Loads every card under `decoders/` (unparseable cards are skipped and counted).
pub async fn load_cards<S: BlobStore>(store: &S, listing: &[ObjectMeta]) -> (Vec<CardView>, usize) {
    let mut cards = Vec::new();
    let mut bad = 0;
    for o in listing.iter().filter(|o| o.key.starts_with(DECODER_PREFIX) && o.key.ends_with(CARD_SUFFIX)) {
        let key = ObjectKey { bucket: buckets::MODELS, key: o.key.clone() };
        match store.get(&key).await.ok().and_then(|b| serde_json::from_slice::<CardView>(&b).ok()) {
            Some(c) => cards.push(c),
            None => bad += 1,
        }
    }
    (cards, bad)
}

/// Bundle keys referenced by any snapshot manifest (`fa-models/snapshots/*/manifest.json`).
pub async fn snapshot_refs<S: BlobStore>(store: &S) -> Result<BTreeSet<String>, BlobError> {
    let mut out = BTreeSet::new();
    for o in store.list(buckets::MODELS, "snapshots/").await? {
        if !o.key.ends_with("/manifest.json") {
            continue;
        }
        let key = ObjectKey { bucket: buckets::MODELS, key: o.key };
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&store.get(&key).await?) {
            collect_bundle_refs(&v, &mut out);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{JSON, MemoryStore, OCTET_STREAM};

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-11-11T00:00:00Z").unwrap().with_timezone(&Utc)
    }

    fn obj(key: &str, days_ago: i64, size: u64) -> ObjectMeta {
        ObjectMeta { key: key.into(), size, last_modified: Some((now() - Duration::days(days_ago)).to_rfc3339()), etag: None }
    }

    fn w(generation: &str, n: u32) -> String {
        format!("decoders/{generation}/nn_forecast-0000000{n}-J{n}.safetensors")
    }

    /// Generations g0 (oldest, 60 d) … g6 (newest, 0 d), two bundles + cards each.
    fn listing() -> Vec<ObjectMeta> {
        let mut out = Vec::new();
        for g in 0..7i64 {
            let days = 60 - g * 10;
            for n in 0..2 {
                let k = w(&format!("g{g}"), n);
                out.push(obj(&card_key(&k), days, 1));
                out.push(obj(&k, days, 100));
            }
        }
        out
    }

    #[test]
    fn weight_key_filter_only_matches_decoder_bundles() {
        assert!(is_weight_key("decoders/01M7/nn_forecast-794652ca-01M8.safetensors"));
        for k in [
            "decoders/01M7/nn_forecast-794652ca-01M8.card.json",
            "decoders/x.safetensors",
            "decoders//a.safetensors",
            "decoders/g/sub/a.safetensors",
            "decoders/g/.safetensors",
            "adapters/s/1.lora.safetensors",
            "snapshots/s/weights.bin",
            "genomes/g1/weights",
            "fa-models/decoders/g/a.safetensors",
        ] {
            assert!(!is_weight_key(k), "{k}");
        }
        assert_eq!(normalize_key("fa-models/decoders/g/a.safetensors"), "decoders/g/a.safetensors");
        assert_eq!(normalize_key("/decoders/g/a.safetensors"), "decoders/g/a.safetensors");
        assert_eq!(card_key("decoders/g/a.safetensors"), "decoders/g/a.card.json");
    }

    #[test]
    fn keeps_newest_generations_and_deletes_old_non_elites() {
        let p = plan(&listing(), &GcPolicy::default(), &Protection::default(), &BTreeSet::new(), now());
        assert_eq!(p.generations, ["g6", "g5", "g4", "g3", "g2", "g1", "g0"]);
        let del: BTreeSet<&str> = p.delete.iter().map(|d| d.generation.as_str()).collect();
        assert_eq!(del, BTreeSet::from(["g0", "g1"]));
        assert_eq!(p.delete.len(), 4);
        assert_eq!(p.delete_bytes(), 400);
        assert_eq!(p.keep.len(), 10);
        assert!(p.keep.iter().all(|d| d.reason == Reason::RecentGeneration));
        assert!(p.delete.iter().chain(&p.keep).all(|d| is_weight_key(&d.key)), "cards never planned");
    }

    #[test]
    fn protection_elites_and_age_guard() {
        let mut prot = Protection::default();
        prot.insert(&format!("fa-models/{}", w("g0", 0)));
        let elites = BTreeSet::from([w("g1", 0)]);
        let objs = listing();
        let policy = GcPolicy { keep_generations: 2, min_age: Duration::days(25), ..Default::default() };
        let p = plan(&objs, &policy, &prot, &elites, now());
        let reason = |k: &str| p.delete.iter().chain(&p.keep).find(|d| d.key == k).unwrap().reason;
        assert_eq!(reason(&w("g0", 0)), Reason::Protected);
        assert_eq!(reason(&w("g0", 1)), Reason::NonEliteOld);
        assert_eq!(reason(&w("g1", 0)), Reason::Elite);
        assert_eq!(reason(&w("g1", 1)), Reason::NonEliteOld);
        assert_eq!(reason(&w("g3", 0)), Reason::NonEliteOld, "30 d, outside the newest 2 generations");
        assert_eq!(reason(&w("g4", 0)), Reason::TooYoung, "20 d < min_age 25 d");
        assert_eq!(reason(&w("g5", 0)), Reason::RecentGeneration);
    }

    #[test]
    fn age_guard_keeps_young_objects_in_old_generations() {
        let policy = GcPolicy { keep_generations: 1, min_age: Duration::days(25), ..Default::default() };
        let p = plan(&listing(), &policy, &Protection::default(), &BTreeSet::new(), now());
        let reason = |g: &str| p.delete.iter().chain(&p.keep).find(|d| d.generation == g).unwrap().reason;
        assert_eq!(reason("g6"), Reason::RecentGeneration);
        assert_eq!(reason("g5"), Reason::TooYoung); // 10 d
        assert_eq!(reason("g4"), Reason::TooYoung); // 20 d
        assert_eq!(reason("g3"), Reason::NonEliteOld); // 30 d
        // undated objects are never deleted
        let mut objs = listing();
        for o in &mut objs {
            o.last_modified = None;
        }
        let p = plan(&objs, &GcPolicy { keep_generations: 0, ..Default::default() }, &Protection::default(), &BTreeSet::new(), now());
        assert!(p.delete.is_empty());
    }

    #[test]
    fn generation_dated_by_its_newest_object() {
        let mut objs = listing();
        objs.push(obj("decoders/g0/late.card.json", 0, 1)); // g0 touched today → newest generation
        let p = plan(&objs, &GcPolicy::default(), &Protection::default(), &BTreeSet::new(), now());
        assert_eq!(p.generations[..2], ["g6", "g0"], "same day as g6, tie broken by name");
        assert!(p.delete.iter().all(|d| d.generation == "g1" || d.generation == "g2"));
    }

    #[test]
    fn card_elites_top_k_per_family_and_step() {
        let card = |k: &str, fam: &str, step: i64, score: f64, created: i64| CardView {
            key: k.into(),
            created_ms: created,
            meta: CardMeta { output_family: fam.into(), step_ms: step, score },
        };
        let cards = vec![
            card("fa-models/decoders/g/a.safetensors", "forecast", 1000, 0.3, 1),
            card("decoders/g/b.safetensors", "forecast", 1000, 0.2, 1),
            card("decoders/g/c.safetensors", "forecast", 1000, 0.2, 2),
            card("decoders/g/d.safetensors", "forecast", 60_000, 0.1, 1),
            card("decoders/g/e.safetensors", "forecast", 1000, -0.1, 9),
            card("decoders/g/f.safetensors", "text", 1000, f64::NAN, 9),
        ];
        let e = card_elites(&cards, 2);
        assert_eq!(
            e,
            BTreeSet::from(
                ["decoders/g/a.safetensors", "decoders/g/c.safetensors", "decoders/g/d.safetensors"].map(String::from)
            )
        );
        let parsed: CardView = serde_json::from_str(
            r#"{"meta":{"format":"x","genome":"g","family":"nn_forecast","output_family":"forecast","step_ms":1000,"score":0.5,"objectives":{}},"bucket":"fa-models","key":"decoders/g/a.safetensors","created_ms":5}"#,
        )
        .unwrap();
        assert_eq!((parsed.meta.score, parsed.created_ms), (0.5, 5));
        let no_score: CardView = serde_json::from_str(r#"{"meta":{},"key":"decoders/g/a.safetensors"}"#).unwrap();
        assert!(card_elites(&[no_score], 8).is_empty());
    }

    #[test]
    fn bundle_refs_from_manifest_and_serving_status() {
        let v = serde_json::json!({
            "feedback": {"decoders": [{"name": "elite_decoders", "elites": [
                {"key": "decoders/3/a.safetensors"}, {"key": "fa-models/decoders/3/b.safetensors"}]}]},
            "slots": [{"serving": {"key": "decoders/9/s.safetensors"}, "pinned": "decoders/1/p.safetensors"}],
            "weights": "snapshots/x/weights.bin",
            "card": "decoders/3/a.card.json"
        });
        let mut out = BTreeSet::new();
        collect_bundle_refs(&v, &mut out);
        assert_eq!(
            out,
            BTreeSet::from(
                ["decoders/1/p.safetensors", "decoders/3/a.safetensors", "decoders/3/b.safetensors", "decoders/9/s.safetensors"]
                    .map(String::from)
            )
        );
    }

    fn block_on<F: Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
    }

    #[test]
    fn execute_against_memory_store() {
        block_on(async {
            let s = MemoryStore::default();
            for o in listing() {
                let ct = if o.key.ends_with(CARD_SUFFIX) { JSON } else { OCTET_STREAM };
                s.put(&ObjectKey { bucket: buckets::MODELS, key: o.key.clone() }, vec![0; o.size as usize], ct)
                    .await
                    .unwrap();
            }
            let other = ObjectKey { bucket: buckets::MODELS, key: "snapshots/s/manifest.json".into() };
            s.put(&other, br#"{"feedback":{"x":"decoders/g0/nn_forecast-00000000-J0.safetensors"}}"#.to_vec(), JSON)
                .await
                .unwrap();
            let mut prot = Protection::default();
            prot.extend(snapshot_refs(&s).await.unwrap());
            assert!(prot.contains(&w("g0", 0)));

            let listing_live = s.list(buckets::MODELS, DECODER_PREFIX).await.unwrap();
            let (cards, bad) = load_cards(&s, &listing_live).await;
            assert_eq!((cards.len(), bad), (0, 14), "zero-filled cards do not parse");
            let p = plan(&listing(), &GcPolicy::default(), &prot, &BTreeSet::new(), now());
            assert_eq!(p.delete.len(), 3);

            let dry = execute(&s, &p, true).await.unwrap();
            assert_eq!((dry.deleted, dry.deleted_bytes, dry.dry_run), (3, 300, true));
            assert_eq!(s.list(buckets::MODELS, "").await.unwrap().len(), 29, "dry run deletes nothing");

            let real = execute(&s, &p, false).await.unwrap();
            assert_eq!((real.deleted, real.deleted_bytes), (3, 300));
            let left: BTreeSet<String> = s.list(buckets::MODELS, "").await.unwrap().into_iter().map(|o| o.key).collect();
            assert_eq!(left.len(), 26);
            assert!(left.contains(&w("g0", 0)), "protected by the manifest");
            assert!(!left.contains(&w("g0", 1)) && !left.contains(&w("g1", 0)));
            assert!(left.contains(&card_key(&w("g0", 1))), "cards survive");
            assert!(left.contains("snapshots/s/manifest.json"));

            // a tampered plan cannot reach a non-weight key
            let mut bad_plan = p.clone();
            bad_plan.delete.push(Decision {
                key: "snapshots/s/manifest.json".into(),
                size: 1,
                generation: String::new(),
                reason: Reason::NonEliteOld,
            });
            assert!(execute(&s, &bad_plan, false).await.is_err());
            assert!(s.exists(&other).await.unwrap());
        });
    }
}
