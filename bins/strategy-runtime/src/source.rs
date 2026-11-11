//! Where the population comes from (`--strategies` at start, `ctl.strategy.reload`):
//! the built-in baselines, a local file, a Garage key or the newest generation in Garage.
//!
//! | spec | meaning |
//! |---|---|
//! | (none), `managed` | the baselines **plus** the managed live set `fa-artifacts/live/strategies.toml` written by the risk-engine pipeline (evolved strategies past the locked holdout, spec 35 §6.4); a missing live set or Garage means baselines only |
//! | `baselines` | `crates/fa-strategies/baselines.toml` (hand-written, not evolved) |
//! | `garage:fa-artifacts/strategies/<gen>.toml` (bucket prefix optional) | that object in `fa-artifacts` |
//! | `latest` | the object under `fa-artifacts/strategies/` with the greatest key (`.toml` only) |
//! | anything else | a local file path |
//!
//! At start any failure falls back to the baselines with a warning ([`load_or_baselines`]);
//! a reload failure leaves the running population untouched.
use std::path::PathBuf;

use anyhow::Context;
use fa_blob::{BlobStore, ObjectKey, buckets};
use fa_strategies::population::{LIVE_SET_KEY, StrategyFile};

/// Key prefix of promoted strategy populations inside `fa-artifacts`.
pub const STRATEGIES_PREFIX: &str = "strategies/";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PopulationSpec {
    /// Baselines + the managed live set (default).
    Managed,
    Baselines,
    Path(PathBuf),
    /// Key inside `fa-artifacts` (no bucket prefix).
    Garage(String),
    Latest,
}

impl PopulationSpec {
    pub fn parse(s: &str) -> Self {
        let s = s.trim();
        match s {
            "" | "managed" => Self::Managed,
            "baselines" => Self::Baselines,
            "latest" | "garage:latest" => Self::Latest,
            _ => match s.strip_prefix("garage:") {
                Some(k) => {
                    let k = k.trim_start_matches('/');
                    Self::Garage(k.strip_prefix(&format!("{}/", buckets::ARTIFACTS)).unwrap_or(k).to_owned())
                }
                None => Self::Path(PathBuf::from(s)),
            },
        }
    }
}

/// Newest population key under `fa-artifacts/strategies/` by key order.
pub async fn latest_key<B: BlobStore>(blob: &B) -> anyhow::Result<String> {
    let list = blob.list(buckets::ARTIFACTS, STRATEGIES_PREFIX).await.context("listing fa-artifacts/strategies/")?;
    list.into_iter()
        .map(|m| m.key)
        .filter(|k| k.ends_with(".toml"))
        .max()
        .ok_or_else(|| anyhow::anyhow!("not_found: no population under {}/{STRATEGIES_PREFIX}", buckets::ARTIFACTS))
}

/// Loads and validates a population; returns it with a label (`baselines`, the path or the key).
pub async fn load<B: BlobStore>(spec: &PopulationSpec, blob: Option<&B>) -> anyhow::Result<(StrategyFile, String)> {
    match spec {
        PopulationSpec::Baselines => Ok((StrategyFile::baselines(), "baselines".into())),
        PopulationSpec::Managed => {
            let Some(blob) = blob else {
                return Ok((StrategyFile::baselines(), "baselines (managed; Garage not configured)".into()));
            };
            let key = ObjectKey { bucket: buckets::ARTIFACTS, key: LIVE_SET_KEY.into() };
            if !blob.exists(&key).await.with_context(|| format!("checking {}/{LIVE_SET_KEY}", buckets::ARTIFACTS))? {
                return Ok((StrategyFile::baselines(), "baselines (managed; no live set yet)".into()));
            }
            let bytes = blob
                .get(&key)
                .await
                .with_context(|| format!("not_found: fetching {}/{LIVE_SET_KEY}", buckets::ARTIFACTS))?;
            let src = String::from_utf8(bytes).context("invalid: live set is not UTF-8")?;
            let live = StrategyFile::parse(&src).with_context(|| format!("invalid: parsing {LIVE_SET_KEY}"))?;
            let n = live.strategy.len();
            Ok((
                StrategyFile::baselines().merged(live),
                format!("baselines+{}/{LIVE_SET_KEY} ({n} evolved)", buckets::ARTIFACTS),
            ))
        }
        PopulationSpec::Path(p) => {
            let src = std::fs::read_to_string(p).with_context(|| format!("not_found: reading {}", p.display()))?;
            let file = StrategyFile::parse(&src).with_context(|| format!("invalid: parsing {}", p.display()))?;
            Ok((file, p.display().to_string()))
        }
        PopulationSpec::Garage(_) | PopulationSpec::Latest => {
            let blob =
                blob.ok_or_else(|| anyhow::anyhow!("Garage is not configured (config/secrets.toml [garage])"))?;
            let key = match spec {
                PopulationSpec::Garage(k) => k.clone(),
                _ => latest_key(blob).await?,
            };
            let bytes = blob
                .get(&ObjectKey { bucket: buckets::ARTIFACTS, key: key.clone() })
                .await
                .with_context(|| format!("not_found: fetching {}/{key}", buckets::ARTIFACTS))?;
            let src = String::from_utf8(bytes).context("invalid: population is not UTF-8")?;
            let file = StrategyFile::parse(&src).with_context(|| format!("invalid: parsing {key}"))?;
            Ok((file, format!("{}/{key}", buckets::ARTIFACTS)))
        }
    }
}

/// [`load`], falling back to the built-in baselines; the second value is the warning, if any.
pub async fn load_or_baselines<B: BlobStore>(
    spec: &PopulationSpec,
    blob: Option<&B>,
) -> (StrategyFile, String, Option<String>) {
    match load(spec, blob).await {
        Ok((f, label)) => (f, label, None),
        Err(e) => (
            StrategyFile::baselines(),
            "baselines".into(),
            Some(format!("population {spec:?} unavailable, running the built-in baselines: {e:#}")),
        ),
    }
}

#[cfg(test)]
mod tests {
    use fa_blob::MemoryStore;

    use super::*;

    const GEN: &str = "[[strategy]]\nid='gp-a'\ngenome_id='gp-a'\ntimescale='1s'\nsignal='(text_score news)'\n";

    #[test]
    fn parses_specs() {
        assert_eq!(PopulationSpec::parse("latest"), PopulationSpec::Latest);
        assert_eq!(PopulationSpec::parse(""), PopulationSpec::Managed);
        assert_eq!(PopulationSpec::parse("baselines"), PopulationSpec::Baselines);
        assert_eq!(
            PopulationSpec::parse("garage:fa-artifacts/strategies/7.toml"),
            PopulationSpec::Garage("strategies/7.toml".into())
        );
        assert_eq!(
            PopulationSpec::parse("garage:strategies/7.toml"),
            PopulationSpec::Garage("strategies/7.toml".into())
        );
        assert_eq!(PopulationSpec::parse("my.toml"), PopulationSpec::Path("my.toml".into()));
    }

    #[tokio::test]
    async fn loads_from_a_local_file() {
        let dir = std::env::temp_dir().join(format!("fa-strat-{}", ulid::Ulid::generate()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("pop.toml");
        std::fs::write(&p, GEN).unwrap();
        let (f, label) = load::<MemoryStore>(&PopulationSpec::Path(p.clone()), None).await.unwrap();
        assert_eq!((f.strategy.len(), label), (1, p.display().to_string()));
        std::fs::write(&p, "[[strategy]]\nid='x'\ntimescale='1s'\nsignal='(rsi 14)'\n").unwrap();
        let err = load::<MemoryStore>(&PopulationSpec::Path(p.clone()), None).await.unwrap_err();
        assert!(format!("{err:#}").starts_with("invalid:"), "{err:#}");
        let (f, label, warn) =
            load_or_baselines::<MemoryStore>(&PopulationSpec::Path(dir.join("nope.toml")), None).await;
        assert_eq!(label, "baselines");
        assert_eq!(f.strategy.len(), StrategyFile::baselines().strategy.len());
        assert!(warn.unwrap().contains("not_found"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn managed_population_is_baselines_plus_the_live_set() {
        let blob = MemoryStore::default();
        let n = StrategyFile::baselines().strategy.len();
        let (f, label) = load(&PopulationSpec::Managed, Some(&blob)).await.unwrap();
        assert_eq!((f.strategy.len(), label.contains("no live set")), (n, true));
        let (f, _) = load::<MemoryStore>(&PopulationSpec::Managed, None).await.unwrap();
        assert_eq!(f.strategy.len(), n);
        let key = ObjectKey { bucket: buckets::ARTIFACTS, key: LIVE_SET_KEY.into() };
        blob.put(&key, GEN.as_bytes().to_vec(), "application/toml").await.unwrap();
        let (f, label) = load(&PopulationSpec::Managed, Some(&blob)).await.unwrap();
        assert_eq!(f.strategy.len(), n + 1);
        assert!(f.strategy.iter().any(|s| s.id == "gp-a"));
        assert!(label.contains("(1 evolved)"), "{label}");
        // a corrupt live set is an error (start falls back to the baselines with a warning)
        blob.put(&key, b"[[strategy]]\nid='x'".to_vec(), "application/toml").await.unwrap();
        assert!(load(&PopulationSpec::Managed, Some(&blob)).await.is_err());
    }

    #[tokio::test]
    async fn loads_a_key_and_the_latest_generation_from_the_blob_store() {
        let blob = MemoryStore::default();
        let key = |k: &str| ObjectKey { bucket: buckets::ARTIFACTS, key: k.into() };
        // nothing stored → latest falls back
        let (_, label, warn) = load_or_baselines(&PopulationSpec::Latest, Some(&blob)).await;
        assert_eq!(label, "baselines");
        assert!(warn.unwrap().contains("no population"));
        blob.put(&key("strategies/0003.toml"), GEN.as_bytes().to_vec(), "application/toml").await.unwrap();
        blob.put(&key("strategies/0012.toml"), GEN.replace("gp-a", "gp-b").into_bytes(), "application/toml")
            .await
            .unwrap();
        blob.put(&key("strategies/zz-notes.txt"), b"x".to_vec(), "text/plain").await.unwrap();
        blob.put(&key("evals/9/j.json"), b"{}".to_vec(), "application/json").await.unwrap();
        assert_eq!(latest_key(&blob).await.unwrap(), "strategies/0012.toml");
        let (f, label) = load(&PopulationSpec::Latest, Some(&blob)).await.unwrap();
        assert_eq!((f.strategy[0].id.as_str(), label.as_str()), ("gp-b", "fa-artifacts/strategies/0012.toml"));
        let (f, _) =
            load(&PopulationSpec::parse("garage:fa-artifacts/strategies/0003.toml"), Some(&blob)).await.unwrap();
        assert_eq!(f.strategy[0].id, "gp-a");
        // a missing key or no store → error, start falls back with a warning
        assert!(load(&PopulationSpec::Garage("strategies/9.toml".into()), Some(&blob)).await.is_err());
        let (_, label, warn) = load_or_baselines::<MemoryStore>(&PopulationSpec::Latest, None).await;
        assert_eq!(label, "baselines");
        assert!(warn.unwrap().contains("Garage is not configured"));
    }
}
