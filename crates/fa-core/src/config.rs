//! Platform configuration loaded from `config/platform.toml` (spec 60), plus the operator-tunable
//! `[runtime]` section that `PUT /api/config/runtime` (spec 50 §4.1) changes at run time and
//! persists to `config/runtime.toml` next to the platform file (git-ignored, merged on load).
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Timescale;

/// File name of the runtime overlay, resolved next to the platform config ([`runtime_path`]).
pub const RUNTIME_FILE: &str = "runtime.toml";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlatformConfig {
    pub timescales: Vec<Timescale>,
    pub nats: NatsConfig,
    pub helix: Endpoint,
    pub victoriametrics: Endpoint,
    pub postgres: PostgresConfig,
    pub garage: GarageConfig,
    pub control_api: ControlApiConfig,
    /// Operator-tunable keys; absent = defaults, overridden by `runtime.toml` when present.
    #[serde(default)]
    pub runtime: RuntimeConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NatsConfig {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Endpoint {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PostgresConfig {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GarageConfig {
    /// S3 endpoint the services talk to (`http://garage:3900` in compose).
    pub endpoint: String,
    /// Endpoint a browser on the host can reach, used for presigned URLs handed to the UI
    /// (`http://127.0.0.1:3900` in compose). Defaults to `endpoint`.
    #[serde(default)]
    pub public_endpoint: String,
    pub region: String,
    pub bucket_raw: String,
    pub bucket_media: String,
    pub bucket_models: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ControlApiConfig {
    pub bind: String,
}

// ── runtime-tunable section ────────────────────────────────────────────────

/// `[runtime]`: the only section `PUT /api/config/{section}` accepts. Every key has a default
/// and a bound checked by [`RuntimeConfig::validate`]; nothing here is a secret.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub ui: UiRuntime,
    pub bus_tap: BusTapRuntime,
    pub live: LiveRuntime,
    pub ws: WsRuntime,
}

/// `[runtime.ui]`
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct UiRuntime {
    /// REST polling interval of the operator UI, milliseconds (D-032: 2 s).
    pub poll_ms: u64,
}

/// `[runtime.bus_tap]`
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BusTapRuntime {
    /// Messages forwarded per second per `/ws/bus` socket; the rest count as `lagged`.
    pub max_msgs_per_sec: u32,
}

/// `[runtime.live]`: bounds of the in-memory live views (D-032).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct LiveRuntime {
    pub max_jobs: usize,
    pub max_orders: usize,
    pub max_decoder_outputs: usize,
    /// Generation summaries kept for `/api/evolution/generations`.
    pub max_generations: usize,
}

/// `[runtime.ws]`: coalescing rates of the latest-value WebSocket streams (spec 50 §4.8).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct WsRuntime {
    /// `/ws/latent` and `/ws/evolution` flush rate, Hz.
    pub latent_hz: u32,
    /// `/ws/trading` flush rate, Hz.
    pub trading_hz: u32,
}

impl Default for UiRuntime {
    fn default() -> Self {
        Self { poll_ms: 2000 }
    }
}

impl Default for BusTapRuntime {
    fn default() -> Self {
        Self { max_msgs_per_sec: 200 }
    }
}

impl Default for LiveRuntime {
    fn default() -> Self {
        Self { max_jobs: 500, max_orders: 500, max_decoder_outputs: 200, max_generations: 100 }
    }
}

impl Default for WsRuntime {
    fn default() -> Self {
        Self { latent_hz: 10, trading_hz: 5 }
    }
}

/// Bounds of every runtime key: `(key, min, max)`.
pub const RUNTIME_BOUNDS: [(&str, u64, u64); 8] = [
    ("ui.poll_ms", 250, 60_000),
    ("bus_tap.max_msgs_per_sec", 1, 10_000),
    ("live.max_jobs", 10, 10_000),
    ("live.max_orders", 10, 10_000),
    ("live.max_decoder_outputs", 10, 5_000),
    ("live.max_generations", 1, 1_000),
    ("ws.latent_hz", 1, 10),
    ("ws.trading_hz", 1, 5),
];

impl RuntimeConfig {
    /// Checks every key against [`RUNTIME_BOUNDS`]; the error names the first offending key.
    pub fn validate(&self) -> Result<(), String> {
        let values = [
            ("ui.poll_ms", self.ui.poll_ms),
            ("bus_tap.max_msgs_per_sec", u64::from(self.bus_tap.max_msgs_per_sec)),
            ("live.max_jobs", self.live.max_jobs as u64),
            ("live.max_orders", self.live.max_orders as u64),
            ("live.max_decoder_outputs", self.live.max_decoder_outputs as u64),
            ("live.max_generations", self.live.max_generations as u64),
            ("ws.latent_hz", u64::from(self.ws.latent_hz)),
            ("ws.trading_hz", u64::from(self.ws.trading_hz)),
        ];
        for (key, v) in values {
            let (_, min, max) = RUNTIME_BOUNDS.iter().find(|(k, _, _)| *k == key).expect("bounds cover every key");
            if v < *min || v > *max {
                return Err(format!("runtime.{key} = {v} is outside [{min}, {max}]"));
            }
        }
        Ok(())
    }

    /// Parses a complete or partial `[runtime]` document, filling the rest from `self` (keys not
    /// mentioned keep their current value), then validates. Unknown keys are rejected.
    pub fn merged_with(&self, partial: &toml::Value) -> Result<Self, String> {
        let mut base = toml::Value::try_from(self).map_err(|e| e.to_string())?;
        merge_tables(&mut base, partial)?;
        let merged: Self = base.try_into().map_err(|e: toml::de::Error| e.message().to_owned())?;
        merged.validate()?;
        Ok(merged)
    }

    /// Loads `path` (missing file = `None`) and validates it.
    pub fn load_overlay(path: impl AsRef<Path>) -> Result<Option<Self>, ConfigError> {
        let path = path.as_ref();
        let p = path.display().to_string();
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(ConfigError::Io { path: p, source }),
        };
        let cfg: Self = toml::from_str(&text).map_err(|source| ConfigError::Parse { path: p.clone(), source })?;
        cfg.validate().map_err(|msg| ConfigError::Invalid { path: p, msg })?;
        Ok(Some(cfg))
    }

    /// Writes the section as TOML to `path` (atomically via a sibling temp file).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        let p = path.display().to_string();
        let text = format!(
            "# Runtime-tunable keys written by control-api (`PUT /api/config/runtime`, spec 50 §4.1).\n# Merged over `[runtime]` of platform.toml on start; git-ignored.\n\n{}",
            toml::to_string(self).unwrap_or_default()
        );
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, text).map_err(|source| ConfigError::Io { path: p.clone(), source })?;
        std::fs::rename(&tmp, path).map_err(|source| ConfigError::Io { path: p, source })
    }
}

/// Deep-merges `over` into `base`; both must be tables at every shared key.
fn merge_tables(base: &mut toml::Value, over: &toml::Value) -> Result<(), String> {
    let toml::Value::Table(o) = over else {
        return Err("runtime config must be a table".into());
    };
    let toml::Value::Table(b) = base else {
        return Err("runtime config must be a table".into());
    };
    for (k, v) in o {
        match (b.get_mut(k), v) {
            (Some(existing @ toml::Value::Table(_)), toml::Value::Table(_)) => merge_tables(existing, v)?,
            (Some(_), toml::Value::Table(_)) => return Err(format!("`{k}` is a value, not a table")),
            (_, _) => {
                b.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(())
}

/// `runtime.toml` next to the platform config file.
pub fn runtime_path(platform_config: impl AsRef<Path>) -> PathBuf {
    platform_config.as_ref().parent().map(Path::to_path_buf).unwrap_or_default().join(RUNTIME_FILE)
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {path}: {source}")]
    Io { path: String, source: std::io::Error },
    #[error("parsing {path}: {source}")]
    Parse { path: String, source: toml::de::Error },
    #[error("{path}: {msg}")]
    Invalid { path: String, msg: String },
}

impl PlatformConfig {
    /// Loads the platform file, fills `garage.public_endpoint`, and merges `runtime.toml` from the
    /// same directory when present ([`runtime_path`]).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let p = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io { path: p.clone(), source })?;
        let mut cfg: Self = toml::from_str(&text).map_err(|source| ConfigError::Parse { path: p.clone(), source })?;
        if cfg.garage.public_endpoint.is_empty() {
            cfg.garage.public_endpoint = cfg.garage.endpoint.clone();
        }
        cfg.runtime.validate().map_err(|msg| ConfigError::Invalid { path: p, msg })?;
        if let Some(overlay) = RuntimeConfig::load_overlay(runtime_path(path))? {
            cfg.runtime = overlay;
        }
        Ok(cfg)
    }

    /// Section names of the platform config in file order.
    pub fn section_names() -> &'static [&'static str] {
        &["timescales", "nats", "helix", "victoriametrics", "postgres", "garage", "control_api", "runtime"]
    }

    /// One section as a TOML value (`None` for an unknown name). Secrets are not part of this
    /// struct, but callers should still pass the result through [`crate::secrets::redact`].
    pub fn section(&self, name: &str) -> Option<toml::Value> {
        let doc = toml::Value::try_from(self).ok()?;
        doc.get(name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_platform_config_parses() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/platform.toml");
        let cfg = PlatformConfig::load(path).expect("config/platform.toml must parse");
        assert_eq!(cfg.timescales, Timescale::LADDER.to_vec());
        assert_eq!(cfg.garage.public_endpoint, cfg.garage.endpoint, "public endpoint defaults to endpoint");
        assert!(cfg.section("garage").is_some());
        assert!(cfg.section("nope").is_none());
        assert!(PlatformConfig::section_names().contains(&"runtime"));
    }

    #[test]
    fn docker_config_sets_host_reachable_garage_endpoint() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/platform.docker.toml");
        let cfg = PlatformConfig::load(path).expect("config/platform.docker.toml must parse");
        assert_eq!(cfg.garage.endpoint, "http://garage:3900");
        assert_eq!(cfg.garage.public_endpoint, "http://127.0.0.1:3900");
    }

    #[test]
    fn runtime_defaults_validate_and_bounds_bite() {
        let r = RuntimeConfig::default();
        r.validate().unwrap();
        assert_eq!(r.ui.poll_ms, 2000);
        let mut bad = r.clone();
        bad.bus_tap.max_msgs_per_sec = 0;
        assert!(bad.validate().unwrap_err().contains("bus_tap.max_msgs_per_sec"));
        let mut bad = r;
        bad.ws.latent_hz = 11;
        assert!(bad.validate().unwrap_err().contains("ws.latent_hz"));
    }

    #[test]
    fn partial_merge_keeps_unmentioned_keys_and_rejects_unknown() {
        let base = RuntimeConfig::default();
        let partial: toml::Value = toml::from_str("[ui]\npoll_ms = 500\n").unwrap();
        let merged = base.merged_with(&partial).unwrap();
        assert_eq!(merged.ui.poll_ms, 500);
        assert_eq!(merged.live.max_jobs, 500);
        let unknown: toml::Value = toml::from_str("[ui]\ncolour = \"red\"\n").unwrap();
        assert!(base.merged_with(&unknown).is_err());
        let wrong_shape: toml::Value = toml::from_str("ui = 3\n").unwrap();
        assert!(base.merged_with(&wrong_shape).is_err());
        let out_of_range: toml::Value = toml::from_str("[live]\nmax_jobs = 1\n").unwrap();
        assert!(base.merged_with(&out_of_range).unwrap_err().contains("live.max_jobs"));
    }

    #[test]
    fn overlay_round_trips_through_a_file() {
        let dir = std::env::temp_dir().join(format!("fa-core-runtime-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let platform = dir.join("platform.toml");
        let path = runtime_path(&platform);
        assert_eq!(path, dir.join(RUNTIME_FILE));
        assert!(RuntimeConfig::load_overlay(&path).unwrap().is_none());
        let mut r = RuntimeConfig::default();
        r.ui.poll_ms = 750;
        r.save(&path).unwrap();
        assert_eq!(RuntimeConfig::load_overlay(&path).unwrap(), Some(r));
        std::fs::write(&path, "[ui]\npoll_ms = 1\n").unwrap();
        assert!(matches!(RuntimeConfig::load_overlay(&path), Err(ConfigError::Invalid { .. })));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
