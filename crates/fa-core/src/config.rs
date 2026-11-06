//! Platform configuration loaded from `config/platform.toml` (spec 60).
use std::path::Path;

use serde::Deserialize;

use crate::Timescale;

#[derive(Debug, Clone, Deserialize)]
pub struct PlatformConfig {
    pub timescales: Vec<Timescale>,
    pub nats: NatsConfig,
    pub helix: Endpoint,
    pub victoriametrics: Endpoint,
    pub postgres: PostgresConfig,
    pub garage: GarageConfig,
    pub control_api: ControlApiConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NatsConfig {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Endpoint {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PostgresConfig {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GarageConfig {
    pub endpoint: String,
    pub region: String,
    pub bucket_raw: String,
    pub bucket_media: String,
    pub bucket_models: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ControlApiConfig {
    pub bind: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {path}: {source}")]
    Io { path: String, source: std::io::Error },
    #[error("parsing {path}: {source}")]
    Parse { path: String, source: toml::de::Error },
}

impl PlatformConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let p = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io { path: p.clone(), source })?;
        toml::from_str(&text).map_err(|source| ConfigError::Parse { path: p, source })
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
    }
}
