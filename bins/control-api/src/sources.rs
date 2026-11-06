//! Source catalog loader for `config/sources.yaml` (spec 11).
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub version: u32,
    pub sources: Vec<Source>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub name: String,
    pub category: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub asset_classes: Vec<String>,
    #[serde(default)]
    pub transport: Vec<String>,
    pub auth: String,
    pub status: String,
    #[serde(default)]
    pub tos_notes: String,
    #[serde(default)]
    pub notes: String,
}

pub fn load(path: &Path) -> anyhow::Result<Catalog> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_yaml_ng::from_str(&text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_catalog_parses() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/sources.yaml");
        let cat = load(Path::new(path)).expect("config/sources.yaml must parse");
        assert!(cat.sources.len() >= 70, "expected the full catalog");
        let mut ids: Vec<_> = cat.sources.iter().map(|s| s.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), cat.sources.len(), "source ids must be unique");
    }
}
