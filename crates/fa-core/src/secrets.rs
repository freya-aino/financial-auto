//! Secrets loaded from `config/secrets.toml` with `FA_<SECTION>_<KEY>` environment
//! overrides (spec 60 §6.7, D-020).
//!
//! Schema (see `config/secrets.example.toml`):
//! ```toml
//! [garage]
//! access_key_id = ""
//! secret_access_key = ""
//!
//! [sources.lemon]
//! api_key = ""
//! ```
//! `FA_GARAGE_ACCESS_KEY_ID` overrides `[garage] access_key_id`; `FA_LEMON_API_KEY`
//! overrides `[sources.lemon] api_key`. A missing file yields empty secrets, not an error,
//! so env-only deployments work. Values are never printed by `Debug`; use [`redact`] for
//! config dumps.
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use serde::Deserialize;

pub const ENV_PATH: &str = "FA_SECRETS";
pub const DEFAULT_PATH: &str = "config/secrets.toml";

/// Garage S3 credentials (`[garage]`).
#[derive(Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct GarageSecrets {
    pub access_key_id: String,
    pub secret_access_key: String,
}

impl GarageSecrets {
    /// Both values present and non-empty.
    pub fn is_configured(&self) -> bool {
        !self.access_key_id.is_empty() && !self.secret_access_key.is_empty()
    }
}

impl std::fmt::Debug for GarageSecrets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GarageSecrets")
            .field("access_key_id", &redact_value("access_key_id", &self.access_key_id))
            .field("secret_access_key", &redact_value("secret_access_key", &self.secret_access_key))
            .finish()
    }
}

/// All secrets. `sources` maps `<source_id>` → `<key>` → value (free-form per source).
#[derive(Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct Secrets {
    pub garage: GarageSecrets,
    pub sources: BTreeMap<String, BTreeMap<String, String>>,
}

impl std::fmt::Debug for Secrets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sources: BTreeMap<&str, BTreeMap<&str, String>> = self
            .sources
            .iter()
            .map(|(s, kv)| (s.as_str(), kv.iter().map(|(k, v)| (k.as_str(), redact_value(k, v))).collect()))
            .collect();
        f.debug_struct("Secrets").field("garage", &self.garage).field("sources", &sources).finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SecretsError {
    #[error("reading {path}: {source}")]
    Io { path: String, source: std::io::Error },
    #[error("parsing {path}: {source}")]
    Parse { path: String, source: toml::de::Error },
}

/// The secrets file path: `$FA_SECRETS` or `config/secrets.toml`.
pub fn default_path() -> PathBuf {
    std::env::var_os(ENV_PATH).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(DEFAULT_PATH))
}

impl Secrets {
    /// Loads [`default_path`] and applies environment overrides.
    pub fn load() -> Result<Self, SecretsError> {
        Self::load_from(default_path())
    }

    /// Loads `path` (missing file = empty) and applies `FA_*` overrides from the process
    /// environment.
    pub fn load_from(path: impl AsRef<Path>) -> Result<Self, SecretsError> {
        let mut s = Self::from_file(path)?;
        s.apply_env(std::env::vars());
        Ok(s)
    }

    /// Parses the file only; a missing file yields `Secrets::default()`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, SecretsError> {
        let path = path.as_ref();
        let p = path.display().to_string();
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text).map_err(|source| SecretsError::Parse { path: p, source }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(SecretsError::Io { path: p, source }),
        }
    }

    pub fn parse(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Applies `FA_<SECTION>_<KEY>` overrides. `FA_GARAGE_<KEY>` targets `[garage]`;
    /// `FA_<SOURCE>_<KEY>` targets `[sources.<source>]` with the source id lower-cased.
    /// Since both source ids and keys may contain `_`, the split is the longest prefix
    /// matching an already known source, else the first `_`; keys are lower-cased.
    /// Variables with an empty value are ignored.
    pub fn apply_env<I, K, V>(&mut self, vars: I)
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        for (k, v) in vars {
            let Some(rest) = k.as_ref().strip_prefix("FA_") else { continue };
            let value: String = v.into();
            if value.is_empty() || rest.is_empty() {
                continue;
            }
            if let Some(key) = rest.strip_prefix("GARAGE_") {
                match key.to_ascii_lowercase().as_str() {
                    "access_key_id" => self.garage.access_key_id = value,
                    "secret_access_key" => self.garage.secret_access_key = value,
                    _ => {}
                }
                continue;
            }
            let lower = rest.to_ascii_lowercase();
            // Longest known source id that is a proper prefix followed by `_`.
            let known = self
                .sources
                .keys()
                .filter(|s| {
                    lower.len() > s.len() + 1 && lower.starts_with(s.as_str()) && lower.as_bytes()[s.len()] == b'_'
                })
                .max_by_key(|s| s.len())
                .cloned();
            let (source, key) = match known {
                Some(s) => {
                    let key = lower[s.len() + 1..].to_owned();
                    (s, key)
                }
                None => match lower.split_once('_') {
                    Some((s, key)) if !s.is_empty() && !key.is_empty() => (s.to_owned(), key.to_owned()),
                    _ => continue,
                },
            };
            if is_reserved(&source) {
                continue;
            }
            self.sources.entry(source).or_default().insert(key, value);
        }
    }

    /// `[sources.<source>] <key>`, if set and non-empty.
    pub fn source(&self, source: &str, key: &str) -> Option<&str> {
        self.sources.get(source).and_then(|kv| kv.get(key)).map(String::as_str).filter(|v| !v.is_empty())
    }
}

/// Env prefixes that address process configuration, not a source's secrets.
fn is_reserved(section: &str) -> bool {
    matches!(section, "config" | "secrets" | "garage" | "gpu" | "evo" | "log" | "nats" | "helix" | "postgres")
}

/// Keys whose values must never be printed: match `*key*|*secret*|*token*` (spec 60 §6.7).
pub fn is_sensitive_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    k.contains("key") || k.contains("secret") || k.contains("token")
}

pub const REDACTED: &str = "***";

/// `value` unchanged unless `key` is sensitive, in which case `***` (or `""` when empty, so
/// dumps still show which secrets are unset).
pub fn redact_value(key: &str, value: &str) -> String {
    if !is_sensitive_key(key) {
        value.to_owned()
    } else if value.is_empty() {
        String::new()
    } else {
        REDACTED.to_owned()
    }
}

/// Redacts every `key = value` pair of a parsed TOML document whose key is sensitive, at any
/// depth. Use before dumping a config.
pub fn redact(doc: &mut toml::Value) {
    redact_in(doc, None);
}

fn redact_in(v: &mut toml::Value, key: Option<&str>) {
    match v {
        toml::Value::Table(t) => {
            for (k, child) in t.iter_mut() {
                redact_in(child, Some(k));
            }
        }
        toml::Value::Array(a) => {
            for child in a.iter_mut() {
                redact_in(child, key);
            }
        }
        toml::Value::String(s) => {
            if let Some(k) = key
                && is_sensitive_key(k)
                && !s.is_empty()
            {
                *s = REDACTED.to_owned();
            }
        }
        _ => {}
    }
}

/// Parses `text` as TOML and returns it re-serialised with sensitive values redacted.
pub fn redact_toml(text: &str) -> Result<String, toml::de::Error> {
    let mut doc: toml::Value = toml::from_str(text)?;
    redact(&mut doc);
    Ok(toml::to_string(&doc).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"
[garage]
access_key_id = "GK1"
secret_access_key = "s3cr3t"

[sources.lemon]
api_key = "lemon-key"

[sources.oanda]
api_key = "oanda-key"
account_id = "001-004"
"#;

    #[test]
    fn parses_example_schema() {
        let s = Secrets::parse(EXAMPLE).unwrap();
        assert_eq!(s.garage.access_key_id, "GK1");
        assert_eq!(s.garage.secret_access_key, "s3cr3t");
        assert!(s.garage.is_configured());
        assert_eq!(s.source("lemon", "api_key"), Some("lemon-key"));
        assert_eq!(s.source("oanda", "account_id"), Some("001-004"));
        assert_eq!(s.source("oanda", "missing"), None);
        assert_eq!(s.source("nope", "api_key"), None);
    }

    #[test]
    fn repo_example_file_parses_and_is_unconfigured() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/secrets.example.toml");
        let s = Secrets::from_file(path).unwrap();
        assert!(!s.garage.is_configured());
        assert!(s.sources.contains_key("lemon"));
        assert_eq!(s.source("lemon", "api_key"), None, "empty values read as unset");
    }

    #[test]
    fn missing_file_is_empty() {
        let s = Secrets::from_file("/nonexistent/dir/secrets.toml").unwrap();
        assert_eq!(s, Secrets::default());
        assert!(!s.garage.is_configured());
    }

    #[test]
    fn empty_text_is_empty() {
        assert_eq!(Secrets::parse("").unwrap(), Secrets::default());
    }

    #[test]
    fn env_overrides() {
        let mut s = Secrets::parse(EXAMPLE).unwrap();
        s.apply_env([
            ("FA_GARAGE_ACCESS_KEY_ID", "GK-env"),
            ("FA_GARAGE_SECRET_ACCESS_KEY", "env-secret"),
            ("FA_LEMON_API_KEY", "lemon-env"),
            ("FA_OANDA_ACCOUNT_ID", "env-acct"),
            ("FA_NEWSRC_API_KEY", "fresh"),
            ("FA_GARAGE_UNKNOWN", "ignored"),
            ("FA_CONFIG", "config/platform.toml"),
            ("FA_", "nothing"),
            ("FA_LEMON_API_KEY_EMPTY", ""),
            ("HOME", "/home/x"),
        ]);
        assert_eq!(s.garage.access_key_id, "GK-env");
        assert_eq!(s.garage.secret_access_key, "env-secret");
        assert_eq!(s.source("lemon", "api_key"), Some("lemon-env"));
        assert_eq!(s.source("oanda", "account_id"), Some("env-acct"));
        assert_eq!(s.source("oanda", "api_key"), Some("oanda-key"));
        assert_eq!(s.source("newsrc", "api_key"), Some("fresh"));
        assert!(!s.sources.contains_key("config"));
        assert_eq!(s.source("lemon", "api_key_empty"), None);
    }

    #[test]
    fn env_prefers_known_source_with_underscore() {
        let mut s = Secrets::parse("[sources.alpha_vantage]\napi_key = \"a\"\n").unwrap();
        s.apply_env([("FA_ALPHA_VANTAGE_API_KEY", "b")]);
        assert_eq!(s.source("alpha_vantage", "api_key"), Some("b"));
        assert!(!s.sources.contains_key("alpha"));
    }

    #[test]
    fn env_path_default() {
        // Cannot mutate the process env safely in parallel tests; only check the fallback.
        if std::env::var_os(ENV_PATH).is_none() {
            assert_eq!(default_path(), PathBuf::from(DEFAULT_PATH));
        }
    }

    #[test]
    fn sensitive_key_matcher() {
        for k in ["api_key", "access_key_id", "secret_access_key", "SECRET", "auth_token", "TokenValue", "keyring"] {
            assert!(is_sensitive_key(k), "{k}");
        }
        for k in ["account_id", "url", "endpoint", "region", "bind"] {
            assert!(!is_sensitive_key(k), "{k}");
        }
    }

    #[test]
    fn debug_never_prints_values() {
        let s = Secrets::parse(EXAMPLE).unwrap();
        let dump = format!("{s:?}");
        for leak in ["GK1", "s3cr3t", "lemon-key", "oanda-key"] {
            assert!(!dump.contains(leak), "leaked {leak}: {dump}");
        }
        assert!(dump.contains("001-004"), "non-sensitive values stay visible: {dump}");
        assert!(dump.contains(REDACTED));
    }

    #[test]
    fn redact_toml_dump() {
        let out = redact_toml(EXAMPLE).unwrap();
        for leak in ["GK1", "s3cr3t", "lemon-key", "oanda-key"] {
            assert!(!out.contains(leak), "leaked {leak}: {out}");
        }
        assert!(out.contains("account_id = \"001-004\""));
        assert!(out.contains("secret_access_key = \"***\""));
        // Empty secrets stay empty so the dump shows they are unset.
        let out = redact_toml("[garage]\naccess_key_id = \"\"\n").unwrap();
        assert!(out.contains("access_key_id = \"\""));
        // Nested arrays of tables are covered too.
        let out = redact_toml("[[a]]\ntoken = \"t\"\nname = \"n\"\n").unwrap();
        assert!(!out.contains("\"t\""));
        assert!(out.contains("name = \"n\""));
    }
}
