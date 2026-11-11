//! Log initialisation shared by every Rust binary (spec 60 §4).
//!
//! `FA_LOG_FORMAT=json` writes one JSON object per line (`timestamp`, `level`, `target`, the
//! event's fields flattened, `span`); anything else (default `text`) keeps the human-readable
//! `tracing_subscriber::fmt` lines. `RUST_LOG` filters as usual (default `info`). Logs go to
//! stdout and are kept by Docker's `json-file` driver (`docker compose logs`; spec 60 §5, Q3).
use tracing_subscriber::{EnvFilter, filter::LevelFilter};

/// Log line format selected by `FA_LOG_FORMAT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    Text,
    Json,
}

impl LogFormat {
    /// `json` (any case) → [`LogFormat::Json`], anything else → [`LogFormat::Text`].
    pub fn parse(v: Option<&str>) -> Self {
        match v.map(str::trim) {
            Some(v) if v.eq_ignore_ascii_case("json") => Self::Json,
            _ => Self::Text,
        }
    }

    pub fn from_env() -> Self {
        Self::parse(std::env::var("FA_LOG_FORMAT").ok().as_deref())
    }
}

/// Installs the global subscriber for `service`; a second call (tests) is a no-op.
pub fn init(service: &str) {
    // `RUST_LOG` directives on top of a global `info` default (what the binaries did before)
    let filter = || EnvFilter::builder().with_default_directive(LevelFilter::INFO.into()).from_env_lossy();
    let format = LogFormat::from_env();
    let installed = match format {
        LogFormat::Json => tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(false)
            .with_env_filter(filter())
            .try_init()
            .is_ok(),
        LogFormat::Text => tracing_subscriber::fmt().with_env_filter(filter()).try_init().is_ok(),
    };
    if installed {
        tracing::debug!(service, ?format, "logging initialised");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_is_json_only_when_asked() {
        assert_eq!(LogFormat::parse(Some("json")), LogFormat::Json);
        assert_eq!(LogFormat::parse(Some(" JSON ")), LogFormat::Json);
        assert_eq!(LogFormat::parse(Some("text")), LogFormat::Text);
        assert_eq!(LogFormat::parse(Some("")), LogFormat::Text);
        assert_eq!(LogFormat::parse(None), LogFormat::Text);
        init("test");
        init("test"); // second call must not panic
    }
}
