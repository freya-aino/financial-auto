//! Runtime contract-version check (spec 21 §6 "Versioning", §8 "Proto incompatibility").
//!
//! Every payload today is package `fa.v1`. A breaking change ships as `fa.v2` **side by side**
//! (spec 21 §6); its producers mark each message with the header `Fa-Schema: fa.v2`. A message
//! without the header is `fa.v1` — so v1 producers add no bytes (market data stays header-free).
//! Consumers drop a message whose header names a major version this build does not decode
//! ([`check`]), log it once per subject root and count it in
//! `fa_bus_rejected_schema_total{major}` instead of mis-decoding v2 bytes as v1 (protobuf would
//! often "succeed" with garbage). `Bus::subscribe`, `Bus::subscribe_traced` and
//! [`crate::jobs::run_worker`] apply it; raw consumers call [`check`] themselves.
use async_nats::HeaderMap;

/// Header naming the protobuf package of the payload.
pub const SCHEMA_HEADER: &str = "Fa-Schema";

/// Major versions this build decodes (`fa_proto::MAJOR` and nothing else yet).
pub const SUPPORTED_MAJORS: &[u32] = &[fa_proto::MAJOR];

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SchemaError {
    #[error("unsupported contract major version {0} (this build decodes {SUPPORTED_MAJORS:?})")]
    UnknownMajor(u32),
    #[error("malformed {SCHEMA_HEADER} header {0:?}")]
    Malformed(String),
}

/// `fa.v2` → 2 (also accepts a bare `2` or `v2`).
pub fn parse_major(v: &str) -> Option<u32> {
    let v = v.trim();
    let v = v.strip_prefix("fa.").unwrap_or(v);
    let v = v.strip_prefix('v').unwrap_or(v);
    v.parse().ok()
}

/// `Ok` when the message carries no schema header (v1) or names a supported major.
pub fn check(headers: Option<&HeaderMap>) -> Result<(), SchemaError> {
    let Some(v) = headers.and_then(|h| h.get(SCHEMA_HEADER)) else { return Ok(()) };
    let s = v.as_str();
    let major = parse_major(s).ok_or_else(|| SchemaError::Malformed(s.to_owned()))?;
    if SUPPORTED_MAJORS.contains(&major) { Ok(()) } else { Err(SchemaError::UnknownMajor(major)) }
}

/// Header value for a package major (`fa.v<major>`); v1 producers need not send it.
pub fn header_value(major: u32) -> String {
    format!("fa.v{major}")
}

/// [`check`] for a consumer: on rejection, counts and logs (warn) and returns `false`.
pub fn accept(subject: &str, headers: Option<&HeaderMap>) -> bool {
    match check(headers) {
        Ok(()) => true,
        Err(e) => {
            let major = match &e {
                SchemaError::UnknownMajor(m) => m.to_string(),
                SchemaError::Malformed(_) => "malformed".to_owned(),
            };
            fa_metrics::counter("fa_bus_rejected_schema_total", "messages dropped for an unknown contract major version")
                .inc_with(&[("major", major.as_str())]);
            tracing::warn!(%subject, error = %e, "dropping message of an unknown contract version");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(v: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(SCHEMA_HEADER, v);
        h
    }

    #[test]
    fn absent_header_is_v1_and_v1_is_accepted() {
        assert_eq!(check(None), Ok(()));
        assert_eq!(check(Some(&HeaderMap::new())), Ok(()));
        assert_eq!(check(Some(&h("fa.v1"))), Ok(()));
        assert_eq!(check(Some(&h(&header_value(fa_proto::MAJOR)))), Ok(()));
    }

    #[test]
    fn unknown_major_and_garbage_are_rejected() {
        assert_eq!(check(Some(&h("fa.v2"))), Err(SchemaError::UnknownMajor(2)));
        assert_eq!(check(Some(&h("v3"))), Err(SchemaError::UnknownMajor(3)));
        assert!(matches!(check(Some(&h("fa.vX"))), Err(SchemaError::Malformed(_))));
        assert!(!accept("md.trade.x.y", Some(&h("fa.v2"))));
        assert!(accept("md.trade.x.y", None));
        assert!(fa_metrics::render().contains("fa_bus_rejected_schema_total{major=\"2\"}"));
        assert_eq!(fa_proto::PACKAGE, "fa.v1");
    }
}
