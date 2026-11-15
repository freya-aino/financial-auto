//! External connections & credentials (owner request 2026-11-15): every keyed catalog source,
//! the Garage S3 credential and the configured infra endpoints in one read-only view, so the
//! owner sees at a glance which credentials are missing and exactly where to set them.
//!
//! Secret values NEVER leave the process: a credential is reported as `{name, set, hint}` where
//! the hint is at most the last 4 characters of a value that is set (and only when the value is
//! long enough that the tail identifies without revealing); endpoint URLs go through
//! [`crate::config_routes::mask_url_password`] plus sensitive-query redaction
//! (`fa_core::secrets::is_sensitive_key`). There is no write route — secrets stay owner-edited
//! in `config/secrets.toml` (D-020), and each missing credential names its `secret_ref`
//! (e.g. `[sources.oanda] api_key`) and `env` override so the UI can show actionable guidance.
//!
//! | Route | Answer |
//! |---|---|
//! | `GET /api/connections` | one row per keyed catalog source (credential fields as `{name, required, set, hint}` with the secrets.toml ref and env override, plus non-secret account info), the `[garage]` S3 credential and the infra endpoint URLs with passwords redacted |
use std::collections::BTreeMap;

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use fa_core::{Secrets, config::PlatformConfig, secrets};
use serde_json::{Map, Value, json};

use crate::{sources, state::AppState};

pub fn router() -> Router<AppState> {
    Router::new().route("/api/connections", get(connections))
}

async fn connections(State(s): State<AppState>) -> axum::response::Response {
    let cat = match sources::load(&s.sources_path) {
        Ok(c) => c,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    };
    // Same loading path as the connectors (config/secrets.toml + FA_* env, D-020).
    let secrets = match Secrets::load() {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    };
    let statuses = db_statuses(&s).await;
    Json(build_response(&cat, &statuses, &secrets, &s.cfg)).into_response()
}

/// `catalog.source` status per id (Postgres wins over the YAML once synced, spec 11 §8);
/// empty without Postgres.
async fn db_statuses(s: &AppState) -> BTreeMap<String, String> {
    let Some(pool) = s.pg.get() else { return BTreeMap::new() };
    match sqlx::query_as::<_, (String, String)>("SELECT id, status FROM catalog.source").fetch_all(pool).await {
        Ok(rows) => rows.into_iter().collect(),
        Err(e) => {
            tracing::warn!(error = %e, "catalog.source read failed; serving yaml statuses");
            BTreeMap::new()
        }
    }
}

/// The credential keys a source needs. The implemented connectors define theirs exactly
/// (`go/connectors/lemon`: `api_key`; `go/connectors/oanda`: `api_key` + `account_id`); for the
/// rest of the catalog the keys are first guesses per auth scheme — `config/sources.yaml`
/// records the scheme (`auth:`), not the key names.
pub fn required_keys(id: &str, auth: &str) -> &'static [&'static str] {
    match id {
        "lemon" => &["api_key"],
        "oanda" => &["api_key", "account_id"],
        _ => match auth {
            "api_key" => &["api_key"],
            "hmac" => &["api_key", "api_secret"],
            "oauth2" => &["client_id", "client_secret"],
            "session" => &["username", "password"],
            "wallet" => &["wallet_key"],
            // `none` needs nothing; `gateway` authenticates against a locally running gateway.
            _ => &[],
        },
    }
}

/// Masked tail of a set value: at most the last 4 characters, and only when the value is long
/// enough (≥ 8 chars) that the tail cannot reconstruct it. Never the value itself.
pub fn mask_tail(value: &str) -> Option<String> {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() < 8 {
        return None;
    }
    Some(format!("…{}", chars[chars.len() - 4..].iter().collect::<String>()))
}

/// `mask_url_password` plus redaction of sensitive query values (`*key*|*secret*|*token*` and
/// `password`), so `postgres://u:pw@h/db?password=x` leaks neither.
pub fn redact_url(url: &str) -> String {
    let masked = crate::config_routes::mask_url_password(url);
    let Some((base, query)) = masked.split_once('?') else { return masked };
    let redacted: Vec<String> = query
        .split('&')
        .map(|pair| match pair.split_once('=') {
            Some((k, v)) if !v.is_empty() && (secrets::is_sensitive_key(k) || k.eq_ignore_ascii_case("password")) => {
                format!("{k}={}", secrets::REDACTED)
            }
            _ => pair.to_owned(),
        })
        .collect();
    format!("{base}?{}", redacted.join("&"))
}

/// One credential field: whether it is set and where to set it — never its value.
fn credential(section: &str, env_section: &str, key: &str, required: bool, value: Option<&str>) -> Value {
    let set = value.is_some_and(|v| !v.is_empty());
    json!({
        "name": key,
        "required": required,
        "set": set,
        "hint": if set { value.and_then(mask_tail) } else { None },
        "secret_ref": format!("[{section}] {key}"),
        "env": format!("FA_{}_{}", env_section.to_uppercase(), key.to_uppercase()),
    })
}

/// Non-secret account info: every configured non-sensitive secret key in full (e.g. the OANDA
/// `account_id` — an identifier, not a secret) plus what the connector derives from
/// `config.extra` (host, practice flag, MIC; mirrors `go/connectors/{lemon,oanda}`).
fn account_info(id: &str, extra: &Map<String, Value>, kv: Option<&BTreeMap<String, String>>) -> Value {
    let mut out = Map::new();
    if let Some(kv) = kv {
        for (k, v) in kv {
            if !secrets::is_sensitive_key(k) && !v.is_empty() {
                out.insert(k.clone(), json!(v));
            }
        }
    }
    match id {
        "oanda" => {
            let env = extra.get("env").and_then(Value::as_str).unwrap_or("practice");
            out.insert("env".into(), json!(env));
            out.insert("practice".into(), json!(env != "live"));
            let host = match env {
                "live" => "https://api-fxtrade.oanda.com",
                _ => "https://api-fxpractice.oanda.com",
            };
            out.insert("host".into(), json!(host));
        }
        "lemon" => {
            out.insert("mic".into(), json!(extra.get("mic").and_then(Value::as_str).unwrap_or("XMUN")));
            out.insert("host".into(), json!("https://data.lemon.markets/v1"));
        }
        _ => {}
    }
    Value::Object(out)
}

/// One connection row for a catalog source, or `None` when it needs no credentials and has none
/// configured.
fn source_row(src: &sources::Source, status: Option<&String>, secrets: &Secrets) -> Option<Value> {
    let required = required_keys(&src.id, &src.auth);
    let kv = secrets.sources.get(&src.id);
    // Keys actually present in secrets.toml beyond the required ones (e.g. sec_edgar user_agent).
    let extra_keys: Vec<&String> = kv
        .map(|kv| kv.iter().filter(|(k, v)| !required.contains(&k.as_str()) && !v.is_empty()).map(|(k, _)| k))
        .into_iter()
        .flatten()
        .collect();
    if required.is_empty() && extra_keys.is_empty() {
        return None;
    }
    let section = format!("sources.{}", src.id);
    let creds: Vec<Value> = required
        .iter()
        .map(|k| credential(&section, &src.id, k, true, secrets.source(&src.id, k)))
        .chain(extra_keys.into_iter().map(|k| credential(&section, &src.id, k, false, secrets.source(&src.id, k))))
        .collect();
    let configured = required.iter().all(|k| secrets.source(&src.id, k).is_some());
    let extra =
        src.config.as_ref().and_then(|c| c.get("extra")).and_then(Value::as_object).cloned().unwrap_or_default();
    Some(json!({
        "id": src.id,
        "name": src.name,
        "category": src.category,
        "auth": src.auth,
        "status": status.unwrap_or(&src.status),
        "configured": configured,
        "credentials": creds,
        "account": account_info(&src.id, &extra, kv),
    }))
}

/// The whole `GET /api/connections` document, pure for the offline tests.
pub fn build_response(
    cat: &sources::Catalog,
    statuses: &BTreeMap<String, String>,
    secrets: &Secrets,
    cfg: &PlatformConfig,
) -> Value {
    let mut connections: Vec<Value> =
        cat.sources.iter().filter_map(|src| source_row(src, statuses.get(&src.id), secrets)).collect();
    // Secrets sections for sources the catalog does not know still show up (never silently held).
    for (id, kv) in &secrets.sources {
        if cat.sources.iter().any(|s| &s.id == id) || kv.values().all(String::is_empty) {
            continue;
        }
        let section = format!("sources.{id}");
        let creds: Vec<Value> = kv
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, _)| credential(&section, id, k, false, secrets.source(id, k)))
            .collect();
        connections.push(json!({
            "id": id, "name": id, "category": Value::Null, "auth": Value::Null,
            "status": Value::Null, "configured": true, "credentials": creds,
            "account": account_info(id, &Map::new(), Some(kv)),
            "note": "not in config/sources.yaml",
        }));
    }
    let g = &secrets.garage;
    let garage = json!({
        "configured": g.is_configured(),
        // The access key id is an identifier (like an S3 access key id), not a secret.
        "access_key_id": if g.access_key_id.is_empty() { Value::Null } else { json!(g.access_key_id) },
        "endpoint": cfg.garage.endpoint,
        "credentials": [
            credential("garage", "garage", "access_key_id", true, Some(&g.access_key_id)),
            credential("garage", "garage", "secret_access_key", true, Some(&g.secret_access_key)),
        ],
    });
    let endpoints: Vec<Value> = [
        ("nats", cfg.nats.url.as_str()),
        ("postgres", cfg.postgres.url.as_str()),
        ("victoriametrics", cfg.victoriametrics.url.as_str()),
        ("garage", cfg.garage.endpoint.as_str()),
        ("qdrant", cfg.qdrant.url.as_str()),
        ("helix", cfg.helix.url.as_str()),
    ]
    .into_iter()
    .map(|(name, url)| json!({ "name": name, "url": redact_url(url) }))
    .collect();
    json!({ "connections": connections, "garage": garage, "endpoints": endpoints })
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    const SECRETS: &str = r#"
[garage]
access_key_id = "GK31c2f218a2e44f329df7"
secret_access_key = "very-secret-garage-key"

[sources.lemon]
api_key = ""

[sources.oanda]
api_key = "oanda-api-key-123456" # secret-scan: allow (test fixture, not a real key)
account_id = "001-004-1234567-001"

[sources.sec_edgar]
user_agent = "financial-auto owner@example.com"

[sources.mystery]
api_key = "mystery-key-98765" # secret-scan: allow (test fixture, not a real key)
"#;

    fn catalog() -> sources::Catalog {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        sources::load(std::path::Path::new(&format!("{root}/sources.yaml"))).unwrap()
    }

    fn cfg() -> PlatformConfig {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        PlatformConfig::load(format!("{root}/platform.toml")).unwrap()
    }

    fn response() -> Value {
        build_response(&catalog(), &BTreeMap::new(), &Secrets::parse(SECRETS).unwrap(), &cfg())
    }

    fn row<'a>(v: &'a Value, id: &str) -> &'a Value {
        v["connections"].as_array().unwrap().iter().find(|c| c["id"] == id).unwrap()
    }

    #[test]
    fn masked_tail_is_at_most_four_chars_and_only_for_long_values() {
        assert_eq!(mask_tail("oanda-api-key-123456").as_deref(), Some("…3456"));
        assert_eq!(mask_tail("12345678").as_deref(), Some("…5678"));
        assert_eq!(mask_tail("1234567"), None, "short values give no tail");
        assert_eq!(mask_tail(""), None);
        assert_eq!(mask_tail("käßöü-秘密キー12").as_deref(), Some("…キー12"), "char boundaries, not bytes");
    }

    #[test]
    fn urls_lose_passwords_and_sensitive_query_values() {
        assert_eq!(redact_url("postgres://fa:hunter2@127.0.0.1:5432/fa"), "postgres://fa:***@127.0.0.1:5432/fa");
        assert_eq!(redact_url("nats://127.0.0.1:4222"), "nats://127.0.0.1:4222");
        assert_eq!(
            redact_url("postgres://h/db?password=x&sslmode=require&api_key=k&empty="),
            "postgres://h/db?password=***&sslmode=require&api_key=***&empty="
        );
        assert_eq!(redact_url("http://u@host/x"), "http://u@host/x", "userinfo without password is kept");
    }

    #[test]
    fn keyed_sources_report_set_missing_and_the_exact_secret_ref() {
        let v = response();
        let oanda = row(&v, "oanda");
        assert_eq!(oanda["configured"], true);
        assert_eq!(oanda["status"], "stub");
        let creds = oanda["credentials"].as_array().unwrap();
        let api_key = creds.iter().find(|c| c["name"] == "api_key").unwrap();
        assert_eq!(api_key["set"], true);
        assert_eq!(api_key["hint"], "…3456");
        assert_eq!(api_key["secret_ref"], "[sources.oanda] api_key");
        assert_eq!(api_key["env"], "FA_OANDA_API_KEY");
        // lemon's key is empty = unset, with the guidance to set it
        let lemon = row(&v, "lemon");
        assert_eq!(lemon["configured"], false);
        let key = &lemon["credentials"][0];
        assert_eq!((key["set"].clone(), key["hint"].clone()), (json!(false), Value::Null));
        assert_eq!(key["secret_ref"], "[sources.lemon] api_key");
        // planned keyed sources are listed too, per their auth scheme
        let kraken = row(&v, "kraken");
        assert_eq!(kraken["auth"], "hmac");
        let names: Vec<&str> =
            kraken["credentials"].as_array().unwrap().iter().map(|c| c["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["api_key", "api_secret"]);
        // keyless sources have no row
        assert!(!v["connections"].as_array().unwrap().iter().any(|c| c["id"] == "yahoo"));
    }

    #[test]
    fn account_info_is_non_secret_only() {
        let v = response();
        let oanda = row(&v, "oanda");
        assert_eq!(oanda["account"]["account_id"], "001-004-1234567-001", "an id, shown in full");
        assert_eq!(oanda["account"]["practice"], true);
        assert_eq!(oanda["account"]["env"], "practice");
        assert_eq!(oanda["account"]["host"], "https://api-fxpractice.oanda.com");
        assert_eq!(row(&v, "lemon")["account"]["mic"], "XMUN");
        // sec_edgar (auth: none) appears because its optional user_agent secret is set
        let edgar = row(&v, "sec_edgar");
        assert_eq!(edgar["account"]["user_agent"], "financial-auto owner@example.com");
        assert_eq!(edgar["credentials"][0]["required"], false);
        // a secrets section the catalog does not know still shows up
        let mystery = row(&v, "mystery");
        assert_eq!(mystery["status"], Value::Null);
        assert_eq!(mystery["credentials"][0]["secret_ref"], "[sources.mystery] api_key");
    }

    #[test]
    fn garage_and_endpoints() {
        let v = response();
        assert_eq!(v["garage"]["configured"], true);
        assert_eq!(
            v["garage"]["access_key_id"], "GK31c2f218a2e44f329df7",
            "access key id in full: an id, not a secret"
        );
        let sk = &v["garage"]["credentials"][1];
        assert_eq!(sk["name"], "secret_access_key");
        assert_eq!(sk["hint"], "…-key");
        assert_eq!(sk["secret_ref"], "[garage] secret_access_key");
        let names: Vec<&str> = v["endpoints"].as_array().unwrap().iter().map(|e| e["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["nats", "postgres", "victoriametrics", "garage", "qdrant", "helix"]);
        let pg = &v["endpoints"][1]["url"];
        assert_eq!(pg, &json!(format!("postgres://fa:{}@127.0.0.1:5432/fa", secrets::REDACTED)));
    }

    /// The non-negotiable rule: no secret value ever appears anywhere in the response.
    #[test]
    fn no_secret_value_ever_leaves() {
        let dump = response().to_string();
        for leak in ["oanda-api-key-123456", "very-secret-garage-key", "mystery-key-98765", "hunter2"] {
            assert!(!dump.contains(leak), "leaked {leak}: {dump}");
        }
        // db status overlay wins over yaml
        let statuses = BTreeMap::from([("oanda".to_owned(), "live".to_owned())]);
        let v = build_response(&catalog(), &statuses, &Secrets::parse(SECRETS).unwrap(), &cfg());
        assert_eq!(row(&v, "oanda")["status"], "live");
    }

    /// Route shape offline (no Postgres, whatever secrets the environment has).
    #[tokio::test]
    async fn route_answers_with_the_three_parts() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let mut cfg = cfg();
        cfg.postgres.url = String::new();
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        let app = router().with_state(state);
        let res = app.oneshot(Request::get("/api/connections").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert!(v["connections"].as_array().is_some_and(|c| !c.is_empty()));
        assert!(v["garage"]["credentials"].as_array().is_some_and(|c| c.len() == 2));
        assert_eq!(v["endpoints"].as_array().unwrap().len(), 6);
        // every credential row is {set, hint?} — never a value field
        for c in v["connections"].as_array().unwrap() {
            for f in c["credentials"].as_array().unwrap() {
                assert!(f.get("value").is_none());
                assert!(f["set"].is_boolean());
            }
        }
    }
}
