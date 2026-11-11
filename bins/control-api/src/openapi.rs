//! `GET /api/openapi.json` (spec 50 §3, §9 "OpenAPI document covers every REST endpoint"; §10 Q3).
//!
//! The document is **generated from the route declarations themselves**: every route module's
//! source is embedded at compile time ([`SOURCES`]) and its `.route("<path>", get(h).post(h2))`
//! calls are parsed (test modules excluded), so a route cannot exist without an operation —
//! no hand-kept list, no annotation crate on ~100 handlers. Summaries come from the modules'
//! doc tables (`//! | \`GET /api/...\` | answer |`) when a row names the route, else from the
//! handler name. Request and response bodies are documented as JSON objects (`x-handler` names
//! the Rust function that defines them). Placeholder routes answering 501 are listed with
//! `x-status: not_implemented` unless a real handler serves the same method and path (the
//! `install`ed routers layer in front of them).
//!
//! The frontend keeps its hand-written fetch layer (spec 51 Q2): `frontend/src/lib/openapi.json`
//! is a committed copy of this document (`openapi_snapshot_is_current` fails when it is stale;
//! `FA_UPDATE_OPENAPI=1 cargo test -p control-api openapi` rewrites it) and the vitest
//! `openapi.test.ts` checks every `/api/…` call of the frontend against it.

use std::{collections::BTreeMap, sync::OnceLock};

use axum::{Json, Router, response::IntoResponse, routing::get};
use serde_json::{Value, json};

use crate::state::AppState;

/// Route modules in install order (later modules win on a duplicate method + path, like the
/// `install`ed routers that layer in front of `routes.rs`).
pub const SOURCES: [(&str, &str); 19] = [
    ("routes", include_str!("routes.rs")),
    ("graph", include_str!("graph_routes.rs")),
    ("mcmc", include_str!("mcmc_routes.rs")),
    ("streams", include_str!("streams_routes.rs")),
    ("sources", include_str!("sources_routes.rs")),
    ("refdata", include_str!("refdata_routes.rs")),
    ("latent", include_str!("latent_routes.rs")),
    ("evolution", include_str!("evolution_routes.rs")),
    ("config", include_str!("config_routes.rs")),
    ("ws", include_str!("ws_routes.rs")),
    ("validation", include_str!("validation_routes.rs")),
    ("alerts", include_str!("alerts_routes.rs")),
    ("console", include_str!("console_routes.rs")),
    ("ops", include_str!("ops_routes.rs")),
    ("openapi", include_str!("openapi.rs")),
    ("ledger", include_str!("ledger_routes.rs")),
    ("services", include_str!("services_routes.rs")),
    ("trading", include_str!("trading_routes.rs")),
    ("live", include_str!("live_routes.rs")),
];

const METHODS: [&str; 5] = ["get", "post", "put", "delete", "patch"];
const PLACEHOLDERS: [&str; 2] = ["not_implemented", "api_not_found"];

pub fn router() -> Router<AppState> {
    Router::new().route("/api/openapi.json", get(serve))
}

async fn serve() -> impl IntoResponse {
    Json(document().clone())
}

/// One parsed route declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecl {
    pub path: String,
    pub method: String,
    pub handler: String,
    pub module: &'static str,
}

/// The source without its `#[cfg(test)]` modules.
fn non_test(src: &str) -> &str {
    src.find("#[cfg(test)]").map_or(src, |i| &src[..i])
}

/// The text inside the parentheses that open just before `s` (depth 1), up to the match.
fn balanced(s: &str) -> &str {
    let mut depth = 1i32;
    let mut in_str = false;
    let mut prev = '\0';
    for (i, c) in s.char_indices() {
        if in_str {
            if c == '"' && prev != '\\' {
                in_str = false;
            }
        } else if c == '"' {
            in_str = true;
        } else if c == '(' {
            depth += 1;
        } else if c == ')' {
            depth -= 1;
            if depth == 0 {
                return &s[..i];
            }
        }
        prev = c;
    }
    s
}

/// Every `.route("path", method(handler)…)` of one module.
pub fn parse_routes(module: &'static str, src: &str) -> Vec<RouteDecl> {
    let src = non_test(src);
    let mut out = Vec::new();
    let mut rest = src;
    while let Some(i) = rest.find(".route(") {
        let after = &rest[i + ".route(".len()..];
        rest = after;
        let trimmed = after.trim_start();
        let Some(lit) = trimmed.strip_prefix('"') else { continue };
        let Some(end) = lit.find('"') else { continue };
        let path = &lit[..end];
        if !path.starts_with('/') {
            continue;
        }
        let args = balanced(&lit[end + 1..]);
        // `get(h)`, `.post(h)`, `any(h)`: an identifier directly followed by `(`
        let bytes = args.as_bytes();
        let mut j = 0;
        while j < args.len() {
            let start = j;
            while j < args.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            if j > start && j < args.len() && bytes[j] == b'(' {
                let word = &args[start..j];
                let prev_ok = start == 0 || !(bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b':');
                let inner = balanced(&args[j + 1..]);
                if prev_ok && METHODS.contains(&word) {
                    out.push(RouteDecl {
                        path: path.to_owned(),
                        method: word.to_owned(),
                        handler: inner.trim().to_owned(),
                        module,
                    });
                }
                j += 1 + inner.len();
            } else {
                j = j.max(start + 1);
            }
        }
    }
    out
}

/// `(METHODS, path) → summary` from a module's doc table rows `//! | \`GET /api/x\` | text |`
/// and doc bullets `//! * \`GET /api/x\` …`.
fn doc_summaries(src: &str) -> BTreeMap<(String, String), String> {
    let mut out = BTreeMap::new();
    for line in src.lines().map(str::trim) {
        let Some(body) = line.strip_prefix("//!") else { continue };
        let body = body.trim().replace("\\|", "\u{1}");
        let (spec, desc) = if let Some(row) = body.strip_prefix('|') {
            let cells: Vec<&str> = row.split('|').map(str::trim).collect();
            if cells.len() < 2 {
                continue;
            }
            (cells[0].to_owned(), cells[1].to_owned())
        } else {
            continue;
        };
        let spec = spec.trim_matches('`').replace('\u{1}', "|");
        let Some((methods, path)) = spec.split_once(' ') else { continue };
        let path = path.split('?').next().unwrap_or(path).trim().to_owned();
        let desc = desc.replace('\u{1}', "|").replace('`', "");
        for m in methods.split(['/', ',', '|']) {
            let m = m.trim().to_ascii_lowercase();
            if METHODS.contains(&m.as_str()) {
                out.insert((m, path.clone()), desc.clone());
            }
        }
    }
    out
}

fn path_params(path: &str) -> Vec<Value> {
    path.split('/')
        .filter_map(|seg| seg.strip_prefix('{').and_then(|s| s.strip_suffix('}')))
        .map(|name| {
            json!({ "name": name.trim_start_matches('*'), "in": "path", "required": true, "schema": { "type": "string" } })
        })
        .collect()
}

/// All declarations across [`SOURCES`], deduplicated (a real handler beats a placeholder,
/// a later module beats an earlier one).
pub fn all_routes() -> Vec<RouteDecl> {
    let mut by: BTreeMap<(String, String), RouteDecl> = BTreeMap::new();
    for (module, src) in SOURCES {
        for r in parse_routes(module, src) {
            let key = (r.path.clone(), r.method.clone());
            let placeholder = PLACEHOLDERS.contains(&r.handler.as_str());
            match by.get(&key) {
                Some(old) if placeholder && !PLACEHOLDERS.contains(&old.handler.as_str()) => {}
                _ => {
                    by.insert(key, r);
                }
            }
        }
    }
    by.into_values().collect()
}

fn humanise(handler: &str) -> String {
    let name = handler.rsplit("::").next().unwrap_or(handler);
    let s = name.replace('_', " ");
    let mut c = s.chars();
    c.next().map_or_else(String::new, |f| f.to_uppercase().chain(c).collect())
}

/// Builds the OpenAPI 3.1 document.
pub fn build() -> Value {
    let summaries: BTreeMap<(String, String), String> = SOURCES.iter().flat_map(|(_, s)| doc_summaries(s)).collect();
    let mut paths = serde_json::Map::new();
    for r in all_routes() {
        let placeholder = PLACEHOLDERS.contains(&r.handler.as_str());
        let ws = r.path.starts_with("/ws/");
        let summary =
            summaries.get(&(r.method.clone(), r.path.clone())).cloned().unwrap_or_else(|| humanise(&r.handler));
        let mut op = json!({
            "operationId": format!("{}_{}", r.module, r.handler.rsplit("::").next().unwrap_or(&r.handler)),
            "tags": [r.module],
            "summary": summary.chars().take(300).collect::<String>(),
            "x-handler": format!("bins/control-api/src/{}.rs::{}", module_file(r.module), r.handler),
            "responses": if ws {
                json!({ "101": { "description": "WebSocket upgrade (spec 50 §4.8 frames `{stream, seq, ts, topic, data}`)" } })
            } else if placeholder {
                json!({ "501": { "description": "not implemented", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } } })
            } else {
                json!({
                    "200": { "description": "OK", "content": { "application/json": { "schema": { "type": "object" } } } },
                    "default": { "description": "error", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } }
                })
            },
        });
        let params = path_params(&r.path);
        if !params.is_empty() {
            op["parameters"] = Value::Array(params);
        }
        if matches!(r.method.as_str(), "post" | "put" | "patch") {
            op["requestBody"] =
                json!({ "required": false, "content": { "application/json": { "schema": { "type": "object" } } } });
        }
        if placeholder {
            op["x-status"] = json!("not_implemented");
        }
        let entry = paths.entry(r.path.clone()).or_insert_with(|| json!({}));
        entry[r.method.as_str()] = op;
    }
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "financial-auto control-api",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Local control plane (spec 50). Generated from the route declarations of bins/control-api/src (see openapi.rs)."
        },
        "servers": [{ "url": "/" }],
        "paths": paths,
        "components": { "schemas": { "Error": {
            "type": "object",
            "properties": { "error": { "type": "string" }, "detail": {}, "path": { "type": "string" }, "spec": { "type": "string" } }
        } } }
    })
}

fn module_file(module: &str) -> String {
    match module {
        "routes" => "routes".into(),
        "live" => "live_routes".into(),
        m => format!("{m}_routes").replace("openapi_routes", "openapi"),
    }
}

/// The document, built once.
pub fn document() -> &'static Value {
    static DOC: OnceLock<Value> = OnceLock::new();
    DOC.get_or_init(build)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_chained_methods_and_skips_tests() {
        let src = r#"
            Router::new()
                .route("/api/a/{id}", get(h1).post(mod_x::h2))
                .route(
                    "/api/b",
                    put(h3),
                )
                .route("/ws/c", get(ws))
            #[cfg(test)]
            mod tests { fn t() { Router::new().route("/api/test", get(x)); } }
        "#;
        let r = parse_routes("m", src);
        let got: Vec<(&str, &str, &str)> =
            r.iter().map(|d| (d.path.as_str(), d.method.as_str(), d.handler.as_str())).collect();
        assert_eq!(
            got,
            vec![
                ("/api/a/{id}", "get", "h1"),
                ("/api/a/{id}", "post", "mod_x::h2"),
                ("/api/b", "put", "h3"),
                ("/ws/c", "get", "ws")
            ]
        );
    }

    /// Spec 50 §9: the document covers every REST endpoint of the running router.
    #[test]
    fn document_covers_the_known_routes() {
        let doc = build();
        let has = |p: &str, m: &str| doc["paths"][p][m].is_object();
        for (p, m) in [
            ("/api/health", "get"),
            ("/api/openapi.json", "get"),
            ("/api/bus/streams", "get"),
            ("/api/latent/snapshots/{id}", "get"),
            ("/api/latent/snapshots/{id}/shadow", "delete"),
            ("/api/evolution/generations", "post"),
            ("/api/trading/orphans/{id}/resolve", "post"),
            ("/api/trading/payouts", "post"),
            ("/api/trading/limits", "put"),
            ("/api/trading/halts/{scope}/rearm", "post"),
            ("/api/mcmc/{snapshot}/k-step", "get"),
            ("/api/config/{section}", "put"),
            ("/ws/latent", "get"),
        ] {
            assert!(has(p, m), "{m} {p} missing");
        }
        // real handlers beat the 501 placeholders of routes.rs
        assert!(doc["paths"]["/api/trading/strategies"]["get"]["x-status"].is_null());
        assert_eq!(doc["paths"]["/api/trading/strategies"]["get"]["tags"][0], "trading");
        // doc-table summaries are used
        let s = doc["paths"]["/api/trading/halts/{scope}/rearm"]["post"]["summary"].as_str().unwrap();
        assert!(s.contains("Resume"), "{s}");
        // path parameters are declared
        assert_eq!(doc["paths"]["/api/mcmc/{snapshot}/k-step"]["get"]["parameters"][0]["name"], "snapshot");
        assert!(all_routes().len() > 90, "{} routes", all_routes().len());
    }

    /// Spec 51 Q2: the frontend's committed copy is the current document.
    #[test]
    fn openapi_snapshot_is_current() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../frontend/src/lib/openapi.json");
        let want = serde_json::to_string_pretty(&build()).unwrap() + "\n";
        if std::env::var("FA_UPDATE_OPENAPI").is_ok_and(|v| v == "1") {
            std::fs::write(path, &want).unwrap();
            return;
        }
        let have = std::fs::read_to_string(path).unwrap_or_default();
        assert!(
            have == want,
            "frontend/src/lib/openapi.json is stale: run `FA_UPDATE_OPENAPI=1 cargo test -p control-api openapi`"
        );
    }
}
