//! D-043: every `QUERY` of `helix/queries.hx` has a same-named (snake_case) typed function on
//! `fa_hub::HubClient`. Source-level check, no server needed.

const QUERIES_HX: &str = include_str!("../../../helix/queries.hx");
const SOURCES: &[&str] = &[
    include_str!("../src/lib.rs"),
    include_str!("../src/queries.rs"),
    include_str!("../src/vectors.rs"),
    include_str!("../src/asof.rs"),
    include_str!("../src/retention.rs"),
    include_str!("../src/meta.rs"),
    include_str!("../src/catalog.rs"),
    include_str!("../src/regions.rs"),
    include_str!("../src/experts.rs"),
];

fn snake(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn hx_queries() -> Vec<String> {
    QUERIES_HX
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("QUERY "))
        .map(|rest| rest.split('(').next().unwrap_or("").trim().to_owned())
        .collect()
}

#[test]
fn every_hx_query_has_a_typed_function() {
    let names = hx_queries();
    assert!(names.len() >= 40, "parser found too few queries: {names:?}");
    let missing: Vec<String> = names
        .iter()
        .map(|n| snake(n))
        .filter(|f| !SOURCES.iter().any(|src| src.contains(&format!("pub async fn {f}("))))
        .collect();
    assert!(missing.is_empty(), "queries.hx entries without a same-named fa-hub function: {missing:?}");
}

#[test]
fn hx_query_names_are_unique() {
    let mut names = hx_queries();
    let n = names.len();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), n, "duplicate QUERY names in queries.hx");
}

#[test]
fn catalog_names_are_hx_queries_or_documented_extras() {
    let hx = hx_queries();
    for q in fa_hub::catalog::NAMED_QUERIES {
        // genomeAncestry / neighborhood are depth-unrolled batches documented in queries.hx comments.
        assert!(
            hx.iter().any(|h| h == q.name) || QUERIES_HX.contains(q.name),
            "catalog entry `{}` is not described in queries.hx",
            q.name
        );
    }
}

#[test]
fn snake_case_conversion() {
    assert_eq!(snake("addFitnessRecord"), "add_fitness_record");
    assert_eq!(snake("knn"), "knn");
}
