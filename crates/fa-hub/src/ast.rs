//! Builder for HelixDB `/v2/query` operation trees.
//!
//! Everything here produces plain `serde_json::Value`s in the exact shape the server's
//! `helix-ast` crate deserializes (externally tagged `snake_case` enums). The shapes were
//! verified against `ghcr.io/helixdb/helixdb:v0.0.14`; see the crate README for examples.
//!
//! A request is a [`Batch`] of named entries (`var`), each an [`Op`] tree, plus the list of
//! names to return. Read batches may only contain read-only ops; write batches may mix reads
//! and mutations but **not** index DDL (`create_index`/`drop_index`) with data mutations.

use std::collections::BTreeMap;

use serde_json::{Value as Json, json};

/// A literal property value in the tagged request form (`{"string": "x"}`, `{"f64": 1.5}`, ...).
///
/// Responses render these untagged (plain JSON scalars/arrays; `DateTime` as RFC 3339).
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    Str(String),
    /// Unix milliseconds. Stored as a real datetime; read back as an RFC 3339 string.
    DateTime(i64),
    F32Vec(Vec<f32>),
    F64Vec(Vec<f64>),
    StrVec(Vec<String>),
}

impl Val {
    pub fn to_json(&self) -> Json {
        match self {
            Val::Null => json!("null"),
            Val::Bool(b) => json!({ "bool": b }),
            Val::I64(i) => json!({ "i64": i }),
            Val::F64(f) => json!({ "f64": f }),
            Val::Str(s) => json!({ "string": s }),
            Val::DateTime(ms) => json!({ "date_time": ms }),
            Val::F32Vec(v) => json!({ "f32_array": v }),
            Val::F64Vec(v) => json!({ "f64_array": v }),
            Val::StrVec(v) => json!({ "string_array": v }),
        }
    }
}

macro_rules! val_from {
    ($($t:ty => $f:expr),* $(,)?) => { $(impl From<$t> for Val { fn from(v: $t) -> Self { $f(v) } })* };
}
val_from! {
    bool => Val::Bool,
    i64 => Val::I64,
    i32 => |v: i32| Val::I64(v.into()),
    u32 => |v: u32| Val::I64(v.into()),
    f64 => Val::F64,
    String => Val::Str,
    &str => |v: &str| Val::Str(v.to_owned()),
    Vec<f32> => Val::F32Vec,
    Vec<f64> => Val::F64Vec,
    Vec<String> => Val::StrVec,
}

/// A property input: a literal, a request parameter, or the server clock.
#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    Lit(Val),
    /// Reference to an entry in the request's `parameters` map.
    Param(String),
    /// Server-side `DateTimeNow` (what `DEFAULT NOW` meant in `schema.hx`).
    Now,
}

impl Input {
    fn to_json(&self) -> Json {
        match self {
            Input::Lit(v) => json!({ "value": v.to_json() }),
            Input::Param(p) => json!({ "expr": { "param": p } }),
            Input::Now => json!({ "expr": "date_time_now" }),
        }
    }

    /// The same input in expression position (predicates, bounds).
    fn to_expr(&self) -> Json {
        match self {
            Input::Lit(v) => json!({ "constant": v.to_json() }),
            Input::Param(p) => json!({ "param": p }),
            Input::Now => json!("date_time_now"),
        }
    }
}

impl<T: Into<Val>> From<T> for Input {
    fn from(v: T) -> Self {
        Input::Lit(v.into())
    }
}

/// Shorthand for [`Input::Param`].
pub fn param(name: impl Into<String>) -> Input {
    Input::Param(name.into())
}

fn prop_expr(property: &str) -> Json {
    json!({ "property": property })
}

/// A filter predicate over the current element's properties (`$label` and `$id` are virtual).
#[derive(Debug, Clone, PartialEq)]
pub struct Pred(Json);

impl Pred {
    fn cmp(op: &str, property: &str, value: Input) -> Self {
        Pred(json!({ op: { "left": prop_expr(property), "right": value.to_expr() } }))
    }
    pub fn eq(property: &str, value: impl Into<Input>) -> Self {
        Self::cmp("eq", property, value.into())
    }
    pub fn neq(property: &str, value: impl Into<Input>) -> Self {
        Self::cmp("neq", property, value.into())
    }
    pub fn gt(property: &str, value: impl Into<Input>) -> Self {
        Self::cmp("gt", property, value.into())
    }
    pub fn gte(property: &str, value: impl Into<Input>) -> Self {
        Self::cmp("gte", property, value.into())
    }
    pub fn lt(property: &str, value: impl Into<Input>) -> Self {
        Self::cmp("lt", property, value.into())
    }
    pub fn lte(property: &str, value: impl Into<Input>) -> Self {
        Self::cmp("lte", property, value.into())
    }
    pub fn between(property: &str, min: impl Into<Input>, max: impl Into<Input>) -> Self {
        Pred(
            json!({ "between": { "value": prop_expr(property), "min": min.into().to_expr(), "max": max.into().to_expr() } }),
        )
    }
    /// `$label == label`; how the SDKs scope a scan to one node/edge type.
    pub fn label(label: &str) -> Self {
        Self::eq("$label", label)
    }
    pub fn has_key(property: &str) -> Self {
        Pred(json!({ "has_key": { "property": property } }))
    }
    pub fn starts_with(property: &str, prefix: &str) -> Self {
        Pred(json!({ "starts_with": { "value": prop_expr(property), "prefix": Input::from(prefix).to_expr() } }))
    }
    pub fn and(preds: impl IntoIterator<Item = Pred>) -> Self {
        Pred(json!({ "and": { "predicates": preds.into_iter().map(|p| p.0).collect::<Vec<_>>() } }))
    }
    pub fn or(preds: impl IntoIterator<Item = Pred>) -> Self {
        Pred(json!({ "or": { "predicates": preds.into_iter().map(|p| p.0).collect::<Vec<_>>() } }))
    }
    pub fn negate(pred: Pred) -> Self {
        Pred(json!({ "not": { "predicate": pred.0 } }))
    }
    pub fn into_json(self) -> Json {
        self.0
    }
}

/// How to reference existing nodes: everything, explicit ids, or a batch variable.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeRef {
    All,
    Ids(Vec<u64>),
    Var(String),
    Param(String),
}

impl NodeRef {
    pub fn id(id: u64) -> Self {
        NodeRef::Ids(vec![id])
    }
    pub fn var(name: impl Into<String>) -> Self {
        NodeRef::Var(name.into())
    }
    fn to_json(&self) -> Json {
        match self {
            NodeRef::All => json!("all"),
            NodeRef::Ids(ids) => json!({ "ids": ids }),
            NodeRef::Var(v) => json!({ "var": v }),
            NodeRef::Param(p) => json!({ "param": p }),
        }
    }
}

impl From<u64> for NodeRef {
    fn from(id: u64) -> Self {
        NodeRef::id(id)
    }
}

/// Edge references mirror [`NodeRef`].
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeRef {
    All,
    Ids(Vec<u64>),
    Var(String),
}

impl EdgeRef {
    fn to_json(&self) -> Json {
        match self {
            EdgeRef::All => json!("all"),
            EdgeRef::Ids(ids) => json!({ "ids": ids }),
            EdgeRef::Var(v) => json!({ "var": v }),
        }
    }
}

/// Vector distance metric of a vector index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Cosine,
    Euclidean,
    Manhattan,
}

impl Metric {
    fn as_str(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
            Metric::Manhattan => "manhattan",
        }
    }
}

/// An index definition. The server keys indexes by `(label, property)`; re-creating with a
/// different definition fails with `index_definition_conflict`.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexSpec(Json);

impl IndexSpec {
    pub fn node_equality(label: &str, property: &str, unique: bool) -> Self {
        IndexSpec(json!({ "node_equality": { "label": label, "property": property, "unique": unique } }))
    }
    pub fn node_range(label: &str, property: &str) -> Self {
        IndexSpec(json!({ "node_range": { "label": label, "property": property } }))
    }
    pub fn edge_equality(label: &str, property: &str) -> Self {
        IndexSpec(json!({ "edge_equality": { "label": label, "property": property } }))
    }
    pub fn edge_range(label: &str, property: &str) -> Self {
        IndexSpec(json!({ "edge_range": { "label": label, "property": property } }))
    }
    /// Vector index over a node property. `tenant_property` partitions the index: searches
    /// then require a `tenant_value`, and writes with the vector must carry the tenant property.
    pub fn node_vector(
        label: &str,
        property: &str,
        dimension: usize,
        metric: Metric,
        tenant_property: Option<&str>,
    ) -> Self {
        let mut spec =
            json!({ "label": label, "property": property, "dimension": dimension, "metric": metric.as_str() });
        if let Some(t) = tenant_property {
            spec["tenant_property"] = json!(t);
        }
        IndexSpec(json!({ "node_vector": spec }))
    }
    pub fn node_text(label: &str, property: &str, tenant_property: Option<&str>) -> Self {
        let mut spec = json!({ "label": label, "property": property });
        if let Some(t) = tenant_property {
            spec["tenant_property"] = json!(t);
        }
        IndexSpec(json!({ "node_text": spec }))
    }
    pub fn into_json(self) -> Json {
        self.0
    }
}

fn props_json(props: Vec<(&str, Input)>) -> Json {
    Json::Array(props.into_iter().map(|(k, v)| json!([k, v.to_json()])).collect())
}

/// One traversal / mutation tree. Build from a source (`Op::nodes*`, `Op::vector_search_nodes`,
/// `Op::add_n`, `Op::create_index`, ...) and chain steps.
#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    tree: Json,
    write: bool,
}

impl Op {
    fn new(tree: Json, write: bool) -> Self {
        Op { tree, write }
    }

    fn step(self, name: &str, mut fields: serde_json::Map<String, Json>, write: bool) -> Self {
        fields.insert("input".into(), self.tree);
        Op { tree: json!({ name: Json::Object(fields) }), write: self.write || write }
    }

    fn step0(self, name: &str) -> Self {
        self.step(name, serde_json::Map::new(), false)
    }

    fn obj(pairs: Vec<(&str, Json)>) -> serde_json::Map<String, Json> {
        pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()
    }

    /// Whether this tree contains a mutation (then it must run in a write batch).
    pub fn is_write(&self) -> bool {
        self.write
    }

    pub fn into_json(self) -> Json {
        self.tree
    }

    // ── sources ────────────────────────────────────────────────────────────

    pub fn nodes(r: impl Into<NodeRef>) -> Self {
        Self::new(json!({ "nodes": { "reference": r.into().to_json() } }), false)
    }
    pub fn nodes_where(pred: Pred) -> Self {
        Self::new(json!({ "nodes_where": { "predicate": pred.0 } }), false)
    }
    /// All nodes with the label (a `$label` predicate scan, as the SDKs do).
    pub fn nodes_with_label(label: &str) -> Self {
        Self::nodes_where(Pred::label(label))
    }
    pub fn nodes_with_label_where(label: &str, pred: Pred) -> Self {
        Self::nodes_where(Pred::and([Pred::label(label), pred]))
    }
    pub fn edges(r: EdgeRef) -> Self {
        Self::new(json!({ "edges": { "reference": r.to_json() } }), false)
    }
    pub fn edges_where(pred: Pred) -> Self {
        Self::new(json!({ "edges_where": { "predicate": pred.0 } }), false)
    }
    /// Top-k nearest nodes by the vector index on `label.property`. Hits carry `$distance`
    /// (ascending; smaller is closer) until a traversal step leaves the hit stream.
    pub fn vector_search_nodes(
        label: &str,
        property: &str,
        query: impl Into<Input>,
        k: usize,
        tenant: Option<Input>,
    ) -> Self {
        let mut fields = json!({ "label": label, "property": property, "query_vector": query.into().to_json(), "k": { "literal": k } });
        if let Some(t) = tenant {
            fields["tenant_value"] = t.to_json();
        }
        Self::new(json!({ "vector_search_nodes": fields }), false)
    }
    /// Index DDL. Cannot share a write batch with `add_n`/`add_e`/`set_property`/`drop`.
    pub fn create_index(spec: IndexSpec, if_not_exists: bool) -> Self {
        Self::new(json!({ "create_index": { "spec": spec.0, "if_not_exists": if_not_exists } }), true)
    }
    pub fn drop_index(spec: IndexSpec) -> Self {
        Self::new(json!({ "drop_index": { "spec": spec.0 } }), true)
    }
    /// Status of an asynchronous index operation (from a `create_index`/`drop_index` receipt).
    pub fn get_index_operation(operation_id: &str) -> Self {
        Self::new(json!({ "get_index_operation": { "operation_id": operation_id } }), false)
    }
    /// Insert a node. Returns the new node (`[{"$id": n}]`; chain `.value_map(None)` for properties).
    pub fn add_n(label: &str, props: Vec<(&str, Input)>) -> Self {
        Self::new(json!({ "add_n": { "label": label, "properties": props_json(props) } }), true)
    }

    // ── node steps ─────────────────────────────────────────────────────────

    fn labeled(self, name: &str, label: Option<&str>) -> Self {
        let mut f = serde_json::Map::new();
        if let Some(l) = label {
            f.insert("label".into(), json!(l));
        }
        self.step(name, f, false)
    }
    pub fn out(self, label: Option<&str>) -> Self {
        self.labeled("out", label)
    }
    pub fn r#in(self, label: Option<&str>) -> Self {
        self.labeled("in", label)
    }
    pub fn both(self, label: Option<&str>) -> Self {
        self.labeled("both", label)
    }
    pub fn out_e(self, label: Option<&str>) -> Self {
        self.labeled("out_e", label)
    }
    pub fn in_e(self, label: Option<&str>) -> Self {
        self.labeled("in_e", label)
    }
    pub fn both_e(self, label: Option<&str>) -> Self {
        self.labeled("both_e", label)
    }
    /// Edge → target node.
    pub fn out_n(self) -> Self {
        self.step0("out_n")
    }
    /// Edge → source node.
    pub fn in_n(self) -> Self {
        self.step0("in_n")
    }
    pub fn where_(self, pred: Pred) -> Self {
        self.step("where", Self::obj(vec![("predicate", pred.0)]), false)
    }
    pub fn has_label(self, label: &str) -> Self {
        self.step("has_label", Self::obj(vec![("label", json!(label))]), false)
    }
    pub fn dedup(self) -> Self {
        self.step0("dedup")
    }
    pub fn limit(self, n: usize) -> Self {
        self.step("limit", Self::obj(vec![("count", json!({ "literal": n }))]), false)
    }
    pub fn skip(self, n: usize) -> Self {
        self.step("skip", Self::obj(vec![("count", json!({ "literal": n }))]), false)
    }
    pub fn order_by(self, property: &str, descending: bool) -> Self {
        let order = if descending { "desc" } else { "asc" };
        self.step("order_by", Self::obj(vec![("property", json!(property)), ("order", json!(order))]), false)
    }

    // ── mutations ──────────────────────────────────────────────────────────

    /// Add an edge from every node in the current stream to `to`.
    pub fn add_e(self, label: &str, to: impl Into<NodeRef>, props: Vec<(&str, Input)>) -> Self {
        let f =
            Self::obj(vec![("label", json!(label)), ("to", to.into().to_json()), ("properties", props_json(props))]);
        self.step("add_e", f, true)
    }
    pub fn set_property(self, name: &str, value: impl Into<Input>) -> Self {
        self.step("set_property", Self::obj(vec![("name", json!(name)), ("value", value.into().to_json())]), true)
    }
    pub fn remove_property(self, name: &str) -> Self {
        self.step("remove_property", Self::obj(vec![("name", json!(name))]), true)
    }
    /// Delete the current edges, or nodes with their incident edges. Yields an empty stream.
    pub fn drop(self) -> Self {
        self.step("drop", serde_json::Map::new(), true)
    }

    // ── terminals ──────────────────────────────────────────────────────────

    pub fn count(self) -> Self {
        self.step0("count")
    }
    pub fn exists(self) -> Self {
        self.step0("exists")
    }
    /// Properties of each element as an object (`$id`, `$label`, plus `$distance` on vector
    /// hits). `None` returns all properties.
    pub fn value_map(self, properties: Option<&[&str]>) -> Self {
        let mut f = serde_json::Map::new();
        if let Some(p) = properties {
            f.insert("properties".into(), json!(p));
        }
        self.step("value_map", f, false)
    }
    /// Edge rows with `$id`, `$label`, `$from`, `$to` and properties.
    pub fn edge_properties(self) -> Self {
        self.step0("edge_properties")
    }
}

/// Runs an entry only when an earlier variable satisfies the condition.
#[derive(Debug, Clone, PartialEq)]
pub enum Cond {
    VarNotEmpty(String),
    VarEmpty(String),
    VarMinSize(String, usize),
    PrevNotEmpty,
}

impl Cond {
    fn to_json(&self) -> Json {
        match self {
            Cond::VarNotEmpty(v) => json!({ "var_not_empty": v }),
            Cond::VarEmpty(v) => json!({ "var_empty": v }),
            Cond::VarMinSize(v, n) => json!({ "var_min_size": [v, n] }),
            Cond::PrevNotEmpty => json!("prev_not_empty"),
        }
    }
}

/// Parameter type hints (`parameter_types`). Untyped parameters are accepted as plain JSON;
/// typed ones are validated/coerced by the server (note: a typed `date_time` stays a string).
#[derive(Debug, Clone, PartialEq)]
pub enum ParamType {
    Bool,
    I64,
    F64,
    F32,
    String,
    DateTime,
    Array(Box<ParamType>),
}

impl ParamType {
    fn to_json(&self) -> Json {
        match self {
            ParamType::Bool => json!("bool"),
            ParamType::I64 => json!("i64"),
            ParamType::F64 => json!("f64"),
            ParamType::F32 => json!("f32"),
            ParamType::String => json!("string"),
            ParamType::DateTime => json!("date_time"),
            ParamType::Array(t) => json!({ "array": t.to_json() }),
        }
    }
}

/// A `/v2/query` request body.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    write: bool,
    entries: Vec<Json>,
    returns: Vec<String>,
    params: BTreeMap<String, Json>,
    param_types: BTreeMap<String, Json>,
    name: Option<String>,
}

impl Batch {
    pub fn read() -> Self {
        Self {
            write: false,
            entries: vec![],
            returns: vec![],
            params: BTreeMap::new(),
            param_types: BTreeMap::new(),
            name: None,
        }
    }
    pub fn write() -> Self {
        Self { write: true, ..Self::read() }
    }
    /// Read or write batch depending on whether `op` mutates.
    pub fn for_op(op: &Op) -> Self {
        if op.is_write() { Self::write() } else { Self::read() }
    }
    pub fn is_write(&self) -> bool {
        self.write
    }
    /// Optional `query_name` (shows up in server metrics/logs).
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    /// Add a named entry. Later entries can reference it via `NodeRef::var(name)`.
    pub fn var(mut self, name: &str, op: Op) -> Self {
        self.entries.push(json!({ "query": { "name": name, "root": op.tree } }));
        self
    }
    pub fn var_if(mut self, name: &str, cond: Cond, op: Op) -> Self {
        self.entries.push(json!({ "query": { "name": name, "root": op.tree, "condition": cond.to_json() } }));
        self
    }
    /// Which variables the response includes. Also marks them as returned by `var`.
    pub fn returning<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        self.returns.extend(names.into_iter().map(Into::into));
        self
    }
    /// Convenience: add an entry and return it.
    pub fn ret(self, name: &str, op: Op) -> Self {
        self.var(name, op).returning([name])
    }
    pub fn param(mut self, name: &str, value: impl Into<Json>, ty: Option<ParamType>) -> Self {
        self.params.insert(name.to_owned(), value.into());
        if let Some(t) = ty {
            self.param_types.insert(name.to_owned(), t.to_json());
        }
        self
    }
    pub fn returns(&self) -> &[String] {
        &self.returns
    }

    pub fn to_json(&self) -> Json {
        let kind = if self.write { "write" } else { "read" };
        let mut body = json!({
            "request_type": kind,
            "query_name": self.name,
            "query": { kind: { "entries": self.entries, "returns": self.returns } },
        });
        if !self.params.is_empty() {
            body["parameters"] = json!(self.params);
            if !self.param_types.is_empty() {
                body["parameter_types"] = json!(self.param_types);
            }
        }
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_tagged() {
        assert_eq!(Val::from("x").to_json(), json!({ "string": "x" }));
        assert_eq!(Val::from(3u32).to_json(), json!({ "i64": 3 }));
        assert_eq!(Val::from(0.5).to_json(), json!({ "f64": 0.5 }));
        assert_eq!(Val::DateTime(7).to_json(), json!({ "date_time": 7 }));
        assert_eq!(Val::from(vec![1.0f32, 0.0]).to_json(), json!({ "f32_array": [1.0, 0.0] }));
    }

    #[test]
    fn add_n_shape() {
        let op =
            Op::add_n("Snapshot", vec![("snapshot_id", "s1".into()), ("created_at", Input::Now), ("gen", param("g"))]);
        assert!(op.is_write());
        assert_eq!(
            op.into_json(),
            json!({ "add_n": { "label": "Snapshot", "properties": [
                ["snapshot_id", { "value": { "string": "s1" } }],
                ["created_at", { "expr": "date_time_now" }],
                ["gen", { "expr": { "param": "g" } }],
            ] } })
        );
    }

    #[test]
    fn label_scan_with_filter_and_value_map() {
        let op =
            Op::nodes_with_label_where("Snapshot", Pred::eq("snapshot_id", "s1")).value_map(Some(&["$id", "status"]));
        assert!(!op.is_write());
        assert_eq!(
            op.into_json(),
            json!({ "value_map": {
                "input": { "nodes_where": { "predicate": { "and": { "predicates": [
                    { "eq": { "left": { "property": "$label" }, "right": { "constant": { "string": "Snapshot" } } } },
                    { "eq": { "left": { "property": "snapshot_id" }, "right": { "constant": { "string": "s1" } } } },
                ] } } } },
                "properties": ["$id", "status"],
            } })
        );
    }

    #[test]
    fn traversal_and_edge_shapes() {
        let op = Op::nodes(NodeRef::var("src")).out_e(Some("Transition")).edge_properties();
        assert_eq!(
            op.into_json(),
            json!({ "edge_properties": { "input": { "out_e": { "input": { "nodes": { "reference": { "var": "src" } } }, "label": "Transition" } } } })
        );
        let e = Op::nodes(4u64).add_e("InSpace", 9u64, vec![]);
        assert!(e.is_write());
        assert_eq!(
            e.into_json(),
            json!({ "add_e": { "input": { "nodes": { "reference": { "ids": [4] } } }, "label": "InSpace", "to": { "ids": [9] }, "properties": [] } })
        );
    }

    #[test]
    fn vector_search_and_index_shapes() {
        let op = Op::vector_search_nodes("LatentVec", "vec3", vec![1.0f32, 0.0, 0.0], 5, Some("sp1".into()));
        assert_eq!(
            op.into_json(),
            json!({ "vector_search_nodes": { "label": "LatentVec", "property": "vec3", "k": { "literal": 5 },
                "query_vector": { "value": { "f32_array": [1.0, 0.0, 0.0] } }, "tenant_value": { "value": { "string": "sp1" } } } })
        );
        let idx = Op::create_index(
            IndexSpec::node_vector("LatentVec", "vec3", 3, Metric::Cosine, Some("expert_space_id")),
            true,
        );
        assert!(idx.is_write());
        assert_eq!(
            idx.into_json(),
            json!({ "create_index": { "if_not_exists": true, "spec": { "node_vector": {
                "label": "LatentVec", "property": "vec3", "dimension": 3, "metric": "cosine", "tenant_property": "expert_space_id" } } } })
        );
    }

    #[test]
    fn batch_envelope() {
        let b = Batch::write()
            .var("existing", Op::nodes_with_label_where("Snapshot", Pred::eq("snapshot_id", param("sid"))))
            .var_if(
                "created",
                Cond::VarEmpty("existing".into()),
                Op::add_n("Snapshot", vec![("snapshot_id", param("sid"))]),
            )
            .returning(["existing", "created"])
            .param("sid", "s1", Some(ParamType::String));
        let j = b.to_json();
        assert_eq!(j["request_type"], "write");
        assert_eq!(j["query"]["write"]["returns"], json!(["existing", "created"]));
        assert_eq!(j["query"]["write"]["entries"][1]["query"]["condition"], json!({ "var_empty": "existing" }));
        assert_eq!(j["parameters"], json!({ "sid": "s1" }));
        assert_eq!(j["parameter_types"], json!({ "sid": "string" }));
        assert!(j.get("search_consistency").is_none());

        let r = Batch::read().ret("n", Op::nodes(NodeRef::All).count()).to_json();
        assert_eq!(r["request_type"], "read");
        assert_eq!(
            r["query"]["read"]["entries"][0]["query"]["root"],
            json!({ "count": { "input": { "nodes": { "reference": "all" } } } })
        );
        assert!(r.get("parameters").is_none());
    }
}
