//! Reference mirror (spec 12 §5, spec 20 §5.1): the Postgres `ref` graph copied into Helix as
//! `Instrument`, `Entity`, `Venue` and `Sector` nodes with `IssuedBy`, `ListedOn`,
//! `SameExposure` and `InSector` (entity → sector, from `ref.entity.sector`) edges.
//! Postgres stays the source of truth; control-api (`ref_mirror.rs`) runs a full sync at start
//! and periodically, and an incremental sync on `ref.changed.*`.
//!
//! Business ids are the Postgres UUIDs (`instrument_id`, `entity_id`, `venue_id`). Every write is
//! idempotent: nodes are upserted by their unique key (properties overwritten, `mirrored_at`
//! stamped), and the out-edges of an instrument are made equal to the given set — edges are
//! upserted by a key stored on the edge (`ListedOn.listing_key = <venue_id>|<venue_symbol>`,
//! `IssuedBy.entity_id`, `SameExposure.exposure_key = <to_instrument_id>|<relation kind>`) and
//! out-edges not in the set are dropped. An instrument that left Postgres (or its validity) is
//! kept as a node with `status = inactive` and no out-edges ([`HubClient::retire_ref_instruments`]).
//!
//! Nodes are written in one batch per chunk before any edge batch, so an edge never depends on
//! a node created in the same transaction; an edge whose target node is missing is skipped by
//! the server-side probe.

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::{
    Batch, Cond, EdgeRow, HubClient, Input, NodeRef, Op, Pred, Result, Val,
    regions::edge_upsert,
    schema::{edge, node},
};

/// Rows per write batch (each row is a handful of entries).
pub const CHUNK: usize = 25;

/// Status given to a mirrored instrument that is no longer current in Postgres.
pub const INACTIVE: &str = "inactive";

/// `ref.entity` row.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefEntity {
    pub entity_id: String,
    /// Empty when unknown.
    pub lei: String,
    pub name: String,
    pub country: String,
    pub sector: String,
}

/// `Sector.sector_id` of a sector name: lowercase ASCII alphanumerics joined by `_`
/// ("Information Technology" → `information_technology`); empty for an empty name.
pub fn sector_id(name: &str) -> String {
    name.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>()
        .join("_")
}

/// `ref.venue` row (`venue_id` = UUID, `code` = the subject token).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefVenue {
    pub venue_id: String,
    pub code: String,
    /// Empty when the venue has no MIC.
    pub mic: String,
    pub name: String,
    pub kind: String,
    pub timezone: String,
}

/// A current `ref.listing` of an instrument.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefListing {
    pub venue_id: String,
    pub venue_symbol: String,
    /// Feeding source (`ref.listing.source_id`), empty when none.
    pub source_id: String,
}

impl RefListing {
    pub fn key(&self) -> String {
        format!("{}|{}", self.venue_id, self.venue_symbol)
    }
}

/// A current `ref.instrument_relation` of kind `same_economic_exposure` (from this instrument).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefExposure {
    pub to_instrument_id: String,
    /// `<from kind>-<to kind>`, e.g. `crypto_spot-perp` (the `schema.hx` "spot-perp" example).
    pub relation: String,
}

impl RefExposure {
    pub fn key(&self) -> String {
        format!("{}|{}", self.to_instrument_id, self.relation)
    }
}

/// `ref.instrument` row with its current out-edges.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefInstrument {
    pub instrument_id: String,
    pub asset_class: String,
    pub kind: String,
    pub name: String,
    /// `active | provisional | merged | …`
    pub status: String,
    pub base_ccy: String,
    pub quote_ccy: String,
    pub issuer_id: Option<String>,
    pub listings: Vec<RefListing>,
    pub exposures: Vec<RefExposure>,
}

/// What one mirror run wrote.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RefSyncReport {
    pub entities: usize,
    pub venues: usize,
    pub instruments: usize,
    pub issued_by: usize,
    pub listed_on: usize,
    pub same_exposure: usize,
    /// Distinct `Sector` nodes and `InSector` edges written.
    #[serde(default)]
    pub sectors: usize,
    #[serde(default)]
    pub in_sector: usize,
    /// Instruments marked `inactive` (absent from Postgres).
    pub retired: usize,
    pub batches: usize,
}

/// An instrument with its issuer, listings (edge rows + venues) and exposures
/// (`instrumentReference`, catalog read).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentReference {
    pub instrument: Json,
    pub issuer: Vec<Json>,
    pub listings: Vec<EdgeRow>,
    pub venues: Vec<Json>,
    pub exposures: Vec<EdgeRow>,
    /// The issuer's sector (`IssuedBy` → `InSector`).
    #[serde(default)]
    pub sectors: Vec<Json>,
}

fn s(v: &str) -> Input {
    Input::Lit(Val::Str(v.to_owned()))
}

fn by_key(label: &str, key: &str, value: &str) -> Op {
    Op::nodes_with_label_where(label, Pred::eq(key, value))
}

/// Node upsert by unique key as entries `ex{i}` / `up{i}` / `cr{i}` (prefixed with `tag`).
fn node_upsert_i(
    batch: Batch,
    tag: &str,
    i: usize,
    label: &str,
    key: &str,
    value: &str,
    props: Vec<(&str, Input)>,
) -> Batch {
    let (ex, up, cr) = (format!("{tag}ex{i}"), format!("{tag}up{i}"), format!("{tag}cr{i}"));
    let mut set = Op::nodes(NodeRef::var(&ex));
    for (k, v) in props.clone() {
        set = set.set_property(k, v);
    }
    set = set.set_property("mirrored_at", Input::Now);
    let mut create = vec![(key, s(value))];
    create.extend(props);
    create.push(("mirrored_at", Input::Now));
    batch.var(&ex, by_key(label, key, value)).var_if(&up, Cond::VarNotEmpty(ex.clone()), set).var_if(
        &cr,
        Cond::VarEmpty(ex),
        Op::add_n(label, create),
    )
}

/// Drops the `label` out-edges of the instrument in `inst` whose `key` is not in `keep`.
fn prune_out_edges(batch: Batch, name: &str, inst: &str, label: &str, key: &str, keep: &[String]) -> Batch {
    let edges = Op::nodes(NodeRef::var(inst)).out_e(Some(label));
    let stale = if keep.is_empty() {
        edges
    } else {
        edges.where_(Pred::negate(Pred::or(keep.iter().map(|k| Pred::eq(key, k.as_str())))))
    };
    batch.var(name, stale.drop())
}

fn entity_props(e: &RefEntity) -> Vec<(&'static str, Input)> {
    vec![("lei", s(&e.lei)), ("name", s(&e.name)), ("country", s(&e.country)), ("sector", s(&e.sector))]
}

fn venue_props(v: &RefVenue) -> Vec<(&'static str, Input)> {
    vec![
        ("code", s(&v.code)),
        ("mic", s(&v.mic)),
        ("name", s(&v.name)),
        ("kind", s(&v.kind)),
        ("timezone", s(&v.timezone)),
    ]
}

fn instrument_props(i: &RefInstrument) -> Vec<(&'static str, Input)> {
    vec![
        ("asset_class", s(&i.asset_class)),
        ("kind", s(&i.kind)),
        ("name", s(&i.name)),
        ("status", s(&i.status)),
        ("base_ccy", s(&i.base_ccy)),
        ("quote_ccy", s(&i.quote_ccy)),
    ]
}

/// The node-upsert batch of one chunk of entities, venues or instruments.
pub fn entity_batch(rows: &[RefEntity]) -> Batch {
    rows.iter().enumerate().fold(Batch::write().named("upsert_ref_entities"), |b, (i, e)| {
        node_upsert_i(b, "e", i, node::ENTITY, "entity_id", &e.entity_id, entity_props(e))
    })
}

pub fn venue_batch(rows: &[RefVenue]) -> Batch {
    rows.iter().enumerate().fold(Batch::write().named("upsert_ref_venues"), |b, (i, v)| {
        node_upsert_i(b, "v", i, node::VENUE, "venue_id", &v.venue_id, venue_props(v))
    })
}

pub fn instrument_batch(rows: &[RefInstrument]) -> Batch {
    rows.iter().enumerate().fold(Batch::write().named("upsert_ref_instruments"), |b, (i, r)| {
        node_upsert_i(b, "i", i, node::INSTRUMENT, "instrument_id", &r.instrument_id, instrument_props(r))
    })
}

/// The distinct sectors of a set of entities as `(sector_id, name)`, first name per id wins.
pub fn sectors_of(rows: &[RefEntity]) -> Vec<(String, String)> {
    let mut seen = std::collections::BTreeMap::new();
    for e in rows {
        let id = sector_id(&e.sector);
        if !id.is_empty() {
            seen.entry(id).or_insert_with(|| e.sector.trim().to_owned());
        }
    }
    seen.into_iter().collect()
}

/// The node-upsert batch of one chunk of sectors `(sector_id, name)`.
pub fn sector_batch(rows: &[(String, String)]) -> Batch {
    rows.iter().enumerate().fold(Batch::write().named("upsert_ref_sectors"), |b, (i, (id, name))| {
        node_upsert_i(b, "s", i, node::SECTOR, "sector_id", id, vec![("name", s(name))])
    })
}

/// The `InSector` edge batch of one chunk of entities: each entity's out-edge made equal to its
/// current sector (none when the sector is empty).
pub fn entity_edge_batch(rows: &[RefEntity]) -> Batch {
    let mut b = Batch::write().named("sync_ref_sector_edges");
    for (i, e) in rows.iter().enumerate() {
        let ent = format!("ent{i}");
        b = b.var(&ent, by_key(node::ENTITY, "entity_id", &e.entity_id));
        let sid = sector_id(&e.sector);
        let keep: Vec<String> = if sid.is_empty() { Vec::new() } else { vec![sid.clone()] };
        b = prune_out_edges(b, &format!("px_sec{i}"), &ent, edge::IN_SECTOR, "sector_id", &keep);
        if !sid.is_empty() {
            b = edge_upsert(
                b,
                i,
                edge::IN_SECTOR,
                Op::nodes(NodeRef::var(&ent)),
                by_key(node::SECTOR, "sector_id", &sid),
                Pred::eq("sector_id", sid.as_str()),
                vec![("entity_id", s(&e.entity_id)), ("sector_id", s(&sid))],
                vec![("mirrored_at", Input::Now)],
            );
        }
    }
    b.returning(Vec::<String>::new())
}

/// The edge batch of one chunk of instruments: prune stale out-edges, upsert the current ones.
pub fn instrument_edge_batch(rows: &[RefInstrument]) -> Batch {
    let mut b = Batch::write().named("sync_ref_edges");
    let mut k = 0usize; // edge_upsert entry index, unique across the batch
    for (i, r) in rows.iter().enumerate() {
        let inst = format!("inst{i}");
        b = b.var(&inst, by_key(node::INSTRUMENT, "instrument_id", &r.instrument_id));
        // IssuedBy
        let issuer: Vec<String> = r.issuer_id.iter().cloned().collect();
        b = prune_out_edges(b, &format!("px_iss{i}"), &inst, edge::ISSUED_BY, "entity_id", &issuer);
        if let Some(e) = &r.issuer_id {
            b = edge_upsert(
                b,
                k,
                edge::ISSUED_BY,
                Op::nodes(NodeRef::var(&inst)),
                by_key(node::ENTITY, "entity_id", e),
                Pred::eq("entity_id", e.as_str()),
                vec![("instrument_id", s(&r.instrument_id)), ("entity_id", s(e))],
                vec![("mirrored_at", Input::Now)],
            );
            k += 1;
        }
        // ListedOn
        let keys: Vec<String> = r.listings.iter().map(RefListing::key).collect();
        b = prune_out_edges(b, &format!("px_lst{i}"), &inst, edge::LISTED_ON, "listing_key", &keys);
        for l in &r.listings {
            b = edge_upsert(
                b,
                k,
                edge::LISTED_ON,
                Op::nodes(NodeRef::var(&inst)),
                by_key(node::VENUE, "venue_id", &l.venue_id),
                Pred::eq("listing_key", l.key().as_str()),
                vec![
                    ("listing_key", s(&l.key())),
                    ("instrument_id", s(&r.instrument_id)),
                    ("venue_id", s(&l.venue_id)),
                    ("venue_symbol", s(&l.venue_symbol)),
                ],
                vec![("source_id", s(&l.source_id)), ("mirrored_at", Input::Now)],
            );
            k += 1;
        }
        // SameExposure
        let keys: Vec<String> = r.exposures.iter().map(RefExposure::key).collect();
        b = prune_out_edges(b, &format!("px_exp{i}"), &inst, edge::SAME_EXPOSURE, "exposure_key", &keys);
        for x in &r.exposures {
            b = edge_upsert(
                b,
                k,
                edge::SAME_EXPOSURE,
                Op::nodes(NodeRef::var(&inst)),
                by_key(node::INSTRUMENT, "instrument_id", &x.to_instrument_id),
                Pred::eq("exposure_key", x.key().as_str()),
                vec![
                    ("exposure_key", s(&x.key())),
                    ("instrument_id", s(&r.instrument_id)),
                    ("to_instrument_id", s(&x.to_instrument_id)),
                    ("relation", s(&x.relation)),
                ],
                vec![("mirrored_at", Input::Now)],
            );
            k += 1;
        }
    }
    b.returning(Vec::<String>::new())
}

/// Marks instruments `inactive` and drops their reference out-edges; unknown ids are no-ops.
pub fn retire_batch(instrument_ids: &[String]) -> Batch {
    let mut b = Batch::write().named("retire_ref_instruments");
    for (i, id) in instrument_ids.iter().enumerate() {
        let inst = format!("inst{i}");
        b = b.var(&inst, by_key(node::INSTRUMENT, "instrument_id", id)).var(
            &format!("st{i}"),
            Op::nodes(NodeRef::var(&inst)).set_property("status", INACTIVE).set_property("mirrored_at", Input::Now),
        );
        for (j, label) in [edge::ISSUED_BY, edge::LISTED_ON, edge::SAME_EXPOSURE].into_iter().enumerate() {
            b = b.var(&format!("px{i}_{j}"), Op::nodes(NodeRef::var(&inst)).out_e(Some(label)).drop());
        }
    }
    b.returning(Vec::<String>::new())
}

impl HubClient {
    /// `upsertRefEntities`: `Entity` nodes by `entity_id`, [`CHUNK`] per batch, then the
    /// `Sector` nodes of their sectors and each entity's `InSector` edge (made equal to its
    /// current sector).
    pub async fn upsert_ref_entities(&self, rows: &[RefEntity]) -> Result<usize> {
        for c in rows.chunks(CHUNK) {
            self.batch(&entity_batch(c).returning(Vec::<String>::new())).await?;
        }
        for c in sectors_of(rows).chunks(CHUNK) {
            self.batch(&sector_batch(c).returning(Vec::<String>::new())).await?;
        }
        for c in rows.chunks(CHUNK) {
            self.batch(&entity_edge_batch(c)).await?;
        }
        Ok(rows.len())
    }

    /// `upsertRefVenues`: `Venue` nodes by `venue_id`.
    pub async fn upsert_ref_venues(&self, rows: &[RefVenue]) -> Result<usize> {
        for c in rows.chunks(CHUNK) {
            self.batch(&venue_batch(c).returning(Vec::<String>::new())).await?;
        }
        Ok(rows.len())
    }

    /// `upsertRefInstruments`: `Instrument` nodes, then (after every node batch) their
    /// `IssuedBy` / `ListedOn` / `SameExposure` out-edges made equal to the given sets.
    pub async fn upsert_ref_instruments(&self, rows: &[RefInstrument]) -> Result<RefSyncReport> {
        let mut rep = RefSyncReport::default();
        for c in rows.chunks(CHUNK) {
            self.batch(&instrument_batch(c).returning(Vec::<String>::new())).await?;
            rep.batches += 1;
        }
        for c in rows.chunks(CHUNK) {
            self.batch(&instrument_edge_batch(c)).await?;
            rep.batches += 1;
        }
        rep.instruments = rows.len();
        rep.issued_by = rows.iter().filter(|r| r.issuer_id.is_some()).count();
        rep.listed_on = rows.iter().map(|r| r.listings.len()).sum();
        rep.same_exposure = rows.iter().map(|r| r.exposures.len()).sum();
        Ok(rep)
    }

    /// `retireRefInstruments`: `status = inactive`, reference out-edges dropped.
    pub async fn retire_ref_instruments(&self, instrument_ids: &[String]) -> Result<usize> {
        for c in instrument_ids.chunks(CHUNK) {
            self.batch(&retire_batch(c)).await?;
        }
        Ok(instrument_ids.len())
    }

    /// `refInstrumentIds`: `instrument_id` of every mirrored `Instrument` with its `status`.
    pub async fn ref_instrument_ids(&self) -> Result<Vec<(String, String)>> {
        #[derive(Deserialize)]
        struct Row {
            instrument_id: String,
            #[serde(default)]
            status: Option<String>,
        }
        let op = Op::nodes_with_label(node::INSTRUMENT).value_map(Some(&["instrument_id", "status"]));
        let mut resp = self.batch(&Batch::read().named("ref_instrument_ids").ret("rows", op)).await?;
        let rows: Vec<Row> = resp.take("rows")?;
        Ok(rows.into_iter().map(|r| (r.instrument_id, r.status.unwrap_or_default())).collect())
    }

    /// Mirrors one Postgres reference read. `entities` / `venues` first (edge targets), then the
    /// instruments with their edges. With `full`, mirrored instruments absent from `instruments`
    /// are retired (`inactive`); pass `full = false` for incremental syncs of a subset.
    pub async fn sync_reference(
        &self,
        entities: &[RefEntity],
        venues: &[RefVenue],
        instruments: &[RefInstrument],
        full: bool,
    ) -> Result<RefSyncReport> {
        self.upsert_ref_entities(entities).await?;
        self.upsert_ref_venues(venues).await?;
        let mut rep = self.upsert_ref_instruments(instruments).await?;
        rep.entities = entities.len();
        rep.venues = venues.len();
        rep.sectors = sectors_of(entities).len();
        rep.in_sector = entities.iter().filter(|e| !sector_id(&e.sector).is_empty()).count();
        rep.batches += entities.len().div_ceil(CHUNK) + venues.len().div_ceil(CHUNK);
        if full {
            let current: std::collections::HashSet<&str> =
                instruments.iter().map(|r| r.instrument_id.as_str()).collect();
            let gone: Vec<String> = self
                .ref_instrument_ids()
                .await?
                .into_iter()
                .filter(|(id, status)| !current.contains(id.as_str()) && status != INACTIVE)
                .map(|(id, _)| id)
                .collect();
            rep.retired = self.retire_ref_instruments(&gone).await?;
            rep.batches += gone.len().div_ceil(CHUNK);
        }
        Ok(rep)
    }

    /// `instrumentReference`: an instrument with its issuer, listings (+ venues) and exposures.
    pub async fn instrument_reference(&self, instrument_id: &str) -> Result<Option<InstrumentReference>> {
        let b = Batch::read()
            .named("instrument_reference")
            .var("i", by_key(node::INSTRUMENT, "instrument_id", instrument_id))
            .ret("instrument", Op::nodes(NodeRef::var("i")).value_map(None))
            .ret("issuer", Op::nodes(NodeRef::var("i")).out(Some(edge::ISSUED_BY)).value_map(None))
            .ret("listings", Op::nodes(NodeRef::var("i")).out_e(Some(edge::LISTED_ON)).edge_properties())
            .ret("venues", Op::nodes(NodeRef::var("i")).out(Some(edge::LISTED_ON)).value_map(None))
            .ret("exposures", Op::nodes(NodeRef::var("i")).out_e(Some(edge::SAME_EXPOSURE)).edge_properties())
            .ret(
                "sectors",
                Op::nodes(NodeRef::var("i")).out(Some(edge::ISSUED_BY)).out(Some(edge::IN_SECTOR)).value_map(None),
            );
        let mut resp = self.batch(&b).await?;
        let Some(instrument) = resp.take_opt::<Json>("instrument")? else { return Ok(None) };
        Ok(Some(InstrumentReference {
            instrument,
            issuer: resp.take("issuer")?,
            listings: resp.take("listings")?,
            venues: resp.take("venues")?,
            exposures: resp.take("exposures")?,
            sectors: resp.take("sectors")?,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn inst() -> RefInstrument {
        RefInstrument {
            instrument_id: "i1".into(),
            asset_class: "crypto".into(),
            kind: "crypto_spot".into(),
            name: "BTC-USD".into(),
            status: "active".into(),
            issuer_id: Some("e1".into()),
            listings: vec![RefListing {
                venue_id: "v1".into(),
                venue_symbol: "BTC-USD".into(),
                source_id: "yahoo".into(),
            }],
            exposures: vec![RefExposure { to_instrument_id: "i2".into(), relation: "crypto_spot-perp".into() }],
            ..Default::default()
        }
    }

    #[test]
    fn sectors_are_slugged_deduplicated_and_linked() {
        assert_eq!(sector_id(" Information Technology "), "information_technology");
        assert_eq!(sector_id("Banks & Insurance"), "banks_insurance");
        assert_eq!(sector_id(""), "");
        let ents = [
            RefEntity { entity_id: "e1".into(), sector: "Information Technology".into(), ..Default::default() },
            RefEntity { entity_id: "e2".into(), sector: "information technology".into(), ..Default::default() },
            RefEntity { entity_id: "e3".into(), ..Default::default() },
        ];
        assert_eq!(sectors_of(&ents), vec![("information_technology".into(), "Information Technology".into())]);
        let n = sector_batch(&sectors_of(&ents)).to_json().to_string();
        assert!(n.contains("\"Sector\"") && n.contains("information_technology"));
        let j = entity_edge_batch(&ents).to_json().to_string();
        for want in ["\"px_sec0\"", "\"px_sec2\"", "InSector", "\"Sector\""] {
            assert!(j.contains(want), "{want} missing in {j}");
        }
    }

    #[test]
    fn node_batches_are_conditional_upserts() {
        let j = instrument_batch(&[inst()]).to_json();
        let e = j["query"]["write"]["entries"].as_array().unwrap();
        assert_eq!(e.len(), 3);
        assert_eq!(e[1]["query"]["condition"], json!({ "var_not_empty": "iex0" }));
        assert_eq!(e[2]["query"]["condition"], json!({ "var_empty": "iex0" }));
        assert_eq!(e[2]["query"]["root"]["add_n"]["label"], "Instrument");
        let s = j.to_string();
        assert!(s.contains("mirrored_at") && s.contains("quote_ccy") && s.contains("\"status\""));
        let v = venue_batch(&[RefVenue { venue_id: "v1".into(), code: "xnys".into(), ..Default::default() }]).to_json();
        assert!(v.to_string().contains("\"Venue\"") && v.to_string().contains("xnys"));
        let en = entity_batch(&[RefEntity { entity_id: "e1".into(), ..Default::default() }]).to_json();
        assert!(en.to_string().contains("\"Entity\""));
    }

    #[test]
    fn edge_batch_prunes_then_upserts_every_edge_kind() {
        let j = instrument_edge_batch(&[inst()]).to_json().to_string();
        for want in [
            "\"px_iss0\"",
            "\"px_lst0\"",
            "\"px_exp0\"",
            "IssuedBy",
            "ListedOn",
            "SameExposure",
            "v1|BTC-USD",
            "i2|crypto_spot-perp",
            "\"cr0\"",
            "\"cr1\"",
            "\"cr2\"",
            "\"drop\"",
            "\"not\"",
        ] {
            assert!(j.contains(want), "{want} missing in {j}");
        }
        // no listings / issuer: everything of that kind is pruned, nothing upserted
        let bare = RefInstrument { instrument_id: "i3".into(), ..Default::default() };
        let j = instrument_edge_batch(&[bare]).to_json().to_string();
        assert!(!j.contains("\"cr0\"") && !j.contains("\"not\"") && j.contains("px_lst0"), "{j}");
    }

    #[test]
    fn retire_batch_marks_inactive_and_drops_out_edges() {
        let j = retire_batch(&["i1".into()]).to_json().to_string();
        assert!(j.contains(INACTIVE) && j.contains("px0_0") && j.contains("px0_2"), "{j}");
        assert_eq!(RefListing { venue_id: "v".into(), venue_symbol: "S".into(), ..Default::default() }.key(), "v|S");
    }
}
