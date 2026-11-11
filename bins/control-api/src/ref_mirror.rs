//! Postgres `ref` → Helix reference mirror (spec 12 §5 / §6 change propagation, spec 20 §5.1):
//! `Instrument`, `Entity`, `Venue` nodes with `IssuedBy`, `ListedOn` and `SameExposure` edges.
//!
//! control-api is the writer because it already holds both ends (the Postgres pool it migrates
//! and the Helix client of the graph routes), is the publisher of almost every `ref.changed.*`
//! notice (refdata overrides, quarantine resolution) and runs background tasks off the trading
//! path; the Go ingestor (the other `ref` writer) has no Helix client, and the risk-engine must not
//! take a Helix dependency for a mirror nothing trades on.
//!
//! - **Full sync** once Postgres and Helix are reachable, then every `FA_REF_MIRROR_FULL_SECS`
//!   (default 6 h, `0` = only at start): every current instrument, entity and venue
//!   (`fa_warehouse::refdata::mirror_rows`); mirrored instruments no longer current are marked
//!   `inactive` with their out-edges dropped.
//! - **Incremental sync** on `ref.changed.*` (core NATS, debounced [`DEBOUNCE`]): `instrument` /
//!   `listing` notices with UUID ids re-mirror just those instruments (ids gone from Postgres
//!   are retired); merges, quarantine resolutions, venue / entity / relation changes and
//!   unparseable notices trigger a full sync; corporate actions, identifiers, contract specs and
//!   calendars are not part of the mirror and are ignored. `catalog.changed.*` (source status) is
//!   not subscribed: no mirrored property depends on a source's status. Notices lost while
//!   control-api is down are covered by the start-up and periodic full syncs.
//! - Failures are retried every [`RETRY`] with the pending plan kept; counters
//!   `fa_ref_mirror_syncs_total` / `fa_ref_mirror_failures_total`, gauges
//!   `fa_ref_mirror_instruments` / `fa_ref_mirror_last_success_seconds`.

use std::{
    collections::BTreeSet,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use fa_bus::subjects;
use fa_hub::{HubClient, RefEntity, RefExposure, RefInstrument, RefListing, RefSyncReport, RefVenue};
use fa_proto::v1::RefChanged;
use fa_warehouse::refdata::{MirrorRows, mirror_rows};
use futures::StreamExt;
use sqlx::PgPool;
use tokio::time::Instant;
use uuid::Uuid;

use crate::state::AppState;

/// Quiet period after a `ref.changed.*` notice before syncing (bursts coalesce).
pub const DEBOUNCE: Duration = Duration::from_secs(2);
/// Wait before retrying a failed sync.
pub const RETRY: Duration = Duration::from_secs(30);
const FULL_SECS_ENV: &str = "FA_REF_MIRROR_FULL_SECS";
const DEFAULT_FULL_SECS: u64 = 6 * 3600;

/// What a set of notices asks the mirror to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    Ignore,
    Instruments(BTreeSet<Uuid>),
    Full,
}

impl Plan {
    /// Union of two plans (`Full` absorbs everything).
    pub fn merge(self, other: Plan) -> Plan {
        match (self, other) {
            (Plan::Full, _) | (_, Plan::Full) => Plan::Full,
            (Plan::Ignore, p) | (p, Plan::Ignore) => p,
            (Plan::Instruments(mut a), Plan::Instruments(b)) => {
                a.extend(b);
                Plan::Instruments(a)
            }
        }
    }
}

/// The plan for one `ref.changed.<entity>` message (`notice` = its decoded payload, if any).
pub fn plan_for(subject: &str, notice: Option<&RefChanged>) -> Plan {
    let entity = notice
        .map(|n| n.entity.as_str())
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| subject.strip_prefix("ref.changed.").unwrap_or(subject));
    match entity {
        // not mirrored to Helix
        "corporate_action" | "identifier" | "contract_spec" | "calendar" | "fx" | "fx_rate" => Plan::Ignore,
        "instrument" | "listing" => {
            let Some(n) = notice else { return Plan::Full };
            // a merge moves listings to another instrument the notice may not name
            if matches!(n.change.as_str(), "merged" | "quarantine_resolved") || n.attrs.contains_key("merged_into") {
                return Plan::Full;
            }
            let ids: Option<BTreeSet<Uuid>> = n.ids.iter().map(|s| Uuid::parse_str(s).ok()).collect();
            match ids {
                Some(ids) if !ids.is_empty() => Plan::Instruments(ids),
                _ => Plan::Full,
            }
        }
        // venue, entity, relation, unknown entities: cheap enough to resync everything
        _ => Plan::Full,
    }
}

fn opt(s: &Option<String>) -> String {
    s.clone().unwrap_or_default()
}

/// Postgres rows → fa-hub mirror rows (NULL → empty string, UUIDs as text).
pub fn to_hub(rows: &MirrorRows) -> (Vec<RefEntity>, Vec<RefVenue>, Vec<RefInstrument>) {
    let entities = rows
        .entities
        .iter()
        .map(|e| RefEntity {
            entity_id: e.id.to_string(),
            lei: opt(&e.lei),
            name: e.name.clone(),
            country: opt(&e.country),
            sector: opt(&e.sector),
        })
        .collect();
    let venues = rows
        .venues
        .iter()
        .map(|v| RefVenue {
            venue_id: v.id.to_string(),
            code: v.code.clone(),
            mic: opt(&v.mic),
            name: v.name.clone(),
            kind: v.kind.clone(),
            timezone: v.timezone.clone(),
        })
        .collect();
    let instruments = rows
        .instruments
        .iter()
        .map(|i| RefInstrument {
            instrument_id: i.id.to_string(),
            asset_class: i.asset_class.clone(),
            kind: i.kind.clone(),
            name: i.name.clone(),
            status: i.status.clone(),
            base_ccy: opt(&i.base_ccy),
            quote_ccy: opt(&i.quote_ccy),
            // an issuer that is not a current entity has no node to point at
            issuer_id: i.issuer_id.filter(|id| rows.entities.iter().any(|e| e.id == *id)).map(|id| id.to_string()),
            listings: i
                .listings
                .iter()
                .map(|l| RefListing {
                    venue_id: l.venue_id.to_string(),
                    venue_symbol: l.venue_symbol.clone(),
                    source_id: opt(&l.source_id),
                })
                .collect(),
            exposures: i
                .exposures
                .iter()
                .map(|x| RefExposure { to_instrument_id: x.to_id.to_string(), relation: x.relation.clone() })
                .collect(),
        })
        .collect();
    (entities, venues, instruments)
}

/// Runs one plan against Postgres and Helix.
pub async fn sync(pool: &PgPool, hub: &HubClient, plan: &Plan) -> anyhow::Result<RefSyncReport> {
    match plan {
        Plan::Ignore => Ok(RefSyncReport::default()),
        Plan::Full => {
            let rows = mirror_rows(pool, None).await?;
            let (e, v, i) = to_hub(&rows);
            Ok(hub.sync_reference(&e, &v, &i, true).await?)
        }
        Plan::Instruments(ids) => {
            let ids: Vec<Uuid> = ids.iter().copied().collect();
            let rows = mirror_rows(pool, Some(&ids)).await?;
            let (e, v, i) = to_hub(&rows);
            let mut rep = hub.sync_reference(&e, &v, &i, false).await?;
            let gone: Vec<String> =
                ids.iter().filter(|id| !rows.instruments.iter().any(|r| r.id == **id)).map(Uuid::to_string).collect();
            rep.retired = hub.retire_ref_instruments(&gone).await?;
            Ok(rep)
        }
    }
}

fn full_every() -> Option<Duration> {
    let secs = std::env::var(FULL_SECS_ENV).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(DEFAULT_FULL_SECS);
    (secs > 0).then(|| Duration::from_secs(secs))
}

/// Starts the mirror task (waits for Postgres and Helix; never fails the API).
pub fn spawn(state: &AppState) {
    let state = state.clone();
    tokio::spawn(async move { run(state).await });
}

async fn run(state: AppState) {
    let pool = loop {
        if let Some(p) = state.pg.get() {
            break p.clone();
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    };
    let hub = state.hub.clone();
    while let Err(e) = hub.ensure_schema_ready(Duration::from_secs(30)).await {
        tracing::warn!(error = %e, "Helix not ready; reference mirror waits");
        tokio::time::sleep(RETRY).await;
    }
    let syncs = fa_bus::metrics::counter("fa_ref_mirror_syncs_total", "Successful Postgres ref -> Helix mirror syncs");
    let failures = fa_bus::metrics::counter("fa_ref_mirror_failures_total", "Failed ref -> Helix mirror syncs");
    let mirrored = fa_bus::metrics::gauge("fa_ref_mirror_instruments", "Instruments written by the last mirror sync");
    let last_ok = fa_bus::metrics::gauge("fa_ref_mirror_last_success_seconds", "Unix time of the last mirror sync");

    let mut sub = match &state.nats {
        Some(nc) => match nc.subscribe(subjects::ALL_REF_CHANGED).await {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(error = %e, "ref.changed.> subscription failed; mirror runs full syncs only");
                None
            }
        },
        None => None,
    };
    let every = full_every();
    let mut pending = Plan::Full; // start-up full sync
    let mut due = Some(Instant::now());
    let mut next_full = every.map(|d| Instant::now() + d);
    loop {
        let wake = [due, next_full].into_iter().flatten().min();
        tokio::select! {
            msg = async { sub.as_mut()?.next().await }, if sub.is_some() => {
                let Some(msg) = msg else {
                    tracing::warn!("ref.changed.> subscription ended; mirror runs full syncs only");
                    sub = None;
                    continue;
                };
                let notice = fa_bus::ingest::ref_changed(&msg.payload);
                let plan = plan_for(msg.subject.as_str(), notice.as_ref());
                if plan != Plan::Ignore {
                    pending = std::mem::replace(&mut pending, Plan::Ignore).merge(plan);
                    due = Some(due.map_or(Instant::now() + DEBOUNCE, |d| d.max(Instant::now() + DEBOUNCE)));
                }
            }
            _ = async { tokio::time::sleep_until(wake.expect("guarded")).await }, if wake.is_some() => {
                let now = Instant::now();
                if next_full.is_some_and(|t| t <= now) {
                    pending = std::mem::replace(&mut pending, Plan::Ignore).merge(Plan::Full);
                    next_full = every.map(|d| now + d);
                    due = Some(now);
                }
                if due.is_none_or(|d| d > now) {
                    continue;
                }
                let plan = std::mem::replace(&mut pending, Plan::Ignore);
                match sync(&pool, &hub, &plan).await {
                    Ok(rep) => {
                        syncs.inc();
                        mirrored.set(rep.instruments as f64);
                        last_ok.set(SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64()));
                        tracing::info!(full = plan == Plan::Full, report = ?rep, "reference mirror synced to Helix");
                        due = None;
                    }
                    Err(e) => {
                        failures.inc();
                        tracing::warn!(error = %e, "reference mirror sync failed; retrying");
                        pending = plan.merge(std::mem::replace(&mut pending, Plan::Ignore));
                        due = Some(Instant::now() + RETRY);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_warehouse::refdata::{MirrorEntity, MirrorExposure, MirrorInstrument, MirrorListing, MirrorVenue};

    fn notice(entity: &str, ids: &[&str], change: &str) -> RefChanged {
        fa_bus::ingest::ref_changed_notice(entity, ids.iter().map(|s| s.to_string()).collect(), change, "ui")
    }

    #[test]
    fn notices_map_to_plans() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let p = plan_for("ref.changed.listing", Some(&notice("listing", &[&a.to_string()], "symbol_change")));
        assert_eq!(p, Plan::Instruments([a].into()));
        let p = plan_for("ref.changed.instrument", Some(&notice("instrument", &[&b.to_string()], "provisioned")));
        assert_eq!(p, Plan::Instruments([b].into()));
        for (subj, n) in [
            ("ref.changed.instrument", Some(notice("instrument", &[&a.to_string()], "quarantine_resolved"))),
            ("ref.changed.instrument", Some(notice("instrument", &["not-a-uuid"], "created"))),
            ("ref.changed.instrument", Some(notice("instrument", &[], "created"))),
            ("ref.changed.instrument", None),
            ("ref.changed.venue", Some(notice("venue", &["xnys"], "created"))),
            ("ref.changed.entity", None),
        ] {
            assert_eq!(plan_for(subj, n.as_ref()), Plan::Full, "{subj} {n:?}");
        }
        let mut merged = notice("instrument", &[&a.to_string()], "status");
        merged.attrs.insert("merged_into".into(), b.to_string());
        assert_eq!(plan_for("ref.changed.instrument", Some(&merged)), Plan::Full);
        for e in ["corporate_action", "identifier", "contract_spec", "calendar"] {
            assert_eq!(
                plan_for(&format!("ref.changed.{e}"), Some(&notice(e, &[&a.to_string()], "created"))),
                Plan::Ignore
            );
        }
        // entity from the subject when the payload is empty / JSON
        assert_eq!(plan_for("ref.changed.calendar", None), Plan::Ignore);
    }

    #[test]
    fn plans_merge() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let i = |u: Uuid| Plan::Instruments([u].into());
        assert_eq!(i(a).merge(i(b)), Plan::Instruments([a, b].into()));
        assert_eq!(Plan::Ignore.merge(i(a)), i(a));
        assert_eq!(i(a).merge(Plan::Full), Plan::Full);
        assert_eq!(Plan::Full.merge(Plan::Ignore), Plan::Full);
    }

    #[test]
    fn rows_convert_to_hub_rows() {
        let (e, v, i1, i2) = (Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7());
        let rows = MirrorRows {
            entities: vec![MirrorEntity {
                id: e,
                lei: None,
                name: "Apple".into(),
                country: Some("US".into()),
                sector: None,
            }],
            venues: vec![MirrorVenue {
                id: v,
                code: "xnas".into(),
                mic: Some("XNAS".into()),
                name: "Nasdaq".into(),
                kind: "exchange".into(),
                timezone: "America/New_York".into(),
            }],
            instruments: vec![
                MirrorInstrument {
                    id: i1,
                    asset_class: "equity".into(),
                    kind: "equity".into(),
                    name: "AAPL".into(),
                    status: "active".into(),
                    base_ccy: None,
                    quote_ccy: Some("USD".into()),
                    issuer_id: Some(e),
                    listings: vec![MirrorListing { venue_id: v, venue_symbol: "AAPL".into(), source_id: None }],
                    exposures: vec![MirrorExposure { to_id: i2, relation: "equity-future".into() }],
                },
                MirrorInstrument {
                    id: i2,
                    asset_class: "derivative".into(),
                    kind: "future".into(),
                    name: "AAPL fut".into(),
                    status: "provisional".into(),
                    base_ccy: None,
                    quote_ccy: None,
                    issuer_id: Some(Uuid::now_v7()), // not a current entity
                    listings: vec![],
                    exposures: vec![],
                },
            ],
        };
        let (ents, vens, insts) = to_hub(&rows);
        assert_eq!(ents[0].entity_id, e.to_string());
        assert_eq!((ents[0].lei.as_str(), ents[0].country.as_str()), ("", "US"));
        assert_eq!((vens[0].code.as_str(), vens[0].mic.as_str()), ("xnas", "XNAS"));
        assert_eq!(insts[0].issuer_id.as_deref(), Some(e.to_string().as_str()));
        assert_eq!(insts[0].listings[0].key(), format!("{v}|AAPL"));
        assert_eq!(insts[0].exposures[0].to_instrument_id, i2.to_string());
        assert_eq!(insts[0].quote_ccy, "USD");
        assert_eq!(insts[1].issuer_id, None, "dangling issuer dropped");
    }

    #[test]
    fn full_sync_period_from_env_default() {
        // the default applies when the variable is unset (tests never set it)
        if std::env::var(FULL_SECS_ENV).is_err() {
            assert_eq!(full_every(), Some(Duration::from_secs(DEFAULT_FULL_SECS)));
        }
    }
}
