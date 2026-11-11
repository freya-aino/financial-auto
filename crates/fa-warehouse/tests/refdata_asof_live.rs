//! Live point-in-time reference tests (spec 12 §9). Run only with `DATABASE_URL` set (compose
//! Postgres with migrations through 20261111000010 applied); every row uses ids unique to the
//! run and is deleted at the end.
use chrono::{Duration, NaiveDate, TimeZone, Utc};
use fa_warehouse::{
    asof::{self, Resolved},
    calendar::{self, CalendarRule, SessionState},
    corpactions::{self, AdjustMode, BarPoint, NewCorporateAction},
};
use sqlx::PgPool;
use uuid::Uuid;

async fn pool() -> Option<PgPool> {
    let url = std::env::var("DATABASE_URL").ok()?;
    Some(PgPool::connect(&url).await.expect("connect"))
}

struct Fixture {
    tag: String,
    venue: Uuid,
    inst: Uuid,
    other: Uuid,
}

async fn fixture(pool: &PgPool) -> Fixture {
    let tag = format!("t{}", &Uuid::now_v7().simple().to_string()[20..]);
    let (venue, inst, other) = (Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7());
    sqlx::query("INSERT INTO ref.venue (id, code, mic, name, timezone, kind) VALUES ($1, $2, NULL, 'test venue', 'UTC', 'exchange')")
        .bind(venue)
        .bind(&tag)
        .execute(pool)
        .await
        .unwrap();
    for (id, name) in [(inst, "A"), (other, "B")] {
        sqlx::query("INSERT INTO ref.instrument (id, asset_class, kind, name, quote_ccy, recorded_at) VALUES ($1, 'equity', 'equity', $2, 'USD', '2026-01-01')")
            .bind(id)
            .bind(format!("{tag} {name}"))
            .execute(pool)
            .await
            .unwrap();
    }
    Fixture { tag, venue, inst, other }
}

async fn cleanup(pool: &PgPool, f: &Fixture) {
    for q in [
        "DELETE FROM ref.corporate_action WHERE instrument_id IN ($1, $2)",
        "DELETE FROM ref.contract_spec WHERE instrument_id IN ($1, $2)",
        "DELETE FROM ref.identifier WHERE instrument_id IN ($1, $2)",
        "DELETE FROM ref.listing WHERE instrument_id IN ($1, $2)",
        "DELETE FROM ref.instrument WHERE id IN ($1, $2)",
    ] {
        sqlx::query(q).bind(f.inst).bind(f.other).execute(pool).await.unwrap();
    }
    sqlx::query("DELETE FROM ref.calendar_session WHERE venue_id = $1").bind(f.venue).execute(pool).await.unwrap();
    sqlx::query("DELETE FROM ref.venue WHERE id = $1").bind(f.venue).execute(pool).await.unwrap();
}

#[tokio::test]
async fn as_of_listing_symbol_change_and_ticker_recycling() {
    let Some(pool) = pool().await else { return };
    let f = fixture(&pool).await;
    let t = |d: u32| Utc.with_ymd_and_hms(2026, 3, d, 12, 0, 0).unwrap();
    // OLD listed for inst since 2026-01-01, recorded then
    sqlx::query("INSERT INTO ref.listing (instrument_id, venue_id, venue_symbol, valid_from, recorded_at) VALUES ($1, $2, 'OLD', '2026-01-01', '2026-01-01')")
        .bind(f.inst)
        .bind(f.venue)
        .execute(&pool)
        .await
        .unwrap();
    assert_eq!(asof::resolve(&pool, &f.tag, "OLD", t(1), None).await.unwrap().one(), Some(f.inst));

    // ticker change OLD → NEW effective 2026-03-10
    let ch = corpactions::symbol_change(&pool, &f.tag, "OLD", "NEW", t(10)).await.unwrap();
    assert_eq!((ch.instrument_id, ch.listings_closed), (f.inst, 1));
    assert_eq!(asof::resolve(&pool, &f.tag, "OLD", t(9), None).await.unwrap().one(), Some(f.inst));
    assert_eq!(asof::resolve(&pool, &f.tag, "OLD", t(11), None).await.unwrap(), Resolved::None);
    assert_eq!(asof::resolve(&pool, &f.tag, "NEW", t(11), None).await.unwrap().one(), Some(f.inst));
    // the new listing was recorded now: a reader as of 2026-03-11 did not know it yet
    assert_eq!(asof::resolve(&pool, &f.tag, "NEW", t(11), Some(t(11))).await.unwrap(), Resolved::None);
    // the change is a symbol_change corporate action
    let acts = corpactions::list_as_of(&pool, f.inst, None).await.unwrap();
    assert!(acts.iter().any(|a| a.kind == "symbol_change" && a.new_symbol.as_deref() == Some("NEW")));

    // OLD is recycled for another instrument from 2026-03-20 (spec 12 §8)
    sqlx::query("INSERT INTO ref.listing (instrument_id, venue_id, venue_symbol, valid_from) VALUES ($1, $2, 'OLD', $3)")
        .bind(f.other)
        .bind(f.venue)
        .bind(t(20))
        .execute(&pool)
        .await
        .unwrap();
    assert_eq!(asof::resolve(&pool, &f.tag, "OLD", t(9), None).await.unwrap().one(), Some(f.inst));
    assert_eq!(asof::resolve(&pool, &f.tag, "OLD", t(21), None).await.unwrap().one(), Some(f.other));
    // a change onto a symbol another instrument holds is refused
    let err = corpactions::symbol_change(&pool, &f.tag, "NEW", "OLD", t(25)).await.unwrap_err();
    assert!(err.to_string().starts_with("invalid:"), "{err}");

    // an overlapping second listing makes the mapping ambiguous, never silently picked
    sqlx::query("INSERT INTO ref.listing (instrument_id, venue_id, venue_symbol, valid_from) VALUES ($1, $2, 'NEW', $3)")
        .bind(f.other)
        .bind(f.venue)
        .bind(t(12))
        .execute(&pool)
        .await
        .unwrap();
    assert!(matches!(asof::resolve(&pool, &f.tag, "NEW", t(13), None).await.unwrap(), Resolved::Ambiguous { .. }));

    // instrument view as of a time
    let view = asof::instrument(&pool, f.inst, t(11), None).await.unwrap().unwrap();
    assert_eq!(view.listings.iter().map(|l| l.venue_symbol.as_str()).collect::<Vec<_>>(), ["NEW"]);
    cleanup(&pool, &f).await;
}

#[tokio::test]
async fn corporate_actions_contract_specs_and_identifiers() {
    let Some(pool) = pool().await else { return };
    let f = fixture(&pool).await;
    let split = NewCorporateAction {
        instrument_id: f.inst,
        kind: "split".into(),
        ex_date: NaiveDate::from_ymd_opt(2026, 3, 4).unwrap(),
        ratio: Some(4.0),
        ..Default::default()
    };
    split.validate().unwrap();
    let (id, new) = corpactions::record(&pool, &split).await.unwrap();
    assert!(new);
    assert!(!corpactions::record(&pool, &split).await.unwrap().1, "idempotent");
    let from = NaiveDate::from_ymd_opt(2026, 3, 1).unwrap();
    let to = NaiveDate::from_ymd_opt(2026, 3, 31).unwrap();
    assert_eq!(corpactions::adjustment_factor_db(&pool, f.inst, from, to, None).await.unwrap(), 0.25);
    // not yet known an hour ago (recorded now)
    let before = Utc::now() - Duration::hours(1);
    assert_eq!(corpactions::adjustment_factor_db(&pool, f.inst, from, to, Some(before)).await.unwrap(), 1.0);
    let acts = corpactions::list_as_of(&pool, f.inst, None).await.unwrap();
    let raw: Vec<BarPoint> = [400.0, 404.0, 101.5]
        .iter()
        .enumerate()
        .map(|(i, c)| BarPoint {
            ts: Utc.with_ymd_and_hms(2026, 3, 2 + i as u32, 21, 0, 0).unwrap(),
            open: *c,
            high: *c,
            low: *c,
            close: *c,
            volume: 1.0,
        })
        .collect();
    let adj = corpactions::adjust_bars(&raw, &acts, AdjustMode::SplitsOnly);
    assert_eq!(adj.iter().map(|b| b.close).collect::<Vec<_>>(), [100.0, 101.0, 101.5]);
    assert!(corpactions::cancel(&pool, id).await.unwrap());
    assert_eq!(corpactions::adjustment_factor_db(&pool, f.inst, from, to, None).await.unwrap(), 1.0);

    // contract spec versions
    let spec = asof::NewContractSpec {
        instrument_id: f.inst,
        venue: f.tag.clone(),
        tick_size: Some(0.01),
        lot_size: Some(1.0),
        valid_from: Some(Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()),
        ..Default::default()
    };
    assert!(asof::upsert_contract_spec(&pool, &spec).await.unwrap());
    let v2 = asof::NewContractSpec { tick_size: Some(0.005), valid_from: Some(Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap()), ..spec };
    assert!(asof::upsert_contract_spec(&pool, &v2).await.unwrap());
    let at = |m| Utc.with_ymd_and_hms(2026, m, 15, 0, 0, 0).unwrap();
    let s1 = asof::contract_specs(&pool, f.inst, Some(&f.tag), at(3), None).await.unwrap();
    let s2 = asof::contract_specs(&pool, f.inst, Some(&f.tag), at(7), None).await.unwrap();
    assert_eq!((s1[0].tick_size, s2[0].tick_size), (Some(0.01), Some(0.005)));
    assert_eq!(s1[0].multiplier, 1.0);

    // identifiers: one owner per scheme/value
    let isin = format!("XX{}", &f.tag[1..]);
    assert!(asof::add_identifier(&pool, f.inst, "isin", &isin, None).await.unwrap());
    assert!(!asof::add_identifier(&pool, f.other, "isin", &isin, None).await.unwrap());
    assert_eq!(asof::resolve_identifier(&pool, "isin", &isin, Utc::now() + Duration::seconds(1), None).await.unwrap().one(), Some(f.inst));

    // EUR converts to itself
    assert_eq!(asof::fx_to_eur(&pool, "EUR", Utc::now(), None).await.unwrap().unwrap().0, 1.0);
    cleanup(&pool, &f).await;
}

#[tokio::test]
async fn calendar_materialize_and_session_state() {
    let Some(pool) = pool().await else { return };
    let f = fixture(&pool).await;
    let from = NaiveDate::from_ymd_opt(2026, 12, 21).unwrap();
    let to = NaiveDate::from_ymd_opt(2026, 12, 27).unwrap();
    assert_eq!(calendar::materialize(&pool, &f.tag, CalendarRule::Us, from, to).await.unwrap(), 7);
    assert_eq!(calendar::materialize(&pool, &f.tag, CalendarRule::Us, from, to).await.unwrap(), 7, "re-run updates in place");
    let at = |d, h| Utc.with_ymd_and_hms(2026, 12, d, h, 0, 0).unwrap();
    let open = calendar::session_state_at(&pool, &f.tag, at(22, 15)).await.unwrap();
    assert_eq!((open.state, open.basis), (SessionState::Open, "table"));
    assert_eq!(calendar::session_state_at(&pool, &f.tag, at(25, 15)).await.unwrap().state, SessionState::Holiday);
    // 24 Dec: early close at 13:00 EST = 18:00 UTC → post-market at 19:00 UTC
    assert_eq!(calendar::session_state_at(&pool, &f.tag, at(24, 19)).await.unwrap().state, SessionState::PostMarket);
    // unknown venue → closed, missing calendar (spec 12 §8)
    let missing = calendar::session_state_at(&pool, "no-such-venue", at(22, 15)).await.unwrap();
    assert_eq!((missing.state, missing.basis), (SessionState::Closed, "missing_calendar"));
    // seeded crypto venue is 24/7
    assert_eq!(calendar::session_state_at(&pool, "crypto", at(25, 3)).await.unwrap().state, SessionState::AlwaysOpen);
    cleanup(&pool, &f).await;
}
