//! News → instrument matching for the text decoder families (spec 34 §6.1 Text) and the
//! decoder trade cost (spec 35 venue cost model).
//!
//! **Matching.** Every news item of a text genome's window is tagged with the instruments it
//! mentions, by the same matcher model-runtime uses for the baseline text decoder
//! (`fa_decoders::text_baseline::InstrumentIndex`: upper-case ticker tokens, cleaned instrument
//! names, `entities_raw`). The reference snapshot (`ref.instrument` names, current
//! `ref.listing` venue symbols and `ref.identifier` values, merged instruments excluded) is
//! loaded from Postgres and refreshed every [`REF_REFRESH`]; without Postgres every item stays
//! market-wide. Genomes with `ninst=1` read only their instrument's items.
//!
//! **Trade cost.** The naive sign policy of `fa_backtest::eval` is charged, per unit of position
//! change, the round-trip-half cost the paper venue would charge a taker fill of
//! [`TRADE_NOTIONAL_EUR`] on the series' venue and asset class (`costs.yaml`: half spread +
//! square-root impact with the series' own daily volatility + commission incl. its minimum),
//! instead of a flat 1 bp.

use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

use fa_backtest::cost::{CostInputs, CostModel};
use fa_bus::replay::RecordedMessage;
use fa_decoders::{
    baseline::Series,
    neural::data::TextTimeline,
    text_baseline::{InstrumentIndex, InstrumentRef, NewsDoc},
};
use fa_proto::{
    Message,
    v1::{Envelope, envelope::Payload},
};
use sqlx::{Row, postgres::PgPoolOptions};

/// Reference snapshot refresh period.
pub const REF_REFRESH: Duration = Duration::from_secs(600);
/// Notional of one unit of the naive policy's position (EUR).
pub const TRADE_NOTIONAL_EUR: f64 = 10_000.0;

/// Shared, refreshed instrument matcher.
#[derive(Clone, Default)]
pub struct NewsRefs {
    inner: Arc<RwLock<Arc<InstrumentIndex>>>,
}

impl NewsRefs {
    pub fn index(&self) -> Arc<InstrumentIndex> {
        self.inner.read().map(|g| g.clone()).unwrap_or_default()
    }

    pub fn set(&self, idx: InstrumentIndex) {
        if let Ok(mut g) = self.inner.write() {
            *g = Arc::new(idx);
        }
    }

    /// Keeps the snapshot fresh from Postgres (warns once while unavailable).
    pub fn spawn_refresh(&self, pg_url: String) {
        let me = self.clone();
        tokio::spawn(async move {
            let mut warned = false;
            loop {
                match load_refs(&pg_url).await {
                    Ok(refs) => {
                        tracing::info!(instruments = refs.len(), "reference snapshot loaded for news matching");
                        warned = false;
                        me.set(InstrumentIndex::new(&refs));
                    }
                    Err(e) if !warned => {
                        warned = true;
                        tracing::warn!(error = %e, "reference snapshot unavailable; news stays market-wide");
                    }
                    Err(e) => tracing::debug!(error = %e, "reference snapshot still unavailable"),
                }
                tokio::time::sleep(REF_REFRESH).await;
            }
        });
    }
}

async fn load_refs(pg_url: &str) -> anyhow::Result<Vec<InstrumentRef>> {
    let pool = PgPoolOptions::new().max_connections(1).acquire_timeout(Duration::from_secs(5)).connect(pg_url).await?;
    let rows = sqlx::query(
        "SELECT i.id::text AS id, i.name, \
                coalesce((SELECT array_agg(l.venue_symbol) FROM ref.listing l \
                           WHERE l.instrument_id = i.id AND now() >= l.valid_from AND now() < l.valid_to), '{}') AS symbols, \
                coalesce((SELECT array_agg(d.value) FROM ref.identifier d \
                           WHERE d.instrument_id = i.id AND d.scheme <> 'contract_address' \
                             AND now() >= d.valid_from AND now() < d.valid_to), '{}') AS idents \
           FROM ref.instrument i WHERE i.status <> 'merged' ORDER BY i.recorded_at DESC LIMIT 5000",
    )
    .fetch_all(&pool)
    .await?;
    pool.close().await;
    Ok(rows
        .iter()
        .map(|r| {
            let mut symbols: Vec<String> = r.try_get("symbols").unwrap_or_default();
            symbols.extend(r.try_get::<Vec<String>, _>("idents").unwrap_or_default());
            InstrumentRef { id: r.get("id"), symbols, names: vec![r.get::<String, _>("name")] }
        })
        .collect())
}

/// Timeline from recorded `news.item.*` envelopes (knowledge time = JetStream receive time),
/// each item tagged with the instruments `index` matches. Returns `(timeline, undecodable)`.
pub fn timeline_of(msgs: &[RecordedMessage], index: &InstrumentIndex) -> (TextTimeline, usize) {
    let mut t = TextTimeline::new();
    let mut bad = 0;
    for m in msgs {
        match Envelope::decode(m.payload.as_slice()) {
            Ok(env) => match env.payload {
                Some(Payload::News(n)) => {
                    let matched = if index.is_empty() {
                        Vec::new()
                    } else {
                        index.match_doc(&NewsDoc {
                            ts_ms: m.published_ms,
                            source: m.subject.trim_start_matches("news.item.").to_owned(),
                            title: n.title.clone(),
                            body: n.body.clone(),
                            entities: n.entities_raw.clone(),
                        })
                    };
                    t.push_tagged(m.published_ms, &n.title, &n.body, &matched);
                }
                _ => bad += 1,
            },
            Err(_) => bad += 1,
        }
    }
    t.finish();
    (t, bad)
}

/// Daily volatility (fraction) of a series from its one-step log returns; 0 when unknown.
pub fn daily_sigma(s: &Series) -> f64 {
    let n = s.len();
    if n < 3 || s.step_ms <= 0 {
        return 0.0;
    }
    let rets: Vec<f64> = s.log_price.windows(2).map(|w| w[1] - w[0]).collect();
    let m = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (rets.len() - 1) as f64;
    let v = var.sqrt() * (86_400_000.0 / s.step_ms as f64).sqrt();
    if v.is_finite() { v } else { 0.0 }
}

/// Cost (bp of notional) of one unit of position change of the naive decoder policy on `s`
/// at `venue` (see the module docs).
pub fn trade_cost_bp(model: &CostModel, venue: &str, s: &Series) -> f64 {
    let class = model.class_of(&s.instrument_id);
    let r = model.price(&CostInputs {
        venue: venue.to_owned(),
        class,
        side: "buy".into(),
        price: 1.0,
        qty: TRADE_NOTIONAL_EUR,
        sigma: daily_sigma(s),
        ..Default::default()
    });
    let bp = r.total_cost(TRADE_NOTIONAL_EUR) / TRADE_NOTIONAL_EUR * 1e4;
    if bp.is_finite() { bp.max(0.0) } else { 1.0 }
}

/// Venue of an instrument from the data loader's `(instrument, venue)` list ("" = unknown).
pub fn venue_of<'a>(venues: &'a [(String, String)], instrument: &str) -> &'a str {
    venues.iter().find(|(i, _)| i == instrument).map_or("", |(_, v)| v.as_str())
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::NewsItem;

    use super::*;

    #[test]
    fn items_are_tagged_with_matched_instruments() {
        let idx = InstrumentIndex::new(&[InstrumentRef {
            id: "0190aaaa-0000-7000-8000-000000000001".into(),
            symbols: vec!["AAPL".into()],
            names: vec!["Apple Inc.".into()],
        }]);
        let env = |title: &str| Envelope {
            payload: Some(Payload::News(NewsItem { title: title.into(), ..Default::default() })),
            ..Default::default()
        };
        let msgs = vec![
            RecordedMessage {
                subject: "news.item.x".into(),
                stream_seq: 1,
                published_ms: 1000,
                payload: env("AAPL rallies on results").encode_to_vec(),
            },
            RecordedMessage {
                subject: "news.item.x".into(),
                stream_seq: 2,
                published_ms: 2000,
                payload: env("Oil prices slump").encode_to_vec(),
            },
            RecordedMessage {
                subject: "news.item.x".into(),
                stream_seq: 3,
                published_ms: 3000,
                payload: vec![0xff, 0xff],
            },
        ];
        let (t, bad) = timeline_of(&msgs, &idx);
        assert_eq!((t.len(), t.matched(), bad), (2, 1, 1));
        assert!(t.embedding_for(Some("0190AAAA000070008000000000000001"), 0, 5000, 64).is_some(), "token form matches");
        assert_eq!(
            t.embedding_for(Some("0190aaaa-0000-7000-8000-000000000001"), 0, 5000, 64),
            t.embedding(0, 1000, 64),
            "the instrument reads only its own item"
        );
        let (u, _) = timeline_of(&msgs, &InstrumentIndex::default());
        assert_eq!(u.matched(), 0);
    }

    #[test]
    fn trade_cost_follows_the_venue_schedule() {
        let model = CostModel::from_yaml(
            "default:\n  default: { commission_bps: 5, commission_min_eur: 1, spread_bps: 5 }\n\
             venues:\n  sim:\n    default: { commission_bps: 0, commission_min_eur: 0, spread_bps: 0, impact_k: 0 }\n",
        )
        .unwrap();
        let s = Series {
            instrument_id: "x".into(),
            ts_ms: (0..100).map(|i| i * 1000).collect(),
            log_price: (0..100).map(|i| 4.0 + 0.0001 * f64::from(i % 3)).collect(),
            step_ms: 1000,
        };
        assert!(daily_sigma(&s) > 0.0);
        let free = trade_cost_bp(&model, "sim", &s);
        assert!(free.abs() < 1e-9, "{free}");
        let retail = trade_cost_bp(&model, "yahoo", &s);
        // 5 bp commission + 2.5 bp half spread + impact
        assert!(retail > 7.5 && retail < 50.0, "{retail}");
        assert_eq!(venue_of(&[("x".into(), "sim".into())], "x"), "sim");
        assert_eq!(venue_of(&[], "x"), "");
    }
}
