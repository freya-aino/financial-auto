//! Baseline decoders served by the runtime (spec 34 §6.1, D-062): the text family over
//! `news.item.>` ([`fa_decoders::text_baseline`]) and the visual family over the bar windows
//! ([`fa_decoders::visual_baseline`]), published as `DecoderOutput` on `decoder.output.<family>`.
//!
//! Output contract (what `fa_strategies::decoders` consumes): `family` ∈ `text.news`,
//! `text.summary`, `visual.chart`, `visual.future`; `quantiles[0]` = the score in `[-1, 1]`
//! (further entries are the components); `output_refs` = `inst:<instrument uuid>` (or `inst:*`
//! for market-wide text), `topic:<news|summary>` or `kind:<chart|future>`, `src:<source>` …,
//! `n:<items>` (text); `timescale`, `snapshot_id` of the timescale's encoder, `as_of`.

use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Duration,
};

use fa_core::Timescale;
use fa_decoders::{
    text_baseline::{self, InstrumentIndex, InstrumentRef, NewsDoc, TextWindow, WindowScore},
    visual_baseline::{self, ChartBar, ChartScores},
};
use fa_encoders::MarketBar;
use fa_proto::v1::{DecoderOutput, Envelope, envelope::Payload};
use tokio::sync::watch;

use crate::pipeline::ms_to_ts;

/// Market-wide key (items that match no instrument).
pub const MARKET: &str = "*";
/// Bars of the visual window.
pub const VISUAL_BARS: usize = 60;
/// How often the reference snapshot is reloaded.
pub const REF_REFRESH: Duration = Duration::from_secs(300);
/// Remembered dedup keys.
const DEDUP_CAP: usize = 20_000;

/// Text window per timescale: 1m → 1 h, 1h → 1 d, 1d → 1 w, 1w → 4 w (1s → 15 min).
pub fn text_window_ms(ts: Timescale) -> i64 {
    let steps = match ts {
        Timescale::Second => 900,
        Timescale::Minute => 60,
        Timescale::Hour => 24,
        Timescale::Day => 7,
        Timescale::Week => 4,
    };
    ts.duration().as_millis() as i64 * steps
}

fn env_ms(env: &Envelope) -> Option<i64> {
    env.event_time
        .as_ref()
        .or(env.publish_time.as_ref())
        .or(env.ingest_time.as_ref())
        .map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)
}

/// `(dedup key, document)` of a `news.item.<source>` envelope.
pub fn news_doc(subject: &str, env: &Envelope) -> Option<(String, NewsDoc)> {
    let Some(Payload::News(n)) = &env.payload else {
        return None;
    };
    if n.title.trim().is_empty() && n.body.trim().is_empty() {
        return None;
    }
    let source = subject.strip_prefix("news.item.").unwrap_or(env.source_id.as_str()).to_owned();
    let dedup = if !n.dedup_key.is_empty() {
        n.dedup_key.clone()
    } else if !env.event_id.is_empty() {
        env.event_id.clone()
    } else {
        format!("{source}:{}", n.url)
    };
    let doc = NewsDoc {
        ts_ms: env_ms(env).unwrap_or_else(|| fa_proto::now().seconds * 1000),
        source,
        title: n.title.clone(),
        body: n.body.clone(),
        entities: n.entities_raw.clone(),
    };
    Some((dedup, doc))
}

/// Text decoder state: one [`TextWindow`] per (instrument | `*`, timescale).
pub struct TextDecoder {
    timescales: Vec<Timescale>,
    windows: HashMap<(String, Timescale), TextWindow>,
    seen: HashSet<String>,
    seen_order: VecDeque<String>,
}

impl TextDecoder {
    pub fn new(timescales: Vec<Timescale>) -> Self {
        Self { timescales, windows: HashMap::new(), seen: HashSet::new(), seen_order: VecDeque::new() }
    }

    pub fn timescales(&self) -> &[Timescale] {
        &self.timescales
    }

    /// Adds a document under its matched instruments (or `*`); `None` for a duplicate. Returns
    /// the keys it landed in.
    pub fn ingest(&mut self, dedup: String, doc: &NewsDoc, matched: &[String], now_ms: i64) -> Option<Vec<String>> {
        if !self.seen.insert(dedup.clone()) {
            return None;
        }
        self.seen_order.push_back(dedup);
        while self.seen_order.len() > DEDUP_CAP {
            if let Some(old) = self.seen_order.pop_front() {
                self.seen.remove(&old);
            }
        }
        let keys: Vec<String> = if matched.is_empty() { vec![MARKET.to_owned()] } else { matched.to_vec() };
        for key in &keys {
            for &ts in &self.timescales {
                let w = self.windows.entry((key.clone(), ts)).or_default();
                w.push(doc);
                w.prune(now_ms, text_window_ms(ts));
            }
        }
        Some(keys)
    }

    /// Keys with any item in the scoring range.
    pub fn keys(&self) -> BTreeSet<String> {
        self.windows.iter().filter(|(_, w)| !w.is_empty()).map(|((k, _), _)| k.clone()).collect()
    }

    pub fn score(&self, key: &str, ts: Timescale, now_ms: i64) -> Option<WindowScore> {
        let w = self.windows.get(&(key.to_owned(), ts))?;
        let s = w.score_at(now_ms, text_window_ms(ts));
        (s.n_items > 0 || s.n_trailing > 0).then_some(s)
    }

    pub fn prune(&mut self, now_ms: i64) {
        for ((_, ts), w) in &mut self.windows {
            w.prune(now_ms, text_window_ms(*ts));
        }
        self.windows.retain(|_, w| !w.is_empty());
    }
}

/// `text.news` and `text.summary` outputs of one key at one time.
pub fn text_outputs(key: &str, ts: Timescale, snapshot_id: &str, now_ms: i64, s: &WindowScore) -> [DecoderOutput; 2] {
    let refs = |topic: &str| {
        let mut r = vec![format!("inst:{key}"), format!("topic:{topic}")];
        r.extend(s.sources.iter().map(|src| format!("src:{src}")));
        r.push(format!("n:{}", s.n_items));
        r
    };
    let out = |family: &str, topic: &str, quantiles: Vec<f64>| DecoderOutput {
        decoder_id: text_baseline::DECODER_ID.into(),
        genome_id: String::new(),
        family: family.into(),
        snapshot_id: snapshot_id.into(),
        timescale: ts.to_string(),
        as_of: Some(ms_to_ts(now_ms)),
        output_refs: refs(topic),
        quantiles,
        analogs: vec![],
    };
    [
        out("text.news", "news", vec![s.news]),
        out("text.summary", "summary", vec![s.novelty, f64::from(s.hits), s.n_trailing as f64]),
    ]
}

/// Chart scores of the last [`VISUAL_BARS`] closed bars.
pub fn chart_of(bars: &[MarketBar]) -> ChartScores {
    let w = &bars[bars.len().saturating_sub(VISUAL_BARS)..];
    let cb: Vec<ChartBar> = w.iter().map(|b| ChartBar::new(b.open, b.high, b.low, b.close, b.volume)).collect();
    visual_baseline::chart_scores(&cb)
}

/// `visual.chart` and `visual.future` outputs of one instrument at a bar close; `None` while the
/// window is shorter than [`visual_baseline::MIN_BARS`].
pub fn visual_outputs(
    inst: &str,
    ts: Timescale,
    snapshot_id: &str,
    as_of_ms: i64,
    bars: &[MarketBar],
    source: &str,
) -> Option<[DecoderOutput; 2]> {
    if bars.len() < visual_baseline::MIN_BARS {
        return None;
    }
    let s = chart_of(bars);
    let n = bars.len().min(VISUAL_BARS);
    let out = |family: &str, kind: &str, quantiles: Vec<f64>| DecoderOutput {
        decoder_id: visual_baseline::DECODER_ID.into(),
        genome_id: String::new(),
        family: family.into(),
        snapshot_id: snapshot_id.into(),
        timescale: ts.to_string(),
        as_of: Some(ms_to_ts(as_of_ms)),
        output_refs: vec![format!("inst:{inst}"), format!("kind:{kind}"), format!("src:{source}"), format!("n:{n}")],
        quantiles,
        analogs: vec![],
    };
    Some([out("visual.chart", "chart", s.chart_quantiles()), out("visual.future", "future", s.future_quantiles())])
}

/// Reference snapshot from Postgres (`ref.instrument` names, `ref.identifier` values,
/// `ref.listing` venue symbols) via `fa_warehouse::refdata::search_instruments`, refreshed every
/// [`REF_REFRESH`]. Failures keep the previous index (items then fall back to `inst:*`).
pub fn spawn_ref_refresh(pg_url: String, tx: watch::Sender<Arc<InstrumentIndex>>) {
    tokio::spawn(async move {
        let mut warned = false;
        loop {
            match load_refs(&pg_url).await {
                Ok(refs) => {
                    tracing::info!(instruments = refs.len(), "reference snapshot loaded for the text decoder");
                    warned = false;
                    let _ = tx.send(Arc::new(InstrumentIndex::new(&refs)));
                }
                Err(e) if !warned => {
                    warned = true;
                    tracing::warn!(error = %e, "reference snapshot unavailable; news maps to inst:* until Postgres answers");
                }
                Err(e) => tracing::debug!(error = %e, "reference snapshot still unavailable"),
            }
            tokio::time::sleep(REF_REFRESH).await;
        }
    });
}

async fn load_refs(pg_url: &str) -> anyhow::Result<Vec<InstrumentRef>> {
    let wh = fa_warehouse::Warehouse::connect(pg_url).await?;
    let hits = fa_warehouse::refdata::search_instruments(wh.pool(), "", None, 500).await?;
    wh.pool().close().await;
    Ok(hits
        .into_iter()
        .filter(|h| h.status != "merged")
        .map(|h| InstrumentRef {
            id: h.id.to_string(),
            symbols: h
                .listings
                .iter()
                .map(|l| l.venue_symbol.clone())
                .chain(h.identifiers.iter().filter(|i| i.scheme != "contract_address").map(|i| i.value.clone()))
                .collect(),
            names: vec![h.name],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::NewsItem;

    use super::*;

    fn env(title: &str, dedup: &str, ms: i64, entities: &[&str]) -> Envelope {
        Envelope {
            event_id: "e1".into(),
            event_time: Some(ms_to_ts(ms)),
            payload: Some(Payload::News(NewsItem {
                title: title.into(),
                dedup_key: dedup.into(),
                entities_raw: entities.iter().map(|s| (*s).to_owned()).collect(),
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    #[test]
    fn news_items_map_dedupe_and_publish_per_contract() {
        let idx = InstrumentIndex::new(&[InstrumentRef {
            id: "0190a3b4-0000-7000-8000-000000000001".into(),
            symbols: vec!["SPY".into()],
            names: vec!["SPDR S&P 500 ETF Trust".into()],
        }]);
        let mut td = TextDecoder::new(vec![Timescale::Minute, Timescale::Hour]);
        let t0 = 1_800_000_000_000;
        let (k1, d1) = news_doc("news.item.google_news", &env("SPY rallies to record high", "g:1", t0, &[])).unwrap();
        assert_eq!(d1.source, "google_news");
        let m1 = idx.match_doc(&d1);
        assert_eq!(m1, vec!["0190a3b4-0000-7000-8000-000000000001"]);
        assert_eq!(td.ingest(k1.clone(), &d1, &m1, t0).unwrap(), m1);
        assert!(td.ingest(k1, &d1, &m1, t0).is_none(), "duplicate dedup key");
        let (k2, d2) =
            news_doc("news.item.hackernews", &env("Show HN: a tiny database", "hn:2", t0, &["author:x"])).unwrap();
        let m2 = idx.match_doc(&d2);
        assert_eq!(td.ingest(k2, &d2, &m2, t0).unwrap(), vec![MARKET]);
        assert_eq!(td.keys().len(), 2);
        let s = td.score(&m1[0], Timescale::Minute, t0 + 1_000).unwrap();
        assert!(s.news > 0.0);
        let [news, summary] = text_outputs(&m1[0], Timescale::Minute, "stat-v1-1m", t0 + 1_000, &s);
        assert_eq!(news.family, "text.news");
        assert_eq!(news.decoder_id, "text-stat-v1");
        assert_eq!(
            news.output_refs,
            vec!["inst:0190a3b4-0000-7000-8000-000000000001", "topic:news", "src:google_news", "n:1"]
        );
        assert!(news.quantiles[0] > 0.0 && news.quantiles[0] <= 1.0);
        assert_eq!(summary.family, "text.summary");
        assert_eq!(summary.output_refs[1], "topic:summary");
        assert_eq!((news.timescale.as_str(), news.snapshot_id.as_str()), ("1m", "stat-v1-1m"));
        // the 1m window (1 h) forgets the item after two windows; the 1h window (1 d) keeps it
        td.prune(t0 + 3 * 3_600_000);
        assert!(td.score(&m1[0], Timescale::Minute, t0 + 3 * 3_600_000).is_none());
        assert!(td.score(&m1[0], Timescale::Hour, t0 + 3 * 3_600_000).is_some());
        // empty items are skipped
        assert!(news_doc("news.item.x", &env("  ", "k", t0, &[])).is_none());
    }

    #[test]
    fn visual_outputs_follow_the_contract() {
        let bars: Vec<MarketBar> = (0..30).map(|i| MarketBar::from_trade(100.0 + i as f64, 10.0)).collect();
        assert!(visual_outputs("x", Timescale::Minute, "s", 0, &bars[..5], "bus").is_none());
        let [chart, future] = visual_outputs("inst-1", Timescale::Minute, "stat-v1-1m", 60_000, &bars, "bus").unwrap();
        assert_eq!(chart.family, "visual.chart");
        assert_eq!(future.family, "visual.future");
        assert_eq!(future.output_refs, vec!["inst:inst-1", "kind:future", "src:bus", "n:30"]);
        assert_eq!(chart.output_refs[1], "kind:chart");
        assert!(future.quantiles[0] > 0.3, "a steady ramp points up: {:?}", future.quantiles);
        assert!(future.quantiles.iter().all(|q| (-1.0..=1.0).contains(q)));
        assert_eq!(text_window_ms(Timescale::Minute), 3_600_000);
        assert_eq!(text_window_ms(Timescale::Week), 4 * 604_800_000);
    }
}
