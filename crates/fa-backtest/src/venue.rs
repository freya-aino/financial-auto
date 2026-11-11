//! The paper venue's market state, ported from `go/services/broker-gateway/paper.go`, so a
//! recorded live window can be replayed and its paper fills reproduced (spec 41 §9 "replaying a
//! recorded live day reproduces the live paper decisions"; spec 40 §6.7 "paper uses the same cost
//! model as backtest").
//!
//! Per instrument token (the last token of `md.trade.<venue>.<token>`): the last trade and its
//! event time, best bid/ask from `md.quote.*`, an EWMA of squared trade-to-trade log returns per
//! second (decay 0.97, `σ_day = sqrt(var · 86400)` after 5 returns) and the traded notional of the
//! last hour, extrapolated to 24 h while the window is not yet full. A market order fills at the
//! last trade (its arrival price) ± half spread ± square-root impact through
//! [`crate::cost::CostModel::price`] — the same arithmetic the gateway runs.
//!
//! [`replay_window`] drives a recorded window ([`VenueRecord`] lines, `testdata/venue_fixture.jsonl`)
//! through the venue and compares every recorded fill of a market intent with the replayed one.

use std::collections::{BTreeMap, HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::cost::{CostInputs, CostModel};

#[derive(Debug, Clone, Default)]
struct Market {
    venue: String,
    last: f64,
    last_at_ms: i64,
    bid: f64,
    ask: f64,
    var_per_sec: f64,
    n_returns: u64,
    vol: VecDeque<(i64, f64)>,
    first_vol_ms: Option<i64>,
}

impl Market {
    fn sigma(&self) -> f64 {
        if self.n_returns < 5 || self.var_per_sec <= 0.0 { 0.0 } else { (self.var_per_sec * 86_400.0).sqrt() }
    }

    fn adv(&self, now_ms: i64, window_ms: i64) -> f64 {
        let Some(first) = self.first_vol_ms else { return 0.0 };
        if self.vol.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.vol.iter().map(|(_, n)| n).sum();
        let elapsed = (now_ms - first).clamp(1_000, window_ms.max(1_000));
        sum * 86_400_000.0 / elapsed as f64
    }
}

/// A fill the venue would give a market order now.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VenueFill {
    pub arrival_price: f64,
    pub fill_price: f64,
    pub fee: f64,
    pub slippage_bps: f64,
    pub sigma_day: f64,
    pub adv_eur: f64,
}

/// The paper venue's market state (`PaperEngine` without the order book).
#[derive(Debug, Clone)]
pub struct PaperVenue {
    markets: HashMap<String, Market>,
    pub costs: CostModel,
    pub volume_window_ms: i64,
    pub sigma_decay: f64,
}

/// Lower-case, dash-free instrument token (`bus.SubjectToken`).
pub fn subject_token(instrument_id: &str) -> String {
    instrument_id.replace('-', "").to_lowercase()
}

impl PaperVenue {
    pub fn new(costs: CostModel) -> Self {
        Self { markets: HashMap::new(), costs, volume_window_ms: 3_600_000, sigma_decay: 0.97 }
    }

    /// A trade at event time `at_ms` (`ObserveTrade`).
    pub fn observe_trade(&mut self, venue: &str, token: &str, price: f64, size: f64, at_ms: i64) {
        if !(price.is_finite() && price > 0.0) {
            return;
        }
        let decay = self.sigma_decay;
        let window = self.volume_window_ms;
        let m = self.markets.entry(token.to_owned()).or_default();
        if m.last > 0.0 && at_ms > m.last_at_ms {
            let dt = (at_ms - m.last_at_ms) as f64 / 1000.0;
            let r = (price / m.last).ln();
            let x = r * r / dt;
            m.var_per_sec = if m.n_returns == 0 { x } else { decay * m.var_per_sec + (1.0 - decay) * x };
            m.n_returns += 1;
        }
        m.venue = venue.to_owned();
        m.last = price;
        m.last_at_ms = at_ms;
        if size > 0.0 {
            m.first_vol_ms.get_or_insert(at_ms);
            m.vol.push_back((at_ms, price * size));
            while m.vol.front().is_some_and(|(t, _)| *t < at_ms - window) {
                m.vol.pop_front();
            }
        }
    }

    /// Best bid/ask (`ObserveQuote`).
    pub fn observe_quote(&mut self, token: &str, bid: f64, ask: f64) {
        if bid <= 0.0 || ask <= 0.0 || ask < bid {
            return;
        }
        let m = self.markets.entry(token.to_owned()).or_default();
        m.bid = bid;
        m.ask = ask;
    }

    /// The fill of a market order at `now_ms`; `None` when no trade was seen for the instrument
    /// (the gateway rejects with "no market price observed").
    pub fn market_fill(&self, instrument_id: &str, sell: bool, qty: f64, now_ms: i64) -> Option<VenueFill> {
        let m = self.markets.get(&subject_token(instrument_id)).filter(|m| m.last > 0.0)?;
        let spread = if m.ask > m.bid && m.bid > 0.0 { m.ask - m.bid } else { 0.0 };
        let (sigma, adv) = (m.sigma(), m.adv(now_ms, self.volume_window_ms));
        let r = self.costs.price(&CostInputs {
            venue: m.venue.clone(),
            class: self.costs.class_of(instrument_id),
            side: if sell { "sell" } else { "buy" }.into(),
            price: m.last,
            qty,
            spread,
            sigma,
            adv_eur: adv,
            passive: false,
        });
        Some(VenueFill {
            arrival_price: m.last,
            fill_price: r.fill_price,
            fee: r.fee,
            slippage_bps: r.slippage_bps,
            sigma_day: sigma,
            adv_eur: adv,
        })
    }
}

/// One line of a recorded venue window, ordered by `t` (ms) = the time the message became known
/// (receive time of a live capture). A trade also carries its event time `at`, which the
/// gateway's σ / ADV estimators use; `reset` marks a (re)start of the broker-gateway: its market
/// state begins empty there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "k", rename_all = "snake_case")]
pub enum VenueRecord {
    Trade {
        t: i64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        at: Option<i64>,
        venue: String,
        token: String,
        price: f64,
        size: f64,
    },
    Quote {
        t: i64,
        token: String,
        bid: f64,
        ask: f64,
    },
    Intent {
        t: i64,
        intent_id: String,
        instrument_id: String,
        sell: bool,
        qty: f64,
        exec_algo: String,
    },
    Fill {
        t: i64,
        intent_id: String,
        fill_price: f64,
        fill_qty: f64,
        fee: f64,
        arrival_price: f64,
        slippage_bps: f64,
    },
    Reset {
        t: i64,
    },
}

impl VenueRecord {
    pub fn t(&self) -> i64 {
        match self {
            Self::Trade { t, .. }
            | Self::Quote { t, .. }
            | Self::Intent { t, .. }
            | Self::Fill { t, .. }
            | Self::Reset { t } => *t,
        }
    }

    /// Sort rank at equal times: market data first (the gateway saw the trade before acting).
    fn rank(&self) -> u8 {
        match self {
            Self::Reset { .. } => 0,
            Self::Trade { .. } | Self::Quote { .. } => 1,
            Self::Intent { .. } => 2,
            Self::Fill { .. } => 3,
        }
    }
}

/// Recorded vs replayed fill of one market intent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FillComparison {
    pub intent_id: String,
    pub recorded: (f64, f64, f64, f64),
    /// `(arrival, fill, fee, slippage_bps)` replayed; `None` when the replay had no price.
    pub replayed: Option<VenueFill>,
    /// |replayed − recorded| fill price in bps of the recorded fill.
    pub fill_diff_bps: f64,
    pub arrival_equal: bool,
}

/// Summary of [`replay_window`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IdentityReport {
    pub intents: usize,
    pub market_fills: usize,
    pub compared: usize,
    pub arrival_equal: usize,
    pub max_fill_diff_bps: f64,
    pub mean_fill_diff_bps: f64,
    pub max_fee_diff: f64,
    pub max_slippage_diff_bps: f64,
    pub comparisons: Vec<FillComparison>,
}

/// Replays a recorded window through a fresh [`PaperVenue`] and compares every recorded fill of
/// a market intent (exec algo `market` or empty, a single `FILL` event) with the replayed fill
/// at the intent's processing time. Records are sorted by `(t, market data first)`; the replay
/// starts at the last `reset` (the gateway's start) when the window holds one.
pub fn replay_window(costs: CostModel, records: &[VenueRecord]) -> IdentityReport {
    let mut recs: Vec<&VenueRecord> = records.iter().collect();
    recs.sort_by_key(|r| (r.t(), r.rank()));
    let mut venue = PaperVenue::new(costs.clone());
    let mut quoted: BTreeMap<String, VenueFill> = BTreeMap::new();
    let mut report = IdentityReport::default();
    for r in recs {
        match r {
            VenueRecord::Reset { .. } => {
                venue = PaperVenue::new(costs.clone());
                quoted.clear();
                report = IdentityReport::default();
            }
            VenueRecord::Trade { t, at, venue: v, token, price, size } => {
                venue.observe_trade(v, token, *price, *size, at.unwrap_or(*t));
            }
            VenueRecord::Quote { token, bid, ask, .. } => venue.observe_quote(token, *bid, *ask),
            VenueRecord::Intent { t, intent_id, instrument_id, sell, qty, exec_algo } => {
                report.intents += 1;
                let algo = exec_algo.split(':').next().unwrap_or_default();
                if (algo.is_empty() || algo == "market")
                    && let Some(f) = venue.market_fill(instrument_id, *sell, *qty, *t)
                {
                    quoted.insert(intent_id.clone(), f);
                }
            }
            VenueRecord::Fill { intent_id, fill_price, fee, arrival_price, slippage_bps, .. } => {
                let Some(f) = quoted.remove(intent_id) else { continue };
                report.market_fills += 1;
                let diff = (f.fill_price - fill_price).abs() / fill_price.abs().max(f64::MIN_POSITIVE) * 10_000.0;
                let arrival_equal = (f.arrival_price - arrival_price).abs() <= 1e-9 * arrival_price.abs().max(1.0);
                report.compared += 1;
                report.arrival_equal += usize::from(arrival_equal);
                report.max_fill_diff_bps = report.max_fill_diff_bps.max(diff);
                report.mean_fill_diff_bps += diff;
                report.max_fee_diff = report.max_fee_diff.max((f.fee - fee).abs());
                report.max_slippage_diff_bps = report.max_slippage_diff_bps.max((f.slippage_bps - slippage_bps).abs());
                report.comparisons.push(FillComparison {
                    intent_id: intent_id.clone(),
                    recorded: (*arrival_price, *fill_price, *fee, *slippage_bps),
                    replayed: Some(f),
                    fill_diff_bps: diff,
                    arrival_equal,
                });
            }
        }
    }
    if report.compared > 0 {
        report.mean_fill_diff_bps /= report.compared as f64;
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn costs() -> CostModel {
        CostModel::from_yaml(include_str!("../../../go/services/broker-gateway/costs.yaml")).unwrap()
    }

    #[test]
    fn estimators_follow_the_gateway() {
        let mut v = PaperVenue::new(costs());
        let inst = "9daa8cea-67a6-5f6b-8347-f9e5a30f7463";
        let tok = subject_token(inst);
        assert!(v.market_fill(inst, false, 1.0, 0).is_none(), "no price yet");
        for (k, p) in [100.0, 101.0, 100.5, 100.2, 100.9, 101.3].iter().enumerate() {
            v.observe_trade("yahoo", &tok, *p, 10.0, 1_000 * k as i64);
        }
        let f = v.market_fill(inst, false, 10.0, 5_000).unwrap();
        assert_eq!(f.arrival_price, 101.3);
        assert!(f.sigma_day > 0.0, "5 returns seen");
        // 6 trades of ~1 000 EUR over 5 s, extrapolated to a day
        let notional: f64 = [100.0, 101.0, 100.5, 100.2, 100.9, 101.3].iter().map(|p| p * 10.0).sum();
        assert!((f.adv_eur - notional * 86_400.0 / 5.0).abs() < 1e-6);
        assert!(f.fill_price > 101.3 && f.slippage_bps > 0.0 && f.fee >= 1.0);
        v.observe_quote(&tok, 101.0, 101.2);
        let g = v.market_fill(inst, true, 10.0, 5_000).unwrap();
        assert!(g.fill_price < 101.3 - 0.1 + 1e-12, "sell pays half the quoted spread plus impact");
    }

    #[test]
    fn the_sim_venue_is_costless_and_replays_exactly() {
        let inst = "01900000-0000-7000-8000-000000000001";
        let tok = subject_token(inst);
        let recs = vec![
            VenueRecord::Trade { t: -10, at: None, venue: "sim".into(), token: tok.clone(), price: 1.0, size: 1.0 },
            VenueRecord::Reset { t: -5 },
            VenueRecord::Trade { t: 0, at: None, venue: "sim".into(), token: tok.clone(), price: 100.0, size: 1.0 },
            VenueRecord::Intent {
                t: 5,
                intent_id: "i1".into(),
                instrument_id: inst.into(),
                sell: false,
                qty: 3.0,
                exec_algo: "market".into(),
            },
            VenueRecord::Trade { t: 5, at: Some(4), venue: "sim".into(), token: tok, price: 100.5, size: 1.0 },
            VenueRecord::Fill {
                t: 6,
                intent_id: "i1".into(),
                fill_price: 100.5,
                fill_qty: 3.0,
                fee: 0.0,
                arrival_price: 100.5,
                slippage_bps: 0.0,
            },
        ];
        let r = replay_window(costs(), &recs);
        assert_eq!((r.intents, r.compared, r.arrival_equal), (1, 1, 1), "trade at the same ms is seen first");
        assert_eq!(r.max_fill_diff_bps, 0.0);
    }
}
