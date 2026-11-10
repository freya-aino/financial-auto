//! Pure strategy stepping logic (no I/O): price/latent state in, exposure distributions out.
use std::collections::{BTreeMap, BTreeSet};

use fa_core::{InstrumentId, Timescale};
use fa_proto::v1::LatentState;
use fa_strategies::{
    ExposureDistribution, GpStrategy, Strategy, StrategyContext,
    features::{FallbackContext, LatentInfo, PriceSeries, VirtualBook},
    population::{PopulationError, StrategyDef, StrategyFile},
};
use uuid::Uuid;

/// Max trades kept per instrument (enough for 64 one-minute bars at ~10 trades/s).
const MAX_POINTS: usize = 40_000;

struct Member {
    def: StrategyDef,
    strategy: GpStrategy,
    next_step_ms: i64,
    books: BTreeMap<InstrumentId, VirtualBook>,
    last_price: BTreeMap<InstrumentId, f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct Latent {
    info: LatentInfo,
}

/// Deterministic strategy population runtime.
pub struct Runtime {
    members: Vec<Member>,
    /// Instruments restricted by the strategy file; empty = all seen.
    allowed: BTreeSet<InstrumentId>,
    prices: BTreeMap<InstrumentId, PriceSeries>,
    latent: BTreeMap<(Timescale, InstrumentId), Latent>,
    pub signals_emitted: u64,
    pub trades_seen: u64,
    pub latent_seen: u64,
}

impl Runtime {
    pub fn new(file: &StrategyFile) -> Result<Self, PopulationError> {
        let members = file
            .strategy
            .iter()
            .map(|def| {
                let strategy = def.build()?;
                Ok::<_, PopulationError>(Member {
                    def: def.clone(),
                    strategy,
                    next_step_ms: 0,
                    books: BTreeMap::new(),
                    last_price: BTreeMap::new(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let allowed = file.instruments.iter().filter_map(|s| Uuid::parse_str(s).ok()).map(InstrumentId).collect();
        Ok(Self {
            members,
            allowed,
            prices: BTreeMap::new(),
            latent: BTreeMap::new(),
            signals_emitted: 0,
            trades_seen: 0,
            latent_seen: 0,
        })
    }

    pub fn strategy_ids(&self) -> Vec<&str> {
        self.members.iter().map(|m| m.def.id.as_str()).collect()
    }

    pub fn instruments(&self) -> usize {
        self.prices.len()
    }

    fn tracks(&self, inst: InstrumentId) -> bool {
        self.allowed.is_empty() || self.allowed.contains(&inst)
    }

    pub fn on_trade(&mut self, inst: InstrumentId, at_ms: i64, price: f64) {
        if !self.tracks(inst) {
            return;
        }
        self.trades_seen += 1;
        self.prices.entry(inst).or_insert_with(|| PriceSeries::new(MAX_POINTS)).push(at_ms, price);
    }

    pub fn on_latent(&mut self, state: &LatentState) {
        let (Ok(ts), Ok(uuid)) = (state.timescale.parse::<Timescale>(), Uuid::parse_str(&state.instrument_id)) else {
            return;
        };
        let inst = InstrumentId(uuid);
        if !self.tracks(inst) {
            return;
        }
        let Some(e) = state.experts.first() else {
            return;
        };
        self.latent_seen += 1;
        let uncertainty = if e.log_var.is_empty() {
            0.0
        } else {
            e.log_var.iter().map(|lv| (f64::from(*lv) / 2.0).exp()).sum::<f64>() / e.log_var.len() as f64
        };
        let dwell = match self.latent.get(&(ts, inst)) {
            Some(prev) if prev.info.region_id == e.region_id => prev.info.dwell.saturating_add(1),
            _ => 0,
        };
        self.latent
            .insert((ts, inst), Latent { info: LatentInfo { region_id: e.region_id.clone(), uncertainty, dwell } });
    }

    /// Steps every strategy whose cadence is due at `now_ms`; returns the distributions to publish.
    pub fn tick(&mut self, now_ms: i64) -> Vec<ExposureDistribution> {
        let mut out = Vec::new();
        let instruments: Vec<InstrumentId> = self.prices.keys().copied().collect();
        if instruments.is_empty() {
            return out;
        }
        for m in &mut self.members {
            if now_ms < m.next_step_ms {
                continue;
            }
            let step_ms = (m.def.step_secs() * 1000) as i64;
            m.next_step_ms = now_ms.div_euclid(step_ms) * step_ms + step_ms;

            let ts = m.def.timescale;
            let contexts: Vec<(InstrumentId, FallbackContext)> = instruments
                .iter()
                .filter_map(|inst| {
                    let series = self.prices.get(inst)?;
                    let closes = series.closes(ts, m.def.lookback_bars + 1);
                    let price = *closes.last()?;
                    let mut book = m.books.get(inst).copied().unwrap_or_default();
                    if let Some(prev) = m.last_price.get(inst)
                        && *prev > 0.0
                    {
                        book.unrealised += book.exposure * (price / prev).ln();
                    }
                    m.last_price.insert(*inst, price);
                    let latent = self.latent.get(&(ts, *inst)).map(|l| l.info.clone());
                    Some((*inst, FallbackContext::new(&closes, latent, book)))
                })
                .collect();
            let refs: Vec<(InstrumentId, &dyn StrategyContext)> =
                contexts.iter().map(|(i, c)| (*i, c as &dyn StrategyContext)).collect();
            let mut dist = m.strategy.step(now_ms, &refs);
            for e in &mut dist.exposures {
                // spec 35 §8: NaN / out-of-range → 0 exposure
                if !e.mean.is_finite() {
                    e.mean = 0.0;
                }
                e.mean = e.mean.clamp(-1.0, 1.0);
                if !e.stddev.is_finite() {
                    e.stddev = 1.0;
                }
                e.stddev = e.stddev.clamp(0.0, 1.0);
                let (_, ctx) = contexts.iter().find(|(i, _)| *i == e.instrument).expect("context exists");
                let prev = m.books.get(&e.instrument).copied().unwrap_or_default();
                let flipped = (e.mean == 0.0) != (prev.exposure == 0.0) || e.mean.signum() != prev.exposure.signum();
                let book = if flipped {
                    VirtualBook { exposure: e.mean, unrealised: 0.0, steps_in_trade: 0 }
                } else {
                    VirtualBook {
                        exposure: e.mean,
                        unrealised: ctx.unrealised_pnl(),
                        steps_in_trade: prev.steps_in_trade.saturating_add(1),
                    }
                };
                m.books.insert(e.instrument, book);
            }
            self.signals_emitted += 1;
            out.push(dist);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::ExpertLatent;

    use super::*;

    fn runtime() -> Runtime {
        Runtime::new(&StrategyFile::baselines()).unwrap()
    }

    #[test]
    fn no_signals_without_trades_then_signals_at_cadence() {
        let mut rt = runtime();
        assert!(rt.tick(1_000).is_empty());
        let inst = InstrumentId::new();
        for i in 0..40 {
            rt.on_trade(inst, i * 1_000, 100.0 + i as f64 * 0.1);
        }
        let first = rt.tick(40_000);
        assert_eq!(first.len(), rt.strategy_ids().len(), "every strategy steps on first due tick");
        let s = first.iter().find(|d| d.strategy_id == "baseline-momentum-1s").unwrap();
        assert_eq!(s.exposures.len(), 1);
        assert!(s.exposures[0].mean > 0.0, "rising prices → positive momentum exposure");
        let rev = first.iter().find(|d| d.strategy_id == "baseline-reversion-1s").unwrap();
        assert!(rev.exposures[0].mean < 0.0);
        assert!(rt.tick(40_500).is_empty(), "nothing due half a second later");
        let second = rt.tick(41_000);
        assert_eq!(
            second.iter().filter(|d| d.horizon.timescale == Timescale::Second).count(),
            4,
            "1s strategies step every second"
        );
        assert!(!second.iter().any(|d| d.strategy_id == "baseline-regime-1m"), "10 s cadence");
        assert!(rt.tick(50_000).iter().any(|d| d.strategy_id == "baseline-regime-1m"));
    }

    #[test]
    fn instrument_filter_and_latent_dwell() {
        let allowed = InstrumentId::new();
        let other = InstrumentId::new();
        let mut file = StrategyFile::baselines();
        file.instruments = vec![allowed.to_string()];
        let mut rt = Runtime::new(&file).unwrap();
        rt.on_trade(other, 0, 1.0);
        rt.on_trade(allowed, 0, 1.0);
        assert_eq!(rt.instruments(), 1);
        let st = |region: &str| LatentState {
            timescale: "1s".into(),
            instrument_id: allowed.to_string(),
            experts: vec![ExpertLatent { region_id: region.into(), log_var: vec![0.0], ..Default::default() }],
            ..Default::default()
        };
        rt.on_latent(&st("r1"));
        rt.on_latent(&st("r1"));
        rt.on_latent(&st("r2"));
        assert_eq!(rt.latent_seen, 3);
        assert_eq!(rt.latent[&(Timescale::Second, allowed)].info.dwell, 0);
        rt.on_latent(&st("r2"));
        assert_eq!(rt.latent[&(Timescale::Second, allowed)].info.dwell, 1);
        assert_eq!(rt.latent[&(Timescale::Second, allowed)].info.uncertainty, 1.0);
    }

    #[test]
    fn virtual_book_tracks_own_exposure() {
        let mut rt = runtime();
        let inst = InstrumentId::new();
        for i in 0..40 {
            rt.on_trade(inst, i * 1_000, 100.0 + i as f64 * 0.1);
        }
        rt.tick(40_000);
        rt.on_trade(inst, 41_000, 105.0);
        rt.tick(41_000);
        let m = rt.members.iter().find(|m| m.def.id == "baseline-momentum-1s").unwrap();
        let b = m.books[&inst];
        assert!(b.exposure > 0.0);
        assert_eq!(b.steps_in_trade, 1);
        assert!(b.unrealised > 0.0, "long exposure into an up move");
    }
}
