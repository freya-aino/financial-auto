//! Pure strategy stepping logic (no I/O): prices, latent graph and decoder outputs in,
//! exposure distributions out. Population hot reload and pause/resume live here too.
use std::collections::{BTreeMap, BTreeSet};

use fa_core::{InstrumentId, Timescale};
use fa_proto::v1::{DecoderOutput, LatentState, RegionEvent, TransitionUpdate};
use fa_strategies::{
    AssetExposure, ExposureDistribution, GpStrategy, Strategy, StrategyContext,
    decoders::{DEFAULT_MAX_AGE_MS, DecoderOutputs},
    features::{FallbackContext, LatentInfo, LiveContext, PriceSeries, Sources, VirtualBook},
    graph::LatentGraph,
    population::{PopulationError, StrategyDef, StrategyFile},
};
use uuid::Uuid;

/// Max trades kept per instrument (enough for 64 one-minute bars at ~10 trades/s).
const MAX_POINTS: usize = 40_000;

/// Per-strategy gauges for the heartbeat.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MemberStats {
    /// Mean of the last emitted exposure means across instruments.
    pub last_mean: f64,
    /// Mean of the last emitted stddevs across instruments.
    pub last_std: f64,
    pub steps: u64,
    pub sources: Sources,
}

struct Member {
    def: StrategyDef,
    strategy: GpStrategy,
    next_step_ms: i64,
    books: BTreeMap<InstrumentId, VirtualBook>,
    last_price: BTreeMap<InstrumentId, f64>,
    stats: MemberStats,
}

impl Member {
    fn new(def: &StrategyDef) -> Result<Self, PopulationError> {
        Ok(Self {
            def: def.clone(),
            strategy: def.build()?,
            next_step_ms: 0,
            books: BTreeMap::new(),
            last_price: BTreeMap::new(),
            stats: MemberStats::default(),
        })
    }
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
    /// Legacy expert-0 summary per (timescale, instrument), the fallback's `LatentInfo`.
    latent: BTreeMap<(Timescale, InstrumentId), Latent>,
    /// Last closed bar bucket fed to the graph per (timescale, instrument).
    last_bucket: BTreeMap<(Timescale, InstrumentId), i64>,
    pub graph: LatentGraph,
    pub decoders: DecoderOutputs,
    paused: BTreeSet<String>,
    /// Label of the loaded population (`baselines`, a path or an artifact key).
    pub population: String,
    pub reloads: u64,
    pub signals_emitted: u64,
    pub trades_seen: u64,
    pub latent_seen: u64,
    pub decoder_max_age_ms: i64,
}

impl Runtime {
    pub fn new(file: &StrategyFile) -> Result<Self, PopulationError> {
        let members = file.strategy.iter().map(Member::new).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            members,
            allowed: Self::allowed_of(file),
            prices: BTreeMap::new(),
            latent: BTreeMap::new(),
            last_bucket: BTreeMap::new(),
            graph: LatentGraph::new(),
            decoders: DecoderOutputs::new(),
            paused: BTreeSet::new(),
            population: "baselines".into(),
            reloads: 0,
            signals_emitted: 0,
            trades_seen: 0,
            latent_seen: 0,
            decoder_max_age_ms: DEFAULT_MAX_AGE_MS,
        })
    }

    fn allowed_of(file: &StrategyFile) -> BTreeSet<InstrumentId> {
        file.instruments.iter().filter_map(|s| Uuid::parse_str(s).ok()).map(InstrumentId).collect()
    }

    pub fn strategy_ids(&self) -> Vec<&str> {
        self.members.iter().map(|m| m.def.id.as_str()).collect()
    }

    pub fn instruments(&self) -> usize {
        self.prices.len()
    }

    pub fn paused(&self) -> impl Iterator<Item = &str> {
        self.paused.iter().map(String::as_str)
    }

    pub fn is_paused(&self, id: &str) -> bool {
        self.paused.contains(id)
    }

    /// Per-strategy gauges (id, stats).
    pub fn member_stats(&self) -> Vec<(&str, MemberStats)> {
        self.members.iter().map(|m| (m.def.id.as_str(), m.stats)).collect()
    }

    /// Primitive source counters summed over the population.
    pub fn sources(&self) -> Sources {
        let mut s = Sources::default();
        for m in &self.members {
            s.add(m.stats.sources);
        }
        s
    }

    /// Replaces the population atomically (hot reload). Every program is validated first; on
    /// any error the running population is untouched. Strategies whose id survives keep their
    /// virtual books and cadence; the paused set is kept for surviving ids.
    pub fn replace_population(&mut self, file: &StrategyFile, label: &str) -> Result<Vec<String>, PopulationError> {
        let mut fresh = file.strategy.iter().map(Member::new).collect::<Result<Vec<_>, _>>()?;
        let mut old: BTreeMap<String, Member> =
            std::mem::take(&mut self.members).into_iter().map(|m| (m.def.id.clone(), m)).collect();
        for m in &mut fresh {
            if let Some(prev) = old.remove(&m.def.id) {
                m.books = prev.books;
                m.last_price = prev.last_price;
                m.next_step_ms = prev.next_step_ms;
                m.stats = prev.stats;
            }
        }
        let ids: Vec<String> = fresh.iter().map(|m| m.def.id.clone()).collect();
        self.paused.retain(|p| ids.contains(p));
        self.members = fresh;
        self.allowed = Self::allowed_of(file);
        self.population = label.to_owned();
        self.reloads += 1;
        Ok(ids)
    }

    /// Pauses a strategy: it keeps stepping but emits `mean = 0, stddev = 1` so downstream flattens.
    pub fn pause(&mut self, id: &str) -> bool {
        if self.members.iter().any(|m| m.def.id == id) {
            self.paused.insert(id.to_owned());
            true
        } else {
            false
        }
    }

    pub fn resume(&mut self, id: &str) -> bool {
        if self.members.iter().any(|m| m.def.id == id) {
            self.paused.remove(id);
            true
        } else {
            false
        }
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
        self.graph.on_state(state);
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

    pub fn on_region(&mut self, ev: &RegionEvent, now_ms: i64) {
        self.graph.on_region(ev, now_ms);
    }

    pub fn on_transition(&mut self, tu: &TransitionUpdate) {
        self.graph.on_transition(tu);
    }

    pub fn on_decoder_output(&mut self, out: &DecoderOutput, now_ms: i64) {
        if let Some(inst) = DecoderOutputs::instrument_of(out)
            && !self.tracks(inst)
        {
            return;
        }
        self.decoders.on_output(out, now_ms);
    }

    /// Feeds closed bars to the graph's region statistics: for every timescale in the population
    /// and every instrument, the bar before the current bucket closes once.
    fn feed_closes(&mut self, now_ms: i64) {
        let timescales: BTreeSet<Timescale> = self.members.iter().map(|m| m.def.timescale).collect();
        let instruments: Vec<InstrumentId> = self.prices.keys().copied().collect();
        for ts in timescales {
            let bar_ms = ts.duration().as_millis() as i64;
            let bucket = now_ms.div_euclid(bar_ms);
            for inst in &instruments {
                let key = (ts, *inst);
                let last = self.last_bucket.get(&key).copied();
                if last.is_some_and(|b| b >= bucket) {
                    continue;
                }
                let close_at = bucket * bar_ms;
                let Some(price) = self.prices.get(inst).and_then(|s| s.price_at(close_at - 1)) else {
                    continue;
                };
                self.last_bucket.insert(key, bucket);
                self.graph.record_close(ts, *inst, close_at, price);
            }
        }
    }

    /// Steps every strategy whose cadence is due at `now_ms`; returns the distributions to publish.
    pub fn tick(&mut self, now_ms: i64) -> Vec<ExposureDistribution> {
        let mut out = Vec::new();
        let instruments: Vec<InstrumentId> = self.prices.keys().copied().collect();
        if instruments.is_empty() {
            return out;
        }
        self.feed_closes(now_ms);
        let paused = self.paused.clone();
        let (graph, decoders, prices, latent) = (&self.graph, &self.decoders, &self.prices, &self.latent);
        let max_age = self.decoder_max_age_ms;
        for m in &mut self.members {
            if now_ms < m.next_step_ms {
                continue;
            }
            let step_ms = (m.def.step_secs() * 1000) as i64;
            m.next_step_ms = now_ms.div_euclid(step_ms) * step_ms + step_ms;

            let ts = m.def.timescale;
            let view = graph.at(ts, now_ms);
            let contexts: Vec<(InstrumentId, LiveContext<'_>)> = instruments
                .iter()
                .filter_map(|inst| {
                    let series = prices.get(inst)?;
                    let closes = series.closes(ts, m.def.lookback_bars + 1);
                    let price = *closes.last()?;
                    let mut book = m.books.get(inst).copied().unwrap_or_default();
                    if let Some(prev) = m.last_price.get(inst)
                        && *prev > 0.0
                    {
                        book.unrealised += book.exposure * (price / prev).ln();
                    }
                    m.last_price.insert(*inst, price);
                    let info = latent.get(&(ts, *inst)).map(|l| l.info.clone());
                    let ctx = LiveContext::new(FallbackContext::new(&closes, info, book), *inst, m.def.steps)
                        .with_graph(&view)
                        .with_decoders(decoders.at(*inst, now_ms, max_age))
                        .with_prices(series);
                    Some((*inst, ctx))
                })
                .collect();
            let refs: Vec<(InstrumentId, &dyn StrategyContext)> =
                contexts.iter().map(|(i, c)| (*i, c as &dyn StrategyContext)).collect();
            let mut dist = m.strategy.step(now_ms, &refs);
            let is_paused = paused.contains(&m.def.id);
            for e in &mut dist.exposures {
                sanitize(e);
                if is_paused {
                    e.mean = 0.0;
                    e.stddev = 1.0;
                }
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
            let n = dist.exposures.len().max(1) as f64;
            m.stats.last_mean = dist.exposures.iter().map(|e| e.mean).sum::<f64>() / n;
            m.stats.last_std = dist.exposures.iter().map(|e| e.stddev).sum::<f64>() / n;
            m.stats.steps += 1;
            for (_, c) in &contexts {
                m.stats.sources.add(c.sources());
            }
            self.signals_emitted += 1;
            out.push(dist);
        }
        out
    }
}

/// spec 35 §8: NaN / out-of-range → 0 exposure.
fn sanitize(e: &mut AssetExposure) {
    if !e.mean.is_finite() {
        e.mean = 0.0;
    }
    e.mean = e.mean.clamp(-1.0, 1.0);
    if !e.stddev.is_finite() {
        e.stddev = 1.0;
    }
    e.stddev = e.stddev.clamp(0.0, 1.0);
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::{ExpertLatent, ExpertSpaceRef, TransitionProb};

    use super::*;

    fn runtime() -> Runtime {
        Runtime::new(&StrategyFile::baselines()).unwrap()
    }

    fn ts(ms: i64) -> fa_proto::Timestamp {
        fa_proto::Timestamp { seconds: ms.div_euclid(1000), nanos: (ms.rem_euclid(1000) * 1_000_000) as i32 }
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
            5,
            "1s strategies step every second"
        );
        assert!(!second.iter().any(|d| d.strategy_id == "baseline-regime-1m"), "10 s cadence");
        assert!(rt.tick(50_000).iter().any(|d| d.strategy_id == "baseline-regime-1m"));
        // price only → every primitive came from the fallback
        let s = rt.sources();
        assert!(s.fallback > 0 && s.graph == 0 && s.decoder == 0, "{s:?}");
        let (_, st) = rt.member_stats().into_iter().find(|(id, _)| *id == "baseline-momentum-1s").unwrap();
        assert_eq!(st.steps, 3);
        assert!(st.last_mean > 0.0);
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
        assert_eq!(rt.graph.seen.states, 4);
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

    /// Feeds a 1s graph: regions a/b, rows a→b 0.8, b→a 0.9, the instrument in `a`.
    fn feed_graph(rt: &mut Runtime, inst: InstrumentId, at_ms: i64) {
        let space = Some(ExpertSpaceRef { snapshot_id: "stat-v1-1s".into(), timescale: "1s".into(), expert_index: 0 });
        for r in ["a", "b"] {
            rt.on_region(
                &RegionEvent {
                    kind: 1,
                    space: space.clone(),
                    region_id: r.into(),
                    lineage_id: r.into(),
                    ..Default::default()
                },
                at_ms,
            );
        }
        let row = |from: &str, to: &[(&str, f64)]| TransitionUpdate {
            space: space.clone(),
            from_region_id: from.into(),
            to: to
                .iter()
                .map(|(r, m)| TransitionProb { region_id: (*r).into(), mean: *m, lower: m - 0.05, upper: m + 0.05 })
                .collect(),
            sampler_id: "conjugate".into(),
            as_of: Some(ts(at_ms)),
        };
        rt.on_transition(&row("a", &[("b", 0.8), ("a", 0.1), ("unknown", 0.1)]));
        rt.on_transition(&row("b", &[("a", 0.9), ("unknown", 0.1)]));
        rt.on_latent(&LatentState {
            timescale: "1s".into(),
            instrument_id: inst.to_string(),
            as_of: Some(ts(at_ms)),
            experts: vec![ExpertLatent { region_id: "a".into(), log_var: vec![-4.0], ..Default::default() }],
            ..Default::default()
        });
    }

    #[test]
    fn graph_feed_makes_graph_primitives_real_and_feeds_region_returns() {
        let mut rt = runtime();
        let inst = InstrumentId::new();
        for i in 0..40 {
            rt.on_trade(inst, i * 1_000, 100.0 + i as f64 * 0.1);
        }
        feed_graph(&mut rt, inst, 30_000);
        let dists = rt.tick(40_000);
        let nb = dists.iter().find(|d| d.strategy_id == "baseline-graph-neighbors-1s").unwrap();
        // p_to_set(a, neighbors(a, 3), 1) = mass to b = 0.8 → signal 4·(0.8 − 0.5) = 1.2 → tanh
        assert!((nb.exposures[0].mean - 1.2f64.tanh()).abs() < 0.02, "{}", nb.exposures[0].mean);
        let s = rt.sources();
        assert!(s.graph > 0, "{s:?}");
        let (_, st) = rt.member_stats().into_iter().find(|(id, _)| *id == "baseline-graph-neighbors-1s").unwrap();
        assert_eq!(st.sources.fallback, 0, "{:?}", st.sources);
        assert_eq!(st.sources.graph, 2);
        // the bar before bucket 40 closed → one close recorded per population timescale
        assert_eq!(rt.graph.seen.closes, 0, "first close only seeds the previous price");
        rt.on_trade(inst, 41_000, 104.5);
        rt.tick(41_000);
        assert_eq!(rt.graph.seen.closes, 1);
        rt.tick(41_000);
        assert_eq!(rt.graph.seen.closes, 1, "a bucket closes once");
        // region a now has a realised return → region_feature answers from the graph
        let view = rt.graph.at(Timescale::Second, 42_000);
        assert!(view.region_feature(0, "a", fa_strategies::RegionFeature::MeanFwdReturn).is_some());
    }

    #[test]
    fn reload_swaps_population_atomically_and_keeps_books() {
        let mut rt = runtime();
        let inst = InstrumentId::new();
        for i in 0..40 {
            rt.on_trade(inst, i * 1_000, 100.0 + i as f64 * 0.1);
        }
        rt.tick(40_000);
        let before = rt.members.iter().find(|m| m.def.id == "baseline-momentum-1s").unwrap().books[&inst];
        assert!(rt.pause("baseline-reversion-1s"));
        assert!(!rt.pause("nope"));

        // invalid file → untouched
        let bad = StrategyFile::parse("[[strategy]]\nid='x'\ntimescale='1s'\nsignal='1'\n").unwrap();
        let mut bad = bad;
        bad.strategy[0].signal = "(rsi 14)".into();
        assert!(rt.replace_population(&bad, "bad").is_err());
        assert_eq!(rt.strategy_ids().len(), 7);
        assert_eq!(rt.reloads, 0);

        let next = StrategyFile::parse(
            "[[strategy]]\nid='baseline-momentum-1s'\ngenome_id='gen-3-abc'\ntimescale='1s'\nsignal='(mul 400 analog_forecast)'\n\
             [[strategy]]\nid='evolved-1'\ngenome_id='gen-3-def'\ntimescale='1s'\nsignal='(p_to_set (cur_region 0) (neighbors (cur_region 0) 2) 1)'\n",
        )
        .unwrap();
        let ids = rt.replace_population(&next, "strategies/strategies.3.toml").unwrap();
        assert_eq!(ids, vec!["baseline-momentum-1s", "evolved-1"]);
        assert_eq!(rt.reloads, 1);
        assert_eq!(rt.population, "strategies/strategies.3.toml");
        assert_eq!(rt.paused().count(), 0, "paused id that left the population is forgotten");
        let kept = rt.members.iter().find(|m| m.def.id == "baseline-momentum-1s").unwrap();
        assert_eq!(kept.books[&inst], before, "surviving id keeps its virtual book");
        assert_eq!(kept.strategy.genome_id, "gen-3-abc");
        let dists = rt.tick(41_000);
        assert_eq!(dists.len(), 2);
        assert_eq!(dists[0].genome_id, "gen-3-abc");

        // pause → flat with full uncertainty; resume → signal again
        assert!(rt.pause("evolved-1"));
        let d = rt.tick(42_000);
        let e = d.iter().find(|d| d.strategy_id == "evolved-1").unwrap().exposures[0];
        assert_eq!((e.mean, e.stddev), (0.0, 1.0));
        assert!(rt.resume("evolved-1"));
        assert!(!rt.is_paused("evolved-1"));
    }
}
