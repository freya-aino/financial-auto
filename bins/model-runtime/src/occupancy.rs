//! Region occupancy over time (spec 30 §5): the `region_weight` series in VictoriaMetrics and
//! the region episodes in Helix.
//!
//! Every **closed bar** of an occupancy timescale (`--occupancy-timescales`, default
//! `1m,1h,1d,1w`; spec 30 §10 q6: not 1s) the hub sink hands the step's soft memberships to
//! [`spawn`]'s task, which
//!
//! - writes one sample per (expert, region in the soft membership) as
//!   `fa_region_weight{tf, inst, expert, region}` = membership weight at the bar close
//!   (`expert` = `<snapshot>-e<k>`, `inst` = the instrument token). Only the `k` (4) regions of
//!   each membership are written, so active series per instrument are bounded by the regions it
//!   visits; samples are batched every 10 s, a failed import is retried with the next batch and
//!   the buffer is bounded (oldest dropped, `occupancy.vm_dropped`);
//! - tracks an **episode** per (timescale, instrument, expert): the run of consecutive closed bars
//!   whose hard region (largest weight) is the same. When the hard region changes, the finished
//!   episode (`region_id`, `instrument_id`, `t_enter` = close of its first bar, `t_exit` = close
//!   of the bar that left it, `bars`, `max_weight`, `mean_weight`) goes to the sink as a
//!   `RegionEpisode` row (deferred like every Helix write while Helix is down). Episodes still
//!   open at shutdown are not written. Retention: model-runtime's prune job drops episodes older
//!   than the timescale's `--hub-retention` horizon (the observation latents' horizon).

use std::collections::{BTreeSet, HashMap, VecDeque};

use fa_core::Timescale;
use fa_hub::NewRegionEpisode;
use fa_proto::v1::RegionMembership;
use fa_tsdb::{Sample, Tsdb, inst_token};
use tokio::sync::mpsc;

use crate::hub_sink::{Counters, HubEvent, expert_space_id};

/// Metric name of the occupancy series.
pub const REGION_WEIGHT: &str = "fa_region_weight";
const FLUSH_EVERY: std::time::Duration = std::time::Duration::from_secs(10);
const MAX_BUFFER: usize = 200_000;

#[derive(Debug, Clone)]
pub struct OccupancyCfg {
    pub timescales: BTreeSet<Timescale>,
    /// VictoriaMetrics base URL; `None` → no `region_weight` series (episodes still tracked).
    pub vm_url: Option<String>,
}

#[derive(Debug, Clone)]
struct Open {
    region: String,
    space: String,
    t_enter_ms: i64,
    bars: u32,
    max_w: f64,
    sum_w: f64,
}

/// Pure part: samples and finished episodes of one membership.
#[derive(Debug, Default)]
pub struct Occupancy {
    open: HashMap<(String, String, u32), Open>,
}

fn ms_of(m: &RegionMembership) -> i64 {
    m.as_of.as_ref().map_or(0, |t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)
}

impl Occupancy {
    /// The VM samples of one closed-bar membership and the episodes it closes.
    pub fn observe(&mut self, m: &RegionMembership) -> (Vec<Sample>, Vec<NewRegionEpisode>) {
        let ts_ms = ms_of(m);
        let inst = inst_token(&m.instrument_id);
        let mut samples = Vec::new();
        let mut done = Vec::new();
        for e in &m.experts {
            let space = expert_space_id(&m.snapshot_id, e.expert_index);
            for w in &e.weights {
                samples.push(
                    Sample::new(REGION_WEIGHT, f64::from(w.weight), ts_ms)
                        .label("tf", &m.timescale)
                        .label("inst", &inst)
                        .label("expert", &space)
                        .label("region", &w.region_id),
                );
            }
            let Some(top) = e.weights.iter().max_by(|a, b| a.weight.total_cmp(&b.weight)) else { continue };
            let w = f64::from(top.weight);
            let key = (m.timescale.clone(), m.instrument_id.clone(), e.expert_index);
            match self.open.get_mut(&key) {
                Some(o) if o.region == top.region_id && o.space == space => {
                    o.bars += 1;
                    o.max_w = o.max_w.max(w);
                    o.sum_w += w;
                }
                prev => {
                    let next = Open {
                        region: top.region_id.clone(),
                        space: space.clone(),
                        t_enter_ms: ts_ms,
                        bars: 1,
                        max_w: w,
                        sum_w: w,
                    };
                    if let Some(o) = prev {
                        let o = std::mem::replace(o, next);
                        done.push(NewRegionEpisode {
                            region_id: o.region,
                            expert_space_id: o.space,
                            timescale: m.timescale.clone(),
                            instrument_id: m.instrument_id.clone(),
                            t_enter_ms: o.t_enter_ms,
                            t_exit_ms: ts_ms,
                            bars: o.bars,
                            max_weight: o.max_w,
                            mean_weight: o.sum_w / f64::from(o.bars.max(1)),
                        });
                    } else {
                        self.open.insert(key, next);
                    }
                }
            }
        }
        (samples, done)
    }

    pub fn open_episodes(&self) -> usize {
        self.open.len()
    }
}

/// Handle used by the hub sink.
#[derive(Clone)]
pub struct OccupancyHandle {
    tx: mpsc::Sender<RegionMembership>,
    timescales: BTreeSet<Timescale>,
}

impl OccupancyHandle {
    pub fn wants(&self, ts: Timescale) -> bool {
        self.timescales.contains(&ts)
    }

    pub fn send(&self, m: RegionMembership) {
        let _ = self.tx.try_send(m);
    }
}

async fn bump(counters: &Counters, key: &str, by: f64) {
    *counters.lock().await.entry(key.to_owned()).or_default() += by;
}

/// Starts the occupancy task; finished episodes go to `sink` as [`HubEvent::Episodes`].
pub fn spawn(cfg: OccupancyCfg, sink: mpsc::Sender<HubEvent>, counters: Counters) -> OccupancyHandle {
    let (tx, mut rx) = mpsc::channel::<RegionMembership>(4096);
    let timescales = cfg.timescales.clone();
    tokio::spawn(async move {
        let vm = cfg.vm_url.as_deref().map(Tsdb::new);
        tracing::info!(timescales = ?cfg.timescales, vm = ?cfg.vm_url, "occupancy stage started (region_weight series, region episodes)");
        let mut occ = Occupancy::default();
        let mut buf: VecDeque<Sample> = VecDeque::new();
        let mut tick = tokio::time::interval(FLUSH_EVERY);
        let mut warned = false;
        loop {
            tokio::select! {
                m = rx.recv() => {
                    let Some(m) = m else { break };
                    let (samples, episodes) = occ.observe(&m);
                    if vm.is_some() {
                        buf.extend(samples);
                        let over = buf.len().saturating_sub(MAX_BUFFER);
                        if over > 0 {
                            buf.drain(..over);
                            bump(&counters, "occupancy.vm_dropped", over as f64).await;
                        }
                    }
                    if !episodes.is_empty() {
                        bump(&counters, "occupancy.episodes", episodes.len() as f64).await;
                        if sink.try_send(HubEvent::Episodes(episodes)).is_err() {
                            bump(&counters, "hub.dropped", 1.0).await;
                        }
                    }
                }
                _ = tick.tick() => {
                    counters.lock().await.insert("occupancy.open_episodes".into(), occ.open_episodes() as f64);
                    let Some(vm) = vm.as_ref() else { continue };
                    if buf.is_empty() {
                        continue;
                    }
                    let batch: Vec<Sample> = buf.iter().cloned().collect();
                    match vm.write(&batch).await {
                        Ok(()) => {
                            buf.clear();
                            warned = false;
                            bump(&counters, "occupancy.vm_samples", batch.len() as f64).await;
                        }
                        Err(e) => {
                            bump(&counters, "occupancy.vm_errors", 1.0).await;
                            if !warned {
                                warned = true;
                                tracing::warn!(error = %e, buffered = buf.len(), "region_weight import failed; retrying with the next batch");
                            }
                        }
                    }
                }
            }
        }
    });
    OccupancyHandle { tx, timescales }
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::{ExpertMembership, RegionWeight};

    use super::*;

    fn m(ts_s: i64, regions: &[(&str, f32)]) -> RegionMembership {
        RegionMembership {
            snapshot_id: "stat-v1-1m".into(),
            timescale: "1m".into(),
            instrument_id: "0190-AB".into(),
            as_of: Some(fa_proto::Timestamp { seconds: ts_s, nanos: 0 }),
            experts: vec![ExpertMembership {
                expert_index: 0,
                weights: regions.iter().map(|(r, w)| RegionWeight { region_id: (*r).into(), weight: *w }).collect(),
                ..Default::default()
            }],
        }
    }

    #[test]
    fn samples_are_labelled_and_episodes_close_on_a_region_change() {
        let mut o = Occupancy::default();
        let (s, e) = o.observe(&m(60, &[("a", 0.7), ("b", 0.3)]));
        assert!(e.is_empty());
        assert_eq!(s.len(), 2);
        assert!(s.iter().all(|x| x.validate().is_ok()), "labels are whitelisted");
        assert_eq!(s[0].labels["inst"], "0190ab");
        assert_eq!(s[0].labels["expert"], "stat-v1-1m-e0");
        assert_eq!((s[0].ts_ms, s[0].labels["region"].as_str()), (60_000, "a"));
        assert!(o.observe(&m(120, &[("a", 0.9), ("b", 0.1)])).1.is_empty());
        let (_, e) = o.observe(&m(180, &[("b", 0.6), ("a", 0.4)]));
        assert_eq!(e.len(), 1);
        let ep = &e[0];
        assert_eq!((ep.region_id.as_str(), ep.t_enter_ms, ep.t_exit_ms, ep.bars), ("a", 60_000, 180_000, 2));
        assert!((ep.max_weight - 0.9).abs() < 1e-6 && (ep.mean_weight - 0.8).abs() < 1e-6);
        assert_eq!(o.open_episodes(), 1);
    }
}
