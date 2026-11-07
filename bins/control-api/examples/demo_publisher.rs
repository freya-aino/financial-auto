//! Dev helper: publishes synthetic traffic for every operator view, so the UI can be
//! exercised before real models exist. Everything here is fake.
//!
//! Streams: trades (3 instruments), latent states (1h, 1d), region events, transition
//! posteriors (computed with `fa-mcmc` from the simulated region walk), decoder
//! outputs, strategy exposures, risk state, and evolution jobs with progress/results
//! (kinds `demo.eval.*`, so the real evo-worker queue on `jobs.evo.>` is not touched).
//!
//! cargo run -p control-api --example demo_publisher -- nats://127.0.0.1:4222
use std::{collections::HashMap, time::Duration};

use fa_core::rng::Rng;
use fa_mcmc::MarkovGraph;
use fa_proto::{
    Message, now,
    v1::{
        Analog, AssetExposure, DecoderOutput, Envelope, ExpertLatent, ExpertSpaceRef, ExposureDistribution, Heartbeat,
        Job, JobProgress, JobResult, LatentState, RegionEvent, RiskState, Trade, TransitionProb, TransitionUpdate,
        envelope::Payload, job_result, region_event,
    },
};

const INSTRUMENTS: [(&str, f64); 3] = [
    // the ingestor heartbeat connector's sim instrument, so paper orders fill against it too
    ("01900000-0000-7000-8000-000000000001", 100.0),
    ("01900000-0000-7000-8000-0000000000a1", 42.0),
    ("01900000-0000-7000-8000-0000000000b2", 1800.0),
];
const TIMESCALES: [&str; 2] = ["1h", "1d"];
const REGIONS: usize = 6;
const EXPERTS: u32 = 3;
const DIMS: usize = 8;
const FAMILIES: [&str; 5] = ["retrieval", "rnn_prior", "text.summary", "visual.future", "text.news"];

struct Bus(fa_bus::async_nats::Client);

impl Bus {
    async fn send<M: Message>(&self, subject: String, m: &M) -> anyhow::Result<()> {
        self.0.publish(subject, m.encode_to_vec().into()).await?;
        Ok(())
    }
}

fn region(ts: &str, i: usize) -> String {
    format!("{ts}-r{i}")
}

fn space(ts: &str, expert: u32) -> Option<ExpertSpaceRef> {
    Some(ExpertSpaceRef { snapshot_id: format!("demo-{ts}-g1"), timescale: ts.into(), expert_index: expert })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = std::env::args().nth(1).unwrap_or_else(|| "nats://127.0.0.1:4222".into());
    let bus = Bus(fa_bus::async_nats::connect(&url).await?);
    let mut rng = Rng::seed(7);

    let mut prices: Vec<f64> = INSTRUMENTS.iter().map(|(_, p)| *p).collect();
    let mut graphs: HashMap<&str, MarkovGraph> = TIMESCALES.iter().map(|t| (*t, MarkovGraph::new(0.5, 0.995))).collect();
    // current region per (timescale, instrument)
    let mut current: HashMap<(&str, usize), usize> = HashMap::new();
    let mut equity = 100_000.0_f64;
    let mut peak = equity;
    let mut job_n = 0u64;
    let mut running: Vec<(String, String, f32)> = Vec::new(); // (job id, kind, progress)

    let mut tick = 0u64;
    loop {
        tick += 1;

        // region registry: re-announced every 10 ticks so late subscribers catch up;
        // one region is periodically marked split to exercise lineage display
        if tick % 10 == 1 {
            for ts in TIMESCALES {
                for r in 0..REGIONS {
                    let split = r == REGIONS - 1 && tick > 20;
                    let ev = RegionEvent {
                        kind: if split { region_event::Kind::Split } else { region_event::Kind::Created } as i32,
                        space: space(ts, (r % EXPERTS as usize) as u32),
                        region_id: region(ts, r),
                        lineage_id: format!("lin-{ts}-{r}"),
                        related_region_ids: if split { vec![region(ts, 0)] } else { vec![] },
                    };
                    bus.send(fa_bus::subjects::latent_region(ts.parse()?), &ev).await?;
                }
            }
        }

        // trades: geometric random walk
        for (i, (inst, _)) in INSTRUMENTS.iter().enumerate() {
            prices[i] *= 1.0 + 0.002 * rng.normal();
            let env = Envelope {
                event_id: fa_core::EventId::new().to_string(),
                source_id: "demo".into(),
                instrument_id: (*inst).into(),
                event_time: Some(now()),
                publish_time: Some(now()),
                ingest_time: Some(now()),
                schema_version: 1,
                payload: Some(Payload::Trade(Trade { price: prices[i], size: 1.0, ..Default::default() })),
                ..Default::default()
            };
            let token = inst.replace('-', "");
            bus.send(format!("md.trade.demo.{token}"), &env).await?;
        }

        // latent states + region walk (sticky: mostly stay, sometimes move to a neighbour)
        for ts in TIMESCALES {
            for (i, (inst, _)) in INSTRUMENTS.iter().enumerate() {
                let from = *current.entry((ts, i)).or_insert(i % REGIONS);
                let to = if rng.uniform() < 0.7 { from } else { (from + 1 + rng.below(2)) % REGIONS };
                current.insert((ts, i), to);
                graphs.get_mut(ts).expect("graph").observe(&region(ts, from), &region(ts, to));

                let experts = (0..EXPERTS)
                    .map(|e| {
                        let r = (to + e as usize) % REGIONS;
                        ExpertLatent {
                            expert_index: e,
                            mean: (0..DIMS).map(|d| (r as f64 + 0.3 * rng.normal() + d as f64 * 0.05) as f32).collect(),
                            log_var: (0..DIMS).map(|_| (-1.0 + 0.2 * rng.normal()) as f32).collect(),
                            region_id: region(ts, r),
                            region_lineage_id: format!("lin-{ts}-{r}"),
                        }
                    })
                    .collect();
                let raw: Vec<f64> = (0..EXPERTS).map(|_| rng.uniform() + 0.2).collect();
                let sum: f64 = raw.iter().sum();
                let st = LatentState {
                    snapshot_id: format!("demo-{ts}-g1"),
                    timescale: ts.into(),
                    instrument_id: (*inst).into(),
                    as_of: Some(now()),
                    experts,
                    router_weights: raw.iter().map(|w| (w / sum) as f32).collect(),
                };
                bus.send(fa_bus::subjects::latent_state(ts.parse()?, &st.snapshot_id), &st).await?;
            }
        }

        // transition posteriors every 3 ticks
        if tick.is_multiple_of(3) {
            for ts in TIMESCALES {
                let g = &graphs[ts];
                for r in 0..REGIONS {
                    let from = region(ts, r);
                    let to: Vec<TransitionProb> = g
                        .summarize(&from, 0.9, 200, &mut rng)
                        .into_iter()
                        .map(|(k, p)| TransitionProb { region_id: k, mean: p.mean, lower: p.lower, upper: p.upper })
                        .collect();
                    if to.is_empty() {
                        continue;
                    }
                    let up = TransitionUpdate {
                        space: space(ts, (r % EXPERTS as usize) as u32),
                        from_region_id: from,
                        to,
                        sampler_id: "conjugate-dirichlet".into(),
                        as_of: Some(now()),
                    };
                    bus.send(fa_bus::subjects::latent_transition(ts.parse()?), &up).await?;
                }
            }
        }

        // decoder outputs
        let family = FAMILIES[rng.below(FAMILIES.len())];
        let center = prices[0];
        let out = DecoderOutput {
            decoder_id: format!("dec-{}", rng.below(20)),
            genome_id: format!("g-{}", rng.below(200)),
            family: family.into(),
            snapshot_id: "demo-1h-g1".into(),
            timescale: "1h".into(),
            as_of: Some(now()),
            output_refs: if family.starts_with("text") || family.starts_with("visual") {
                vec![format!("decoder-outputs/demo/{family}/{tick}.bin")]
            } else {
                vec![]
            },
            quantiles: if family == "rnn_prior" || family == "visual.future" {
                [-1.64, -0.67, 0.0, 0.67, 1.64].iter().map(|z| center * (1.0 + 0.01 * z)).collect()
            } else {
                vec![]
            },
            analogs: if family == "retrieval" {
                (0..5)
                    .map(|k| Analog {
                        window_id: format!("w-{}", rng.below(10_000)),
                        window_end: Some(now()),
                        similarity: 0.95 - 0.07 * f64::from(k) + 0.01 * rng.normal(),
                        time_distance_days: 30.0 + 400.0 * rng.uniform(),
                    })
                    .collect()
            } else {
                vec![]
            },
        };
        bus.send(fa_bus::subjects::decoder_output(family), &out).await?;

        // strategy exposures
        for (s, kind) in [("gp-0007", "1h"), ("np-0013", "1h"), ("gp-0042", "1d")] {
            let exp = ExposureDistribution {
                strategy_id: s.into(),
                genome_id: format!("genome-{s}"),
                horizon: if kind == "1h" { "2h" } else { "2d" }.into(),
                as_of: Some(now()),
                exposures: INSTRUMENTS
                    .iter()
                    .map(|(inst, _)| AssetExposure {
                        instrument_id: (*inst).into(),
                        mean: (0.4 * rng.normal()).clamp(-1.0, 1.0),
                        stddev: 0.05 + 0.2 * rng.uniform(),
                    })
                    .collect(),
            };
            bus.send(fa_bus::subjects::signal_exposure(s), &exp).await?;
        }

        // risk state of the (fake) risk service
        equity *= 1.0 + 0.0008 * rng.normal();
        peak = peak.max(equity);
        let risk = RiskState {
            as_of: Some(now()),
            equity_eur: equity,
            deployable_eur: equity - 500.0,
            gross_exposure_eur: equity * (0.3 + 0.1 * rng.uniform()),
            drawdown: 1.0 - equity / peak,
            daily_pnl_eur: equity - 100_000.0,
            active_halts: vec![],
        };
        bus.send(fa_bus::subjects::RISK_STATE.to_owned(), &risk).await?;

        // evolution: submit a job every 2 ticks, advance running ones, finish at 100%
        if tick.is_multiple_of(2) {
            job_n += 1;
            // `demo.*` kinds stay out of the real `jobs.evo.>` work queue (evo-worker)
            let kind = if job_n.is_multiple_of(3) { "demo.eval.strategy" } else { "demo.eval.decoder" };
            let job = Job {
                job_id: fa_core::JobId::new().to_string(),
                kind: kind.into(),
                priority: 2,
                snapshot_id: "demo-1h-g1".into(),
                genome_id: format!("g-{job_n}"),
                budget_gpu_ms: 60_000,
                seed: job_n,
                ..Default::default()
            };
            bus.send(fa_bus::subjects::jobs(kind), &job).await?;
            running.push((job.job_id, kind.into(), 0.0));
        }
        let mut finished = Vec::new();
        for (idx, (id, kind, p)) in running.iter_mut().enumerate() {
            *p += 0.15 + 0.2 * rng.uniform() as f32;
            if *p < 1.0 {
                let prog = JobProgress { job_id: id.clone(), fraction: *p, message: format!("rung {}", (*p * 4.0) as u32) };
                bus.send(fa_bus::subjects::job_progress(id), &prog).await?;
                continue;
            }
            let failed = rng.uniform() < 0.1;
            let recon = 0.05 + 0.3 * rng.uniform();
            let skill = 0.6 - recon + 0.1 * rng.normal();
            let result = serde_json::json!({
                "score": skill,
                "objectives": {
                    "reconstruction": recon,
                    "forecast_skill": skill,
                    "trading_value": 0.5 * skill + 0.1 * rng.normal(),
                    "compute_cost": 0.2 + rng.uniform(),
                }
            });
            let res = JobResult {
                job_id: id.clone(),
                status: if failed { job_result::Status::Failed } else { job_result::Status::Ok } as i32,
                result: if failed { vec![] } else { result.to_string().into_bytes() },
                used_gpu_ms: 30_000 + rng.below(30_000) as u64,
                worker_id: "demo-worker".into(),
                error: if failed { "synthetic failure".into() } else { String::new() },
                ..Default::default()
            };
            bus.send(fa_bus::subjects::job_result(kind), &res).await?;
            finished.push(idx);
        }
        for idx in finished.into_iter().rev() {
            running.swap_remove(idx);
        }

        let hb = Heartbeat {
            service: "demo-publisher".into(),
            instance_id: "local".into(),
            at: Some(now()),
            version: env!("CARGO_PKG_VERSION").into(),
            counters: [("ticks".to_owned(), tick as f64), ("jobs_submitted".to_owned(), job_n as f64)].into(),
        };
        bus.send(fa_bus::subjects::heartbeat("demo-publisher"), &hb).await?;
        bus.0.flush().await?;
        tokio::time::sleep(Duration::from_millis(1000)).await;
    }
}
