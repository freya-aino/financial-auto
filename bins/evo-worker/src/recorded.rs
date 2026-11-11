//! Recorded bus traffic for replays (spec 41 §3/§6.1, D-053, D-059): the `LATENT` JetStream
//! stream (`latent.state.>`, `latent.region.>`, `latent.transition.>`, `decoder.output.>`) for
//! strategy replays and decoder evaluations, and the `SIGNAL` stream (`signal.exposure.>`) for
//! allocator replays, read by time range through `fa_bus::replay`.
//!
//! Both go through an **incremental** cache (`fa_backtest::latents::RecordingCache`): a window
//! inside a cached span is a hit (sliced, no read); a window overlapping or touching it reads
//! only the missing ranges (`[start, cached start)` and `(cached end − 5 s, end]`) and merges
//! them in knowledge-time order (extend); anything else is a miss (full read, replaces the
//! recording). The budget is counted in messages (`FA_EVO_LATENT_CACHE_MESSAGES`, default
//! 2 000 000, signals a quarter of it). The `LATENT` read is unfiltered, so one recording serves
//! every timescale and instrument set. Now-relative windows are pinned to the minute by the
//! callers (`DataWindow::pinned`), so the candidates of one rung share a window (hits) and the
//! next rung only reads the new minute(s) (extend). Heartbeat counters: `latents.cache_hit|
//! extend|miss`, `latents.messages_read`, `signals.cache_hit|extend|miss`, `signals.messages_read`.
//!
//! Environment: `FA_EVO_NO_LATENTS=1` disables the feed (fallback only),
//! `FA_EVO_LATENT_PRELOAD_HOURS` (default 1) is read before the window start so rows and regions
//! published earlier are known at the first step, `FA_EVO_LATENT_MAX_MESSAGES` (default 500 000
//! per filter) caps a read.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use fa_backtest::{
    allocator_replay::{SIGNAL_FILTER, SIGNAL_STREAM, SignalRecord, decode_signals},
    latents::{CacheKey, CachePlan, DEFAULT_CACHE_MESSAGES, LatentCache, RecordedLatents, RecordingCache, RecordingId},
    replay::ReplaySpec,
};
use fa_bus::{
    async_nats::jetstream,
    replay::{LatentMessage, ReadOptions, RecordedMessage, read_latents, read_range},
};
use fa_core::Timescale;
use fa_decoders::{DecodeCtx, LatentPoint};
use serde_json::{Value, json};

use crate::latent::vector_of;

pub const DEFAULT_PRELOAD_MS: i64 = 3_600_000;
const READ_TIMEOUT: Duration = Duration::from_secs(120);

/// Hit / extend / miss / messages-read counters of one cache.
#[derive(Debug, Default)]
pub struct CacheCounters {
    pub hit: AtomicU64,
    pub extend: AtomicU64,
    pub miss: AtomicU64,
    pub messages_read: AtomicU64,
}

impl CacheCounters {
    fn bump(&self, plan: &str, read: usize) {
        match plan {
            "hit" => &self.hit,
            "extend" => &self.extend,
            _ => &self.miss,
        }
        .fetch_add(1, Ordering::Relaxed);
        self.messages_read.fetch_add(read as u64, Ordering::Relaxed);
    }

    /// `(prefix.cache_hit, …)` heartbeat entries.
    pub fn export(&self, prefix: &str, out: &mut HashMap<String, f64>) {
        for (k, v) in [
            ("cache_hit", &self.hit),
            ("cache_extend", &self.extend),
            ("cache_miss", &self.miss),
            ("messages_read", &self.messages_read),
        ] {
            out.insert(format!("{prefix}.{k}"), v.load(Ordering::Relaxed) as f64);
        }
    }
}

/// Shared loader + caches.
#[derive(Clone)]
pub struct RecordedStore {
    cache: Arc<Mutex<LatentCache>>,
    signals: Arc<Mutex<RecordingCache>>,
    pub latent_counters: Arc<CacheCounters>,
    pub signal_counters: Arc<CacheCounters>,
    pub enabled: bool,
    pub preload_ms: i64,
    pub opts: ReadOptions,
}

impl Default for RecordedStore {
    fn default() -> Self {
        Self::with_budget(DEFAULT_CACHE_MESSAGES)
    }
}

/// Result of one load.
#[derive(Debug, Default)]
pub struct LoadedLatents {
    pub latents: Option<Arc<RecordedLatents>>,
    /// `hit` | `extend` | `miss` (empty when nothing was attempted).
    pub plan: &'static str,
    /// Messages read from the stream for this load (0 on a hit).
    pub messages_read: usize,
    pub error: Option<String>,
    pub read_start_ms: i64,
    pub read_end_ms: i64,
    pub elapsed_ms: u64,
}

impl LoadedLatents {
    pub fn cached(&self) -> bool {
        self.plan == "hit"
    }
}

/// Result of a signal load.
#[derive(Debug, Default)]
pub struct LoadedSignals {
    pub signals: Vec<SignalRecord>,
    pub undecodable: usize,
    pub plan: &'static str,
    pub messages_read: usize,
    pub error: Option<String>,
    pub read_start_ms: i64,
    pub read_end_ms: i64,
    pub elapsed_ms: u64,
}

fn plan_name(p: &CachePlan) -> &'static str {
    match p {
        CachePlan::Hit => "hit",
        CachePlan::Extend { .. } => "extend",
        CachePlan::Miss => "miss",
    }
}

/// The one key every `LATENT` load uses: the read is unfiltered.
fn latent_key(start_ms: i64, end_ms: i64) -> CacheKey {
    CacheKey::new(start_ms, end_ms, Timescale::Second, &[])
}

fn signal_id() -> RecordingId {
    RecordingId { timescale: Timescale::Second, instruments: vec!["signal.exposure".into()] }
}

impl RecordedStore {
    pub fn with_budget(max_messages: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(LatentCache::new(max_messages))),
            signals: Arc::new(Mutex::new(RecordingCache::new((max_messages / 4).max(1)))),
            latent_counters: Arc::default(),
            signal_counters: Arc::default(),
            enabled: true,
            preload_ms: DEFAULT_PRELOAD_MS,
            opts: ReadOptions::default(),
        }
    }

    /// A store that never reads the stream (fallback-only replays).
    #[cfg(test)]
    pub fn disabled() -> Self {
        Self { enabled: false, ..Default::default() }
    }

    pub fn from_env() -> Self {
        let budget = std::env::var("FA_EVO_LATENT_CACHE_MESSAGES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_CACHE_MESSAGES);
        let mut s = Self {
            enabled: std::env::var("FA_EVO_NO_LATENTS").map_or(true, |v| v != "1"),
            ..Self::with_budget(budget)
        };
        if let Some(h) = std::env::var("FA_EVO_LATENT_PRELOAD_HOURS").ok().and_then(|v| v.parse::<f64>().ok()) {
            s.preload_ms = (h.max(0.0) * 3_600_000.0) as i64;
        }
        if let Some(n) = std::env::var("FA_EVO_LATENT_MAX_MESSAGES").ok().and_then(|v| v.parse::<usize>().ok()) {
            s.opts.max_messages = n.max(1);
        }
        s
    }

    /// Recorded latents for a strategy replay: `[window start − preload, window end]`.
    pub async fn load(&self, js: &jetstream::Context, spec: &ReplaySpec) -> LoadedLatents {
        self.load_window(js, spec.window_ms.0 - self.preload_ms, spec.window_ms.1).await
    }

    /// Recorded `LATENT` traffic published in `[start, end]`, through the incremental cache.
    /// Never fails the evaluation: an unreadable stream yields `latents: None` with `error`.
    pub async fn load_window(&self, js: &jetstream::Context, start: i64, end: i64) -> LoadedLatents {
        let started = Instant::now();
        let mut out = LoadedLatents { read_start_ms: start, read_end_ms: end, ..Default::default() };
        if !self.enabled {
            out.error = Some("disabled (FA_EVO_NO_LATENTS)".into());
            return out;
        }
        let key = latent_key(start, end);
        let plan = self.cache.lock().unwrap_or_else(|e| e.into_inner()).plan(&key);
        out.plan = plan_name(&plan);
        let ranges: Vec<(i64, i64)> = match &plan {
            CachePlan::Hit => {
                if let Some(hit) = self.cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
                    out.latents = Some(hit);
                    out.elapsed_ms = started.elapsed().as_millis() as u64;
                    self.latent_counters.bump("hit", 0);
                    return out;
                }
                out.plan = "miss";
                vec![(start, end)]
            }
            CachePlan::Extend { read } => read.clone(),
            CachePlan::Miss => vec![(start, end)],
        };
        let mut raw: Vec<RecordedMessage> = Vec::new();
        for (a, b) in &ranges {
            match tokio::time::timeout(READ_TIMEOUT, read_latents(js, *a, *b, &self.opts)).await {
                Ok(Ok(v)) => raw.extend(v),
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "recorded latents unavailable; falling back");
                    out.error = Some(e.to_string());
                    break;
                }
                Err(_) => {
                    tracing::warn!("recorded latents read timed out; falling back");
                    out.error = Some(format!("read timed out ({READ_TIMEOUT:?})"));
                    break;
                }
            }
        }
        if out.error.is_some() {
            out.elapsed_ms = started.elapsed().as_millis() as u64;
            return out;
        }
        out.messages_read = raw.len();
        let rec = {
            let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if out.plan == "extend" { c.extend(&key, raw) } else { c.insert(&key, raw) }
        };
        self.latent_counters.bump(out.plan, out.messages_read);
        tracing::info!(
            plan = out.plan,
            read = out.messages_read,
            messages = rec.len(),
            states = rec.summary.states,
            transitions = rec.summary.transitions,
            decoder_outputs = rec.summary.decoder_outputs,
            start,
            end,
            "recorded latents loaded from LATENT"
        );
        out.latents = Some(rec);
        out.elapsed_ms = started.elapsed().as_millis() as u64;
        out
    }

    /// Recorded `signal.exposure.*` published in `[start, end]`, through the incremental cache.
    pub async fn load_signals(&self, js: &jetstream::Context, start: i64, end: i64) -> LoadedSignals {
        let started = Instant::now();
        let mut out = LoadedSignals { read_start_ms: start, read_end_ms: end, ..Default::default() };
        let id = signal_id();
        let plan = self.signals.lock().unwrap_or_else(|e| e.into_inner()).plan(&id, start, end);
        out.plan = plan_name(&plan);
        let ranges: Vec<(i64, i64)> = match &plan {
            CachePlan::Hit => Vec::new(),
            CachePlan::Extend { read } => read.clone(),
            CachePlan::Miss => vec![(start, end)],
        };
        let mut raw = Vec::new();
        for (a, b) in &ranges {
            match tokio::time::timeout(READ_TIMEOUT, read_range(js, SIGNAL_STREAM, SIGNAL_FILTER, *a, *b, &self.opts))
                .await
            {
                Ok(Ok(v)) => raw.extend(v),
                Ok(Err(e)) => {
                    out.error = Some(e.to_string());
                    break;
                }
                Err(_) => {
                    out.error = Some(format!("read timed out ({READ_TIMEOUT:?})"));
                    break;
                }
            }
        }
        if out.error.is_some() {
            out.elapsed_ms = started.elapsed().as_millis() as u64;
            return out;
        }
        out.messages_read = raw.len();
        let window = {
            let mut c = self.signals.lock().unwrap_or_else(|e| e.into_inner());
            match out.plan {
                "extend" => c.extend(&id, start, end, raw),
                "miss" => c.insert(&id, start, end, raw),
                _ => {}
            }
            c.slice(&id, start, end).unwrap_or_default()
        };
        self.signal_counters.bump(out.plan, out.messages_read);
        let (signals, bad) = decode_signals(&window);
        out.signals = signals;
        out.undecodable = bad;
        out.elapsed_ms = started.elapsed().as_millis() as u64;
        out
    }

    /// Heartbeat counters of both caches plus their sizes.
    pub fn export(&self, out: &mut HashMap<String, f64>) {
        self.latent_counters.export("latents", out);
        self.signal_counters.export("signals", out);
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        out.insert("latents.cache_messages".into(), c.total_messages() as f64);
        let s = self.signals.lock().unwrap_or_else(|e| e.into_inner());
        out.insert("signals.cache_messages".into(), s.total_messages() as f64);
    }

    /// Payload block describing a load.
    pub fn json(&self, l: &LoadedLatents) -> Value {
        let c = &self.latent_counters;
        json!({
            "source": if l.latents.is_some() { "jetstream LATENT (by_start_time ordered consumer)" } else { "none (price fallback)" },
            "enabled": self.enabled,
            "cached": l.cached(),
            "plan": l.plan,
            "messages_read": l.messages_read,
            "read_start_ms": l.read_start_ms,
            "read_end_ms": l.read_end_ms,
            "preload_ms": self.preload_ms,
            "elapsed_ms": l.elapsed_ms,
            "error": l.error,
            "summary": l.latents.as_ref().map(|r| &r.summary),
            "cache": {"hits": c.hit.load(Ordering::Relaxed), "extends": c.extend.load(Ordering::Relaxed),
                      "misses": c.miss.load(Ordering::Relaxed), "messages_read": c.messages_read.load(Ordering::Relaxed)},
        })
    }
}

/// Decoder conditioning from a recording: every `LatentState` becomes a `LatentPoint` (concatenated
/// expert means, as the live buffer builds them), keyed by instrument token at its `as_of`.
pub fn decode_ctx(rec: &RecordedLatents) -> DecodeCtx {
    let mut ctx = DecodeCtx::default();
    for r in rec.records() {
        if let LatentMessage::State(s) = &r.message
            && !s.instrument_id.is_empty()
            && let Some(vec) = vector_of(s)
        {
            let as_of_ms =
                s.as_of.as_ref().map_or(r.published_ms, |t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000);
            ctx.set_layout(&s.instrument_id, crate::latent::layout_of(s));
            ctx.push(&s.instrument_id, LatentPoint { as_of_ms, vec });
        }
    }
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_describes_a_disabled_store() {
        let store = RecordedStore::disabled();
        assert_eq!(store.preload_ms, DEFAULT_PRELOAD_MS);
        let l =
            LoadedLatents { read_start_ms: 1, read_end_ms: 2, error: Some("disabled".into()), ..Default::default() };
        let j = store.json(&l);
        assert_eq!(j["enabled"], false);
        assert_eq!(j["source"], "none (price fallback)");
        assert_eq!(j["cache"]["hits"], 0);
        assert_eq!(j["summary"], Value::Null);
        assert_eq!(j["error"], "disabled");
        let mut hb = HashMap::new();
        store.latent_counters.bump("extend", 12);
        store.export(&mut hb);
        assert_eq!(
            (hb["latents.cache_extend"], hb["latents.messages_read"], hb["latents.cache_hit"]),
            (1.0, 12.0, 0.0)
        );
        assert!(hb.contains_key("signals.cache_miss"));
    }

    #[test]
    fn decode_ctx_from_the_fixture_recording() {
        let text = include_str!("../../../crates/fa-backtest/testdata/latent_fixture.jsonl");
        let rec = RecordedLatents::from_jsonl(text).unwrap();
        let ctx = decode_ctx(&rec);
        let (inst, points) = ctx.stats();
        assert!(inst > 0 && points == rec.summary.states, "{inst} {points} {:?}", rec.summary);
    }

    /// `FA_NATS_TEST=1`: two overlapping now-relative windows against the live `LATENT` and
    /// `SIGNAL` streams → miss, then hit (inner window), then extend (moved forward).
    #[tokio::test]
    async fn live_incremental_loads() {
        if std::env::var("FA_NATS_TEST").is_err() {
            return;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let js = jetstream::new(fa_bus::async_nats::connect(&url).await.unwrap());
        let store = RecordedStore { preload_ms: 0, ..Default::default() };
        let now = crate::market::now_ms();
        let a = store.load_window(&js, now - 120_000, now - 30_000).await;
        let b = store.load_window(&js, now - 100_000, now - 40_000).await;
        let c = store.load_window(&js, now - 90_000, now).await;
        eprintln!(
            "latents: {} read {} / {} read {} / {} read {}",
            a.plan, a.messages_read, b.plan, b.messages_read, c.plan, c.messages_read
        );
        assert_eq!((a.plan, b.plan, c.plan), ("miss", "hit", "extend"), "{:?} {:?}", a.error, c.error);
        assert_eq!(b.messages_read, 0);
        let s1 = store.load_signals(&js, now - 120_000, now - 30_000).await;
        let s2 = store.load_signals(&js, now - 60_000, now).await;
        eprintln!(
            "signals: {} {} ({} read) / {} {} ({} read)",
            s1.plan,
            s1.signals.len(),
            s1.messages_read,
            s2.plan,
            s2.signals.len(),
            s2.messages_read
        );
        assert_eq!((s1.plan, s2.plan), ("miss", "extend"));
        assert!(s2.messages_read < s1.messages_read + s2.signals.len());
    }
}
