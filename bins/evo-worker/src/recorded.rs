//! Recorded latents for strategy replays (spec 41 §3/§6.1, D-053): the `LATENT` JetStream
//! stream is read by time range (`fa_bus::replay::read_latents`) for the replay window plus a
//! preload margin, decoded into `fa_backtest::latents::RecordedLatents` and kept in a bounded
//! per-worker [`LatentCache`] keyed by `(window, timescale, instrument set)` so every candidate
//! of a generation shares one load. Replays then see the same point-in-time latent graph and
//! decoder outputs the live `strategy-runtime` saw; windows before the recordings fall back to
//! prices (reported in `sources.fallback`).
//!
//! Environment: `FA_EVO_NO_LATENTS=1` disables the feed (fallback only),
//! `FA_EVO_LATENT_PRELOAD_HOURS` (default 1) is read before the window start so rows and
//! regions published earlier are known at the first step, `FA_EVO_LATENT_MAX_MESSAGES`
//! (default 500 000 per filter) caps a read.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use fa_backtest::{
    latents::{CacheKey, LatentCache, RecordedLatents},
    replay::ReplaySpec,
};
use fa_bus::{
    async_nats::jetstream,
    replay::{ReadOptions, read_latents},
};
use serde_json::{Value, json};

pub const DEFAULT_PRELOAD_MS: i64 = 3_600_000;

/// Shared loader + cache.
#[derive(Clone)]
pub struct RecordedStore {
    cache: Arc<Mutex<LatentCache>>,
    pub enabled: bool,
    pub preload_ms: i64,
    pub opts: ReadOptions,
}

impl Default for RecordedStore {
    fn default() -> Self {
        Self {
            cache: Arc::new(Mutex::new(LatentCache::default())),
            enabled: true,
            preload_ms: DEFAULT_PRELOAD_MS,
            opts: ReadOptions::default(),
        }
    }
}

/// Result of one load.
#[derive(Debug, Default)]
pub struct LoadedLatents {
    pub latents: Option<Arc<RecordedLatents>>,
    pub cached: bool,
    pub error: Option<String>,
    pub read_start_ms: i64,
    pub read_end_ms: i64,
    pub elapsed_ms: u64,
}

impl RecordedStore {
    /// A store that never reads the stream (fallback-only replays).
    #[cfg(test)]
    pub fn disabled() -> Self {
        Self { enabled: false, ..Default::default() }
    }

    pub fn from_env() -> Self {
        let mut s =
            Self { enabled: std::env::var("FA_EVO_NO_LATENTS").map_or(true, |v| v != "1"), ..Default::default() };
        if let Some(h) = std::env::var("FA_EVO_LATENT_PRELOAD_HOURS").ok().and_then(|v| v.parse::<f64>().ok()) {
            s.preload_ms = (h.max(0.0) * 3_600_000.0) as i64;
        }
        if let Some(n) = std::env::var("FA_EVO_LATENT_MAX_MESSAGES").ok().and_then(|v| v.parse::<usize>().ok()) {
            s.opts.max_messages = n.max(1);
        }
        s
    }

    /// Recorded latents for `spec`'s window (`[start − preload, end]`), from the cache when the
    /// same window / timescale / instrument set was loaded before. Never fails the evaluation:
    /// an unreadable stream yields `latents: None` with `error` set (fallback-only replay).
    pub async fn load(&self, js: &jetstream::Context, spec: &ReplaySpec) -> LoadedLatents {
        let started = Instant::now();
        let (start, end) = (spec.window_ms.0 - self.preload_ms, spec.window_ms.1);
        let mut out = LoadedLatents { read_start_ms: start, read_end_ms: end, ..Default::default() };
        if !self.enabled {
            out.error = Some("disabled (FA_EVO_NO_LATENTS)".into());
            return out;
        }
        let key = CacheKey::of(spec);
        if let Some(hit) = self.cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            out.latents = Some(hit);
            out.cached = true;
            out.elapsed_ms = started.elapsed().as_millis() as u64;
            return out;
        }
        match tokio::time::timeout(Duration::from_secs(120), read_latents(js, start, end, &self.opts)).await {
            Ok(Ok(raw)) => {
                let rec = RecordedLatents::new(raw);
                tracing::info!(
                    messages = rec.len(),
                    states = rec.summary.states,
                    transitions = rec.summary.transitions,
                    decoder_outputs = rec.summary.decoder_outputs,
                    snapshots = ?rec.summary.snapshots,
                    start,
                    end,
                    "recorded latents loaded from LATENT"
                );
                let arc = self.cache.lock().unwrap_or_else(|e| e.into_inner()).insert(key, rec);
                out.latents = Some(arc);
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "recorded latents unavailable; replay falls back to prices");
                out.error = Some(e.to_string());
            }
            Err(_) => {
                tracing::warn!("recorded latents read timed out; replay falls back to prices");
                out.error = Some("read timed out (120 s)".into());
            }
        }
        out.elapsed_ms = started.elapsed().as_millis() as u64;
        out
    }

    /// `(hits, misses, windows)` of the cache.
    pub fn stats(&self) -> (u64, u64, usize) {
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        (c.hits, c.misses, c.len())
    }

    /// Payload block describing a load.
    pub fn json(&self, l: &LoadedLatents) -> Value {
        let (hits, misses, windows) = self.stats();
        json!({
            "source": if l.latents.is_some() { "jetstream LATENT (by_start_time ordered consumer)" } else { "none (price fallback)" },
            "enabled": self.enabled,
            "cached": l.cached,
            "read_start_ms": l.read_start_ms,
            "read_end_ms": l.read_end_ms,
            "preload_ms": self.preload_ms,
            "elapsed_ms": l.elapsed_ms,
            "error": l.error,
            "summary": l.latents.as_ref().map(|r| &r.summary),
            "cache": {"hits": hits, "misses": misses, "windows": windows},
        })
    }
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
        assert_eq!(j["cache"]["windows"], 0);
        assert_eq!(j["summary"], Value::Null);
        assert_eq!(j["error"], "disabled");
    }
}
