//! Serving of elite **neural** decoders (spec 34, `fa_decoders::neural`) next to the baselines,
//! with shadow serving and online promotion.
//!
//! **Elites.** A loader task lists the decoder cards evo-worker writes to Garage
//! (`fa-models/decoders/**/*.card.json`, every `FA_NEURAL_RELOAD_SECS`, default 300 s, and on
//! `ctl.decoders.reload`), ranks them per (output family, timescale) with
//! `fa_decoders::neural::feedback::select_top_k` (Pareto rank, then family score; only decoders that
//! beat their naive baseline) and loads the best [`CANDIDATES`] bundles. Per (family, timescale) the
//! best one **serves** and the next one runs in **shadow**; an operator pin (`ctl.decoders.pin`)
//! overrides the ranking, `ctl.decoders.off` disables a slot.
//!
//! **Outputs.** At every closed bar of an instrument at the decoder's timescale: serving decoders
//! publish `DecoderOutput` on `decoder.output.<family>`, shadow decoders on
//! `decoder.shadow.<family>` (not consumed by strategies). Contract: `rnn_prior` / `retrieval`
//! carry horizon-return quantiles (+ analogs) and `h:<bars>`; `visual.*` / `text.*` carry the score
//! in `[-1, 1]` as `quantiles[0]` (future chart: `2·P(up) − 1` of the predicted last column; chart:
//! direction of the reconstructed raster; news / narrative: sentiment of the decoded embedding vs
//! the training targets) followed by components. `output_refs`: `inst:<uuid>`, `kind:`/`topic:`,
//! `src:neural`, `role:serving|shadow`, `card:<garage key>`. Inputs: the bar closes of the series,
//! the latent the pipeline just produced for the instrument, the news timeline (`news.item.>`).
//! With `FA_NEURAL_REPLACE_BASELINE=1` a serving neural decoder **replaces** the baseline output
//! of its family at its timescale, else both publish (alongside).
//!
//! **Online scoring and promotion.** Forecasts are resolved when their horizon closes (CRPS from the
//! quantiles; future chart and retrieval included); chart and news outputs are scored immediately
//! against the realised raster / news embedding. Per slot, once serving and shadow both have
//! `FA_NEURAL_PROMOTE_MIN` (default 200) scored outputs and the shadow has run at least
//! `FA_NEURAL_PROMOTE_MINUTES` (default 30), a shadow whose mean loss is lower by more than
//! `FA_NEURAL_PROMOTE_MARGIN` (default 0.02, relative) is promoted (roles swap, statistics restart,
//! the promotion becomes a pin). Heartbeat: `neural.cards`, `neural.loaded`, `neural.slots`,
//! `neural.serving.<family>.<ts>`, `neural.outputs.<family>.<role>`,
//! `neural.online.<family>.<ts>.<role>.{n,loss}`, `neural.promotions`, `neural.load_errors`,
//! `neural.infer_ms`.
//!
//! `ctl.decoders.<verb>` (request/reply `CtlRequest` / `CtlReply`): `status`, `reload`,
//! `promote {family, timescale}`, `pin {key}`, `unpin {family, timescale}`, `off {family, timescale}`,
//! `on {family, timescale}`. (`ctl.model.>` belongs to the encoder snapshot listener.)

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use fa_blob::{BlobStore, ObjectKey, S3BlobStore};
use fa_bus::Bus;
use fa_core::Timescale;
use fa_decoders::{
    baseline::Series,
    crps_quantiles,
    genome::DecoderGenome,
    neural::{
        Device, NeuralDecoder,
        data::TextTimeline,
        feedback::{self, EliteCard},
    },
};
use fa_proto::v1::{self, CtlReply, CtlRequest, DecoderOutput, LatentState};
use futures::StreamExt;
use serde_json::json;
use tokio::sync::{mpsc, watch};

use crate::pipeline::ms_to_ts;

/// Ranked bundles loaded per (family, timescale): serving + shadow + one spare.
pub const CANDIDATES: usize = 3;
/// News kept for the text decoders.
const TEXT_KEEP_MS: i64 = 7 * 86_400_000;
/// Pending forecasts kept per slot role.
const MAX_PENDING: usize = 50_000;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Tunables (environment).
#[derive(Debug, Clone, Copy)]
pub struct ServingCfg {
    pub reload: Duration,
    pub replace_baseline: bool,
    pub promote_min: usize,
    pub promote_margin: f64,
    pub promote_min_ms: i64,
}

impl ServingCfg {
    pub fn from_env() -> Self {
        Self {
            reload: Duration::from_secs(env_or("FA_NEURAL_RELOAD_SECS", 300u64).max(5)),
            replace_baseline: env_or("FA_NEURAL_REPLACE_BASELINE", 0u8) == 1,
            promote_min: env_or("FA_NEURAL_PROMOTE_MIN", 200usize).max(1),
            promote_margin: env_or("FA_NEURAL_PROMOTE_MARGIN", 0.02f64),
            promote_min_ms: env_or("FA_NEURAL_PROMOTE_MINUTES", 30i64) * 60_000,
        }
    }
}

/// One loaded candidate.
#[derive(Clone)]
pub struct Candidate {
    pub key: String,
    pub card: EliteCard,
    pub decoder: Arc<NeuralDecoder>,
    pub timescale: Timescale,
}

/// What the loader produced.
#[derive(Default)]
pub struct Loaded {
    /// Candidates in rank order within each (family, timescale).
    pub candidates: Vec<Candidate>,
    pub cards: usize,
    pub errors: Vec<String>,
}

type SlotKey = (String, Timescale);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Serving,
    Shadow,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Serving => "serving",
            Self::Shadow => "shadow",
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Slot {
    serving: Option<String>,
    shadow: Option<String>,
    shadow_since_ms: i64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Online {
    n: usize,
    sum: f64,
}

impl Online {
    fn mean(&self) -> f64 {
        if self.n > 0 { self.sum / self.n as f64 } else { f64::NAN }
    }
}

/// A forecast waiting for its horizon to close.
struct Pending {
    slot: SlotKey,
    key: String,
    inst: String,
    due_ms: i64,
    base: f64,
    quantiles: Vec<f64>,
    levels: Vec<f64>,
}

struct State {
    cfg: ServingCfg,
    loaded: Arc<Loaded>,
    by_key: HashMap<String, Candidate>,
    slots: BTreeMap<SlotKey, Slot>,
    pins: HashMap<SlotKey, String>,
    disabled: HashSet<SlotKey>,
    text: TextTimeline,
    pending: VecDeque<Pending>,
    online: HashMap<(SlotKey, String), Online>,
    counters: BTreeMap<String, f64>,
}

impl State {
    /// Recomputes serving / shadow per slot from the ranked candidates, keeping a current serving
    /// decoder while it is still a candidate, honouring pins.
    fn assign(&mut self, now_ms: i64) {
        let mut ranked: BTreeMap<SlotKey, Vec<String>> = BTreeMap::new();
        for c in &self.loaded.candidates {
            ranked.entry((c.card.meta.output_family.clone(), c.timescale)).or_default().push(c.key.clone());
        }
        self.by_key = self.loaded.candidates.iter().map(|c| (c.key.clone(), c.clone())).collect();
        let mut slots = BTreeMap::new();
        for (slot, keys) in ranked {
            let prev = self.slots.get(&slot).cloned().unwrap_or_default();
            let serving = self
                .pins
                .get(&slot)
                .filter(|k| keys.contains(k))
                .or(prev.serving.as_ref().filter(|k| keys.contains(k)))
                .or(keys.first())
                .cloned();
            let shadow = keys.iter().find(|k| Some(*k) != serving.as_ref()).cloned();
            let since = if shadow == prev.shadow { prev.shadow_since_ms } else { now_ms };
            slots.insert(slot, Slot { serving, shadow, shadow_since_ms: since });
        }
        self.slots = slots;
    }

    fn role_keys(&self, slot: &SlotKey) -> Vec<(Role, String)> {
        let Some(s) = self.slots.get(slot) else { return Vec::new() };
        let mut v = Vec::new();
        if let Some(k) = &s.serving {
            v.push((Role::Serving, k.clone()));
        }
        if let Some(k) = &s.shadow {
            v.push((Role::Shadow, k.clone()));
        }
        v
    }

    fn score(&mut self, slot: &SlotKey, key: &str, loss: f64) {
        if !loss.is_finite() {
            return;
        }
        let o = self.online.entry((slot.clone(), key.to_owned())).or_default();
        o.n += 1;
        o.sum += loss;
    }

    /// Promotes the shadow of `slot` (roles swap, statistics restart, pin the new serving one).
    fn promote(&mut self, slot: &SlotKey, now_ms: i64) -> Option<String> {
        let s = self.slots.get_mut(slot)?;
        let new = s.shadow.take()?;
        s.shadow = s.serving.replace(new.clone());
        s.shadow_since_ms = now_ms;
        self.pins.insert(slot.clone(), new.clone());
        self.online.retain(|(sl, _), _| sl != slot);
        *self.counters.entry("neural.promotions".into()).or_default() += 1.0;
        Some(new)
    }

    /// Automatic promotion check of one slot.
    fn maybe_promote(&mut self, slot: &SlotKey, now_ms: i64) {
        let Some(s) = self.slots.get(slot) else { return };
        let (Some(sv), Some(sh)) = (s.serving.clone(), s.shadow.clone()) else { return };
        if now_ms - s.shadow_since_ms < self.cfg.promote_min_ms {
            return;
        }
        let get = |k: &str| self.online.get(&(slot.clone(), k.to_owned())).copied().unwrap_or_default();
        let (a, b) = (get(&sv), get(&sh));
        if a.n < self.cfg.promote_min || b.n < self.cfg.promote_min {
            return;
        }
        if b.mean() < a.mean() * (1.0 - self.cfg.promote_margin) {
            tracing::info!(family = %slot.0, ts = %slot.1, serving = %sv, shadow = %sh, serving_loss = a.mean(),
                           shadow_loss = b.mean(), "shadow decoder promoted");
            self.promote(slot, now_ms);
        }
    }
}

/// Neural decoder serving; cheap to clone (shared state).
#[derive(Clone)]
pub struct NeuralServing {
    state: Arc<Mutex<State>>,
    rx: watch::Receiver<Arc<Loaded>>,
    kick: mpsc::Sender<()>,
}

impl NeuralServing {
    /// Starts the loader (Garage from `config/secrets.toml`; without credentials nothing loads) and
    /// the `ctl.decoders.>` listener.
    pub fn spawn(bus: &Bus, blob: Option<S3BlobStore>, cfg: ServingCfg) -> Self {
        let (tx, rx) = watch::channel(Arc::new(Loaded::default()));
        let (kick, kick_rx) = mpsc::channel(4);
        if let Some(blob) = blob {
            tokio::spawn(run_loader(blob, cfg.reload, kick_rx, tx));
        } else {
            tracing::warn!("no Garage client: neural decoders are not served");
        }
        let me = Self::with_receiver(cfg, rx, kick);
        tokio::spawn(run_ctl(bus.clone(), me.clone()));
        me
    }

    fn with_receiver(cfg: ServingCfg, rx: watch::Receiver<Arc<Loaded>>, kick: mpsc::Sender<()>) -> Self {
        let state = State {
            cfg,
            loaded: Arc::new(Loaded::default()),
            by_key: HashMap::new(),
            slots: BTreeMap::new(),
            pins: HashMap::new(),
            disabled: HashSet::new(),
            text: TextTimeline::new(),
            pending: VecDeque::new(),
            online: HashMap::new(),
            counters: BTreeMap::new(),
        };
        Self { state: Arc::new(Mutex::new(state)), rx, kick }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Picks up a new loader result.
    fn refresh(&mut self, now_ms: i64) {
        if !self.rx.has_changed().unwrap_or(false) {
            return;
        }
        let loaded = self.rx.borrow_and_update().clone();
        let mut st = self.lock();
        st.counters.insert("neural.cards".into(), loaded.cards as f64);
        st.counters.insert("neural.loaded".into(), loaded.candidates.len() as f64);
        st.counters.insert("neural.load_errors".into(), loaded.errors.len() as f64);
        st.loaded = loaded;
        st.assign(now_ms);
        let summary: Vec<String> = st
            .slots
            .iter()
            .map(|((f, ts), s)| format!("{f}@{ts}: {} / shadow {}", s.serving.as_deref().unwrap_or("-"), s.shadow.as_deref().unwrap_or("-")))
            .collect();
        tracing::info!(slots = ?summary, "neural decoder set refreshed");
    }

    /// A news item (`news.item.>`), knowledge time `ts_ms`.
    pub fn on_news(&self, ts_ms: i64, title: &str, body: &str) {
        let mut st = self.lock();
        st.text.push(ts_ms, title, body);
        st.text.prune_before(ts_ms - TEXT_KEEP_MS);
        st.text.finish();
    }

    /// Whether a serving neural decoder replaces the baseline `family` at `ts`.
    pub fn replaces(&self, family: &str, ts: &str) -> bool {
        let st = self.lock();
        st.cfg.replace_baseline
            && st.slots.iter().any(|((f, t), s)| {
                f == family && t.to_string() == ts && s.serving.is_some() && !st.disabled.contains(&(f.clone(), *t))
            })
    }

    /// One closed bar: resolves due forecasts and returns `(subject, output)` for every serving
    /// and shadow decoder of this timescale. `closes` are the closed bars' close prices (oldest
    /// first), the last one closing at `as_of_ms`.
    pub fn on_bar_close(
        &mut self,
        inst: &str,
        ts: Timescale,
        as_of_ms: i64,
        closes: &[f64],
        latent: Option<&LatentState>,
    ) -> Vec<(String, DecoderOutput)> {
        self.refresh(as_of_ms);
        let started = Instant::now();
        let step_ms = ts.duration().as_millis() as i64;
        let series = series_of(inst, closes, as_of_ms, step_ms);
        let latent = latent.and_then(latent_vector);
        let mut st = self.lock();
        resolve(&mut st, inst, ts, as_of_ms, series.log_price.last().copied());
        let slots: Vec<SlotKey> = st.slots.keys().filter(|(_, t)| *t == ts).cloned().collect();
        let mut outs = Vec::new();
        for slot in slots {
            if st.disabled.contains(&slot) {
                continue;
            }
            for (role, key) in st.role_keys(&slot) {
                let Some(c) = st.by_key.get(&key).cloned() else { continue };
                let Some(served) = c.decoder.serve(&series, latent.as_deref(), &st.text) else { continue };
                let g = c.decoder.genome();
                // online scoring
                if !served.quantiles.is_empty() && st.pending.len() < MAX_PENDING {
                    if let Some(base) = series.log_price.last() {
                        st.pending.push_back(Pending {
                            slot: slot.clone(),
                            key: key.clone(),
                            inst: inst.to_owned(),
                            due_ms: as_of_ms + g.horizon as i64 * step_ms,
                            base: *base,
                            quantiles: served.quantiles.clone(),
                            levels: served.levels.clone(),
                        });
                    }
                } else if let Some(loss) = served.components.first().filter(|_| served.quantiles.is_empty()) {
                    // chart: CE vs the realised raster; news: cos(pred, realised) → loss 1 − cos
                    let loss = if c.card.meta.output_family.starts_with("text.") { 1.0 - loss } else { *loss };
                    st.score(&slot, &key, loss);
                }
                let out = output_of(&c, role, inst, ts, as_of_ms, &served);
                *st.counters.entry(format!("neural.outputs.{}.{}", out.family, role.as_str())).or_default() += 1.0;
                let subject = match role {
                    Role::Serving => fa_bus::subjects::decoder_output(&out.family),
                    Role::Shadow => format!("decoder.shadow.{}", out.family),
                };
                outs.push((subject, out));
            }
            st.maybe_promote(&slot, as_of_ms);
        }
        st.counters.insert("neural.infer_ms".into(), started.elapsed().as_secs_f64() * 1e3);
        outs
    }

    /// Heartbeat gauges.
    pub fn gauges(&self) -> Vec<(String, f64)> {
        let st = self.lock();
        let mut g: Vec<(String, f64)> = st.counters.iter().map(|(k, v)| (k.clone(), *v)).collect();
        g.push(("neural.slots".into(), st.slots.len() as f64));
        g.push(("neural.pending".into(), st.pending.len() as f64));
        g.push(("neural.news".into(), st.text.len() as f64));
        for ((f, ts), s) in &st.slots {
            g.push((format!("neural.serving.{f}.{ts}"), f64::from(u8::from(s.serving.is_some() && !st.disabled.contains(&(f.clone(), *ts))))));
            for (role, key) in [(Role::Serving, &s.serving), (Role::Shadow, &s.shadow)] {
                let Some(key) = key else { continue };
                let o = st.online.get(&((f.clone(), *ts), key.clone())).copied().unwrap_or_default();
                g.push((format!("neural.online.{f}.{ts}.{}.n", role.as_str()), o.n as f64));
                if o.n > 0 {
                    g.push((format!("neural.online.{f}.{ts}.{}.loss", role.as_str()), o.mean()));
                }
            }
        }
        g
    }

    /// `ctl.decoders.<verb>`.
    pub fn ctl(&self, verb: &str, args: &HashMap<String, String>) -> CtlReply {
        let now = crate::now_ms();
        let slot_arg = || -> Result<SlotKey, String> {
            let f = args.get("family").cloned().ok_or("missing `family`")?;
            let ts = args.get("timescale").ok_or("missing `timescale`")?.parse::<Timescale>().map_err(|e| e.to_string())?;
            Ok((f, ts))
        };
        let ok = |m: String| CtlReply { ok: true, message: m };
        let err = |m: String| CtlReply { ok: false, message: m };
        match verb {
            "status" => {
                let st = self.lock();
                let slots: Vec<_> = st
                    .slots
                    .iter()
                    .map(|((f, ts), s)| {
                        let stat = |k: &Option<String>| {
                            k.as_ref().map(|k| {
                                let o = st.online.get(&((f.clone(), *ts), k.clone())).copied().unwrap_or_default();
                                let c = st.by_key.get(k);
                                json!({"key": k, "genome": c.map(|c| c.card.meta.genome.clone()),
                                       "score": c.map(|c| c.card.meta.score), "online_n": o.n, "online_loss": o.mean()})
                            })
                        };
                        json!({"family": f, "timescale": ts.to_string(), "serving": stat(&s.serving),
                               "shadow": stat(&s.shadow), "disabled": st.disabled.contains(&(f.clone(), *ts)),
                               "pinned": st.pins.get(&(f.clone(), *ts))})
                    })
                    .collect();
                ok(json!({"cards": st.loaded.cards, "loaded": st.loaded.candidates.len(), "errors": st.loaded.errors,
                          "replace_baseline": st.cfg.replace_baseline, "news": st.text.len(), "slots": slots})
                .to_string())
            }
            "reload" => match self.kick.try_send(()) {
                Ok(()) => ok("reload requested".into()),
                Err(e) => err(format!("reload not queued: {e}")),
            },
            "promote" => match slot_arg() {
                Ok(slot) => match self.lock().promote(&slot, now) {
                    Some(k) => ok(format!("{}@{}: {k} now serving", slot.0, slot.1)),
                    None => err(format!("{}@{}: no shadow decoder to promote", slot.0, slot.1)),
                },
                Err(e) => err(e),
            },
            "pin" => {
                let Some(key) = args.get("key") else { return err("missing `key`".into()) };
                let mut st = self.lock();
                let Some(c) = st.by_key.get(key).cloned() else {
                    return err(format!("`{key}` is not a loaded candidate"));
                };
                let slot = (c.card.meta.output_family.clone(), c.timescale);
                st.pins.insert(slot.clone(), key.clone());
                st.assign(now);
                ok(format!("{}@{}: pinned {key}", slot.0, slot.1))
            }
            "unpin" | "off" | "on" => match slot_arg() {
                Ok(slot) => {
                    let mut st = self.lock();
                    match verb {
                        "unpin" => {
                            st.pins.remove(&slot);
                        }
                        "off" => {
                            st.disabled.insert(slot.clone());
                        }
                        _ => {
                            st.disabled.remove(&slot);
                        }
                    }
                    st.assign(now);
                    ok(format!("{}@{}: {verb}", slot.0, slot.1))
                }
                Err(e) => err(e),
            },
            other => err(format!("unknown verb `{other}` (status|reload|promote|pin|unpin|off|on)")),
        }
    }
}

/// Resolves due forecasts of `inst` at `ts` with the current log close.
fn resolve(st: &mut State, inst: &str, ts: Timescale, now_ms: i64, log_close: Option<f64>) {
    let Some(y_now) = log_close else { return };
    let mut keep = VecDeque::with_capacity(st.pending.len());
    let mut done = Vec::new();
    for p in st.pending.drain(..) {
        if p.inst == inst && p.slot.1 == ts && p.due_ms <= now_ms {
            done.push(p);
        } else {
            keep.push_back(p);
        }
    }
    st.pending = keep;
    for p in done {
        let y = y_now - p.base;
        let loss = crps_quantiles(&p.quantiles, &p.levels, y);
        st.score(&p.slot, &p.key, loss);
    }
}

/// Concatenated expert means of a latent state (expert order), as evo-worker conditions on them.
pub fn latent_vector(state: &LatentState) -> Option<Vec<f32>> {
    let mut experts: Vec<_> = state.experts.iter().collect();
    experts.sort_by_key(|e| e.expert_index);
    let v: Vec<f32> = experts.iter().flat_map(|e| e.mean.iter().copied()).collect();
    (!v.is_empty()).then_some(v)
}

/// Log-price series of closed bars ending at `as_of_ms`.
pub fn series_of(inst: &str, closes: &[f64], as_of_ms: i64, step_ms: i64) -> Series {
    let n = closes.len();
    let mut s = Series { instrument_id: inst.to_owned(), ts_ms: Vec::with_capacity(n), log_price: Vec::with_capacity(n), step_ms };
    for (i, c) in closes.iter().enumerate() {
        if *c > 0.0 && c.is_finite() {
            s.ts_ms.push(as_of_ms - (n - 1 - i) as i64 * step_ms);
            s.log_price.push(c.ln());
        }
    }
    s
}

fn output_of(c: &Candidate, role: Role, inst: &str, ts: Timescale, as_of_ms: i64, s: &fa_decoders::neural::Served) -> DecoderOutput {
    let fam = c.card.meta.output_family.clone();
    let g = c.decoder.genome();
    let mut refs = vec![format!("inst:{inst}")];
    let quantiles = match fam.as_str() {
        "rnn_prior" | "retrieval" => {
            refs.push(format!("h:{}", g.horizon));
            s.quantiles.clone()
        }
        f => {
            if let Some(kind) = f.strip_prefix("visual.") {
                refs.push(format!("kind:{kind}"));
            } else if let Some(topic) = f.strip_prefix("text.") {
                refs.push(format!("topic:{topic}"));
            }
            if f == "visual.future" {
                refs.push(format!("h:{}", g.horizon));
            }
            std::iter::once(s.score.clamp(-1.0, 1.0)).chain(s.components.iter().copied()).collect()
        }
    };
    refs.extend(["src:neural".to_owned(), format!("role:{}", role.as_str()), format!("card:{}", c.key)]);
    DecoderOutput {
        decoder_id: c.decoder.decoder_id(),
        genome_id: c.card.meta.genome.clone(),
        family: fam,
        snapshot_id: String::new(),
        timescale: ts.to_string(),
        as_of: Some(ms_to_ts(as_of_ms)),
        output_refs: refs,
        quantiles,
        analogs: s
            .analogs
            .iter()
            .map(|a| v1::Analog {
                window_id: a.window_id.clone(),
                window_end: Some(ms_to_ts(as_of_ms - (a.time_distance_days * 86_400_000.0) as i64)),
                similarity: a.similarity,
                time_distance_days: a.time_distance_days,
            })
            .collect(),
    }
}

/// Ranks cards into candidates per (output family, timescale).
pub fn rank_cards(cards: &[EliteCard], per_slot: usize) -> Vec<(EliteCard, Timescale)> {
    let mut by_ts: BTreeMap<Timescale, Vec<EliteCard>> = BTreeMap::new();
    for c in cards {
        if let Ok(g) = DecoderGenome::parse(&c.meta.genome) {
            by_ts.entry(g.timescale).or_default().push(c.clone());
        }
    }
    by_ts.into_iter().flat_map(|(ts, cs)| feedback::select_top_k(&cs, per_slot, None).into_iter().map(move |c| (c, ts))).collect()
}

async fn run_loader(blob: S3BlobStore, every: Duration, mut kick: mpsc::Receiver<()>, tx: watch::Sender<Arc<Loaded>>) {
    let mut cards: HashMap<String, EliteCard> = HashMap::new();
    let mut bundles: HashMap<String, Arc<NeuralDecoder>> = HashMap::new();
    let mut tick = tokio::time::interval(every);
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            Some(()) = kick.recv() => {}
        }
        let mut errors = Vec::new();
        let listed = match blob.list(fa_blob::buckets::MODELS, feedback::PREFIX).await {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(error = %e, "decoder card listing failed");
                continue;
            }
        };
        for o in listed.iter().filter(|o| o.key.ends_with(feedback::CARD_SUFFIX)) {
            if cards.contains_key(&o.key) {
                continue;
            }
            let key = ObjectKey { bucket: fa_blob::buckets::MODELS, key: o.key.clone() };
            match blob.get(&key).await.map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<EliteCard>(&b).map_err(|e| e.to_string())) {
                Ok(c) => {
                    cards.insert(o.key.clone(), c);
                }
                Err(e) => errors.push(format!("{}: {e}", o.key)),
            }
        }
        let all: Vec<EliteCard> = cards.values().cloned().collect();
        let mut candidates = Vec::new();
        for (card, ts) in rank_cards(&all, CANDIDATES) {
            let d = match bundles.get(&card.key) {
                Some(d) => d.clone(),
                None => {
                    let key = ObjectKey { bucket: fa_blob::buckets::MODELS, key: card.key.clone() };
                    let bytes = match blob.get(&key).await {
                        Ok(b) => b,
                        Err(e) => {
                            errors.push(format!("{}: {e}", card.key));
                            continue;
                        }
                    };
                    let loaded = tokio::task::spawn_blocking(move || NeuralDecoder::from_bundle(&bytes, &Device::Cpu)).await;
                    match loaded {
                        Ok(Ok(d)) => {
                            let d = Arc::new(d);
                            bundles.insert(card.key.clone(), d.clone());
                            d
                        }
                        Ok(Err(e)) => {
                            errors.push(format!("{}: {e}", card.key));
                            continue;
                        }
                        Err(e) => {
                            errors.push(format!("{}: {e}", card.key));
                            continue;
                        }
                    }
                }
            };
            candidates.push(Candidate { key: card.key.clone(), card, decoder: d, timescale: ts });
        }
        let keep: HashSet<String> = candidates.iter().map(|c| c.key.clone()).collect();
        bundles.retain(|k, _| keep.contains(k));
        tracing::info!(cards = cards.len(), candidates = candidates.len(), errors = errors.len(), "decoder cards scanned");
        if tx.send(Arc::new(Loaded { candidates, cards: cards.len(), errors })).is_err() {
            return;
        }
    }
}

async fn run_ctl(bus: Bus, serving: NeuralServing) {
    let subject = "ctl.decoders.>".to_owned();
    let Ok(mut sub) = bus.subscribe_raw(subject.clone()).await else {
        tracing::error!(subject, "cannot subscribe");
        return;
    };
    while let Some(msg) = sub.next().await {
        let verb = msg.subject.as_str().strip_prefix("ctl.decoders.").unwrap_or_default().to_owned();
        let answer = match <CtlRequest as fa_proto::Message>::decode(msg.payload.as_ref()) {
            Ok(req) => serving.ctl(&verb, &req.args.into_iter().collect()),
            Err(e) => CtlReply { ok: false, message: format!("undecodable CtlRequest: {e}") },
        };
        if let Some(reply) = msg.reply {
            let bytes = fa_proto::Message::encode_to_vec(&answer);
            if let Err(e) = bus.client().publish(reply, bytes.into()).await {
                tracing::warn!(error = %e, "ctl.decoders reply failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fa_decoders::{
        DecodeCtx, LatentPoint,
        neural::{TrainOptions, train},
    };

    use super::*;

    /// SYNTHETIC trending prices + latents, enough for a small decoder.
    fn trained(genome: &str, score: f64, key: &str) -> Candidate {
        let mut ctx = DecodeCtx::default();
        let mut lp = 4.0;
        let mut ts = Vec::new();
        let mut log_price = Vec::new();
        let mut rng = fa_core::rng::Rng::seed(3);
        for i in 0..800 {
            let r = if (i / 100) % 2 == 0 { 1.0 } else { -1.0 };
            lp += 0.0005 * r + 0.001 * rng.normal();
            ts.push(i as i64 * 60_000);
            log_price.push(lp);
            ctx.push("i1", LatentPoint { as_of_ms: i as i64 * 60_000, vec: vec![r as f32, 0.0] });
        }
        let s = Series { instrument_id: "i1".into(), ts_ms: ts, log_price, step_ms: 60_000 };
        let g = DecoderGenome::parse(genome).unwrap();
        let mut d = train(&g, &[s], Arc::new(ctx), Arc::default(), &TrainOptions::default()).unwrap();
        d.meta.score = score;
        d.meta.objectives.insert("crps_skill".into(), score);
        let card = EliteCard { meta: d.meta.clone(), bucket: "fa-models".into(), key: key.into(), created_ms: 0 };
        Candidate { key: key.into(), card, decoder: Arc::new(d), timescale: g.timescale }
    }

    fn serving_with(cands: Vec<Candidate>, cfg: ServingCfg) -> (NeuralServing, watch::Sender<Arc<Loaded>>) {
        let (tx, rx) = watch::channel(Arc::new(Loaded::default()));
        let (kick, _krx) = mpsc::channel(1);
        let s = NeuralServing::with_receiver(cfg, rx, kick);
        tx.send(Arc::new(Loaded { cards: cands.len(), candidates: cands, errors: vec![] })).unwrap();
        (s, tx)
    }

    #[test]
    fn serves_shadows_scores_and_promotes() {
        let a = trained("nn_forecast:ts=1m;h=2;lb=8;width=8;steps=40", 0.2, "decoders/g/a");
        let b = trained("nn_forecast:ts=1m;h=2;lb=8;width=8;steps=40;seed=9", 0.1, "decoders/g/b");
        let cfg = ServingCfg { reload: Duration::from_secs(60), replace_baseline: true, promote_min: 3, promote_margin: -10.0, promote_min_ms: 0 };
        let (mut s, _tx) = serving_with(vec![a, b], cfg);
        let closes: Vec<f64> = (0..40).map(|i| 100.0 + i as f64 * 0.1).collect();
        let latent = LatentState {
            experts: vec![v1::ExpertLatent { expert_index: 0, mean: vec![1.0, 0.0], ..Default::default() }],
            ..Default::default()
        };
        let mut outs = Vec::new();
        for k in 0..8 {
            let n = 30 + k;
            outs = s.on_bar_close("i1", Timescale::Minute, n as i64 * 60_000, &closes[..n], Some(&latent));
        }
        assert_eq!(outs.len(), 2);
        let (sub0, o0) = &outs[0];
        assert_eq!(sub0, "decoder.output.rnn_prior");
        assert!(o0.output_refs.contains(&"role:serving".to_owned()) && o0.output_refs.contains(&"h:2".to_owned()));
        assert!(o0.quantiles.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(outs[1].0, "decoder.shadow.rnn_prior");
        assert!(s.replaces("rnn_prior", "1m"));
        assert!(!s.replaces("rnn_prior", "1h"));
        // promote_margin −10: the shadow wins as soon as both have 3 resolved forecasts
        let g: HashMap<String, f64> = s.gauges().into_iter().collect();
        let promotions = g["neural.promotions"];
        assert!(promotions >= 1.0, "{g:?}");
        let status = s.ctl("status", &HashMap::new());
        assert!(status.ok && status.message.contains("decoders/g/b"), "{}", status.message);
        let v: serde_json::Value = serde_json::from_str(&status.message).unwrap();
        // every promotion swaps the roles (and pins the new serving decoder)
        let expect = if promotions as u64 % 2 == 1 { "decoders/g/b" } else { "decoders/g/a" };
        assert_eq!(v["slots"][0]["serving"]["key"], expect);
        assert_eq!(v["slots"][0]["pinned"], expect);
        let args = |kv: &[(&str, &str)]| kv.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
        assert!(s.ctl("promote", &args(&[("family", "rnn_prior"), ("timescale", "1m")])).ok);
        assert!(s.ctl("off", &args(&[("family", "rnn_prior"), ("timescale", "1m")])).ok);
        assert!(s.on_bar_close("i1", Timescale::Minute, 39 * 60_000, &closes[..39], None).is_empty());
        assert!(!s.ctl("pin", &args(&[("key", "nope")])).ok);
        assert!(!s.ctl("bogus", &HashMap::new()).ok);
    }

    #[test]
    fn ranking_keeps_the_best_per_slot() {
        let a = trained("nn_forecast:ts=1m;h=2;lb=8;width=8;steps=5", 0.3, "a");
        let mut bad = trained("nn_forecast:ts=1m;h=2;lb=8;width=8;steps=5", -0.1, "bad");
        bad.card.meta.score = -0.1;
        let ranked = rank_cards(&[bad.card.clone(), a.card.clone()], 3);
        assert_eq!(ranked.len(), 1, "decoders worse than naive never serve");
        assert_eq!(ranked[0].1, Timescale::Minute);
        let s = series_of("x", &[1.0, -1.0, 2.0], 10_000, 1000);
        assert_eq!(s.ts_ms, vec![8000, 10_000]);
    }
}
