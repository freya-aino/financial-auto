//! In-memory aggregation of live bus traffic for the Latent, Evolution and Trading views
//! (spec 50 §4.5–4.7, spec 51). Pure: `ingest(subject, payload)` is fed by the bus
//! collector and by tests; the `*_view` methods produce the endpoint JSON.
//!
//! Nothing here is persisted. Paper positions are rebuilt from the fills seen since
//! control-api started (D-028).
use std::collections::{BTreeMap, HashMap, VecDeque};

use chrono::{DateTime, Utc};
use fa_core::InstrumentId;
use fa_ledger::{Fill, Ledger};
use fa_proto::{
    Message, Timestamp,
    v1::{
        DecoderOutput, Envelope, ExposureDistribution, Halt, Job, JobProgress, JobResult, LatentState, OrderEvent,
        OrderIntent, RegionEvent, RiskState, Side, TradingMode, TransitionUpdate, envelope::Payload, job_result,
        order_event, region_event,
    },
};
use serde::Serialize;
use serde_json::{Value, json};
use uuid::Uuid;

/// Subjects the collector subscribes to.
pub const SUBJECTS: [&str; 7] =
    ["latent.>", "decoder.output.>", "jobs.>", "signal.exposure.>", "order.>", "risk.>", "md.trade.>"];

/// Starting cash of the in-memory paper book (D-028).
pub const PAPER_CASH_EUR: f64 = 100_000.0;

const MAX_DECODER_OUTPUTS: usize = 200;
const MAX_JOBS: usize = 500;
const MAX_ORDERS: usize = 500;

pub fn ts(t: Option<&Timestamp>) -> Option<DateTime<Utc>> {
    t.and_then(|t| DateTime::from_timestamp(t.seconds, u32::try_from(t.nanos).unwrap_or(0)))
}

// ---------------------------------------------------------------- latent

#[derive(Debug, Clone, Serialize)]
pub struct ExpertStateView {
    pub expert_index: u32,
    pub region_id: String,
    pub region_lineage_id: String,
    pub dims: usize,
    /// L2 norm of the latent mean.
    pub norm: f64,
    /// Mean posterior std (exp(log_var / 2)).
    pub uncertainty: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StateView {
    pub instrument_id: String,
    pub snapshot_id: String,
    pub timescale: String,
    pub as_of: Option<DateTime<Utc>>,
    pub router_weights: Vec<f32>,
    pub experts: Vec<ExpertStateView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegionView {
    pub region_id: String,
    pub lineage_id: String,
    pub snapshot_id: String,
    pub expert_index: u32,
    /// Last event kind: created | merged | split | retired.
    pub status: String,
    pub related_region_ids: Vec<String>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeView {
    pub to: String,
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TransitionRowView {
    pub from: String,
    pub sampler_id: String,
    pub snapshot_id: String,
    pub expert_index: u32,
    pub as_of: Option<DateTime<Utc>>,
    pub to: Vec<EdgeView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecoderOutputView {
    pub received_at: DateTime<Utc>,
    pub decoder_id: String,
    pub genome_id: String,
    pub family: String,
    pub snapshot_id: String,
    pub timescale: String,
    pub as_of: Option<DateTime<Utc>>,
    pub output_refs: Vec<String>,
    pub quantiles: Vec<f64>,
    pub analogs: Vec<Value>,
}

// ------------------------------------------------------------- evolution

#[derive(Debug, Clone, Serialize)]
pub struct JobView {
    pub job_id: String,
    pub kind: String,
    /// queued | running | ok | failed | preempted | budget_exceeded
    pub status: String,
    pub priority: u32,
    pub snapshot_id: String,
    pub genome_id: String,
    pub budget_gpu_ms: u64,
    pub used_gpu_ms: u64,
    pub progress: f32,
    pub message: String,
    pub worker_id: String,
    pub error: String,
    /// Single provisional score (`{"score": x}` results, D-025).
    pub score: Option<f64>,
    /// Pareto objectives when the result carries `{"objectives": {...}}`.
    pub objectives: BTreeMap<String, f64>,
    pub submitted_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// --------------------------------------------------------------- trading

#[derive(Debug, Clone, Serialize)]
pub struct OrderView {
    pub intent_id: String,
    pub account_id: String,
    pub instrument_id: String,
    pub side: String,
    pub quantity: f64,
    pub exec_algo: String,
    pub limit_price: Option<f64>,
    pub mode: String,
    pub strategy_ids: Vec<String>,
    /// intent | approved | ack | partial_fill | filled | canceled | rejected
    pub status: String,
    pub filled_quantity: f64,
    pub avg_fill_price: Option<f64>,
    pub fees: f64,
    pub reason: String,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
    pub events: Vec<OrderEventView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrderEventView {
    pub kind: String,
    pub fill_price: f64,
    pub fill_quantity: f64,
    pub fee: f64,
    pub fee_currency: String,
    pub event_time: Option<DateTime<Utc>>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HaltView {
    pub scope: String,
    pub reason: String,
    pub since: Option<DateTime<Utc>>,
    pub manual: bool,
}

impl From<&Halt> for HaltView {
    fn from(h: &Halt) -> Self {
        Self { scope: h.scope.clone(), reason: h.reason.clone(), since: ts(h.since.as_ref()), manual: h.manual }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SignalView {
    pub strategy_id: String,
    pub genome_id: String,
    pub horizon: String,
    pub as_of: Option<DateTime<Utc>>,
    pub exposures: Vec<Value>,
}

// ------------------------------------------------------------------ state

#[derive(Debug)]
pub struct Live {
    // latent, keyed by timescale
    states: BTreeMap<String, BTreeMap<String, StateView>>,
    regions: BTreeMap<String, BTreeMap<String, RegionView>>,
    transitions: BTreeMap<String, BTreeMap<String, TransitionRowView>>,
    decoder_outputs: VecDeque<DecoderOutputView>,
    // evolution
    jobs: HashMap<String, JobView>,
    job_order: VecDeque<String>,
    // trading
    signals: BTreeMap<String, SignalView>,
    orders: HashMap<String, OrderView>,
    order_order: VecDeque<String>,
    ledger: Ledger,
    /// Last trade price per instrument UUID string.
    marks: HashMap<String, f64>,
    risk: Option<(RiskState, DateTime<Utc>)>,
    halts: BTreeMap<String, HaltView>,
    /// Peak paper equity, for drawdown of the paper book.
    peak_equity: f64,
}

impl Default for Live {
    fn default() -> Self {
        Self {
            states: BTreeMap::new(),
            regions: BTreeMap::new(),
            transitions: BTreeMap::new(),
            decoder_outputs: VecDeque::new(),
            jobs: HashMap::new(),
            job_order: VecDeque::new(),
            signals: BTreeMap::new(),
            orders: HashMap::new(),
            order_order: VecDeque::new(),
            ledger: Ledger::with_cash(PAPER_CASH_EUR),
            marks: HashMap::new(),
            risk: None,
            halts: BTreeMap::new(),
            peak_equity: PAPER_CASH_EUR,
        }
    }
}

impl Live {
    /// Routes one bus message by subject. Undecodable payloads are logged and ignored.
    pub fn ingest(&mut self, subject: &str, payload: &[u8]) {
        let r = if subject.starts_with("latent.state.") {
            LatentState::decode(payload).map(|m| self.on_latent_state(m))
        } else if subject.starts_with("latent.region.") {
            RegionEvent::decode(payload).map(|m| self.on_region(m))
        } else if subject.starts_with("latent.transition.") {
            TransitionUpdate::decode(payload).map(|m| self.on_transition(m))
        } else if subject.starts_with("decoder.output.") {
            DecoderOutput::decode(payload).map(|m| self.on_decoder_output(m))
        } else if let Some(kind) = subject.strip_prefix("jobs.result.") {
            JobResult::decode(payload).map(|m| self.on_job_result(kind, m))
        } else if subject.starts_with("jobs.progress.") {
            JobProgress::decode(payload).map(|m| self.on_job_progress(m))
        } else if subject.starts_with("jobs.") {
            Job::decode(payload).map(|m| self.on_job(m))
        } else if subject.starts_with("signal.exposure.") {
            ExposureDistribution::decode(payload).map(|m| self.on_signal(m))
        } else if subject.starts_with("order.intent.") {
            OrderIntent::decode(payload).map(|m| self.on_intent(m, "intent"))
        } else if subject.starts_with("order.approved.") {
            OrderIntent::decode(payload).map(|m| self.on_intent(m, "approved"))
        } else if subject.starts_with("order.event.") {
            OrderEvent::decode(payload).map(|m| self.on_order_event(m))
        } else if subject == "risk.state" {
            RiskState::decode(payload).map(|m| self.on_risk_state(m))
        } else if subject.starts_with("risk.halt.") {
            Halt::decode(payload).map(|m| self.on_halt(m))
        } else if let Some(scope) = subject.strip_prefix("risk.resume.") {
            self.on_resume(scope);
            Ok(())
        } else if subject.starts_with("md.trade.") {
            Envelope::decode(payload).map(|m| self.on_trade(m))
        } else {
            Ok(())
        };
        if let Err(e) = r {
            tracing::debug!(subject, error = %e, "live: undecodable message");
        }
    }

    // ---- latent

    fn on_latent_state(&mut self, m: LatentState) {
        let experts = m
            .experts
            .iter()
            .map(|e| {
                let norm = e.mean.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                let uncertainty = if e.log_var.is_empty() {
                    0.0
                } else {
                    e.log_var.iter().map(|lv| (f64::from(*lv) / 2.0).exp()).sum::<f64>() / e.log_var.len() as f64
                };
                ExpertStateView {
                    expert_index: e.expert_index,
                    region_id: e.region_id.clone(),
                    region_lineage_id: e.region_lineage_id.clone(),
                    dims: e.mean.len(),
                    norm,
                    uncertainty,
                }
            })
            .collect();
        let v = StateView {
            instrument_id: m.instrument_id.clone(),
            snapshot_id: m.snapshot_id,
            timescale: m.timescale.clone(),
            as_of: ts(m.as_of.as_ref()),
            router_weights: m.router_weights,
            experts,
        };
        self.states.entry(m.timescale).or_default().insert(m.instrument_id, v);
    }

    fn on_region(&mut self, m: RegionEvent) {
        let space = m.space.unwrap_or_default();
        let status = match region_event::Kind::try_from(m.kind).unwrap_or_default() {
            region_event::Kind::Created => "created",
            region_event::Kind::Merged => "merged",
            region_event::Kind::Split => "split",
            region_event::Kind::Retired => "retired",
            region_event::Kind::Unspecified => "unknown",
        };
        let v = RegionView {
            region_id: m.region_id.clone(),
            lineage_id: m.lineage_id,
            snapshot_id: space.snapshot_id,
            expert_index: space.expert_index,
            status: status.into(),
            related_region_ids: m.related_region_ids,
            updated_at: Utc::now(),
        };
        self.regions.entry(space.timescale).or_default().insert(m.region_id, v);
    }

    fn on_transition(&mut self, m: TransitionUpdate) {
        let space = m.space.unwrap_or_default();
        let v = TransitionRowView {
            from: m.from_region_id.clone(),
            sampler_id: m.sampler_id,
            snapshot_id: space.snapshot_id,
            expert_index: space.expert_index,
            as_of: ts(m.as_of.as_ref()),
            to: m
                .to
                .into_iter()
                .map(|p| EdgeView { to: p.region_id, mean: p.mean, lower: p.lower, upper: p.upper })
                .collect(),
        };
        self.transitions.entry(space.timescale).or_default().insert(m.from_region_id, v);
    }

    fn on_decoder_output(&mut self, m: DecoderOutput) {
        let analogs = m
            .analogs
            .iter()
            .map(|a| {
                json!({
                    "window_id": a.window_id,
                    "window_end": ts(a.window_end.as_ref()),
                    "similarity": a.similarity,
                    "time_distance_days": a.time_distance_days,
                })
            })
            .collect();
        self.decoder_outputs.push_front(DecoderOutputView {
            received_at: Utc::now(),
            decoder_id: m.decoder_id,
            genome_id: m.genome_id,
            family: m.family,
            snapshot_id: m.snapshot_id,
            timescale: m.timescale,
            as_of: ts(m.as_of.as_ref()),
            output_refs: m.output_refs,
            quantiles: m.quantiles,
            analogs,
        });
        self.decoder_outputs.truncate(MAX_DECODER_OUTPUTS);
    }

    pub fn timescales_seen(&self) -> Vec<String> {
        let mut v: Vec<String> =
            self.states.keys().chain(self.regions.keys()).chain(self.transitions.keys()).cloned().collect();
        v.sort();
        v.dedup();
        v
    }

    pub fn states_view(&self, timescale: &str) -> Value {
        let states: Vec<&StateView> = self.states.get(timescale).map(|m| m.values().collect()).unwrap_or_default();
        json!({ "timescale": timescale, "states": states })
    }

    pub fn regions_view(&self, timescale: &str) -> Value {
        let occupancy = self.occupancy(timescale);
        let regions: Vec<Value> = self
            .regions
            .get(timescale)
            .map(|m| {
                m.values()
                    .map(|r| {
                        let mut v = serde_json::to_value(r).unwrap_or_default();
                        v["occupancy"] = json!(occupancy.get(&r.region_id).copied().unwrap_or(0));
                        v
                    })
                    .collect()
            })
            .unwrap_or_default();
        json!({ "timescale": timescale, "regions": regions })
    }

    /// Number of instruments whose latest state sits in each region (any expert).
    fn occupancy(&self, timescale: &str) -> HashMap<String, usize> {
        let mut occ = HashMap::new();
        for s in self.states.get(timescale).into_iter().flat_map(|m| m.values()) {
            for e in &s.experts {
                *occ.entry(e.region_id.clone()).or_insert(0) += 1;
            }
        }
        occ
    }

    /// Region graph: nodes = known regions ∪ transition endpoints, edges = posterior rows.
    pub fn transitions_view(&self, timescale: &str) -> Value {
        let occupancy = self.occupancy(timescale);
        let known = self.regions.get(timescale);
        let rows: Vec<&TransitionRowView> =
            self.transitions.get(timescale).map(|m| m.values().collect()).unwrap_or_default();

        let mut ids: Vec<String> = known.map(|m| m.keys().cloned().collect()).unwrap_or_default();
        for r in &rows {
            ids.push(r.from.clone());
            ids.extend(r.to.iter().map(|e| e.to.clone()));
        }
        ids.sort();
        ids.dedup();

        let nodes: Vec<Value> = ids
            .iter()
            .map(|id| {
                let r = known.and_then(|m| m.get(id));
                json!({
                    "id": id,
                    "lineage_id": r.map(|r| r.lineage_id.as_str()),
                    "expert_index": r.map(|r| r.expert_index),
                    "status": r.map_or("unregistered", |r| r.status.as_str()),
                    "occupancy": occupancy.get(id).copied().unwrap_or(0),
                })
            })
            .collect();
        let edges: Vec<Value> = rows
            .iter()
            .flat_map(|r| {
                r.to.iter().map(move |e| {
                    json!({
                        "from": r.from, "to": e.to, "mean": e.mean, "lower": e.lower, "upper": e.upper,
                        "sampler_id": r.sampler_id,
                    })
                })
            })
            .collect();
        json!({ "timescale": timescale, "nodes": nodes, "edges": edges, "rows": rows })
    }

    pub fn decoder_outputs_view(&self, family: Option<&str>, limit: usize) -> Value {
        let outputs: Vec<&DecoderOutputView> = self
            .decoder_outputs
            .iter()
            .filter(|o| family.is_none_or(|f| o.family == f || o.family.starts_with(&format!("{f}."))))
            .take(limit)
            .collect();
        let mut families: Vec<&str> = self.decoder_outputs.iter().map(|o| o.family.as_str()).collect();
        families.sort_unstable();
        families.dedup();
        json!({ "outputs": outputs, "families": families })
    }

    // ---- evolution

    fn job_mut(&mut self, job_id: &str, kind: &str) -> &mut JobView {
        if !self.jobs.contains_key(job_id) {
            self.job_order.push_front(job_id.to_owned());
            if self.job_order.len() > MAX_JOBS
                && let Some(old) = self.job_order.pop_back()
            {
                self.jobs.remove(&old);
            }
        }
        let now = Utc::now();
        self.jobs.entry(job_id.to_owned()).or_insert_with(|| JobView {
            job_id: job_id.to_owned(),
            kind: kind.to_owned(),
            status: "queued".into(),
            priority: 0,
            snapshot_id: String::new(),
            genome_id: String::new(),
            budget_gpu_ms: 0,
            used_gpu_ms: 0,
            progress: 0.0,
            message: String::new(),
            worker_id: String::new(),
            error: String::new(),
            score: None,
            objectives: BTreeMap::new(),
            submitted_at: now,
            updated_at: now,
        })
    }

    fn on_job(&mut self, m: Job) {
        let j = self.job_mut(&m.job_id, &m.kind);
        j.kind = m.kind;
        j.priority = m.priority;
        j.snapshot_id = m.snapshot_id;
        j.genome_id = m.genome_id;
        j.budget_gpu_ms = m.budget_gpu_ms;
        j.updated_at = Utc::now();
    }

    fn on_job_progress(&mut self, m: JobProgress) {
        let j = self.job_mut(&m.job_id, "unknown");
        if j.status == "queued" {
            j.status = "running".into();
        }
        j.progress = m.fraction.clamp(0.0, 1.0);
        j.message = m.message;
        j.updated_at = Utc::now();
    }

    fn on_job_result(&mut self, kind: &str, m: JobResult) {
        let j = self.job_mut(&m.job_id, kind);
        j.status = match job_result::Status::try_from(m.status).unwrap_or(job_result::Status::Failed) {
            job_result::Status::Ok => "ok",
            job_result::Status::Failed => "failed",
            job_result::Status::Preempted => "preempted",
            job_result::Status::BudgetExceeded => "budget_exceeded",
        }
        .into();
        if j.status == "ok" {
            j.progress = 1.0;
        }
        j.used_gpu_ms = m.used_gpu_ms;
        j.worker_id = m.worker_id;
        j.error = m.error;
        if let Ok(v) = serde_json::from_slice::<Value>(&m.result) {
            j.score = v.get("score").and_then(Value::as_f64);
            if let Some(obj) = v.get("objectives").and_then(Value::as_object) {
                j.objectives = obj.iter().filter_map(|(k, v)| v.as_f64().map(|x| (k.clone(), x))).collect();
            }
        }
        j.updated_at = Utc::now();
    }

    pub fn jobs_view(&self, kind: Option<&str>, limit: usize) -> Value {
        let mut counts: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
        for j in self.jobs.values() {
            *counts.entry(&j.kind).or_default().entry(&j.status).or_insert(0) += 1;
        }
        let jobs: Vec<&JobView> = self
            .job_order
            .iter()
            .filter_map(|id| self.jobs.get(id))
            .filter(|j| kind.is_none_or(|k| j.kind == k))
            .take(limit)
            .collect();
        let mut objective_names: Vec<&str> =
            self.jobs.values().flat_map(|j| j.objectives.keys().map(String::as_str)).collect();
        objective_names.sort_unstable();
        objective_names.dedup();
        json!({ "counts": counts, "jobs": jobs, "objectives": objective_names })
    }

    // ---- trading

    fn on_signal(&mut self, m: ExposureDistribution) {
        let v = SignalView {
            strategy_id: m.strategy_id.clone(),
            genome_id: m.genome_id,
            horizon: m.horizon,
            as_of: ts(m.as_of.as_ref()),
            exposures: m
                .exposures
                .iter()
                .map(|e| json!({ "instrument_id": e.instrument_id, "mean": e.mean, "stddev": e.stddev }))
                .collect(),
        };
        self.signals.insert(m.strategy_id, v);
    }

    /// Records an intent (from the bus or a manual paper order). Later stages never
    /// downgrade the status of an order that already has broker events.
    pub fn on_intent(&mut self, m: OrderIntent, stage: &str) {
        let now = Utc::now();
        if !self.orders.contains_key(&m.intent_id) {
            self.order_order.push_front(m.intent_id.clone());
            if self.order_order.len() > MAX_ORDERS
                && let Some(old) = self.order_order.pop_back()
            {
                self.orders.remove(&old);
            }
        }
        let side = match Side::try_from(m.side).unwrap_or_default() {
            Side::Buy => "buy",
            Side::Sell => "sell",
            Side::Unspecified => "unspecified",
        };
        let mode = match TradingMode::try_from(m.mode).unwrap_or_default() {
            TradingMode::Paper => "paper",
            TradingMode::Shadow => "shadow",
            TradingMode::Live => "live",
            TradingMode::Unspecified => "unspecified",
        };
        let o = self.orders.entry(m.intent_id.clone()).or_insert_with(|| OrderView {
            intent_id: m.intent_id.clone(),
            account_id: String::new(),
            instrument_id: String::new(),
            side: String::new(),
            quantity: 0.0,
            exec_algo: String::new(),
            limit_price: None,
            mode: String::new(),
            strategy_ids: Vec::new(),
            status: stage.into(),
            filled_quantity: 0.0,
            avg_fill_price: None,
            fees: 0.0,
            reason: String::new(),
            created_at: None,
            updated_at: now,
            events: Vec::new(),
        });
        o.account_id = m.account_id;
        o.instrument_id = m.instrument_id;
        o.side = side.into();
        o.quantity = m.quantity;
        o.exec_algo = m.exec_algo;
        o.limit_price = m.limit_price;
        o.mode = mode.into();
        o.strategy_ids = m.strategy_ids;
        o.created_at = ts(m.created_at.as_ref());
        if o.events.is_empty() && (o.status == "intent" || stage == "approved") {
            o.status = stage.into();
        }
        o.updated_at = now;
    }

    fn on_order_event(&mut self, m: OrderEvent) {
        let kind = order_event::Kind::try_from(m.kind).unwrap_or_default();
        let kind_s = match kind {
            order_event::Kind::Ack => "ack",
            order_event::Kind::PartialFill => "partial_fill",
            order_event::Kind::Fill => "filled",
            order_event::Kind::Canceled => "canceled",
            order_event::Kind::Rejected => "rejected",
            order_event::Kind::Unspecified => "unknown",
        };
        let Some(o) = self.orders.get_mut(&m.intent_id) else {
            tracing::debug!(intent = %m.intent_id, "order event for unknown intent");
            return;
        };
        o.events.push(OrderEventView {
            kind: kind_s.into(),
            fill_price: m.fill_price,
            fill_quantity: m.fill_quantity,
            fee: m.fee,
            fee_currency: m.fee_currency.clone(),
            event_time: ts(m.event_time.as_ref()),
            reason: m.reason.clone(),
        });
        o.status = kind_s.into();
        o.reason = m.reason;
        o.updated_at = Utc::now();
        if !matches!(kind, order_event::Kind::Fill | order_event::Kind::PartialFill) || m.fill_quantity <= 0.0 {
            return;
        }
        let total = o.filled_quantity + m.fill_quantity;
        o.avg_fill_price =
            Some((o.avg_fill_price.unwrap_or(0.0) * o.filled_quantity + m.fill_price * m.fill_quantity) / total);
        o.filled_quantity = total;
        o.fees += m.fee;

        let Ok(uuid) = Uuid::parse_str(&o.instrument_id) else {
            return;
        };
        let signed = if o.side == "sell" { -m.fill_quantity } else { m.fill_quantity };
        let fill_id = format!(
            "{}:{}:{}",
            m.intent_id,
            o.events.len(),
            m.event_time.as_ref().map_or(0, |t| t.seconds * 1_000_000_000 + i64::from(t.nanos))
        );
        let instrument_id = o.instrument_id.clone();
        // Paper instruments are treated as EUR-priced until FX conversion lands (D-028).
        self.ledger.apply_fill(&Fill {
            fill_id,
            intent_id: m.intent_id,
            instrument: InstrumentId(uuid),
            quantity: signed,
            price: m.fill_price,
            fx_to_eur: 1.0,
            fee_eur: m.fee,
        });
        self.marks.entry(instrument_id).or_insert(m.fill_price);
        self.update_peak();
    }

    fn on_trade(&mut self, m: Envelope) {
        if let Some(Payload::Trade(t)) = m.payload
            && !m.instrument_id.is_empty()
            && t.price > 0.0
        {
            self.marks.insert(m.instrument_id, t.price);
            self.update_peak();
        }
    }

    fn on_risk_state(&mut self, m: RiskState) {
        for h in &m.active_halts {
            self.halts.insert(h.scope.clone(), h.into());
        }
        self.risk = Some((m, Utc::now()));
    }

    pub fn on_halt(&mut self, h: Halt) {
        self.halts.insert(h.scope.clone(), (&h).into());
    }

    pub fn on_resume(&mut self, scope: &str) {
        self.halts.remove(scope);
    }

    pub fn is_halted(&self, scope: &str) -> bool {
        self.halts.contains_key(scope)
    }

    fn marks_eur(&self) -> BTreeMap<InstrumentId, f64> {
        self.marks.iter().filter_map(|(k, v)| Uuid::parse_str(k).ok().map(|u| (InstrumentId(u), *v))).collect()
    }

    fn paper_equity(&self) -> f64 {
        self.ledger.equity(&self.marks_eur())
    }

    fn update_peak(&mut self) {
        let eq = self.paper_equity();
        if eq > self.peak_equity {
            self.peak_equity = eq;
        }
    }

    pub fn positions_view(&self) -> Value {
        let positions: Vec<Value> = self
            .ledger
            .positions
            .iter()
            .filter(|(_, p)| p.quantity != 0.0 || p.realised_eur != 0.0)
            .map(|(id, p)| {
                let key = id.to_string();
                let mark = self.marks.get(&key).copied();
                let value = mark.map(|m| m * p.quantity);
                json!({
                    "instrument_id": key,
                    "quantity": p.quantity,
                    "avg_cost_eur": p.avg_cost_eur,
                    "mark_eur": mark,
                    "market_value_eur": value,
                    "unrealised_eur": mark.map(|m| (m - p.avg_cost_eur) * p.quantity),
                    "realised_eur": p.realised_eur,
                })
            })
            .collect();
        json!({
            "book": "paper",
            "base_currency": fa_core::BASE_CURRENCY,
            "cash_eur": self.ledger.cash_eur,
            "fees_eur": self.ledger.fees_eur,
            "equity_eur": self.paper_equity(),
            "positions": positions,
        })
    }

    pub fn orders_view(&self, fills_only: bool, limit: usize) -> Value {
        let orders: Vec<&OrderView> = self
            .order_order
            .iter()
            .filter_map(|id| self.orders.get(id))
            .filter(|o| !fills_only || o.filled_quantity > 0.0)
            .take(limit)
            .collect();
        json!({ "orders": orders })
    }

    pub fn signals_view(&self) -> Value {
        let signals: Vec<&SignalView> = self.signals.values().collect();
        json!({ "signals": signals })
    }

    /// Reported risk state (from the risk service, when running) plus the paper book's
    /// own figures, and the halts currently known.
    pub fn risk_view(&self) -> Value {
        let reported = self.risk.as_ref().map(|(r, at)| {
            json!({
                "as_of": ts(r.as_of.as_ref()),
                "received_at": at,
                "equity_eur": r.equity_eur,
                "deployable_eur": r.deployable_eur,
                "gross_exposure_eur": r.gross_exposure_eur,
                "drawdown": r.drawdown,
                "daily_pnl_eur": r.daily_pnl_eur,
            })
        });
        let equity = self.paper_equity();
        let marks = self.marks_eur();
        let gross: f64 = self
            .ledger
            .positions
            .iter()
            .map(|(i, p)| (p.quantity * marks.get(i).copied().unwrap_or(p.avg_cost_eur)).abs())
            .sum();
        let halts: Vec<&HaltView> = self.halts.values().collect();
        json!({
            "mode": "paper",
            "live_available": false,
            "reported": reported,
            "paper": {
                "equity_eur": equity,
                "deployable_eur": equity, // payout quota not tracked in the paper book yet
                "gross_exposure_eur": gross,
                "drawdown": if self.peak_equity > 0.0 { 1.0 - equity / self.peak_equity } else { 0.0 },
                "pnl_eur": equity - PAPER_CASH_EUR,
            },
            "halts": halts,
            "portfolio_halted": self.is_halted("portfolio"),
        })
    }
}

#[cfg(test)]
mod tests {
    use fa_proto::{
        now,
        v1::{AssetExposure, ExpertLatent, ExpertSpaceRef, Trade, TransitionProb},
    };

    use super::*;

    fn feed<M: Message>(live: &mut Live, subject: &str, m: &M) {
        live.ingest(subject, &m.encode_to_vec());
    }

    fn space() -> Option<ExpertSpaceRef> {
        Some(ExpertSpaceRef { snapshot_id: "snap1".into(), timescale: "1h".into(), expert_index: 0 })
    }

    #[test]
    fn latent_graph_view() {
        let mut l = Live::default();
        for r in ["r1", "r2"] {
            feed(
                &mut l,
                "latent.region.1h",
                &RegionEvent {
                    kind: region_event::Kind::Created as i32,
                    space: space(),
                    region_id: r.into(),
                    lineage_id: format!("lin-{r}"),
                    related_region_ids: vec![],
                },
            );
        }
        feed(
            &mut l,
            "latent.transition.1h",
            &TransitionUpdate {
                space: space(),
                from_region_id: "r1".into(),
                to: vec![
                    TransitionProb { region_id: "r1".into(), mean: 0.7, lower: 0.6, upper: 0.8 },
                    TransitionProb { region_id: "r3".into(), mean: 0.3, lower: 0.2, upper: 0.4 },
                ],
                sampler_id: "conjugate".into(),
                as_of: Some(now()),
            },
        );
        feed(
            &mut l,
            "latent.state.1h.snap1",
            &LatentState {
                snapshot_id: "snap1".into(),
                timescale: "1h".into(),
                instrument_id: "inst-a".into(),
                as_of: Some(now()),
                experts: vec![ExpertLatent {
                    expert_index: 0,
                    mean: vec![3.0, 4.0],
                    log_var: vec![0.0, 0.0],
                    region_id: "r1".into(),
                    region_lineage_id: "lin-r1".into(),
                }],
                router_weights: vec![1.0],
            },
        );

        let g = l.transitions_view("1h");
        let ids: Vec<&str> = g["nodes"].as_array().unwrap().iter().map(|n| n["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["r1", "r2", "r3"]);
        assert_eq!(g["nodes"][0]["occupancy"], 1);
        assert_eq!(g["nodes"][2]["status"], "unregistered");
        assert_eq!(g["edges"].as_array().unwrap().len(), 2);
        assert_eq!(g["edges"][0]["mean"], 0.7);

        let s = l.states_view("1h");
        assert_eq!(s["states"][0]["experts"][0]["norm"], 5.0);
        assert_eq!(s["states"][0]["experts"][0]["uncertainty"], 1.0);
        assert_eq!(l.regions_view("1h")["regions"][0]["lineage_id"], "lin-r1");
        assert_eq!(l.timescales_seen(), ["1h"]);
        assert!(l.states_view("1d")["states"].as_array().unwrap().is_empty());
    }

    #[test]
    fn decoder_outputs_filter_by_family_prefix() {
        let mut l = Live::default();
        for f in ["retrieval", "text.news", "text.summary"] {
            feed(&mut l, &format!("decoder.output.{f}"), &DecoderOutput { family: f.into(), ..Default::default() });
        }
        assert_eq!(l.decoder_outputs_view(Some("text"), 50)["outputs"].as_array().unwrap().len(), 2);
        assert_eq!(l.decoder_outputs_view(None, 50)["families"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn job_lifecycle_and_counts() {
        let mut l = Live::default();
        let job =
            Job { job_id: "j1".into(), kind: "evo.eval.decoder".into(), budget_gpu_ms: 1000, ..Default::default() };
        feed(&mut l, "jobs.evo.eval.decoder", &job);
        feed(&mut l, "jobs.progress.j1", &JobProgress { job_id: "j1".into(), fraction: 0.5, message: "rung 1".into() });
        let v = l.jobs_view(None, 10);
        assert_eq!(v["jobs"][0]["status"], "running");
        assert_eq!(v["counts"]["evo.eval.decoder"]["running"], 1);

        let result = JobResult {
            job_id: "j1".into(),
            status: job_result::Status::Ok as i32,
            result: br#"{"score": 0.8, "objectives": {"recon": 0.1, "cost": 3}}"#.to_vec(),
            used_gpu_ms: 900,
            worker_id: "w1".into(),
            ..Default::default()
        };
        feed(&mut l, "jobs.result.evo.eval.decoder", &result);
        let v = l.jobs_view(Some("evo.eval.decoder"), 10);
        assert_eq!(v["jobs"][0]["status"], "ok");
        assert_eq!(v["jobs"][0]["score"], 0.8);
        assert_eq!(v["jobs"][0]["objectives"]["cost"], 3.0);
        assert_eq!(v["objectives"], json!(["cost", "recon"]));
        assert_eq!(v["counts"]["evo.eval.decoder"]["ok"], 1);
    }

    #[test]
    fn paper_fill_updates_positions_and_risk() {
        let mut l = Live::default();
        let inst = Uuid::now_v7().to_string();
        let intent = OrderIntent {
            intent_id: "i1".into(),
            account_id: "paper".into(),
            instrument_id: inst.clone(),
            side: Side::Buy as i32,
            quantity: 10.0,
            exec_algo: "market".into(),
            mode: TradingMode::Paper as i32,
            ..Default::default()
        };
        feed(&mut l, "order.approved.paper", &intent);
        assert_eq!(l.orders_view(false, 10)["orders"][0]["status"], "approved");
        feed(
            &mut l,
            "order.event.paper",
            &OrderEvent {
                intent_id: "i1".into(),
                kind: order_event::Kind::Fill as i32,
                fill_price: 100.0,
                fill_quantity: 10.0,
                fee: 1.0,
                fee_currency: "EUR".into(),
                event_time: Some(now()),
                ..Default::default()
            },
        );
        let o = l.orders_view(true, 10);
        assert_eq!(o["orders"][0]["status"], "filled");
        assert_eq!(o["orders"][0]["avg_fill_price"], 100.0);

        // mark moves to 110 via a trade
        feed(
            &mut l,
            "md.trade.sim.x",
            &Envelope {
                instrument_id: inst.clone(),
                payload: Some(Payload::Trade(Trade { price: 110.0, size: 1.0, ..Default::default() })),
                ..Default::default()
            },
        );
        let p = l.positions_view();
        assert_eq!(p["positions"][0]["quantity"], 10.0);
        assert_eq!(p["positions"][0]["unrealised_eur"], 100.0);
        assert_eq!(p["cash_eur"], PAPER_CASH_EUR - 1001.0);
        assert_eq!(p["equity_eur"], PAPER_CASH_EUR - 1001.0 + 1100.0);

        // replaying the intent must not reset the status
        feed(&mut l, "order.intent.paper", &intent);
        assert_eq!(l.orders_view(false, 10)["orders"][0]["status"], "filled");
    }

    #[test]
    fn halts_and_resume() {
        let mut l = Live::default();
        feed(
            &mut l,
            "risk.halt.portfolio",
            &Halt { scope: "portfolio".into(), reason: "manual".into(), since: Some(now()), manual: true },
        );
        assert!(l.is_halted("portfolio"));
        assert_eq!(l.risk_view()["portfolio_halted"], true);
        l.ingest("risk.resume.portfolio", &[]);
        assert!(!l.is_halted("portfolio"));

        feed(
            &mut l,
            "risk.state",
            &RiskState {
                equity_eur: 5.0,
                active_halts: vec![Halt { scope: "strategy:s1".into(), ..Default::default() }],
                ..Default::default()
            },
        );
        let r = l.risk_view();
        assert_eq!(r["reported"]["equity_eur"], 5.0);
        assert_eq!(r["halts"][0]["scope"], "strategy:s1");
        assert_eq!(r["live_available"], false);
    }

    #[test]
    fn signals_keep_latest_per_strategy() {
        let mut l = Live::default();
        for mean in [0.1, 0.4] {
            feed(
                &mut l,
                "signal.exposure.s1",
                &ExposureDistribution {
                    strategy_id: "s1".into(),
                    exposures: vec![AssetExposure { instrument_id: "a".into(), mean, stddev: 0.1 }],
                    ..Default::default()
                },
            );
        }
        let v = l.signals_view();
        assert_eq!(v["signals"].as_array().unwrap().len(), 1);
        assert_eq!(v["signals"][0]["exposures"][0]["mean"], 0.4);
    }
}
