//! Deterministic decision inputs (spec 41 §9, D-105). Every bus message the decision loop acts
//! on is buffered with its receive time instead of being applied on arrival; a tick at grid
//! time `t` first applies, in arrival order, exactly the inputs received at or before `t`, then
//! decides. The live service and the replay share this rule — and the decoding
//! ([`Input::parse`]) and application ([`Input::apply`]) — so a replay that feeds the retained
//! streams on the same grid (knowledge time = stream publish time ≤ t) drives the engine
//! through the same states, however long the live loop's own publishes or Postgres writes took.
//!
//! What stays outside the buffer, deliberately: operator control calls (`ctl.risk.*`), FX and
//! reference-data reloads, calibration, carry and the recovery paths (orphans, reconciliation).
//! Those are either operator actions or Postgres-derived state a replay cannot reproduce; spec
//! 41 §9 lists them as the known nondeterministic inputs.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use fa_core::InstrumentId;
use fa_proto::{
    Message,
    v1::{Envelope, ExposureDistribution, Halt, KernelUpdate, OrderEvent, OrderIntent, envelope::Payload},
};
use uuid::Uuid;

use crate::engine::{Engine, Out};

/// One buffered decision input, decoded at arrival (a malformed message is dropped there, as
/// before).
#[derive(Debug, Clone)]
pub enum Input {
    /// `signal.exposure.>`
    Signal(ExposureDistribution),
    /// `md.trade.<venue>.<token>`: the mark/volume tape. `at_ms` is the event time when the
    /// envelope carries one (the receive time is the fallback at apply).
    Trade { inst: InstrumentId, venue: Option<String>, at_ms: Option<i64>, price: f64, size: f64 },
    /// `order.intent.<account>` / `order.approved.<account>` (gating and memory of stages).
    OrderStage { intent: OrderIntent, approved: bool },
    /// `order.event.<account>`: fills and rejections to book.
    OrderEvent(OrderEvent),
    /// `risk.halt.<scope>` (the engine hears its own halts too, like the live subscription).
    Halt(Halt),
    /// `risk.resume.<scope>`: the scope is the subject's tail.
    Resume { scope: String },
    /// `ingest.health.<source>` (not retained in any stream: live-only; the ingestor's
    /// `risk.halt.source:*` mirrors the downs into the retained ORDERS stream).
    SourceHealth { source: String, status: String, reason: String },
    /// `latent.mcmc.>` kernel updates: snapshot→timescale memory and the model-drift alarm.
    Kernel { snapshot: String, ts: fa_core::Timescale, expert: u32, drift_kl: f64 },
}

impl Input {
    /// Decodes a subject/payload pair into the input the live service would buffer for it;
    /// `None` for subjects the decision loop does not consume or payloads that do not decode.
    /// The replay feeds retained stream messages through this, so both paths share one decoder.
    pub fn parse(subject: &str, payload: &[u8]) -> Option<Input> {
        if subject.starts_with("signal.exposure.") {
            return ExposureDistribution::decode(payload).ok().map(Input::Signal);
        }
        if let Some(rest) = subject.strip_prefix("md.trade.") {
            let env = Envelope::decode(payload).ok()?;
            let Some(Payload::Trade(t)) = &env.payload else { return None };
            let uuid = Uuid::parse_str(&env.instrument_id).ok()?;
            return Some(Input::Trade {
                inst: InstrumentId(uuid),
                venue: rest.split('.').next().map(str::to_owned),
                at_ms: env.event_time.map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000),
                price: t.price,
                size: t.size,
            });
        }
        if subject.starts_with("order.intent.") || subject.starts_with("order.approved.") {
            let approved = subject.starts_with("order.approved.");
            return OrderIntent::decode(payload).ok().map(|intent| Input::OrderStage { intent, approved });
        }
        if subject.starts_with("order.event.") {
            return OrderEvent::decode(payload).ok().map(Input::OrderEvent);
        }
        if subject.starts_with("risk.halt.") {
            return Halt::decode(payload).ok().map(Input::Halt);
        }
        if let Some(scope) = subject.strip_prefix("risk.resume.") {
            return Some(Input::Resume { scope: scope.to_owned() });
        }
        if let Some(source) = subject.strip_prefix("ingest.health.") {
            let (status, reason) = fa_bus::ingest::health_status(payload)?;
            return Some(Input::SourceHealth { source: source.to_owned(), status, reason });
        }
        if subject.starts_with("latent.mcmc.") {
            let k = KernelUpdate::decode(payload).ok()?;
            let space = k.space.as_ref()?;
            let ts = space.timescale.parse().ok()?;
            return Some(Input::Kernel {
                snapshot: space.snapshot_id.clone(),
                ts,
                expert: space.expert_index,
                drift_kl: k.drift_kl,
            });
        }
        None
    }

    /// Applies the input to the engine at its receive time. Booked/unknown fills stay parked in
    /// the engine (`take_booked_fills` / `take_unknown_fills`) for the caller, as before.
    pub fn apply(&self, e: &mut Engine, recv_ms: i64) -> Vec<Out> {
        match self {
            Input::Signal(m) => {
                e.on_signal(m, recv_ms);
                Vec::new()
            }
            Input::Trade { inst, venue, at_ms, price, size } => {
                if let Some(v) = venue {
                    e.note_md_venue(*inst, v);
                }
                e.on_trade(*inst, at_ms.unwrap_or(recv_ms), *price, *size)
            }
            Input::OrderStage { intent, approved } => e.on_intent_seen(intent, *approved, recv_ms),
            Input::OrderEvent(ev) => e.on_order_event(ev, recv_ms),
            Input::Halt(h) => {
                e.on_halt(h, recv_ms);
                Vec::new()
            }
            Input::Resume { scope } => {
                e.on_resume(scope);
                Vec::new()
            }
            Input::SourceHealth { source, status, reason } => {
                if e.on_source_health(source, status, reason, recv_ms) {
                    // the change is audited through the normal out path (`Out::Audit`)
                    vec![Out::Audit {
                        action: (if status == "down" { "halt" } else { "resume" }).into(),
                        subject: format!("source:{source}"),
                        payload: serde_json::json!({"reason": reason, "manual": false, "raised_by": "ingest.health"}),
                    }]
                } else {
                    Vec::new()
                }
            }
            Input::Kernel { snapshot, ts, expert, drift_kl } => {
                e.on_kernel_update(snapshot, *ts, *expert, *drift_kl, recv_ms)
            }
        }
    }
}

/// The arrival-ordered buffer between the subscription tasks and the decision loop. Pushes are
/// lock-cheap (no engine lock, no I/O); the decision loop drains per grid slot.
#[derive(Clone, Default)]
pub struct InputBuffer(Arc<Mutex<VecDeque<(i64, Input)>>>);

impl InputBuffer {
    pub fn push(&self, recv_ms: i64, input: Input) {
        self.0.lock().expect("input buffer").push_back((recv_ms, input));
    }

    /// Removes and returns, in arrival order, every input received at or before `t`. Arrival
    /// order is the queue order; receive times are monotone per pusher, and the queue keeps the
    /// interleaving the subscriptions produced (what the live engine previously acted on).
    pub fn drain_upto(&self, t: i64) -> Vec<(i64, Input)> {
        let mut q = self.0.lock().expect("input buffer");
        let n = q.iter().take_while(|(recv, _)| *recv <= t).count();
        q.drain(..n).collect()
    }

    pub fn len(&self) -> usize {
        self.0.lock().expect("input buffer").len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_respects_the_grid_time() {
        let b = InputBuffer::default();
        b.push(1_000, Input::Resume { scope: "a".into() });
        b.push(1_500, Input::Resume { scope: "b".into() });
        b.push(2_100, Input::Resume { scope: "c".into() });
        let first: Vec<String> = b
            .drain_upto(2_000)
            .into_iter()
            .map(|(_, i)| match i {
                Input::Resume { scope } => scope,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(first, ["a", "b"], "inputs received after t are invisible to tick t");
        assert_eq!(b.len(), 1);
        assert_eq!(b.drain_upto(2_000).len(), 0, "nothing at or before t is left");
        assert_eq!(b.drain_upto(2_100).len(), 1);
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn parse_covers_every_decision_subject() {
        use fa_proto::v1 as pb;
        let sig = pb::ExposureDistribution { strategy_id: "s".into(), ..Default::default() };
        assert!(matches!(
            Input::parse("signal.exposure.1m.s", &sig.encode_to_vec()),
            Some(Input::Signal(m)) if m.strategy_id == "s"
        ));
        let env = pb::Envelope {
            instrument_id: Uuid::nil().to_string(),
            payload: Some(Payload::Trade(pb::Trade { price: 1.5, size: 2.0, ..Default::default() })),
            event_time: Some(fa_proto::Timestamp { seconds: 1, nanos: 0 }),
            ..Default::default()
        };
        let Some(Input::Trade { venue, at_ms, price, .. }) = Input::parse("md.trade.kraken.0190", &env.encode_to_vec())
        else {
            panic!("trade")
        };
        assert_eq!((venue.as_deref(), at_ms, price), (Some("kraken"), Some(1_000), 1.5));
        let intent = pb::OrderIntent { intent_id: "i1".into(), account_id: "paper".into(), ..Default::default() };
        assert!(matches!(
            Input::parse("order.approved.paper", &intent.encode_to_vec()),
            Some(Input::OrderStage { approved: true, .. })
        ));
        assert!(matches!(
            Input::parse("order.intent.paper", &intent.encode_to_vec()),
            Some(Input::OrderStage { approved: false, .. })
        ));
        let ev = pb::OrderEvent { intent_id: "i1".into(), ..Default::default() };
        assert!(matches!(Input::parse("order.event.paper", &ev.encode_to_vec()), Some(Input::OrderEvent(_))));
        let halt = pb::Halt { scope: "portfolio".into(), ..Default::default() };
        assert!(matches!(Input::parse("risk.halt.portfolio", &halt.encode_to_vec()), Some(Input::Halt(_))));
        assert!(matches!(
            Input::parse("risk.resume.broker", b""),
            Some(Input::Resume { scope }) if scope == "broker"
        ));
        assert!(Input::parse("ops.alert", b"").is_none(), "non-decision subjects are not buffered");
        assert!(Input::parse("md.trade.kraken.x", b"garbage").is_none(), "undecodable payloads are dropped");
    }
}
