//! Trace ids carried in NATS headers (spec 60 §6.5).
//!
//! The header is W3C `traceparent` (`00-<32 hex trace id>-<16 hex span id>-01`), so an
//! OpenTelemetry exporter can adopt the ids later without a format change; there is no
//! exporter today — the trace id is a log field (`trace_id`) and an `audit.log` payload key.
//!
//! * Publishing: [`crate::Bus::publish`] attaches a `traceparent` continuing the task's current
//!   trace ([`scope`]) or starting a new one. High-rate subjects (`md.*`, heartbeats) are not
//!   traced ([`traced`]): a market-data tick starts no chain worth correlating and would grow
//!   every stored message by ~70 bytes.
//! * Consuming: [`crate::Bus::subscribe`] logs each traced message at `debug` with its
//!   `trace_id`; [`crate::Bus::subscribe_traced`] hands the context to the consumer, which runs
//!   its handler under [`scope`] so anything it publishes continues the same trace.
//! * Go (`go/pkg/bus/trace.go`) uses the same header.
use std::future::Future;

use async_nats::HeaderMap;

/// Header name (W3C trace context).
pub const TRACEPARENT: &str = "traceparent";

/// One trace context: the trace id shared by every hop, and this hop's span id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceContext {
    pub trace_id: u128,
    pub span_id: u64,
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

fn random_u128() -> u128 {
    // ULIDs are 48 bits of time + 80 random bits; xor two to spread the time bits.
    u128::from(fa_core::EventId::new().0) ^ u128::from(fa_core::EventId::new().0).rotate_left(64)
}

impl TraceContext {
    /// A new trace (new trace id and span id).
    pub fn new() -> Self {
        let r = random_u128();
        Self { trace_id: r.max(1), span_id: (random_u128() as u64).max(1) }
    }

    /// The next hop of the same trace (new span id).
    pub fn child(&self) -> Self {
        Self { trace_id: self.trace_id, span_id: (random_u128() as u64).max(1) }
    }

    /// `00-<trace>-<span>-01`.
    pub fn traceparent(&self) -> String {
        format!("00-{:032x}-{:016x}-01", self.trace_id, self.span_id)
    }

    /// The trace id as 32 lowercase hex digits (the `trace_id` log / audit field).
    pub fn trace_id_hex(&self) -> String {
        format!("{:032x}", self.trace_id)
    }

    /// Parses a `traceparent` value; `None` for anything malformed or all-zero ids.
    pub fn parse(s: &str) -> Option<Self> {
        let mut parts = s.trim().split('-');
        let (version, trace, span, _flags) = (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
        if version.len() != 2 || trace.len() != 32 || span.len() != 16 {
            return None;
        }
        let trace_id = u128::from_str_radix(trace, 16).ok()?;
        let span_id = u64::from_str_radix(span, 16).ok()?;
        (trace_id != 0 && span_id != 0).then_some(Self { trace_id, span_id })
    }
}

tokio::task_local! {
    static CURRENT: TraceContext;
}

/// The trace of the running task, if it runs under [`scope`].
pub fn current() -> Option<TraceContext> {
    CURRENT.try_with(|c| *c).ok()
}

/// Runs `fut` with `ctx` as the current trace.
pub async fn scope<F: Future>(ctx: TraceContext, fut: F) -> F::Output {
    CURRENT.scope(ctx, fut).await
}

/// The context for an outgoing message: a child of the current trace, or a new trace.
pub fn outgoing() -> TraceContext {
    current().map_or_else(TraceContext::new, |c| c.child())
}

/// Whether messages on `subject` carry a trace header (not `md.*`, not heartbeats).
pub fn traced(subject: &str) -> bool {
    !(subject.starts_with("md.") || subject.starts_with(crate::subjects::HEARTBEAT_PREFIX))
}

/// Headers carrying `ctx`.
pub fn headers(ctx: &TraceContext) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert(TRACEPARENT, ctx.traceparent().as_str());
    h
}

/// The trace context of a received message, if it has one.
pub fn from_headers(h: Option<&HeaderMap>) -> Option<TraceContext> {
    h.and_then(|h| h.get(TRACEPARENT)).and_then(|v| TraceContext::parse(v.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traceparent_round_trips_and_rejects_garbage() {
        let t = TraceContext::new();
        let s = t.traceparent();
        assert_eq!(s.len(), 55, "{s}");
        assert_eq!(TraceContext::parse(&s), Some(t));
        let c = t.child();
        assert_eq!(c.trace_id, t.trace_id);
        assert_ne!(c.span_id, t.span_id);
        assert_ne!(TraceContext::new().trace_id, t.trace_id);
        for bad in ["", "00-abc-def-01", "00-00000000000000000000000000000000-0000000000000001-01", "xx"] {
            assert!(TraceContext::parse(bad).is_none(), "{bad}");
        }
        // W3C example value
        let w3c = TraceContext::parse("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01").unwrap();
        assert_eq!(w3c.trace_id_hex(), "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(from_headers(Some(&headers(&w3c))), Some(w3c));
        assert_eq!(from_headers(None), None);
    }

    #[test]
    fn hot_subjects_are_not_traced() {
        assert!(!traced("md.trade.sim.x"));
        assert!(!traced(&crate::subjects::heartbeat("risk-engine")));
        assert!(traced("risk.halt.portfolio"));
        assert!(traced("ctl.evo.start"));
    }

    #[tokio::test]
    async fn scope_sets_the_current_trace() {
        assert!(current().is_none());
        let t = TraceContext::new();
        let inner = scope(t, async { (current(), outgoing()) }).await;
        assert_eq!(inner.0, Some(t));
        assert_eq!(inner.1.trace_id, t.trace_id, "outgoing continues the trace");
        assert_ne!(outgoing().trace_id, t.trace_id, "outside the scope a new trace starts");
    }
}
