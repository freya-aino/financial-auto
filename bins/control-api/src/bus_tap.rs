//! WS `/ws/bus`: live tap of NATS subjects (spec 50 §4.3), decoded to JSON, on the shared
//! socket loop of `ws_routes` (bounded queue: `runtime.bus_tap.max_msgs_per_sec` frames per
//! second per socket spread over `BUS_TICKS_PER_SEC` flushes; overflow → `{op: "lagged"}`).
use axum::{
    extract::{
        Query, State,
        ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
    },
    response::Response,
};
use fa_proto::{
    Message,
    v1::{Envelope, Halt, Heartbeat, Resume, RiskEvent, envelope::Payload},
};
use futures::{StreamExt, stream::select_all};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    state::AppState,
    ws_routes::{self, Coalescer, SourceItem},
};

#[derive(Debug, Deserialize)]
pub struct TapQuery {
    /// Single pattern (`?subject=md.>`).
    subject: Option<String>,
    /// Comma-separated patterns (`?subjects=md.>,risk.>`).
    subjects: Option<String>,
}

impl TapQuery {
    pub fn patterns(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .subjects
            .iter()
            .flat_map(|s| s.split(','))
            .chain(self.subject.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        if out.is_empty() {
            out.push(">".to_owned());
        }
        out
    }
}

pub async fn ws_bus(ws: WebSocketUpgrade, State(state): State<AppState>, Query(q): Query<TapQuery>) -> Response {
    let patterns = q.patterns();
    ws.on_upgrade(move |socket| tap(socket, state, patterns))
}

/// Frames per flush for a per-second budget.
pub fn budget_per_flush(max_per_sec: u32) -> usize {
    (max_per_sec as usize).div_ceil(ws_routes::BUS_TICKS_PER_SEC as usize).max(1)
}

async fn tap(mut socket: WebSocket, state: AppState, patterns: Vec<String>) {
    let Some(client) = state.nats.clone() else {
        let _ = socket.send(WsMessage::text(json!({"op": "error", "error": "nats unavailable"}).to_string())).await;
        return;
    };
    let mut subs = Vec::new();
    for p in &patterns {
        match client.subscribe(p.clone()).await {
            Ok(s) => subs.push(s),
            Err(e) => {
                let _ = socket
                    .send(WsMessage::text(json!({"op": "error", "error": e.to_string(), "subject": p}).to_string()))
                    .await;
                return;
            }
        }
    }
    let max_per_sec = state.live().runtime().bus_tap.max_msgs_per_sec;
    let (tx, rx) = ws_routes::source_channel();
    tokio::spawn(async move {
        let mut merged = select_all(subs);
        while let Some(msg) = merged.next().await {
            let subject = msg.subject.as_str();
            let data = json!({
                "subject": subject,
                "size": msg.payload.len(),
                "decoded": decode(subject, &msg.payload),
            });
            if tx.send(SourceItem::Frame(subject.to_owned(), data)).await.is_err() {
                break;
            }
        }
    });
    let flush = ws_routes::flush_interval(ws_routes::BUS_TICKS_PER_SEC);
    ws_routes::serve(socket, "bus", Vec::new(), rx, Coalescer::queue(budget_per_flush(max_per_sec)), flush).await;
}

const ENVELOPE_PREFIXES: [&str; 6] = ["md.", "news.", "media.", "macro.", "onchain.", "filing."];

/// Decodes known payloads to JSON; unknown subjects yield `null`.
pub fn decode(subject: &str, payload: &[u8]) -> Value {
    if let Some(v) = crate::console_routes::decode_extra(subject, payload) {
        return v; // ops.alert, latent.membership|lifecycle|mcmc|hier|crossprior (D-066, D-069)
    }
    if subject.starts_with("svc.heartbeat.") {
        return Heartbeat::decode(payload).map(heartbeat_json).unwrap_or(Value::Null);
    }
    if ENVELOPE_PREFIXES.iter().any(|p| subject.starts_with(p)) {
        return Envelope::decode(payload).map(envelope_json).unwrap_or(Value::Null);
    }
    if let Some(rest) = subject.strip_prefix("ingest.") {
        let kind = rest.split('.').next().unwrap_or("");
        return fa_bus::ingest::frame_json(kind, payload)
            .map(|v| json!({"type": format!("ingest.{kind}"), "frame": v}))
            .unwrap_or(Value::Null);
    }
    if subject.starts_with("ref.changed.") || subject.starts_with("catalog.changed.") {
        return fa_bus::ingest::ref_changed(payload)
            .map(|r| {
                let mut v = fa_bus::ingest::ref_changed_json(&r);
                v["type"] = json!("ref_changed");
                v
            })
            .unwrap_or(Value::Null);
    }
    if subject.starts_with("risk.halt.") {
        return Halt::decode(payload)
            .map(|h| json!({"type": "halt", "scope": h.scope, "reason": h.reason, "manual": h.manual, "since": ts(&h.since)}))
            .unwrap_or(Value::Null);
    }
    if subject.starts_with("risk.resume.") {
        return Resume::decode(payload)
            .map(|r| json!({"type": "resume", "scope": r.scope, "reason": r.reason, "manual": r.manual, "actor": r.actor, "at": ts(&r.at)}))
            .unwrap_or(Value::Null);
    }
    if subject.starts_with("risk.event.") {
        return RiskEvent::decode(payload)
            .map(|e| {
                json!({
                    "type": "risk_event",
                    "event_id": e.event_id,
                    "kind": e.kind().as_str_name(),
                    "at": ts(&e.at),
                    "account": e.account,
                    "intent_id": e.intent_id,
                    "strategy_id": e.strategy_id,
                    "instrument_id": e.instrument_id,
                    "scope": e.scope,
                    "reason": e.reason,
                    "metrics": e.metrics,
                })
            })
            .unwrap_or(Value::Null);
    }
    Value::Null
}

fn ts(t: &Option<fa_proto::Timestamp>) -> Value {
    t.as_ref().map_or(Value::Null, |t| {
        chrono::DateTime::from_timestamp(t.seconds, t.nanos.max(0) as u32)
            .map_or(Value::Null, |d| Value::String(d.to_rfc3339()))
    })
}

fn heartbeat_json(hb: Heartbeat) -> Value {
    json!({
        "type": "heartbeat",
        "service": hb.service,
        "instance_id": hb.instance_id,
        "version": hb.version,
        "at": ts(&hb.at),
        "counters": hb.counters,
    })
}

fn envelope_json(e: Envelope) -> Value {
    let payload = match e.payload {
        Some(Payload::Trade(t)) => json!({"kind": "trade", "price": t.price, "size": t.size}),
        Some(Payload::Quote(q)) => json!({"kind": "quote", "bid": q.bid_price, "ask": q.ask_price}),
        Some(Payload::Bar(b)) => {
            json!({"kind": "bar", "timeframe": b.timeframe, "close": b.close, "volume": b.volume})
        }
        Some(Payload::Book(b)) => {
            json!({"kind": "book", "bids": b.bids.len(), "asks": b.asks.len()})
        }
        Some(Payload::News(n)) => json!({"kind": "news", "title": n.title, "outlet": n.outlet}),
        Some(Payload::Media(m)) => json!({"kind": "media", "media_id": m.media_id}),
        Some(Payload::Macro(m)) => {
            json!({"kind": "macro", "series_id": m.series_id, "value": m.value})
        }
        Some(Payload::Onchain(o)) => json!({"kind": "onchain", "chain": o.chain, "event": o.kind}),
        Some(Payload::Filing(f)) => json!({"kind": "filing", "form_type": f.form_type}),
        None => Value::Null,
    };
    json!({
        "type": "envelope",
        "event_id": e.event_id,
        "source_id": e.source_id,
        "instrument_id": e.instrument_id,
        "event_time": ts(&e.event_time),
        "payload": payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_proto::v1::Trade;

    #[test]
    fn decodes_trade_envelope() {
        let env = Envelope {
            event_id: "01J".into(),
            source_id: "heartbeat".into(),
            payload: Some(Payload::Trade(Trade { price: 101.5, size: 2.0, ..Default::default() })),
            ..Default::default()
        };
        let v = decode("md.trade.sim.abc", &env.encode_to_vec());
        assert_eq!(v["payload"]["kind"], "trade");
        assert_eq!(v["payload"]["price"], 101.5);
    }

    #[test]
    fn unknown_subject_is_null() {
        assert_eq!(decode("jobs.evo", b"\x01\x02"), Value::Null);
    }

    #[test]
    fn budget_spreads_the_per_second_limit_over_flushes() {
        assert_eq!(budget_per_flush(200), 10);
        assert_eq!(budget_per_flush(1), 1);
        assert_eq!(budget_per_flush(25), 2);
    }

    #[test]
    fn default_pattern_is_everything() {
        let q = TapQuery { subject: None, subjects: None };
        assert_eq!(q.patterns(), vec![">"]);
        let q = TapQuery { subject: Some("md.>".into()), subjects: Some("risk.>, svc.>".into()) };
        assert_eq!(q.patterns(), vec!["risk.>", "svc.>", "md.>"]);
    }
}
