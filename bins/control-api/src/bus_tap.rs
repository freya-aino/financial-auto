//! WS `/ws/bus`: live tap of NATS subjects (spec 50 §4.3).
use std::time::Duration;

use axum::{
    extract::{
        Query, State,
        ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
    },
    response::Response,
};
use chrono::Utc;
use fa_proto::{
    Message,
    v1::{Envelope, Heartbeat, envelope::Payload},
};
use futures::{SinkExt, StreamExt, stream::select_all};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::state::AppState;

/// Max messages forwarded per second per socket; the rest are counted as dropped.
const MAX_PER_SEC: u32 = 200;

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

async fn tap(socket: WebSocket, state: AppState, patterns: Vec<String>) {
    let (mut tx, mut rx) = socket.split();
    let Some(client) = state.nats.clone() else {
        let _ = tx.send(WsMessage::text(json!({"op": "error", "error": "nats unavailable"}).to_string())).await;
        return;
    };

    let mut subs = Vec::new();
    for p in &patterns {
        match client.subscribe(p.clone()).await {
            Ok(s) => subs.push(s),
            Err(e) => {
                let _ = tx
                    .send(WsMessage::text(json!({"op": "error", "error": e.to_string(), "subject": p}).to_string()))
                    .await;
                return;
            }
        }
    }
    let mut merged = select_all(subs);
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    let (mut seq, mut sent_this_sec, mut dropped) = (0u64, 0u32, 0u64);

    loop {
        tokio::select! {
            msg = merged.next() => {
                let Some(msg) = msg else { break };
                if sent_this_sec >= MAX_PER_SEC {
                    dropped += 1;
                    continue;
                }
                sent_this_sec += 1;
                seq += 1;
                let frame = json!({
                    "stream": "bus",
                    "seq": seq,
                    "ts": Utc::now().to_rfc3339(),
                    "data": {
                        "subject": msg.subject.as_str(),
                        "size": msg.payload.len(),
                        "decoded": decode(msg.subject.as_str(), &msg.payload),
                    }
                });
                if tx.send(WsMessage::text(frame.to_string())).await.is_err() {
                    break;
                }
            }
            _ = tick.tick() => {
                if dropped > 0 {
                    let lag = json!({"op": "lagged", "dropped": dropped});
                    if tx.send(WsMessage::text(lag.to_string())).await.is_err() {
                        break;
                    }
                    dropped = 0;
                }
                sent_this_sec = 0;
            }
            incoming = rx.next() => {
                match incoming {
                    None | Some(Err(_)) | Some(Ok(WsMessage::Close(_))) => break,
                    Some(Ok(_)) => {} // subscribe/unsubscribe ops: not yet supported
                }
            }
        }
    }
}

const ENVELOPE_PREFIXES: [&str; 6] = ["md.", "news.", "media.", "macro.", "onchain.", "filing."];

/// Decodes known payloads to JSON; unknown subjects yield `null`.
pub fn decode(subject: &str, payload: &[u8]) -> Value {
    if subject.starts_with("svc.heartbeat.") {
        return Heartbeat::decode(payload).map(heartbeat_json).unwrap_or(Value::Null);
    }
    if ENVELOPE_PREFIXES.iter().any(|p| subject.starts_with(p)) {
        return Envelope::decode(payload).map(envelope_json).unwrap_or(Value::Null);
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
    fn default_pattern_is_everything() {
        let q = TapQuery { subject: None, subjects: None };
        assert_eq!(q.patterns(), vec![">"]);
        let q = TapQuery { subject: Some("md.>".into()), subjects: Some("risk.>, svc.>".into()) };
        assert_eq!(q.patterns(), vec!["risk.>", "svc.>", "md.>"]);
    }
}
