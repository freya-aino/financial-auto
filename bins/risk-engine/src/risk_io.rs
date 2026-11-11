//! I/O of the spec 40 §9 additions: paper carry booking (`exec.carry.<account>`), the
//! risk-chain history writer (allocation snapshots, netted targets, promotion reviews), the
//! exchange calendars behind the daily-loss day, and the side-effect-free pre-trade check
//! (`ctl.risk.pretrade`).
use std::{sync::Arc, time::Duration};

use fa_bus::Bus;
use fa_proto::{
    Message,
    v1::{CtlReply, CtlRequest},
};
use fa_risk::strategy::Checklist;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::{
    Shared, Store,
    engine::{CarryCharge, PretradeRequest, history},
    persist::Persist,
    settings::Settings,
};

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// Books one carry charge: Postgres first (idempotent on (book, venue, kind, ref)), then the
/// paper book with the same exact amount; without Postgres in memory only. A failed write is
/// not applied (the gateway re-announces the charge for six periods; reconciliation would
/// otherwise see a cash row only memory has).
pub async fn book_carry(engine: &Shared, store: &Store, c: &CarryCharge) -> bool {
    let book = store.as_ref().map_or_else(|| "paper".to_owned(), |p| p.book().to_owned());
    let Some(b) = engine.lock().await.carry_booking(c, &book) else { return false };
    if let Some(p) = store
        && let Err(e) = p.pg().record_cash_movement(&b.movement).await
    {
        tracing::warn!(reference = %c.reference, error = %e, "carry charge not persisted (re-announced by the gateway)");
        *engine.lock().await.counters.entry("carry.persist_errors").or_insert(0.0) += 1.0;
        return false;
    }
    engine.lock().await.apply_carry(&b)
}

/// `exec.carry.<account>` → [`book_carry`].
pub async fn spawn_carry(bus: &Bus, engine: Shared, store: Store, account: &str) -> anyhow::Result<()> {
    let mut sub = bus.subscribe_raw(format!("exec.carry.{account}")).await?;
    tokio::spawn(async move {
        while let Some(m) = sub.next().await {
            match serde_json::from_slice::<CarryCharge>(&m.payload) {
                Ok(c) => {
                    if book_carry(&engine, &store, &c).await {
                        tracing::debug!(reference = %c.reference, kind = %c.kind, amount = c.amount, "carry booked");
                    }
                }
                Err(e) => tracing::debug!(error = %e, "undecodable carry charge"),
            }
        }
    });
    Ok(())
}

/// Writes the kept risk-chain history every 5 s and prunes it hourly.
pub fn spawn_history(engine: Shared, p: Arc<Persist>, settings: &Settings) {
    let cfg = settings.history.clone();
    if !cfg.enabled {
        return;
    }
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(5));
        let mut last_prune = 0i64;
        loop {
            tick.tick().await;
            let recs = engine.lock().await.take_history();
            if !recs.is_empty()
                && let Err(e) = history::write(p.pg().pool(), p.book(), &recs).await
            {
                tracing::warn!(records = recs.len(), error = %e, "risk-chain history not written");
                *engine.lock().await.counters.entry("history.write_errors").or_insert(0.0) += 1.0;
            }
            let now = now_ms();
            if now - last_prune > 3_600_000 {
                last_prune = now;
                match history::prune(p.pg().pool(), p.book(), cfg.retention_days).await {
                    Ok(n) if n > 0 => tracing::info!(rows = n, "risk-chain history pruned"),
                    Ok(_) => {}
                    Err(e) => tracing::warn!(error = %e, "risk-chain history not pruned"),
                }
            }
        }
    });
}

/// Appends a promotion review (best-effort, logged).
#[allow(clippy::too_many_arguments)]
pub async fn review(
    p: &Persist,
    strategy_id: &str,
    trigger: &str,
    actor: &str,
    before: Option<&str>,
    after: Option<&str>,
    checklist: &Checklist,
    evidence: Value,
) {
    let r = history::Review {
        strategy_id,
        at_ms: now_ms(),
        trigger,
        actor,
        status_before: before,
        status_after: after,
        checklist,
        evidence,
    };
    if let Err(e) = history::write_review(p.pg().pool(), p.book(), &r).await {
        tracing::warn!(strategy = strategy_id, error = %e, "promotion review not written");
    }
}

/// Loads the exchange calendar per instrument into the engine (start and every reference-data
/// reload).
pub async fn load_calendars(engine: &Shared, p: &Persist) {
    match crate::engine::load_calendars(p.pg().pool()).await {
        Ok(m) => engine.lock().await.set_calendars(m),
        Err(e) => tracing::warn!(error = %e, "exchange calendars not loaded: every instrument on the UTC day"),
    }
}

/// The request of `ctl.risk.pretrade`: a JSON object, or a `CtlRequest` whose `args` carry the
/// same fields as strings.
fn pretrade_request(payload: &[u8]) -> Result<PretradeRequest, String> {
    if payload.first() == Some(&b'{') {
        return serde_json::from_slice(payload).map_err(|e| format!("invalid: {e}"));
    }
    let req = CtlRequest::decode(payload).map_err(|e| format!("invalid: {e}"))?;
    let a = |k: &str| req.args.get(k).map(String::as_str).unwrap_or("");
    Ok(PretradeRequest {
        instrument_id: a("instrument_id").to_owned(),
        side: a("side").to_owned(),
        quantity: a("quantity").parse().map_err(|_| "invalid: quantity is not a number".to_owned())?,
        limit_price: match a("limit_price") {
            "" => None,
            x => Some(x.parse().map_err(|_| "invalid: limit_price is not a number".to_owned())?),
        },
        mode: Some(a("mode").to_owned()).filter(|m| !m.is_empty()),
    })
}

/// `ctl.risk.pretrade` (request/reply): runs the pre-trade checks against the in-memory book
/// without side effects. A JSON request gets the JSON reply `{ok, reasons}`; a `CtlRequest`
/// gets a `CtlReply` whose `ok` is the verdict and whose `message` is that JSON.
pub async fn serve_pretrade(bus: &Bus, engine: Shared) -> anyhow::Result<()> {
    let mut sub = bus.subscribe_raw(fa_bus::subjects::ctl("risk", "pretrade")).await?;
    let bus = bus.clone();
    tokio::spawn(async move {
        while let Some(m) = sub.next().await {
            let Some(reply_to) = m.reply.clone() else { continue };
            let json_req = m.payload.first() == Some(&b'{');
            let verdict = match pretrade_request(&m.payload) {
                Ok(r) => engine.lock().await.pretrade_check(&r, now_ms()),
                Err(e) => crate::engine::PretradeReply { ok: false, reasons: vec![e] },
            };
            let body = json!(verdict).to_string();
            let bytes: Vec<u8> = if json_req {
                body.into_bytes()
            } else {
                CtlReply { ok: verdict.ok, message: body }.encode_to_vec()
            };
            if let Err(e) = bus.client().publish(reply_to, bytes.into()).await {
                tracing::warn!(error = %e, "pretrade reply failed");
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretrade_requests_decode_from_json_and_ctl_args() {
        let r = pretrade_request(br#"{"instrument_id":"x","side":"buy","quantity":2.5,"mode":"paper"}"#).unwrap();
        assert_eq!((r.side.as_str(), r.quantity, r.limit_price), ("buy", 2.5, None));
        let req = CtlRequest {
            args: [("instrument_id", "x"), ("side", "sell"), ("quantity", "1"), ("limit_price", "10.5")]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect(),
            ..Default::default()
        };
        let r = pretrade_request(&req.encode_to_vec()).unwrap();
        assert_eq!((r.side.as_str(), r.limit_price), ("sell", Some(10.5)));
    }
}
