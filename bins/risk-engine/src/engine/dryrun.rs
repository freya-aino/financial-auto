//! Pre-submit risk check (`ctl.risk.pretrade`, spec 51 "pre-submit risk check"): the checks a
//! foreign / manual `order.intent.<account>` would meet ([`Engine::on_intent_seen`]: mode, halts,
//! [`fa_risk::pretrade::check_order`], feed guard, stale FX) against the in-memory book, with no
//! side effect — no intent remembered, no rejection or risk event, no counter or FX-staleness
//! state changed.
use fa_core::InstrumentId;
use fa_proto::v1::TradingMode;
use serde::{Deserialize, Serialize};

use super::Engine;

/// Request of `ctl.risk.pretrade`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PretradeRequest {
    pub instrument_id: String,
    /// `buy` | `sell`
    pub side: String,
    pub quantity: f64,
    #[serde(default)]
    pub limit_price: Option<f64>,
    /// `paper` (default) | `live` | `shadow`
    #[serde(default)]
    pub mode: Option<String>,
}

/// Reply of `ctl.risk.pretrade`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PretradeReply {
    pub ok: bool,
    pub reasons: Vec<String>,
}

impl Engine {
    /// Would this order pass the risk gate now? Same checks as a foreign intent, no effects.
    pub fn pretrade_check(&mut self, r: &PretradeRequest, now_ms: i64) -> PretradeReply {
        let fail = |reason: String| PretradeReply { ok: false, reasons: vec![reason] };
        let Ok(uuid) = uuid::Uuid::parse_str(r.instrument_id.trim()) else {
            return fail(format!("invalid instrument_id {:?}", r.instrument_id));
        };
        let inst = InstrumentId(uuid);
        let sign = match r.side.trim().to_ascii_lowercase().as_str() {
            "buy" | "b" => 1.0,
            "sell" | "s" => -1.0,
            other => return fail(format!("invalid side {other:?}: buy | sell")),
        };
        if !r.quantity.is_finite() || r.quantity <= 0.0 {
            return fail("quantity must be a positive number".into());
        }
        let mode = match r.mode.as_deref().map(str::trim).unwrap_or("paper").to_ascii_lowercase().as_str() {
            "" | "paper" => TradingMode::Paper,
            "live" => TradingMode::Live,
            _ => TradingMode::Shadow,
        };
        if mode != TradingMode::Paper {
            return fail("only paper trading exists; live/shadow intents are refused (D-029)".into());
        }
        let signed = sign * r.quantity;
        let mut reasons = Vec::new();
        if self.instrument_blocked(&inst) {
            let current = self.ledger.positions.get(&inst).map_or(0.0, |p| p.quantity);
            let reducing =
                current != 0.0 && signed.signum() != current.signum() && signed.abs() <= current.abs() + 1e-9;
            if !(self.risk.reduce_only() && reducing) {
                reasons.push("trading halted".to_owned());
            }
        }
        if reasons.is_empty() {
            // `pretrade` touches only FX bookkeeping (staleness set, counters): restore both
            let (counters, stale) = (self.counters.clone(), self.fx_stale.clone());
            let res = self.pretrade(inst, signed, r.limit_price, now_ms);
            self.counters = counters;
            self.fx_stale = stale;
            if let Err(e) = res {
                reasons.push(e);
            }
        }
        PretradeReply { ok: reasons.is_empty(), reasons }
    }
}
