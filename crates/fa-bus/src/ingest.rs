//! Ingestion operability frames (`ingest.health|gap|unresolved.<source>`, D-054) and change
//! notices (`ref.changed.*`, `catalog.changed.*`).
//!
//! The ingestor published these as JSON until the `fa.v1` messages `IngestHealth`, `IngestGap`,
//! `IngestUnresolved` and `RefChanged` existed. Consumers accept both encodings: a JSON object
//! starts with `{`, which is never the first byte of a proto3 encoding (field 15, wire type 3 is
//! a deprecated group start that no encoder emits). [`frame_json`] turns either encoding into
//! the D-054 JSON shape that control-api serves and the frontend reads, so the HTTP contract is
//! unchanged by the switch.
use fa_proto::{
    Message, Timestamp,
    v1::{HealthStatus, IngestGap, IngestHealth, IngestUnresolved, RefChanged},
};
use serde_json::{Map, Value, json};

/// True when the payload is a (legacy) JSON object rather than a protobuf message.
pub fn is_json(payload: &[u8]) -> bool {
    payload.iter().find(|b| !b.is_ascii_whitespace()) == Some(&b'{')
}

/// `ok` | `degraded` | `down` | `""`.
pub fn status_str(s: HealthStatus) -> &'static str {
    match s {
        HealthStatus::Ok => "ok",
        HealthStatus::Degraded => "degraded",
        HealthStatus::Down => "down",
        HealthStatus::Unspecified => "",
    }
}

pub fn parse_status(s: &str) -> HealthStatus {
    match s {
        "ok" => HealthStatus::Ok,
        "degraded" => HealthStatus::Degraded,
        "down" => HealthStatus::Down,
        _ => HealthStatus::Unspecified,
    }
}

/// Status and reason of an `ingest.health.<source>` frame in either encoding (what the
/// risk-engine needs for source halts). `None` when the payload is neither.
pub fn health_status(payload: &[u8]) -> Option<(String, String)> {
    if is_json(payload) {
        let v: Value = serde_json::from_slice(payload).ok()?;
        return Some((
            v.get("status").and_then(Value::as_str).unwrap_or("").to_owned(),
            v.get("reason").and_then(Value::as_str).unwrap_or("").to_owned(),
        ));
    }
    let h = IngestHealth::decode(payload).ok()?;
    Some((status_str(h.status()).to_owned(), h.reason))
}

/// Decodes an `ingest.<kind>.<source>` payload (either encoding) into the D-054 JSON shape.
/// `kind` is `health`, `gap` or `unresolved`; other kinds are returned only if JSON.
pub fn frame_json(kind: &str, payload: &[u8]) -> Option<Value> {
    if is_json(payload) {
        return serde_json::from_slice(payload).ok();
    }
    match kind {
        "health" => IngestHealth::decode(payload).ok().map(|h| health_json(&h)),
        "gap" => IngestGap::decode(payload).ok().map(|g| gap_json(&g)),
        "unresolved" => IngestUnresolved::decode(payload).ok().map(|u| unresolved_json(&u)),
        _ => None,
    }
}

fn rfc3339(t: &Option<Timestamp>) -> Option<String> {
    let t = t.as_ref()?;
    if t.seconds == 0 && t.nanos == 0 {
        return None;
    }
    let st = std::time::UNIX_EPOCH
        .checked_add(std::time::Duration::new(u64::try_from(t.seconds).ok()?, u32::try_from(t.nanos).ok()?))?;
    Some(fmt_rfc3339(st))
}

/// RFC 3339 UTC with nanoseconds trimmed like Go's `time.RFC3339Nano`.
fn fmt_rfc3339(t: std::time::SystemTime) -> String {
    let d = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, dd) = civil_from_days(days);
    let mut s = format!("{y:04}-{m:02}-{dd:02}T{:02}:{:02}:{:02}", rem / 3600, rem % 3600 / 60, rem % 60);
    let n = d.subsec_nanos();
    if n > 0 {
        let frac = format!("{n:09}");
        s.push('.');
        s.push_str(frac.trim_end_matches('0'));
    }
    s.push('Z');
    s
}

/// Howard Hinnant's days → civil date.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn secs(t: &Option<Timestamp>) -> Option<f64> {
    let t = t.as_ref()?;
    if t.seconds == 0 && t.nanos == 0 {
        return None;
    }
    Some(t.seconds as f64 + f64::from(t.nanos) / 1e9)
}

fn put(m: &mut Map<String, Value>, k: &str, v: Option<impl Into<Value>>) {
    if let Some(v) = v {
        m.insert(k.to_owned(), v.into());
    }
}

/// `IngestHealth` → the JSON frame of D-054 (same keys as the ingestor's former `Frame`, ages
/// derived from `at`).
pub fn health_json(h: &IngestHealth) -> Value {
    let mut m = Map::new();
    m.insert("source".into(), json!(h.source));
    m.insert("status".into(), json!(status_str(h.status())));
    m.insert("running".into(), json!(h.running));
    m.insert("mode".into(), json!(h.mode));
    put(&mut m, "at", rfc3339(&h.at));
    if !h.reason.is_empty() {
        m.insert("reason".into(), json!(h.reason));
    }
    m.insert("halted".into(), json!(h.halted));
    put(&mut m, "started_at", rfc3339(&h.started_at));
    m.insert("expected_period_s".into(), json!(h.expected_period_s));
    let at = secs(&h.at);
    for (key, t) in [("last_activity", &h.last_activity_at), ("last_event", &h.last_event_at)] {
        put(&mut m, &format!("{key}_at"), rfc3339(t));
        if let (Some(at), Some(x)) = (at, secs(t)) {
            m.insert(format!("{key}_age_s"), json!((at - x).max(0.0)));
        }
    }
    m.insert("events_total".into(), json!(h.events_total));
    m.insert("errors_total".into(), json!(h.errors_total));
    m.insert("window_s".into(), json!(h.window_s));
    m.insert("events_window".into(), json!(h.events_window));
    m.insert("errors_window".into(), json!(h.errors_window));
    m.insert("error_rate".into(), json!(h.error_rate));
    m.insert("limiter_saturation".into(), json!(h.limiter_saturation));
    m.insert("limiter_requests".into(), json!(h.limiter_requests));
    if !h.last_error.is_empty() {
        m.insert("last_error".into(), json!(h.last_error));
    }
    put(&mut m, "last_error_at", rfc3339(&h.last_error_at));
    Value::Object(m)
}

pub fn gap_json(g: &IngestGap) -> Value {
    json!({
        "source": g.source,
        "instrument_id": g.instrument_id,
        "timeframe": g.timeframe,
        "from": rfc3339(&g.from),
        "to": rfc3339(&g.to),
        "missing_periods": g.missing_periods,
        "at": rfc3339(&g.at),
    })
}

pub fn unresolved_json(u: &IngestUnresolved) -> Value {
    let mut v = json!({
        "source": u.source,
        "venue_symbol": u.venue_symbol,
        "reason": u.reason,
        "at": rfc3339(&u.at),
    });
    if !u.instrument_id.is_empty() {
        v["instrument_id"] = json!(u.instrument_id);
    }
    v
}

/// `ref.changed.<entity>` / `catalog.changed.<id>` notice (empty or JSON payloads tolerated).
pub fn ref_changed(payload: &[u8]) -> Option<RefChanged> {
    if payload.is_empty() || is_json(payload) {
        return None;
    }
    RefChanged::decode(payload).ok()
}

pub fn ref_changed_json(r: &RefChanged) -> Value {
    json!({
        "entity": r.entity,
        "ids": r.ids,
        "change": r.change,
        "at": rfc3339(&r.at),
        "actor": r.actor,
        "source_id": r.source_id,
        "attrs": r.attrs,
    })
}

/// Builds a `RefChanged` notice stamped now.
pub fn ref_changed_notice(entity: &str, ids: Vec<String>, change: &str, actor: &str) -> RefChanged {
    RefChanged {
        entity: entity.to_owned(),
        ids,
        change: change.to_owned(),
        at: Some(fa_proto::now()),
        actor: actor.to_owned(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: i64, n: i32) -> Option<Timestamp> {
        Some(Timestamp { seconds: s, nanos: n })
    }

    #[test]
    fn json_and_proto_are_told_apart() {
        assert!(is_json(br#"{"status":"ok"}"#));
        assert!(is_json(b"  {}"));
        let h = IngestHealth { source: "yahoo".into(), status: HealthStatus::Ok as i32, ..Default::default() };
        let b = h.encode_to_vec();
        assert!(!is_json(&b));
        // every proto3 field tag byte for fields 1..=21 differs from '{'
        assert_ne!(b[0], b'{');
    }

    #[test]
    fn health_proto_maps_to_d054_json() {
        let h = IngestHealth {
            source: "yahoo".into(),
            status: HealthStatus::Degraded as i32,
            running: true,
            mode: "poll".into(),
            at: ts(1_762_819_200, 500_000_000), // 2025-11-11T00:00:00.5Z
            reason: "1 error(s)".into(),
            last_activity_at: ts(1_762_819_190, 500_000_000),
            expected_period_s: 60.0,
            limiter_saturation: 0.25,
            errors_window: 1,
            ..Default::default()
        };
        let v = frame_json("health", &h.encode_to_vec()).unwrap();
        assert_eq!(v["status"], "degraded");
        assert_eq!(v["running"], true);
        assert_eq!(v["at"], "2025-11-11T00:00:00.5Z");
        assert_eq!(v["last_activity_age_s"], 10.0);
        assert!(v.get("last_event_at").is_none());
        assert_eq!(v["limiter_saturation"], 0.25);
        assert_eq!(health_status(&h.encode_to_vec()).unwrap(), ("degraded".into(), "1 error(s)".into()));
    }

    #[test]
    fn legacy_json_frames_pass_through() {
        let raw = br#"{"source":"hn","status":"down","reason":"connector exited","running":false}"#;
        assert_eq!(frame_json("health", raw).unwrap()["status"], "down");
        assert_eq!(health_status(raw).unwrap(), ("down".into(), "connector exited".into()));
    }

    #[test]
    fn gap_and_unresolved_round_trip() {
        let g = IngestGap { source: "yahoo".into(), instrument_id: "i".into(), timeframe: "1m".into(), missing_periods: 3, from: ts(60, 0), to: ts(300, 0), ..Default::default() };
        let v = frame_json("gap", &g.encode_to_vec()).unwrap();
        assert_eq!(v["missing_periods"], 3);
        assert_eq!(v["from"], "1970-01-01T00:01:00Z");
        let u = IngestUnresolved { source: "yahoo".into(), venue_symbol: "X".into(), reason: "provisioned".into(), ..Default::default() };
        let v = frame_json("unresolved", &u.encode_to_vec()).unwrap();
        assert_eq!(v["venue_symbol"], "X");
        assert!(v.get("instrument_id").is_none());
    }

    #[test]
    fn resume_is_wire_compatible_with_halt() {
        use fa_proto::v1::{Halt, Resume};
        let r = Resume { scope: "portfolio".into(), reason: "manual resume".into(), at: ts(5, 0), manual: true, actor: "ui".into() };
        let h = Halt::decode(r.encode_to_vec().as_slice()).unwrap();
        assert_eq!((h.scope.as_str(), h.reason.as_str(), h.manual), ("portfolio", "manual resume", true));
        assert_eq!(h.since, ts(5, 0));
        let back = Resume::decode(h.encode_to_vec().as_slice()).unwrap();
        assert_eq!(back.scope, "portfolio");
    }

    #[test]
    fn ref_changed_notice_round_trips() {
        let n = ref_changed_notice("instrument", vec!["abc".into()], "provisioned", "system:ingestor");
        let d = ref_changed(&n.encode_to_vec()).unwrap();
        assert_eq!(d.entity, "instrument");
        assert_eq!(ref_changed_json(&d)["ids"][0], "abc");
        assert!(ref_changed(b"").is_none());
    }

    #[test]
    fn civil_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_403), (2025, 11, 11));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }
}
