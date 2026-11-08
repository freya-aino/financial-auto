//! Subject taxonomy (spec 21 §4.1). Keep in sync with `go/pkg/bus/subjects.go`.
use fa_core::{InstrumentId, Timescale};

pub fn md_trade(venue: &str, inst: &InstrumentId) -> String {
    format!("md.trade.{venue}.{}", inst.subject_token())
}

pub fn md_quote(venue: &str, inst: &InstrumentId) -> String {
    format!("md.quote.{venue}.{}", inst.subject_token())
}

pub fn md_bar(tf: Timescale, venue: &str, inst: &InstrumentId) -> String {
    format!("md.bar.{tf}.{venue}.{}", inst.subject_token())
}

pub fn md_book(venue: &str, inst: &InstrumentId) -> String {
    format!("md.book.{venue}.{}", inst.subject_token())
}

pub fn news_item(source: &str) -> String {
    format!("news.item.{source}")
}

/// Last token is the D-017 instrument token (32-hex UUID); the snapshot travels inside `LatentState` (D-039).
pub fn latent_state(ts: Timescale, instrument: &str) -> String {
    format!("latent.state.{ts}.{instrument}")
}

pub fn latent_region(ts: Timescale) -> String {
    format!("latent.region.{ts}")
}

pub fn latent_transition(ts: Timescale) -> String {
    format!("latent.transition.{ts}")
}

pub fn decoder_output(family: &str) -> String {
    format!("decoder.output.{family}")
}

pub fn signal_exposure(strategy: &str) -> String {
    format!("signal.exposure.{strategy}")
}

pub fn order_intent(account: &str) -> String {
    format!("order.intent.{account}")
}

pub fn order_approved(account: &str) -> String {
    format!("order.approved.{account}")
}

pub fn order_event(account: &str) -> String {
    format!("order.event.{account}")
}

pub const RISK_STATE: &str = "risk.state";

pub fn risk_halt(scope: &str) -> String {
    format!("risk.halt.{scope}")
}

/// Clears a halt for `scope` (control-api re-arm, D-030).
pub fn risk_resume(scope: &str) -> String {
    format!("risk.resume.{scope}")
}

/// Wildcards for consumers that aggregate whole families (risk-engine, control-api).
pub const ALL_SIGNALS: &str = "signal.exposure.>";
pub const ALL_ORDERS: &str = "order.>";
pub const ALL_RISK_HALTS: &str = "risk.halt.>";
pub const ALL_RISK_RESUMES: &str = "risk.resume.>";
pub const ALL_LATENT_STATES: &str = "latent.state.>";

pub fn jobs(kind: &str) -> String {
    format!("jobs.{kind}")
}

pub fn job_result(kind: &str) -> String {
    format!("jobs.result.{kind}")
}

pub fn job_progress(job_id: &str) -> String {
    format!("jobs.progress.{job_id}")
}

pub fn heartbeat(service: &str) -> String {
    format!("svc.heartbeat.{service}")
}

/// GPU arbiter protocol (spec 60, D-006): request/reply `acquire` (args `lease_id`, `class`, `mem_mb` → reply
/// `granted` | `queued`) and `release` (`lease_id`); notifications `granted` / `preempt` carry the lease id.
pub const CTL_GPU_ACQUIRE: &str = "ctl.gpu.acquire";
pub const CTL_GPU_RELEASE: &str = "ctl.gpu.release";
pub const CTL_GPU_GRANTED: &str = "ctl.gpu.granted";
pub const CTL_GPU_PREEMPT: &str = "ctl.gpu.preempt";

pub fn ctl(service: &str, verb: &str) -> String {
    format!("ctl.{service}.{verb}")
}

/// Wildcards.
pub const ALL_MD_TRADES: &str = "md.trade.>";
pub const ALL_HEARTBEATS: &str = "svc.heartbeat.>";
pub const EVERYTHING: &str = ">";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subjects_match_taxonomy() {
        let i = InstrumentId::new();
        let t = i.subject_token();
        assert_eq!(md_trade("sim", &i), format!("md.trade.sim.{t}"));
        assert_eq!(md_bar(Timescale::Hour, "lemon", &i), format!("md.bar.1h.lemon.{t}"));
        assert_eq!(heartbeat("ingestor"), "svc.heartbeat.ingestor");
        assert_eq!(job_result("evo.eval.decoder"), "jobs.result.evo.eval.decoder");
    }
}
