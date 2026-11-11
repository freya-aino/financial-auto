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

/// Soft region memberships of one instrument, all experts (`RegionMembership`, spec 30 §4.2).
pub fn latent_membership(ts: Timescale, instrument: &str) -> String {
    format!("latent.membership.{ts}.{instrument}")
}

/// Region lifecycle with lineage and event time (`RegionLifecycleEvent`, spec 30 §4.2).
pub fn latent_lifecycle(ts: Timescale) -> String {
    format!("latent.lifecycle.{ts}")
}

/// Graph-level kernel summary of one expert space (`KernelUpdate`, spec 33 §4.3).
pub fn latent_mcmc(ts: Timescale, expert_index: u32) -> String {
    format!("latent.mcmc.{ts}.{expert_index}")
}

/// Cross-timescale region hierarchy (`RegionHierarchy`, spec 30 §6.5).
pub fn latent_hier(fine: Timescale, coarse: Timescale) -> String {
    format!("latent.hier.{fine}.{coarse}")
}

/// Cross-asset prior proposals (`CrossAssetPriors`, spec 33 §6.5).
pub fn latent_crossprior(ts: Timescale) -> String {
    format!("latent.crossprior.{ts}")
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

/// Risk decisions and state changes (`RiskEvent`, D-042 follow-up).
pub fn risk_event(account: &str) -> String {
    format!("risk.event.{account}")
}

/// Ingestion operability frames (`IngestHealth` / `IngestGap` / `IngestUnresolved`, D-054).
pub fn ingest_health(source: &str) -> String {
    format!("ingest.health.{source}")
}

pub fn ingest_gap(source: &str) -> String {
    format!("ingest.gap.{source}")
}

pub fn ingest_unresolved(source: &str) -> String {
    format!("ingest.unresolved.{source}")
}

/// Reference-data change notice (`RefChanged`, spec 12 §6); `entity` = instrument, listing, ...
pub fn ref_changed(entity: &str) -> String {
    format!("ref.changed.{entity}")
}

/// Catalog change notice (`RefChanged` with entity `source`, spec 11 §4).
pub fn catalog_changed(source: &str) -> String {
    format!("catalog.changed.{source}")
}

/// Wildcards for consumers that aggregate whole families (risk-engine, control-api).
pub const ALL_SIGNALS: &str = "signal.exposure.>";
pub const ALL_ORDERS: &str = "order.>";
pub const ALL_RISK_HALTS: &str = "risk.halt.>";
pub const ALL_RISK_RESUMES: &str = "risk.resume.>";
pub const ALL_LATENT_STATES: &str = "latent.state.>";
pub const ALL_RISK_EVENTS: &str = "risk.event.>";
pub const ALL_INGEST_HEALTH: &str = "ingest.health.>";
pub const ALL_REF_CHANGED: &str = "ref.changed.>";
pub const ALL_CATALOG_CHANGED: &str = "catalog.changed.>";

pub fn jobs(kind: &str) -> String {
    format!("jobs.{kind}")
}

pub fn job_result(kind: &str) -> String {
    format!("jobs.result.{kind}")
}

pub fn job_progress(job_id: &str) -> String {
    format!("jobs.progress.{job_id}")
}

/// Prefix of every heartbeat subject (`fa_bus::Bus::publish` mirrors these into metrics).
pub const HEARTBEAT_PREFIX: &str = "svc.heartbeat.";

pub fn heartbeat(service: &str) -> String {
    format!("{HEARTBEAT_PREFIX}{service}")
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
