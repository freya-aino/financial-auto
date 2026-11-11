// Package bus holds the NATS subject taxonomy and typed protobuf publishing (spec 21).
// subjects.go mirrors crates/fa-bus/src/subjects.rs — keep both in sync.
package bus

import (
	"fmt"
	"strings"
)

// SubjectToken renders an instrument UUID as 32 lowercase hex chars, no dashes (D-017).
func SubjectToken(instrumentID string) string {
	return strings.ToLower(strings.ReplaceAll(instrumentID, "-", ""))
}

func MdTrade(venue, inst string) string {
	return fmt.Sprintf("md.trade.%s.%s", venue, SubjectToken(inst))
}
func MdQuote(venue, inst string) string {
	return fmt.Sprintf("md.quote.%s.%s", venue, SubjectToken(inst))
}
func MdBar(tf, venue, inst string) string {
	return fmt.Sprintf("md.bar.%s.%s.%s", tf, venue, SubjectToken(inst))
}
func MdBook(venue, inst string) string {
	return fmt.Sprintf("md.book.%s.%s", venue, SubjectToken(inst))
}

func NewsItem(source string) string         { return "news.item." + source }
func MediaChunk(source string) string       { return "media.chunk." + source }
func MacroObs(source, series string) string { return fmt.Sprintf("macro.obs.%s.%s", source, series) }
func OnchainEvent(chain string) string      { return "onchain.event." + chain }
func FilingEvent(source string) string      { return "filing.event." + source }

// FundamentalFact is a reported XBRL fact (stream TEXT via "filing.>").
func FundamentalFact(source string) string { return "filing.fact." + source }
func LatentState(ts, instrument string) string {
	return fmt.Sprintf("latent.state.%s.%s", ts, instrument)
}                                       // D-039
func LatentRegion(ts string) string     { return "latent.region." + ts }
func LatentTransition(ts string) string { return "latent.transition." + ts }

// D-069 latent subjects outside latent.region.> (specs 30 §4.2, 33 §4.3).
func LatentMembership(ts, instrument string) string {
	return fmt.Sprintf("latent.membership.%s.%s", ts, instrument)
}
func LatentLifecycle(ts string) string { return "latent.lifecycle." + ts }
func LatentMcmc(ts string, expertIndex uint32) string {
	return fmt.Sprintf("latent.mcmc.%s.%d", ts, expertIndex)
}
func LatentHier(fine, coarse string) string { return fmt.Sprintf("latent.hier.%s.%s", fine, coarse) }
func LatentCrossPrior(ts string) string     { return "latent.crossprior." + ts }

// Spec 32 §4.3 expert-space contract: fa.v1.ExpertView per step, fa.v1.ExpertHealth.
func LatentExpert(ts, assetSet string) string {
	return fmt.Sprintf("latent.expert.%s.%s", ts, assetSet)
}

const LatentExpertHealth = "latent.expert.health"

func DecoderOutput(family string) string    { return "decoder.output." + family }
func SignalExposure(strategy string) string { return "signal.exposure." + strategy }
func OrderIntent(account string) string     { return "order.intent." + account }
func OrderApproved(account string) string   { return "order.approved." + account }
func OrderEvent(account string) string      { return "order.event." + account }

const RiskState = "risk.state"

func RiskHalt(scope string) string    { return "risk.halt." + scope }
func RiskResume(scope string) string  { return "risk.resume." + scope }
func RiskEvent(account string) string { return "risk.event." + account }

// Ingestion operability frames (fa.v1 IngestHealth / IngestGap / IngestUnresolved, D-054)
// and change notices (fa.v1 RefChanged, specs 11 §4 and 12 §6).
func IngestHealth(source string) string     { return "ingest.health." + source }
func IngestGap(source string) string        { return "ingest.gap." + source }
func IngestUnresolved(source string) string { return "ingest.unresolved." + source }
func RefChanged(entity string) string       { return "ref.changed." + entity }
func CatalogChanged(source string) string   { return "catalog.changed." + source }

func Jobs(kind string) string         { return "jobs." + kind }
func JobResult(kind string) string    { return "jobs.result." + kind }
func JobProgress(jobID string) string { return "jobs.progress." + jobID }
func Heartbeat(service string) string { return "svc.heartbeat." + service }

// OpsAlert carries alert transitions as JSON (D-066; control-api, dead-letter forwarder).
const OpsAlert = "ops.alert"

// DeadLetter is the DLQ copy of a poison message: dlq.<stream>.<original subject> (spec 21 §8).
func DeadLetter(stream, subject string) string { return "dlq." + stream + "." + subject }
func Ctl(service, verb string) string          { return fmt.Sprintf("ctl.%s.%s", service, verb) }

// Wildcards.
const (
	AllMdTrades      = "md.trade.>"
	AllHeartbeats    = "svc.heartbeat.>"
	AllOrderApproved = "order.approved.>"
	AllJobResults    = "jobs.result.>"
	Everything       = ">"
)
