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

func NewsItem(source string) string          { return "news.item." + source }
func MediaChunk(source string) string        { return "media.chunk." + source }
func MacroObs(source, series string) string  { return fmt.Sprintf("macro.obs.%s.%s", source, series) }
func OnchainEvent(chain string) string       { return "onchain.event." + chain }
func FilingEvent(source string) string       { return "filing.event." + source }
func LatentState(ts, snapshot string) string { return fmt.Sprintf("latent.state.%s.%s", ts, snapshot) }
func LatentRegion(ts string) string          { return "latent.region." + ts }
func LatentTransition(ts string) string      { return "latent.transition." + ts }
func DecoderOutput(family string) string     { return "decoder.output." + family }
func SignalExposure(strategy string) string  { return "signal.exposure." + strategy }
func OrderIntent(account string) string      { return "order.intent." + account }
func OrderApproved(account string) string    { return "order.approved." + account }
func OrderEvent(account string) string       { return "order.event." + account }

const RiskState = "risk.state"

func RiskHalt(scope string) string    { return "risk.halt." + scope }
func Jobs(kind string) string         { return "jobs." + kind }
func JobResult(kind string) string    { return "jobs.result." + kind }
func JobProgress(jobID string) string { return "jobs.progress." + jobID }
func Heartbeat(service string) string { return "svc.heartbeat." + service }
func Ctl(service, verb string) string { return fmt.Sprintf("ctl.%s.%s", service, verb) }

// Wildcards.
const (
	AllMdTrades      = "md.trade.>"
	AllHeartbeats    = "svc.heartbeat.>"
	AllOrderApproved = "order.approved.>"
	AllJobResults    = "jobs.result.>"
	Everything       = ">"
)
