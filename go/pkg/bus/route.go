package bus

import (
	"fmt"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// SubjectFor routes a canonical envelope to its subject (spec 21 §4.1).
// venue is the connector's venue token for market data.
func SubjectFor(env *fav1.Envelope, venue string) (string, error) {
	switch p := env.Payload.(type) {
	case *fav1.Envelope_Trade:
		return MdTrade(venue, env.InstrumentId), nil
	case *fav1.Envelope_Quote:
		return MdQuote(venue, env.InstrumentId), nil
	case *fav1.Envelope_Bar:
		return MdBar(p.Bar.GetTimeframe(), venue, env.InstrumentId), nil
	case *fav1.Envelope_Book:
		return MdBook(venue, env.InstrumentId), nil
	case *fav1.Envelope_News:
		return NewsItem(env.SourceId), nil
	case *fav1.Envelope_Media:
		return MediaChunk(env.SourceId), nil
	case *fav1.Envelope_Macro:
		return MacroObs(env.SourceId, p.Macro.GetSeriesId()), nil
	case *fav1.Envelope_Onchain:
		return OnchainEvent(p.Onchain.GetChain()), nil
	case *fav1.Envelope_Filing:
		return FilingEvent(env.SourceId), nil
	case *fav1.Envelope_Fundamental:
		return FundamentalFact(env.SourceId), nil
	default:
		return "", fmt.Errorf("bus: envelope %s has no payload", env.GetEventId())
	}
}
