package bus

import (
	"fmt"
	"os"
	"strings"
)

// MarketDataMode selects which market data the paper-trading path acts on (D-103). The default
// is real: paper orders fill only against data from the real connectors, and the synthetic
// venues (the heartbeat and synthetic connectors) are ignored. Ephemeral test stacks run
// synthetic. Mirrors fa_core::market_data in Rust.
type MarketDataMode string

const (
	MarketDataReal      MarketDataMode = "real"
	MarketDataSynthetic MarketDataMode = "synthetic"
	MarketDataAll       MarketDataMode = "all"
)

// SyntheticVenues are the venues of locally generated market data (go/connectors/heartbeat and
// go/connectors/synthetic).
var SyntheticVenues = []string{"sim"}

// ParseMarketDataMode accepts real, synthetic or all; empty means real.
func ParseMarketDataMode(s string) (MarketDataMode, error) {
	switch m := MarketDataMode(strings.ToLower(strings.TrimSpace(s))); m {
	case "":
		return MarketDataReal, nil
	case MarketDataReal, MarketDataSynthetic, MarketDataAll:
		return m, nil
	default:
		return "", fmt.Errorf("FA_MARKET_DATA must be real, synthetic or all, not %q", s)
	}
}

// MarketDataModeFromEnv reads FA_MARKET_DATA (default real).
func MarketDataModeFromEnv() (MarketDataMode, error) {
	return ParseMarketDataMode(os.Getenv("FA_MARKET_DATA"))
}

// IsSyntheticVenue reports whether venue carries locally generated data.
func IsSyntheticVenue(venue string) bool {
	for _, v := range SyntheticVenues {
		if venue == v {
			return true
		}
	}
	return false
}

// AdmitsVenue reports whether data from venue is acted on in this mode.
func (m MarketDataMode) AdmitsVenue(venue string) bool {
	switch m {
	case MarketDataAll:
		return true
	case MarketDataSynthetic:
		return IsSyntheticVenue(venue)
	default:
		return !IsSyntheticVenue(venue)
	}
}

// AdmitsSubject applies AdmitsVenue to an `md.<kind>.<venue>.…` subject
// (`md.bar.<tf>.<venue>.…` for bars); other subjects pass.
func (m MarketDataMode) AdmitsSubject(subject string) bool {
	parts := strings.Split(subject, ".")
	if len(parts) < 3 || parts[0] != "md" {
		return true
	}
	if parts[1] == "bar" {
		if len(parts) < 4 {
			return true
		}
		return m.AdmitsVenue(parts[3])
	}
	return m.AdmitsVenue(parts[2])
}
