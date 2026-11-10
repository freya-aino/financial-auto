package connectors_test

import (
	"testing"

	"github.com/fey/financial-auto/go/pkg/connector"

	_ "github.com/fey/financial-auto/go/connectors/googlenews"
	_ "github.com/fey/financial-auto/go/connectors/hackernews"
	_ "github.com/fey/financial-auto/go/connectors/heartbeat"
	_ "github.com/fey/financial-auto/go/connectors/lemon"
	_ "github.com/fey/financial-auto/go/connectors/oanda"
	_ "github.com/fey/financial-auto/go/connectors/yahoo"
)

// TestFirstWaveRegistered checks every first-wave connector registers under
// its catalog id, implements Normalizer and that the declared capabilities
// match the implemented interfaces.
func TestFirstWaveRegistered(t *testing.T) {
	for _, id := range []string{"heartbeat", "yahoo", "lemon", "oanda", "hackernews", "google_news"} {
		c, err := connector.New(id)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		if c.ID() != id {
			t.Errorf("%s: ID() = %q", id, c.ID())
		}
		if _, ok := c.(connector.Normalizer); !ok {
			t.Fatalf("%s: must implement Normalizer", id)
		}
		caps := c.Capabilities()
		if _, isS := c.(connector.Streamer); isS != caps.Stream {
			t.Errorf("%s: Stream capability mismatch", id)
		}
		if _, isP := c.(connector.Poller); isP != caps.Poll {
			t.Errorf("%s: Poll capability mismatch", id)
		}
		if _, isB := c.(connector.Backfiller); isB != caps.Backfill {
			t.Errorf("%s: Backfill capability mismatch", id)
		}
		if len(caps.Events) == 0 {
			t.Errorf("%s: declares no events", id)
		}
	}
}
