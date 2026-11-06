package connectors_test

import (
	"context"
	"errors"
	"testing"

	"github.com/fey/financial-auto/go/pkg/connector"

	_ "github.com/fey/financial-auto/go/connectors/googlenews"
	_ "github.com/fey/financial-auto/go/connectors/hackernews"
	_ "github.com/fey/financial-auto/go/connectors/heartbeat"
	_ "github.com/fey/financial-auto/go/connectors/lemon"
	_ "github.com/fey/financial-auto/go/connectors/oanda"
	_ "github.com/fey/financial-auto/go/connectors/yahoo"
)

func TestFirstWaveRegisteredAndStubbed(t *testing.T) {
	for _, id := range []string{"yahoo", "lemon", "oanda", "hackernews", "google_news"} {
		c, err := connector.New(id)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		if err := c.Init(context.Background(), connector.SourceConfig{}, connector.Deps{}); err != nil {
			t.Fatalf("%s init: %v", id, err)
		}
		n, ok := c.(connector.Normalizer)
		if !ok {
			t.Fatalf("%s: must implement Normalizer", id)
		}
		if _, _, err := n.Normalize(connector.RawMessage{}); !errors.Is(err, connector.ErrNotImplemented) {
			t.Errorf("%s: want ErrNotImplemented, got %v", id, err)
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
	}
}
