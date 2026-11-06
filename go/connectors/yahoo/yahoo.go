// Package yahoo is a first-wave connector stub for catalog id "yahoo" (specs 10, 11).
//
// Notes (UNVERIFIED — confirm against current API docs before implementing):
//   - Chart/OHLC: https://query1.finance.yahoo.com/v8/finance/chart/{symbol}
//   - Quote summary/metadata: https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}
//   - Unofficial endpoints (yfinance); research use only, may break or require crumb/cookie.
package yahoo

import (
	"context"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// ID matches config/sources.yaml.
const ID = "yahoo"

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector is a compile-ready stub; every data method returns ErrNotImplemented.
type Connector struct {
	cfg  connector.SourceConfig
	deps connector.Deps
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Backfill: true, Events: []connector.EventKind{connector.KindBar}, Venue: "yahoo"}
}

func (c *Connector) Init(_ context.Context, cfg connector.SourceConfig, deps connector.Deps) error {
	c.cfg, c.deps = cfg, deps
	return nil
}

func (c *Connector) Normalize(connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	return nil, nil, connector.ErrNotImplemented
}

func (c *Connector) Poll(context.Context, connector.PollRequest) ([]connector.RawMessage, error) {
	return nil, connector.ErrNotImplemented
}

func (c *Connector) Schedule() connector.PollSchedule {
	return connector.PollSchedule{Interval: time.Minute}
}

func (c *Connector) Backfill(context.Context, connector.BackfillRequest, chan<- connector.RawMessage) error {
	return connector.ErrNotImplemented
}
