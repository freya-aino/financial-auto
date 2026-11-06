// Package oanda is a first-wave connector stub for catalog id "oanda" (specs 10, 11).
//
// Notes (UNVERIFIED — confirm against current API docs before implementing):
//   - REST v20: https://api-fxpractice.oanda.com/v3 (practice), https://api-fxtrade.oanda.com/v3 (live)
//   - Pricing stream: https://stream-fxpractice.oanda.com/v3/accounts/{id}/pricing/stream
//   - Candles: /v3/instruments/{instrument}/candles. Bearer token + account id.
package oanda

import (
	"context"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// ID matches config/sources.yaml.
const ID = "oanda"

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector is a compile-ready stub; every data method returns ErrNotImplemented.
type Connector struct {
	cfg  connector.SourceConfig
	deps connector.Deps
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Stream: true, Backfill: true, Events: []connector.EventKind{connector.KindQuote, connector.KindBar}, Venue: "oanda"}
}

func (c *Connector) Init(_ context.Context, cfg connector.SourceConfig, deps connector.Deps) error {
	c.cfg, c.deps = cfg, deps
	return nil
}

func (c *Connector) Normalize(connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	return nil, nil, connector.ErrNotImplemented
}

func (c *Connector) Stream(context.Context, []connector.Subscription, chan<- connector.RawMessage) error {
	return connector.ErrNotImplemented
}

func (c *Connector) Backfill(context.Context, connector.BackfillRequest, chan<- connector.RawMessage) error {
	return connector.ErrNotImplemented
}
