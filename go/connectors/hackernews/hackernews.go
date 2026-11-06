// Package hackernews is a first-wave connector stub for catalog id "hackernews" (specs 10, 11).
//
// Notes (UNVERIFIED — confirm against current API docs before implementing):
//   - Firebase API: https://hacker-news.firebaseio.com/v0/{topstories,newstories,item/{id}}.json
//   - Algolia search: https://hn.algolia.com/api/v1/search_by_date
//   - No auth.
package hackernews

import (
	"context"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// ID matches config/sources.yaml.
const ID = "hackernews"

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector is a compile-ready stub; every data method returns ErrNotImplemented.
type Connector struct {
	cfg  connector.SourceConfig
	deps connector.Deps
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Backfill: true, Events: []connector.EventKind{connector.KindNews}}
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
