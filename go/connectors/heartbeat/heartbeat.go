// Package heartbeat is a synthetic connector proving the ingestion path end to end (D-018).
// It emits one random-walk Trade per interval for a fixed simulated instrument.
package heartbeat

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const (
	ID    = "heartbeat"
	Venue = "sim"
	// InstrumentID is the fixed simulated instrument (UUID v7 layout).
	InstrumentID = "01900000-0000-7000-8000-000000000001"
)

func init() {
	connector.Register(ID, func() connector.Connector { return &Connector{Interval: time.Second} })
}

// Connector implements Connector, Streamer and Normalizer.
type Connector struct {
	Interval time.Duration
	deps     connector.Deps
	price    float64
}

type tick struct {
	Price float64   `json:"price"`
	Size  float64   `json:"size"`
	Buy   bool      `json:"buy"`
	At    time.Time `json:"at"`
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Stream: true, Events: []connector.EventKind{connector.KindTrade}, Venue: Venue}
}

func (c *Connector) Init(_ context.Context, _ connector.SourceConfig, deps connector.Deps) error {
	if deps.Clock == nil {
		deps.Clock = connector.SystemClock{}
	}
	c.deps = deps
	c.price = 100
	return nil
}

// Stream emits one synthetic trade per interval.
func (c *Connector) Stream(ctx context.Context, _ []connector.Subscription, out chan<- connector.RawMessage) error {
	t := time.NewTicker(c.Interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			c.price *= 1 + rand.NormFloat64()*0.001
			now := c.deps.Clock.Now()
			b, err := json.Marshal(tick{Price: c.price, Size: 1 + rand.Float64()*9, Buy: rand.IntN(2) == 0, At: now})
			if err != nil {
				return err
			}
			msg := connector.RawMessage{SourceID: ID, ReceivedAt: now, Bytes: b, ContentType: "application/json"}
			select {
			case out <- msg:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
}

// Normalize turns a synthetic tick into a Trade envelope with all three timestamps.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var tk tick
	if err := json.Unmarshal(raw.Bytes, &tk); err != nil {
		return nil, nil, fmt.Errorf("heartbeat: %w", err)
	}
	side := fav1.Side_SIDE_SELL
	if tk.Buy {
		side = fav1.Side_SIDE_BUY
	}
	env := &fav1.Envelope{
		EventId:       ulid.Make().String(),
		SourceId:      ID,
		InstrumentId:  InstrumentID,
		EventTime:     timestamppb.New(tk.At),
		PublishTime:   timestamppb.New(tk.At),
		IngestTime:    timestamppb.New(raw.ReceivedAt),
		SchemaVersion: 1,
		Payload:       &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: tk.Price, Size: tk.Size, Aggressor: side}},
	}
	return []*fav1.Envelope{env}, nil, nil
}
