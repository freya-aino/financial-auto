package heartbeat

import (
	"context"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/connector"
)

func TestStreamAndNormalize(t *testing.T) {
	c := &Connector{Interval: 5 * time.Millisecond}
	if err := c.Init(context.Background(), connector.SourceConfig{}, connector.Deps{}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	out := make(chan connector.RawMessage, 1)
	go func() { _ = c.Stream(ctx, nil, out) }()
	raw := <-out
	envs, unresolved, err := c.Normalize(raw)
	if err != nil || len(envs) != 1 || len(unresolved) != 0 {
		t.Fatalf("normalize: %v %v %v", envs, unresolved, err)
	}
	e := envs[0]
	if e.GetTrade().GetPrice() <= 0 || e.InstrumentId != InstrumentID {
		t.Fatalf("bad envelope: %v", e)
	}
	if e.EventTime == nil || e.PublishTime == nil || e.IngestTime == nil {
		t.Fatal("all three timestamps must be set")
	}
}
