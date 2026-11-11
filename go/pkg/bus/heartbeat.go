package bus

import (
	"context"
	"sync"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// Counters is a concurrency-safe set of named float counters reported in heartbeats.
type Counters struct {
	mu sync.Mutex
	m  map[string]float64
}

func NewCounters() *Counters { return &Counters{m: map[string]float64{}} }

func (c *Counters) Add(name string, v float64) {
	c.mu.Lock()
	c.m[name] += v
	c.mu.Unlock()
}

func (c *Counters) Set(name string, v float64) {
	c.mu.Lock()
	c.m[name] = v
	c.mu.Unlock()
}

func (c *Counters) Snapshot() map[string]float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make(map[string]float64, len(c.m))
	for k, v := range c.m {
		out[k] = v
	}
	return out
}

// BuildHeartbeat constructs a Heartbeat message.
func BuildHeartbeat(service, instance, version string, counters *Counters) *fav1.Heartbeat {
	hb := &fav1.Heartbeat{
		Service:    service,
		InstanceId: instance,
		At:         timestamppb.Now(),
		Version:    version,
	}
	if counters != nil {
		hb.Counters = counters.Snapshot()
	}
	return hb
}

// RunHeartbeat publishes svc.heartbeat.<service> every interval until ctx is done.
func RunHeartbeat(ctx context.Context, p *Publisher, service, instance, version string, interval time.Duration, counters *Counters, onErr func(error)) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		if err := p.Publish(Heartbeat(service), BuildHeartbeat(service, instance, version, counters)); err != nil {
			if onErr != nil {
				onErr(err)
			}
		} else {
			heartbeatPublished(time.Now())
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}
