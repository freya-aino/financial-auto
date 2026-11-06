package connector

import (
	"context"
	"sync"
	"time"

	"golang.org/x/time/rate"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/catalog"
)

// NewLimiter builds a token bucket from a catalog rate limit.
func NewLimiter(l catalog.RateLimit) RateLimiter {
	return rate.NewLimiter(rate.Limit(l.PerSecond), l.Burst)
}

// SystemClock is the wall clock.
type SystemClock struct{}

func (SystemClock) Now() time.Time { return time.Now().UTC() }

// NopArchive discards raw payloads (until the Garage writer exists).
type NopArchive struct{}

func (NopArchive) Put(context.Context, RawMessage) (*fav1.RawRef, error) { return nil, nil }

// StaticResolver is an in-memory symbol map (until the Postgres resolver exists).
type StaticResolver struct {
	mu sync.RWMutex
	m  map[string]string
}

func NewStaticResolver() *StaticResolver { return &StaticResolver{m: map[string]string{}} }

func (r *StaticResolver) Add(sourceID, venueSymbol, instrumentID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.m[sourceID+"\x00"+venueSymbol] = instrumentID
}

func (r *StaticResolver) Resolve(sourceID, venueSymbol string) (string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	id, ok := r.m[sourceID+"\x00"+venueSymbol]
	return id, ok
}
