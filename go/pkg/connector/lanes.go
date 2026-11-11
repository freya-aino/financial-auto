package connector

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/time/rate"

	"github.com/fey/financial-auto/go/pkg/catalog"
)

// DefaultBackfillShare is the share of a source's rate a backfill lane may use
// at most (spec 10 §6.4 low-priority lanes; tunable `backfill.lane_share`).
const DefaultBackfillShare = 0.5

// LaneLimiter is one token bucket per source with two lanes (spec 10 §6.4):
// the live lane takes tokens first-come; the backfill lane is capped at
// share × rate by its own bucket, never takes a token while a live request is
// waiting, and leaves one token of headroom in the shared bucket (when burst
// ≥ 2) so a live request arriving next is admitted at once. Every admitted
// request of either lane passes the shared bucket, so the catalog limit holds
// for the sum (the 24 h soak criterion of spec 10 §9).
type LaneLimiter struct {
	mu     sync.Mutex
	shared *rate.Limiter
	bf     *rate.Limiter
	share  float64

	liveWaiting atomic.Int64
	liveN       atomic.Int64
	backfillN   atomic.Int64
	// Poll is how often a yielding backfill request re-checks the bucket (default 25 ms).
	Poll time.Duration
}

// NewLaneLimiter builds the lanes for a catalog rate limit; share ≤ 0 uses DefaultBackfillShare.
func NewLaneLimiter(l catalog.RateLimit, share float64) *LaneLimiter {
	if share <= 0 || share > 1 {
		share = DefaultBackfillShare
	}
	ll := &LaneLimiter{share: share, Poll: 25 * time.Millisecond}
	ll.shared = rate.NewLimiter(rate.Limit(l.PerSecond), l.Burst)
	ll.bf = rate.NewLimiter(rate.Limit(l.PerSecond*share), 1)
	return ll
}

// SetLimit applies a new catalog rate limit to both lanes (ctl.ingest.reload / configure).
func (l *LaneLimiter) SetLimit(r catalog.RateLimit) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.shared.SetLimit(rate.Limit(r.PerSecond))
	l.shared.SetBurst(r.Burst)
	l.bf.SetLimit(rate.Limit(r.PerSecond * l.share))
}

// Limit returns the shared bucket's current rate and burst.
func (l *LaneLimiter) Limit() catalog.RateLimit {
	l.mu.Lock()
	defer l.mu.Unlock()
	return catalog.RateLimit{PerSecond: float64(l.shared.Limit()), Burst: l.shared.Burst()}
}

// Counts returns requests admitted per lane since creation.
func (l *LaneLimiter) Counts() (live, backfill int64) { return l.liveN.Load(), l.backfillN.Load() }

// Live is the high-priority lane (live polls and streams).
func (l *LaneLimiter) Live() RateLimiter { return liveLane{l} }

// Backfill is the low-priority lane (backfill jobs, gap-fill).
func (l *LaneLimiter) Backfill() RateLimiter { return backfillLane{l} }

type liveLane struct{ l *LaneLimiter }

func (v liveLane) Wait(ctx context.Context) error {
	v.l.liveWaiting.Add(1)
	defer v.l.liveWaiting.Add(-1)
	if err := v.l.shared.Wait(ctx); err != nil {
		return err
	}
	v.l.liveN.Add(1)
	return nil
}

type backfillLane struct{ l *LaneLimiter }

func (v backfillLane) Wait(ctx context.Context) error {
	l := v.l
	if err := l.bf.Wait(ctx); err != nil {
		return err
	}
	poll := l.Poll
	if poll <= 0 {
		poll = 25 * time.Millisecond
	}
	for {
		if l.liveWaiting.Load() == 0 {
			l.mu.Lock()
			headroom := 0.0
			if l.shared.Burst() >= 2 {
				headroom = 1
			}
			ok := l.shared.Tokens() >= 1+headroom && l.liveWaiting.Load() == 0 && l.shared.Allow()
			l.mu.Unlock()
			if ok {
				l.backfillN.Add(1)
				return nil
			}
		}
		t := time.NewTimer(poll)
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
	}
}
