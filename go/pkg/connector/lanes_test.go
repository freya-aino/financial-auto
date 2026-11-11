package connector

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/catalog"
)

func TestLaneLimiterBackfillYieldsToLive(t *testing.T) {
	l := NewLaneLimiter(catalog.RateLimit{PerSecond: 40, Burst: 2}, 0.5)
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Millisecond)
	defer cancel()
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { // live saturates the bucket
		defer wg.Done()
		for l.Live().Wait(ctx) == nil {
		}
	}()
	go func() {
		defer wg.Done()
		for l.Backfill().Wait(ctx) == nil {
		}
	}()
	wg.Wait()
	live, bf := l.Counts()
	if live < 15 {
		t.Fatalf("live admitted %d, want ≈ 24", live)
	}
	if bf > 1 {
		t.Fatalf("backfill admitted %d while live was always waiting", bf)
	}
	if live+bf > 2+int64(40*0.6)+1 {
		t.Fatalf("sum %d exceeds the bucket", live+bf)
	}
}

func TestLaneLimiterBackfillShareAndReload(t *testing.T) {
	l := NewLaneLimiter(catalog.RateLimit{PerSecond: 40, Burst: 4}, 0.25)
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	for l.Backfill().Wait(ctx) == nil {
	}
	_, bf := l.Counts()
	// 10/s share over 0.5 s plus the lane's burst of one
	if bf < 3 || bf > 7 {
		t.Fatalf("backfill admitted %d, want ≈ 6 at 25 %% of 40/s", bf)
	}
	l.SetLimit(catalog.RateLimit{PerSecond: 2, Burst: 1})
	if got := l.Limit(); got.PerSecond != 2 || got.Burst != 1 {
		t.Fatalf("limit %+v", got)
	}
}
