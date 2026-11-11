package main

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRetryOpenBacksOffUntilSuccess(t *testing.T) {
	calls := 0
	v, err := retryOpen(context.Background(), 10*time.Second, func(context.Context) (int, error) {
		calls++
		if calls < 3 {
			return 0, errors.New("connection refused")
		}
		return 42, nil
	}, nil)
	if err != nil || v != 42 || calls != 3 {
		t.Fatalf("v=%d err=%v calls=%d", v, err, calls)
	}
	// gives up once the budget is spent
	calls = 0
	if _, err := retryOpen(context.Background(), 100*time.Millisecond, func(context.Context) (int, error) {
		calls++
		return 0, errors.New("down")
	}, nil); err == nil || calls != 1 {
		t.Fatalf("want immediate give-up within budget, err=%v calls=%d", err, calls)
	}
}
