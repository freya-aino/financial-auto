package main

import (
	"context"
	"time"
)

// retryOpen calls open until it succeeds, ctx ends or total has elapsed, doubling the wait from 500 ms up to 5 s.
func retryOpen[T any](ctx context.Context, total time.Duration, open func(context.Context) (T, error), onRetry func(error, time.Duration)) (T, error) {
	deadline := time.Now().Add(total)
	wait := 500 * time.Millisecond
	for {
		v, err := open(ctx)
		if err == nil || time.Now().Add(wait).After(deadline) {
			return v, err
		}
		if onRetry != nil {
			onRetry(err, wait)
		}
		select {
		case <-ctx.Done():
			return v, err
		case <-time.After(wait):
		}
		wait = min(2*wait, 5*time.Second)
	}
}
