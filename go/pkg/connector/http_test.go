package connector

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

type countingLimiter struct{ n atomic.Int32 }

func (c *countingLimiter) Wait(context.Context) error { c.n.Add(1); return nil }

func TestFetcherRetriesAndLimits(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("User-Agent") != UserAgent || r.Header.Get("X-Auth") != "yes" {
			t.Errorf("headers: %v", r.Header)
		}
		switch hits.Add(1) {
		case 1:
			w.WriteHeader(429)
		case 2:
			w.WriteHeader(503)
		default:
			w.Write([]byte("ok"))
		}
	}))
	defer srv.Close()
	lim := &countingLimiter{}
	f := NewFetcher(lim)
	f.Sleep = func(context.Context, time.Duration) error { return nil }
	f.Prepare = func(r *http.Request) { r.Header.Set("X-Auth", "yes") }
	b, err := f.Get(context.Background(), srv.URL)
	if err != nil || string(b) != "ok" {
		t.Fatalf("got %q, %v", b, err)
	}
	if hits.Load() != 3 || lim.n.Load() != 3 {
		t.Fatalf("hits=%d limiter=%d", hits.Load(), lim.n.Load())
	}
	// 404 is final, no retry.
	srv404 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(404) }))
	defer srv404.Close()
	if _, err := f.Get(context.Background(), srv404.URL); !IsHTTPStatus(err, 404) {
		t.Fatalf("want 404 error, got %v", err)
	}
}
