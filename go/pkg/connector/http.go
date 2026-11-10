package connector

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"time"
)

// UserAgent identifies the ingestor to HTTP sources.
const UserAgent = "financial-auto-ingestor/0.1 (+https://github.com/fey/financial-auto)"

// HTTPError is a non-2xx response after retries.
type HTTPError struct {
	Status int
	Body   string
}

func (e *HTTPError) Error() string { return fmt.Sprintf("http %d: %s", e.Status, e.Body) }

// Fetcher performs rate-limited HTTP requests with bounded retries
// (spec 10 §6.3: token bucket, exponential backoff with jitter).
type Fetcher struct {
	Client  *http.Client
	Limiter RateLimiter
	Retries int           // attempts after the first (default 3)
	Backoff time.Duration // base delay (default 500 ms)
	MaxBody int64         // response cap (default 16 MiB)
	Sleep   func(context.Context, time.Duration) error
	Prepare func(*http.Request) // sets auth headers etc.
}

// NewFetcher builds a Fetcher with a 20 s client and the given limiter (nil = unlimited).
func NewFetcher(lim RateLimiter) *Fetcher {
	return &Fetcher{Client: &http.Client{Timeout: 20 * time.Second}, Limiter: lim}
}

// Do executes req, retrying on network errors, 429 and 5xx. The body is
// fully read and returned.
func (f *Fetcher) Do(ctx context.Context, req *http.Request) ([]byte, *http.Response, error) {
	retries, backoff, maxBody := f.Retries, f.Backoff, f.MaxBody
	if retries == 0 {
		retries = 3
	}
	if backoff == 0 {
		backoff = 500 * time.Millisecond
	}
	if maxBody == 0 {
		maxBody = 16 << 20
	}
	sleep := f.Sleep
	if sleep == nil {
		sleep = func(ctx context.Context, d time.Duration) error {
			t := time.NewTimer(d)
			defer t.Stop()
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-t.C:
				return nil
			}
		}
	}
	client := f.Client
	if client == nil {
		client = http.DefaultClient
	}
	if req.Header.Get("User-Agent") == "" {
		req.Header.Set("User-Agent", UserAgent)
	}
	if f.Prepare != nil {
		f.Prepare(req)
	}
	var last error
	for attempt := 0; attempt <= retries; attempt++ {
		if attempt > 0 {
			d := backoff << (attempt - 1)
			d += time.Duration(rand.Int64N(int64(d / 2)))
			if err := sleep(ctx, d); err != nil {
				return nil, nil, err
			}
		}
		if f.Limiter != nil {
			if err := f.Limiter.Wait(ctx); err != nil {
				return nil, nil, err
			}
		}
		res, err := client.Do(req.Clone(ctx))
		if err != nil {
			last = err
			continue
		}
		body, err := io.ReadAll(io.LimitReader(res.Body, maxBody))
		res.Body.Close()
		if err != nil {
			last = err
			continue
		}
		if res.StatusCode == http.StatusTooManyRequests || res.StatusCode >= 500 {
			last = &HTTPError{Status: res.StatusCode, Body: trim(body)}
			continue
		}
		if res.StatusCode/100 != 2 {
			return body, res, &HTTPError{Status: res.StatusCode, Body: trim(body)}
		}
		return body, res, nil
	}
	return nil, nil, fmt.Errorf("%s %s: giving up after %d attempts: %w", req.Method, req.URL.Redacted(), retries+1, last)
}

// Get is Do for a GET of url.
func (f *Fetcher) Get(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	b, _, err := f.Do(ctx, req)
	return b, err
}

// IsHTTPStatus reports whether err is an HTTPError with the given status.
func IsHTTPStatus(err error, status int) bool {
	var he *HTTPError
	return errors.As(err, &he) && he.Status == status
}

func trim(b []byte) string {
	if len(b) > 256 {
		b = b[:256]
	}
	return string(b)
}
