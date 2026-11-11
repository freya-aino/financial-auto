package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/connectors/kraken"
	"github.com/fey/financial-auto/go/connectors/yahoo"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// TestRateLimitSoak is the harness for spec 10 §9 "rate limits from catalog
// are never exceeded in a 24 h soak test", run against a local fake server
// (no external API is contacted). For each live market-data source it runs
// the real connector code (yahoo, kraken) with its catalog rate limit through
// the ingestor's lane limiter: live polls on a schedule plus two backfill
// lanes hammering Backfill continuously, with 5 % of responses answered 503
// so the Fetcher's retries are exercised too. The fake server timestamps every
// request; the test asserts that no window of 1 s, 10 s or 60 s ever holds
// more than burst + rate × window (+1 for timer granularity) requests, and
// that backfill never delays a live poll beyond what the live requests alone
// need.
//
// Default: 6 s per source at 10 × the catalog rate and a 6 s live schedule
// (same shape, fast). Real soak: FA_SOAK_DURATION=24h (catalog rates, the
// connectors' own 1 min schedule), e.g.
//
//	FA_SOAK_DURATION=24h go test ./services/ingestor -run TestRateLimitSoak -timeout 25h -v
func TestRateLimitSoak(t *testing.T) {
	if testing.Short() {
		t.Skip("soak harness: skipped with -short")
	}
	dur, scale, interval := 6*time.Second, 10.0, 6*time.Second
	if v := os.Getenv("FA_SOAK_DURATION"); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil {
			t.Fatal(err)
		}
		dur, scale, interval = d, 1, time.Minute
	}
	cat, err := catalog.Load("../../../config/sources.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"yahoo", "kraken"} {
		src, _ := cat.ByID(id)
		t.Run(id, func(t *testing.T) {
			t.Parallel()
			soakSource(t, src, dur, scale, interval)
		})
	}
}

func soakSource(t *testing.T, src catalog.Source, dur time.Duration, scale float64, interval time.Duration) {
	lim := src.Limit()
	lim.PerSecond *= scale
	var mu sync.Mutex
	var stamps []time.Time
	n := 0
	fixtures := map[string][]byte{}
	for name, path := range map[string]string{
		"chart":  "../../connectors/yahoo/testdata/chart_aapl_1m.json",
		"OHLC":   "../../connectors/kraken/testdata/ohlc_xbteur_1m.json",
		"Trades": "../../connectors/kraken/testdata/trades_xbteur.json",
		"Ticker": "../../connectors/kraken/testdata/ticker_xbteur.json",
	} {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		fixtures[name] = b
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		stamps = append(stamps, time.Now())
		n++
		fail := n%20 == 0
		mu.Unlock()
		if fail {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		for k, b := range fixtures {
			if strings.Contains(r.URL.Path, k) {
				w.Write(b)
				return
			}
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	lanes := connector.NewLaneLimiter(lim, connector.DefaultBackfillShare)
	newConn := func(rl connector.RateLimiter) (connector.Poller, connector.Backfiller) {
		deps := connector.Deps{Limiter: rl, Resolver: connector.NewStaticResolver(), Clock: connector.SystemClock{}}
		cfg := connector.SourceConfig{Source: src}
		var c interface {
			connector.Connector
			connector.Poller
			connector.Backfiller
		}
		switch src.ID {
		case "yahoo":
			y := &yahoo.Connector{BaseURL: srv.URL}
			y.Fetch = connector.NewFetcher(rl)
			y.Fetch.Backoff = 10 * time.Millisecond
			c = y
		case "kraken":
			k := &kraken.Connector{BaseURL: srv.URL}
			k.Fetch = connector.NewFetcher(rl)
			k.Fetch.Backoff = 10 * time.Millisecond
			c = k
		}
		if err := c.Init(context.Background(), cfg, deps); err != nil {
			t.Fatal(err)
		}
		return c, c
	}
	ctx, cancel := context.WithTimeout(context.Background(), dur)
	defer cancel()
	var wg sync.WaitGroup
	symbols := src.Params().Symbols
	for range 2 { // two backfill lanes
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, bf := newConn(lanes.Backfill())
			for ctx.Err() == nil {
				for _, s := range symbols {
					out := make(chan connector.RawMessage, 64)
					go func() {
						for range out {
						}
					}()
					to := time.Now()
					_ = bf.Backfill(ctx, connector.BackfillRequest{VenueSymbol: s, Timeframe: "1m", From: to.Add(-30 * 24 * time.Hour), To: to}, out)
					close(out)
				}
			}
		}()
	}
	var polls []time.Duration
	wg.Add(1)
	go func() { // live lane on its schedule
		defer wg.Done()
		p, _ := newConn(lanes.Live())
		for ctx.Err() == nil {
			start := time.Now()
			_, err := p.Poll(ctx, connector.PollRequest{})
			if ctx.Err() != nil {
				return
			}
			if err == nil {
				polls = append(polls, time.Since(start))
			}
			select {
			case <-ctx.Done():
			case <-time.After(max(0, interval-time.Since(start))):
			}
		}
	}()
	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	if len(stamps) < 3 {
		t.Fatalf("only %d requests", len(stamps))
	}
	slices.SortFunc(stamps, func(a, b time.Time) int { return a.Compare(b) })
	for _, w := range []time.Duration{time.Second, 10 * time.Second, time.Minute} {
		if w > dur {
			continue
		}
		bound := float64(lim.Burst) + lim.PerSecond*w.Seconds() + 1
		j, worst := 0, 0
		for i := range stamps {
			for j < len(stamps) && stamps[j].Sub(stamps[i]) < w {
				j++
			}
			worst = max(worst, j-i)
		}
		if float64(worst) > bound {
			t.Errorf("%s: %d requests in a %s window, limit allows %.1f", src.ID, worst, w, bound)
		}
		t.Logf("%s: worst %s window %d requests (bound %.1f)", src.ID, w, worst, bound)
	}
	live, bf := lanes.Counts()
	t.Logf("%s: %d requests (live %d, backfill %d) in %s at %.2f/s burst %d; live polls %v", src.ID, len(stamps), live, bf, dur, lim.PerSecond, lim.Burst, polls)
	if len(polls) == 0 {
		t.Fatalf("%s: no live poll completed", src.ID)
	}
	// one live poll issues len(symbols) × requests-per-symbol requests; alone it needs
	// (requests − 1) / rate. Backfill may add at most one in-flight token, retries one more.
	perPoll := map[string]int{"yahoo": 1, "kraken": 3}[src.ID] * len(symbols)
	limit := time.Duration((float64(perPoll+2)/lim.PerSecond)*float64(time.Second)) + 300*time.Millisecond
	for _, d := range polls {
		if d > limit {
			t.Errorf("%s: live poll took %s, more than %s: backfill starved live", src.ID, d, limit)
		}
	}
}
