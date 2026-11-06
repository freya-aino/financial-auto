// Command ingestor runs connectors concurrently, normalizes their output to
// canonical envelopes and publishes them on NATS (spec 10).
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/oklog/ulid/v2"

	"github.com/fey/financial-auto/go/pkg/bus"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"

	_ "github.com/fey/financial-auto/go/connectors/googlenews"
	_ "github.com/fey/financial-auto/go/connectors/hackernews"
	"github.com/fey/financial-auto/go/connectors/heartbeat"
	_ "github.com/fey/financial-auto/go/connectors/lemon"
	_ "github.com/fey/financial-auto/go/connectors/oanda"
	_ "github.com/fey/financial-auto/go/connectors/yahoo"
)

const service = "ingestor"

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	natsURL := flag.String("nats", env("FA_NATS_URL", "nats://127.0.0.1:4222"), "NATS URL")
	catalogPath := flag.String("catalog", env("FA_CATALOG", "config/sources.yaml"), "source catalog path")
	enabled := flag.String("connectors", env("FA_CONNECTORS", heartbeat.ID), "comma-separated connector ids to run")
	hbEvery := flag.Duration("heartbeat", 5*time.Second, "service heartbeat interval")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, nil)).With("svc", service)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cat, err := catalog.Load(*catalogPath)
	if err != nil {
		log.Error("catalog", "err", err)
		os.Exit(1)
	}
	pub, err := bus.Connect(*natsURL, service)
	if err != nil {
		log.Error("nats connect", "err", err)
		os.Exit(1)
	}
	defer pub.Close()

	counters := bus.NewCounters()
	instance := ulid.Make().String()
	go bus.RunHeartbeat(ctx, pub, service, instance, "0.1.0", *hbEvery, counters, func(err error) { log.Warn("heartbeat", "err", err) })

	resolver := connector.NewStaticResolver()
	var wg sync.WaitGroup
	for _, id := range strings.Split(*enabled, ",") {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		src, ok := cat.ByID(id)
		if !ok {
			// Built-in synthetic sources (heartbeat) are not in the catalog.
			src = catalog.Source{ID: id, Name: id, Status: "live", Auth: "none"}
		}
		c, err := connector.New(id)
		if err != nil {
			log.Error("connector", "id", id, "err", err)
			continue
		}
		deps := connector.Deps{
			Limiter:  connector.NewLimiter(src.Limit()),
			Resolver: resolver,
			Archive:  connector.NopArchive{},
			Clock:    connector.SystemClock{},
		}
		if err := c.Init(ctx, connector.SourceConfig{Source: src}, deps); err != nil {
			log.Error("init", "id", id, "err", err)
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			run(ctx, log.With("source", id), c, pub, counters)
		}()
	}
	log.Info("started", "connectors", connector.Registered(), "enabled", *enabled)
	wg.Wait()
	log.Info("stopped")
}

// run drives one connector: stream or poll, normalize, publish.
func run(ctx context.Context, log *slog.Logger, c connector.Connector, pub *bus.Publisher, counters *bus.Counters) {
	id := c.ID()
	norm, ok := c.(connector.Normalizer)
	if !ok {
		log.Error("connector does not implement Normalizer")
		return
	}
	raws := make(chan connector.RawMessage, 1024)
	go func() {
		defer close(raws)
		var err error
		switch {
		case c.Capabilities().Stream:
			err = c.(connector.Streamer).Stream(ctx, nil, raws)
		case c.Capabilities().Poll:
			err = pollLoop(ctx, c.(connector.Poller), raws)
		default:
			log.Warn("connector has neither stream nor poll capability")
		}
		if err != nil && !errors.Is(err, context.Canceled) {
			counters.Add("errors."+id, 1)
			log.Error("connector stopped", "err", err)
		}
	}()

	venue := c.Capabilities().Venue
	for raw := range raws {
		envs, unresolved, err := norm.Normalize(raw)
		if err != nil {
			counters.Add("errors."+id, 1)
			log.Warn("normalize", "err", err)
			continue
		}
		counters.Add("unresolved."+id, float64(len(unresolved)))
		for _, e := range envs {
			subj, err := bus.SubjectFor(e, venue)
			if err != nil {
				counters.Add("errors."+id, 1)
				continue
			}
			if err := pub.Publish(subj, e); err != nil {
				counters.Add("errors."+id, 1)
				log.Warn("publish", "subject", subj, "err", err)
				continue
			}
			counters.Add("events."+id, 1)
		}
	}
}

func pollLoop(ctx context.Context, p connector.Poller, out chan<- connector.RawMessage) error {
	t := time.NewTicker(p.Schedule().Interval)
	defer t.Stop()
	since := time.Now().UTC()
	for {
		msgs, err := p.Poll(ctx, connector.PollRequest{Since: since})
		if err != nil {
			return err
		}
		since = time.Now().UTC()
		for _, m := range msgs {
			select {
			case out <- m:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
		}
	}
}
