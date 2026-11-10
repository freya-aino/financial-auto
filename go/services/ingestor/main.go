// Command ingestor runs connectors concurrently, archives raw payloads to
// Garage, normalizes their output to canonical envelopes, publishes them on
// NATS and writes market data to VictoriaMetrics (spec 10, spec 20 §5.3).
// With -postgres it resolves symbols through ref.* (provisioning unknown ones),
// records connector runs and persists cursors (go/pkg/refdata).
//
// Control subjects (request/reply, CtlRequest/CtlReply):
//
//	ctl.ingest.start   args: source   start a catalog source's connector
//	ctl.ingest.stop    args: source   stop it
//	ctl.ingest.status  (no args)      reply message = comma list of running sources
//
// Error replies carry a prefix control-api maps to HTTP: `not_found:` (404),
// `conflict:` (409, already running / not running / status disabled|planned),
// anything else 422.
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/proto"

	"github.com/fey/financial-auto/go/pkg/archive"
	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
	"github.com/fey/financial-auto/go/pkg/refdata"
	"github.com/fey/financial-auto/go/pkg/tsdb"

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
	enabled := flag.String("connectors", env("FA_CONNECTORS", heartbeat.ID), "comma-separated connector ids to run at startup")
	tsdbURL := flag.String("tsdb-url", env("FA_TSDB_URL", "http://127.0.0.1:8428"), "VictoriaMetrics base URL for market-data samples (empty disables)")
	pgURL := flag.String("postgres", env("FA_POSTGRES_URL", ""), "Postgres URL for symbol resolution, cursors and run log (empty: in-memory StaticResolver)")
	garageURL := flag.String("garage", env("FA_GARAGE_URL", ""), "Garage S3 endpoint for the raw archive, e.g. http://127.0.0.1:3900 (empty disables)")
	secretsPath := flag.String("secrets", env("FA_SECRETS", "config/secrets.toml"), "secrets file (D-020); FA_<SOURCE>_<KEY> env vars override")
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
	if _, err := bus.EnsureStreamsConn(ctx, pub.NC); err != nil {
		log.Warn("jetstream streams", "err", err) // spec 21 §5; publishing works without them
	}

	counters := bus.NewCounters()
	instance := ulid.Make().String()
	go bus.RunHeartbeat(ctx, pub, service, instance, "0.1.0", *hbEvery, counters, func(err error) { log.Warn("heartbeat", "err", err) })

	var writer *tsdb.Writer
	if *tsdbURL != "" {
		writer = tsdb.New(*tsdbURL, tsdb.Options{})
		go writer.Run(ctx, func(err error) { log.Warn("tsdb", "err", err) })
		go func() { // mirror writer stats into heartbeat counters
			t := time.NewTicker(time.Second)
			defer t.Stop()
			for {
				st := writer.Stats()
				counters.Set("tsdb.written", float64(st.Written))
				counters.Set("tsdb.errors", float64(st.Errors))
				counters.Set("tsdb.dropped", float64(st.Dropped))
				select {
				case <-ctx.Done():
					return
				case <-t.C:
				}
			}
		}()
	}

	secrets, err := connector.LoadSecrets(*secretsPath)
	if err != nil {
		log.Error("secrets", "err", err)
		os.Exit(1)
	}

	// Raw archive: Garage when an endpoint and credentials exist, else discard.
	var arch connector.RawArchive = connector.NopArchive{}
	var archWriter *archive.Writer
	if *garageURL != "" {
		st, err := archive.NewS3(archive.Options{Endpoint: *garageURL, AccessKey: secrets.Get("garage", "access_key_id"), SecretKey: secrets.Get("garage", "secret_access_key")})
		if err != nil {
			log.Error("garage", "err", err)
			os.Exit(1)
		}
		archWriter = archive.NewWriter(st, archive.WriterOptions{})
		go archWriter.Run(ctx, func(err error) { log.Warn("archive", "err", err) })
		arch = archWriter
	}

	// Reference data: Postgres when configured, else the in-memory resolver
	// (which only knows what connectors add themselves — nothing, so every
	// market symbol would be unresolved; the heartbeat connector needs none).
	var resolver connector.SymbolResolver = connector.NewStaticResolver()
	var cursors connector.CursorStore = connector.NewMemCursors()
	var runLog RunLogger
	var store *refdata.Store
	if *pgURL != "" {
		store, err = refdata.Open(ctx, *pgURL)
		if err != nil {
			log.Error("postgres", "err", err)
			os.Exit(1)
		}
		defer store.Close()
		if err := store.SyncCatalog(ctx, cat); err != nil {
			log.Error("catalog sync", "err", err)
			os.Exit(1)
		}
		resolver = refdata.NewResolver(store, nil, func(err error) { counters.Add("refdata.errors", 1); log.Warn("refdata", "err", err) })
		cursors = refdata.NewCursors(store)
		runLog = refdata.NewRunLog(store)
	}

	deps := func(src catalog.Source) connector.Deps {
		return connector.Deps{
			Limiter:  connector.NewLimiter(src.Limit()),
			Resolver: resolver,
			Archive:  arch,
			Clock:    connector.SystemClock{},
			Cursors:  cursors,
		}
	}
	sup := NewSupervisor(cat, log, counters, deps, func(ctx context.Context, log *slog.Logger, c connector.Connector) {
		run(ctx, log, c, pub, writer, arch, counters)
	})
	sup.Secrets = secrets.ForSource
	sup.RunLog = runLog
	if archWriter != nil {
		go func() {
			t := time.NewTicker(time.Second)
			defer t.Stop()
			for {
				st := archWriter.Stats()
				counters.Set("archive.messages", float64(st.Messages))
				counters.Set("archive.objects", float64(st.Objects))
				counters.Set("archive.errors", float64(st.Errors))
				select {
				case <-ctx.Done():
					return
				case <-t.C:
				}
			}
		}()
	}

	for _, id := range strings.Split(*enabled, ",") {
		if id = strings.TrimSpace(id); id == "" {
			continue
		}
		if err := sup.Start(ctx, id); err != nil {
			log.Error("start", "id", id, "err", err)
		}
	}

	handle := func(verb string, fn func(*fav1.CtlRequest) (string, error)) {
		_, err := pub.NC.Subscribe(bus.Ctl("ingest", verb), func(m *nats.Msg) {
			var req fav1.CtlRequest
			reply := &fav1.CtlReply{}
			if err := proto.Unmarshal(m.Data, &req); err != nil {
				reply.Message = "invalid: decode: " + err.Error()
			} else if msg, err := fn(&req); err != nil {
				reply.Message = err.Error()
			} else {
				reply.Ok, reply.Message = true, msg
			}
			if b, err := proto.Marshal(reply); err == nil && m.Reply != "" {
				_ = m.Respond(b)
			}
		})
		if err != nil {
			log.Error("subscribe", "verb", verb, "err", err)
			os.Exit(1)
		}
	}
	handle("start", func(r *fav1.CtlRequest) (string, error) {
		id := r.Args["source"]
		if err := sup.Start(ctx, id); err != nil {
			return "", err
		}
		return "started " + id, nil
	})
	handle("stop", func(r *fav1.CtlRequest) (string, error) {
		id := r.Args["source"]
		if err := sup.Stop(id); err != nil {
			return "", err
		}
		return "stopped " + id, nil
	})
	handle("status", func(*fav1.CtlRequest) (string, error) {
		return strings.Join(sup.Running(), ","), nil
	})

	log.Info("started", "connectors", connector.Registered(), "running", sup.Running(), "tsdb", *tsdbURL, "postgres", *pgURL != "", "garage", *garageURL)
	<-ctx.Done()
	sup.Wait()
	if archWriter != nil {
		fctx, fcancel := context.WithTimeout(context.Background(), 10*time.Second)
		if err := archWriter.FlushSync(fctx); err != nil {
			log.Warn("archive flush", "err", err)
		}
		fcancel()
	}
	log.Info("stopped")
}

// run drives one connector: stream or poll, archive raw, normalize, publish, write to the TSDB.
func run(ctx context.Context, log *slog.Logger, c connector.Connector, pub *bus.Publisher, writer *tsdb.Writer, arch connector.RawArchive, counters *bus.Counters) {
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
		ref, err := arch.Put(ctx, raw)
		if err != nil {
			counters.Add("errors."+id, 1)
			log.Warn("archive", "err", err)
		}
		envs, unresolved, err := norm.Normalize(raw)
		if err != nil {
			counters.Add("errors."+id, 1)
			log.Warn("normalize", "err", err)
			continue
		}
		counters.Add("unresolved."+id, float64(len(unresolved)))
		for _, u := range unresolved {
			log.Warn("unresolved symbol", "symbol", u.VenueSymbol, "reason", u.Reason)
		}
		for _, e := range envs {
			if e.RawRef == nil && ref != nil {
				e.RawRef = ref
			}
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
			if writer != nil {
				writer.AddEnvelope(e, venue)
			}
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
