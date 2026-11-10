// Command ingestor runs connectors concurrently, archives raw payloads to
// Garage, normalizes their output to canonical envelopes, publishes them on
// NATS and writes market data to VictoriaMetrics (spec 10, spec 20 §5.3).
// With -postgres it resolves symbols through ref.* (provisioning unknown ones),
// records connector runs and persists cursors (go/pkg/refdata).
//
// Control subjects (request/reply, CtlRequest/CtlReply):
//
//	ctl.ingest.start    args: source   start a catalog source's connector
//	ctl.ingest.stop     args: source   stop it
//	ctl.ingest.enable   args: source   status → live (persisted) and start
//	ctl.ingest.disable  args: source   status → disabled (persisted) and stop
//	ctl.ingest.backfill args: source, timescale, start, end, instruments (comma list)
//	                    enqueues a jobs.ingest.backfill Job; reply message = job id
//	ctl.ingest.status   (no args)      reply message = JSON {running, health, cursors, ...}
//
// Health (JSON, D-054): ingest.health.<source> every 5 s per running connector,
// ingest.gap.<source> on bar gaps, ingest.unresolved.<source> on provisioned or
// unresolvable symbols; risk.halt.source:<id> / risk.resume.source:<id> when a
// source goes down unexpectedly and recovers.
//
// Error replies carry a prefix control-api maps to HTTP: `not_found:` (404),
// `conflict:` (409, already running / not running / status disabled|planned),
// `invalid:` (400), anything else 422.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"sync"
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
	metricsBind := flag.String("metrics", bus.MetricsBindDefault(service), "Prometheus GET /metrics bind (FA_METRICS_BIND; off disables)")
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
	go bus.ServeMetrics(ctx, *metricsBind, service, "0.1.0", counters, log) // spec 60 §4 metrics endpoint (D-056)

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
	health := NewTracker(pub, func(msg string, args ...any) { log.Info(msg, args...) })
	go health.Run(ctx, HealthInterval)

	var resolver connector.SymbolResolver = connector.NewStaticResolver()
	var cursors connector.CursorStore = connector.NewMemCursors()
	var backfillCursors connector.BackfillCursorStore = connector.NewMemBackfillCursors()
	var runLog RunLogger
	var statuses StatusStore
	var store *refdata.Store
	var seedGaps func(context.Context, string) map[string]time.Time
	if *pgURL != "" {
		store, err = refdata.Open(ctx, *pgURL)
		if err != nil {
			log.Error("postgres", "err", err)
			os.Exit(1)
		}
		defer store.Close()
		// YAML seeds new rows; the Postgres status wins for existing ones (spec 11 §8, D-054).
		if err := store.SyncCatalog(ctx, cat); err != nil {
			log.Error("catalog sync", "err", err)
			os.Exit(1)
		}
		if st, err := store.Statuses(ctx); err != nil {
			log.Warn("catalog statuses", "err", err)
		} else if changed := cat.ApplyStatuses(st); len(changed) > 0 {
			log.Info("catalog statuses from postgres override the yaml seed", "sources", changed)
		}
		r := refdata.NewResolver(store, nil, func(err error) { counters.Add("refdata.errors", 1); log.Warn("refdata", "err", err) })
		r.OnProvision = func(source, symbol, id string) {
			counters.Add("provisioned."+source, 1)
			health.PublishUnresolved(UnresolvedEvent{Source: source, VenueSymbol: symbol, InstrumentID: id, Reason: "provisioned as provisional instrument; review in quarantine", At: time.Now().UTC()})
		}
		resolver = r
		cursors = refdata.NewCursors(store)
		backfillCursors = refdata.NewBackfillCursors(store)
		runLog = refdata.NewRunLog(store)
		statuses = store
		seedGaps = func(ctx context.Context, source string) map[string]time.Time {
			rows, err := store.ListCursors(ctx, source, 1000)
			if err != nil {
				log.Warn("gap seed", "source", source, "err", err)
				return nil
			}
			out := map[string]time.Time{}
			for _, r := range rows {
				if r.Live && r.InstrumentID != "" && r.Cursor != nil {
					out[GapKey(r.InstrumentID, r.Timescale)] = *r.Cursor
				}
			}
			return out
		}
	}

	// One token bucket per source, shared by the live connector and backfill
	// jobs so the catalog limit holds across both (spec 10 §6.4).
	var limMu sync.Mutex
	limiters := map[string]*connector.MeteredLimiter{}
	limiterFor := func(src catalog.Source) *connector.MeteredLimiter {
		limMu.Lock()
		defer limMu.Unlock()
		if l, ok := limiters[src.ID]; ok {
			return l
		}
		l := connector.NewMeteredLimiter(connector.NewLimiter(src.Limit()))
		limiters[src.ID] = l
		return l
	}
	deps := func(src catalog.Source) connector.Deps {
		return connector.Deps{
			Limiter:  limiterFor(src),
			Resolver: resolver,
			Archive:  arch,
			Clock:    connector.SystemClock{},
			Cursors:  cursors,
		}
	}
	pipe := &Pipeline{Pub: pub, Writer: writer, Archive: arch, Counters: counters, Health: health, SeedGaps: seedGaps}
	sup := NewSupervisor(cat, log, counters, deps, pipe.Run)
	sup.Secrets = secrets.ForSource
	sup.RunLog = runLog
	sup.Health = health
	sup.Statuses = statuses

	backfiller, err := NewBackfiller(ctx, pub.NC, sup, pipe, backfillCursors, runLog, log.With("worker", "backfill"), counters, instance)
	if err != nil {
		log.Warn("backfill work queue unavailable (jobs.ingest.backfill disabled)", "err", err)
	} else {
		go backfiller.Run(ctx)
	}
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
	handle("enable", func(r *fav1.CtlRequest) (string, error) {
		id := r.Args["source"]
		started, err := sup.Enable(ctx, id)
		if err != nil {
			return "", err
		}
		if started {
			return "enabled and started " + id, nil
		}
		return "enabled " + id + " (already running)", nil
	})
	handle("disable", func(r *fav1.CtlRequest) (string, error) {
		id := r.Args["source"]
		stopped, err := sup.Disable(ctx, id)
		if err != nil {
			return "", err
		}
		if stopped {
			return "disabled and stopped " + id, nil
		}
		return "disabled " + id, nil
	})
	handle("backfill", func(r *fav1.CtlRequest) (string, error) {
		if backfiller == nil {
			return "", fmt.Errorf("%w: backfill work queue unavailable (JetStream)", ErrConflict)
		}
		spec := BackfillSpec{Source: r.Args["source"], Timescale: r.Args["timescale"], Start: r.Args["start"], End: r.Args["end"]}
		if inst := strings.TrimSpace(r.Args["instruments"]); inst != "" {
			spec.Instruments = strings.Split(inst, ",")
		}
		raw, _ := json.Marshal(spec)
		return backfiller.Enqueue(ctx, raw)
	})
	handle("status", func(*fav1.CtlRequest) (string, error) {
		st := map[string]any{"running": sup.Running(), "connectors": connector.Registered(), "health": health.Frames()}
		if store != nil {
			sctx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()
			if rows, err := store.ListCursors(sctx, "", 200); err == nil {
				st["cursors"] = rows
			}
		}
		limMu.Lock()
		lim := map[string]connector.LimiterStats{}
		for id, l := range limiters {
			lim[id] = l.Snapshot(false)
		}
		limMu.Unlock()
		st["limiters"] = lim
		b, err := json.Marshal(st)
		if err != nil {
			return "", err
		}
		return string(b), nil
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
