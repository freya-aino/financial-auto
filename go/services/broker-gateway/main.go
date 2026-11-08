// Command broker-gateway executes approved order intents (spec 40). v1 runs
// the paper engine only; live venue adapters are stubbed behind VenueAdapter.
package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/proto"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

const service = "broker-gateway"

func main() {
	natsURL := flag.String("nats", envOr("FA_NATS_URL", "nats://127.0.0.1:4222"), "NATS URL")
	feeBps := flag.Float64("fee-bps", 0, "paper fee in basis points of notional (0 = zero-fee default, spec 41 cost model hook)")
	feeMin := flag.Float64("fee-min", 0, "minimum paper fee per fill (EUR)")
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil)).With("svc", service)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pub, err := bus.Connect(*natsURL, service)
	if err != nil {
		log.Error("nats connect", "err", err)
		os.Exit(1)
	}
	defer pub.Close()

	counters := bus.NewCounters()
	go bus.RunHeartbeat(ctx, pub, service, ulid.Make().String(), "0.1.0", 5*time.Second, counters, nil)

	engine := NewPaperEngine()
	if *feeBps > 0 || *feeMin > 0 {
		engine.Fees = BpsFee{Bps: *feeBps, Min: *feeMin}
	}
	if _, err := pub.NC.Subscribe(bus.AllMdTrades, func(m *nats.Msg) {
		var e fav1.Envelope
		if proto.Unmarshal(m.Data, &e) == nil {
			engine.ObserveTrade(m.Subject, e.GetTrade().GetPrice())
		}
	}); err != nil {
		log.Error("subscribe trades", "err", err)
		os.Exit(1)
	}
	if _, err := pub.NC.Subscribe(bus.AllOrderApproved, func(m *nats.Msg) {
		var in fav1.OrderIntent
		if err := proto.Unmarshal(m.Data, &in); err != nil {
			counters.Add("errors.decode", 1)
			return
		}
		ev := engine.Execute(&in)
		if ev == nil {
			counters.Add("intents.duplicate", 1)
			return
		}
		if err := pub.Publish(bus.OrderEvent(in.AccountId), ev); err != nil {
			log.Warn("publish order event", "err", err)
			return
		}
		counters.Add("events."+ev.Kind.String(), 1)
	}); err != nil {
		log.Error("subscribe intents", "err", err)
		os.Exit(1)
	}
	log.Info("started", "mode", "paper")
	<-ctx.Done()
	log.Info("stopped")
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
