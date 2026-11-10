// Command broker-gateway executes approved order intents (spec 40). v1 runs
// the paper engine only; live venue adapters are stubbed behind VenueAdapter.
//
// Subjects: `order.approved.>` in → `order.event.<account>` out; `md.trade.>` and
// `md.quote.>` feed the marks, volatility, volume and spread behind the cost model
// (`costs.yaml`, `-costs` / `FA_COSTS`); `order.cancel.<account>` (payload: an
// `OrderIntent` carrying the intent id — provisional, there is no cancel proto yet)
// cancels a resting limit order.
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

// Subjects not yet in go/pkg/bus (D-051, provisional): quotes feed the spread, cancels pull
// resting limit orders.
const (
	allMdQuotes     = "md.quote.>"
	allOrderCancels = "order.cancel.>"
)

func main() {
	natsURL := flag.String("nats", envOr("FA_NATS_URL", "nats://127.0.0.1:4222"), "NATS URL")
	costsPath := flag.String("costs", envOr("FA_COSTS", ""), "cost + slippage model YAML (spec 41 §6.3); built-in defaults when empty")
	metricsBind := flag.String("metrics", bus.MetricsBindDefault(service), "Prometheus GET /metrics bind (FA_METRICS_BIND; off disables)")
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil)).With("svc", service)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	costs := DefaultCostModel()
	if *costsPath != "" {
		m, err := LoadCostModel(*costsPath)
		if err != nil {
			log.Error("load costs", "err", err)
			os.Exit(1)
		}
		costs = m
	}

	pub, err := bus.Connect(*natsURL, service)
	if err != nil {
		log.Error("nats connect", "err", err)
		os.Exit(1)
	}
	defer pub.Close()

	counters := bus.NewCounters()
	go bus.RunHeartbeat(ctx, pub, service, ulid.Make().String(), "0.2.0", 5*time.Second, counters, nil)
	go bus.ServeMetrics(ctx, *metricsBind, service, "0.2.0", counters, log) // spec 60 §4 metrics endpoint (D-056)

	engine := NewPaperEngine(costs)
	publish := func(items []Published) {
		for _, it := range items {
			if err := pub.Publish(bus.OrderEvent(it.Account), it.Event); err != nil {
				log.Warn("publish order event", "err", err)
				continue
			}
			counters.Add("events."+it.Event.Kind.String(), 1)
		}
		counters.Set("resting", float64(engine.Resting()))
	}
	if _, err := pub.NC.Subscribe(bus.AllMdTrades, func(m *nats.Msg) {
		var e fav1.Envelope
		if proto.Unmarshal(m.Data, &e) != nil || e.GetTrade() == nil {
			return
		}
		var at time.Time
		if e.EventTime != nil {
			at = e.EventTime.AsTime()
		}
		publish(engine.ObserveTrade(m.Subject, e.GetTrade().GetPrice(), e.GetTrade().GetSize(), at))
	}); err != nil {
		log.Error("subscribe trades", "err", err)
		os.Exit(1)
	}
	if _, err := pub.NC.Subscribe(allMdQuotes, func(m *nats.Msg) {
		var e fav1.Envelope
		if proto.Unmarshal(m.Data, &e) == nil && e.GetQuote() != nil {
			engine.ObserveQuote(m.Subject, e.GetQuote().GetBidPrice(), e.GetQuote().GetAskPrice())
			counters.Add("quotes", 1)
		}
	}); err != nil {
		log.Error("subscribe quotes", "err", err)
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
		publish([]Published{{Account: in.AccountId, Event: ev}})
	}); err != nil {
		log.Error("subscribe intents", "err", err)
		os.Exit(1)
	}
	if _, err := pub.NC.Subscribe(allOrderCancels, func(m *nats.Msg) {
		var in fav1.OrderIntent
		if err := proto.Unmarshal(m.Data, &in); err != nil || in.IntentId == "" {
			counters.Add("errors.decode", 1)
			return
		}
		if c := engine.CancelResting(in.IntentId); c != nil {
			publish([]Published{*c})
		} else {
			counters.Add("cancels.unknown", 1)
		}
	}); err != nil {
		log.Error("subscribe cancels", "err", err)
		os.Exit(1)
	}
	log.Info("started", "mode", "paper", "costs", *costsPath)
	<-ctx.Done()
	log.Info("stopped")
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
