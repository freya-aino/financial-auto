// Command broker-gateway executes approved order intents (spec 40). v1 runs
// the paper engine only; live venue adapters are stubbed behind VenueAdapter.
//
// Subjects: `order.approved.>` in → `order.event.<account>` out; `md.trade.>` and
// `md.quote.>` feed the marks, volatility, volume and spread behind the cost model
// (`costs.yaml`, `-costs` / `FA_COSTS`); `order.cancel.<account>` (`OrderCancel`; a legacy
// `OrderIntent` payload decodes identically by intent id) cancels a resting limit order or the
// remainder of a working algo order (algo.go). Live fills recalibrate the impact coefficient
// (calib.go) and the report goes out on `exec.costs.calibrated`.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io/fs"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
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
	costsCalibrated = "exec.costs.calibrated"
)

func main() {
	natsURL := flag.String("nats", envOr("FA_NATS_URL", "nats://127.0.0.1:4222"), "NATS URL")
	costsPath := flag.String("costs", envOr("FA_COSTS", ""), "cost + slippage model YAML (spec 41 §6.3); built-in defaults when empty")
	calibPath := flag.String("costs-calibrated", envOr("FA_COSTS_CALIBRATED", defaultCalibPath()), "calibrated costs YAML written by the live recalibration and loaded at start (empty: not persisted)")
	calibEvery := flag.Duration("recalibrate-every", 15*time.Minute, "live slippage recalibration interval (0 disables)")
	calibApply := flag.Bool("recalibrate-apply", true, "apply the calibrated model in memory (paper)")
	calibMin := flag.Int("recalibrate-min-samples", 30, "informative fills per (venue, class) needed for a fit")
	tick := flag.Duration("algo-tick", time.Second, "algo scheduler tick (TWAP/VWAP slices, timeouts)")
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
	base := costs // the prior of the recalibration: never the calibrated file itself
	if *calibPath != "" {
		switch m, err := LoadCostModel(*calibPath); {
		case err == nil:
			costs = m
			log.Info("loaded calibrated costs", "path", *calibPath)
		case errors.Is(err, fs.ErrNotExist):
		default:
			log.Warn("calibrated costs ignored", "path", *calibPath, "err", err)
		}
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
	engine.Counters = counters
	calib := NewCalibrator()
	calib.MinSamples = *calibMin
	engine.Calib = calib
	publish := func(items []Published) {
		for _, it := range items {
			if err := pub.Publish(bus.OrderEvent(it.Account), it.Event); err != nil {
				log.Warn("publish order event", "err", err)
				continue
			}
			counters.Add("events."+it.Event.Kind.String(), 1)
		}
		counters.Set("resting", float64(engine.Resting()))
		counters.Set("working", float64(engine.Working()))
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
		in, err := decodeCancel(m.Data)
		if err != nil {
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
	go func() {
		t := time.NewTicker(*tick)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				publish(engine.Tick())
			}
		}
	}()
	if *calibEvery > 0 {
		go func() {
			t := time.NewTicker(*calibEvery)
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C:
					rep, m := calib.Calibrate(base, time.Now())
					if m != nil {
						if err := m.Validate(); err != nil {
							rep.Error = err.Error()
						} else {
							if *calibApply {
								engine.SetCosts(m)
								rep.Applied = true
							}
							if *calibPath != "" {
								if err := WriteCalibrated(*calibPath, m, rep); err != nil {
									rep.Error = err.Error()
								} else {
									rep.Path = *calibPath
								}
							}
						}
					}
					if m == nil && *calibPath != "" { // nothing calibrated: the report still records why
						if b, err := json.MarshalIndent(rep, "", "  "); err == nil {
							if err := writeAtomic(*calibPath+".report.json", b); err != nil {
								rep.Error = err.Error()
							}
						}
					}
					counters.Add("calibrations", 1)
					if b, err := json.Marshal(rep); err == nil {
						_ = pub.NC.Publish(costsCalibrated, b)
					}
					log.Info("costs recalibrated", "entries", len(rep.Entries), "applied", rep.Applied, "path", rep.Path, "err", rep.Error)
				}
			}
		}()
	}
	log.Info("started", "mode", "paper", "costs", *costsPath, "calibrated", *calibPath)
	<-ctx.Done()
	log.Info("stopped")
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// decodeCancel reads `order.cancel.<account>`: an `OrderCancel`, or a legacy `OrderIntent`
// (field 1 is the intent id in both).
func decodeCancel(b []byte) (*fav1.OrderCancel, error) {
	var c fav1.OrderCancel
	if err := (proto.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(b, &c); err != nil {
		return nil, err
	}
	if c.IntentId == "" {
		return nil, errors.New("cancel without intent_id")
	}
	return &c, nil
}

// defaultCalibPath: `$FA_STATE_DIR/costs.calibrated.yaml`, else next to the costs file.
func defaultCalibPath() string {
	if d := os.Getenv("FA_STATE_DIR"); d != "" {
		return filepath.Join(d, "costs.calibrated.yaml")
	}
	if c := os.Getenv("FA_COSTS"); c != "" {
		return filepath.Join(filepath.Dir(c), "costs.calibrated.yaml")
	}
	return ""
}
