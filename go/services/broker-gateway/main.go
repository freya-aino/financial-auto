// Command broker-gateway executes approved order intents (spec 40). v1 runs
// the paper engine only; live venue adapters are stubbed behind VenueAdapter.
//
// Subjects: `order.approved.>` in → `order.event.<account>` out; `md.trade.>` and
// `md.quote.>` feed the marks, volatility, volume and spread behind the cost model
// (`costs.yaml`, `-costs` / `FA_COSTS`); `order.cancel.<account>` (`OrderCancel`; a legacy
// `OrderIntent` payload decodes identically by intent id) cancels a resting limit order or the
// remainder of a working algo order (algo.go). Live fills recalibrate the impact coefficient
// (calib.go) and the report goes out on `exec.costs.calibrated`. Open positions accrue the
// schedule's funding / borrow carry, charged per period on `exec.carry.<account>` (carry.go).
//
// Restart safety and fail-closed (state.go, spec 40 §8): working algo orders, resting limits,
// recently seen intent ids and per-account positions persist in `$FA_STATE_DIR/orders.json`;
// while `svc.heartbeat.risk-engine` is older than `-risk-heartbeat-timeout` (15 s) every
// risk-increasing intent is rejected.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io/fs"
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
	statePath := flag.String("orders-state", envOr("FA_ORDERS_STATE", defaultStatePath()), "working/resting orders + positions persisted across restarts (empty: in memory only)")
	riskTimeout := flag.Duration("risk-heartbeat-timeout", 15*time.Second, "fail-closed: reject risk-increasing intents while the risk-engine heartbeat is older than this (0 disables)")
	carryPeriod := flag.Duration("carry-period", time.Hour, "funding / borrow charging period on open paper positions, published on exec.carry.<account> (0 disables)")
	flag.Parse()
	log := bus.NewLogger(service) // FA_LOG_FORMAT=json|text (spec 60 §4)
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
			// only the fitted impact coefficients carry over: fee schedules, classes and
			// instrument mappings always come from the current costs.yaml
			costs = MergeCalibrated(base, m)
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
	engine.CarryPeriod = *carryPeriod
	guard := &RiskGuard{Timeout: *riskTimeout}
	engine.Guard = guard
	var store *StateStore
	if *statePath != "" {
		store = &StateStore{Path: *statePath}
		switch st, err := store.Load(); {
		case err != nil:
			log.Error("orders state unreadable; refusing to start with unknown working orders", "path", *statePath, "err", err)
			os.Exit(1)
		case st != nil:
			if err := engine.Restore(st); err != nil {
				log.Error("orders state", "path", *statePath, "err", err)
				os.Exit(1)
			}
			log.Info("orders state restored", "path", *statePath, "working", len(st.Working), "resting", len(st.Resting),
				"positions", len(st.Positions), "seen", len(st.Seen), "saved_at", st.SavedAt)
			counters.Set("state.restored_working", float64(len(st.Working)))
		}
	}
	if _, err := pub.NC.Subscribe(bus.Heartbeat("risk-engine"), func(*nats.Msg) { guard.Beat(time.Now()) }); err != nil {
		log.Error("subscribe risk heartbeat", "err", err)
		os.Exit(1)
	}
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
		save := time.NewTicker(time.Second)
		defer save.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				publish(engine.Tick())
				for _, ch := range engine.TickCarry() {
					if b, err := json.Marshal(ch); err == nil {
						if err := pub.NC.Publish(CarrySubjectPrefix+ch.Account, b); err != nil {
							counters.Add("carry.publish_errors", 1)
						}
					}
				}
				age := guard.Age(time.Now())
				if age > 1e9 {
					age = -1 // never seen
				}
				counters.Set("risk_heartbeat_age_s", age)
				alive := 0.0
				if guard.Alive(time.Now()) {
					alive = 1
				}
				counters.Set("risk_heartbeat_alive", alive)
			case <-save.C:
				if store != nil {
					if ok, err := store.SaveIfChanged(engine, false); err != nil {
						counters.Add("state.save_errors", 1)
						log.Warn("orders state not saved", "err", err)
					} else if ok {
						counters.Add("state.saves", 1)
					}
				}
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
	log.Info("started", "mode", "paper", "costs", *costsPath, "calibrated", *calibPath, "orders_state", *statePath,
		"risk_heartbeat_timeout", *riskTimeout)
	<-ctx.Done()
	if store != nil {
		if _, err := store.SaveIfChanged(engine, true); err != nil {
			log.Warn("orders state not saved at shutdown", "err", err)
		}
	}
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

// defaultStatePath: `$FA_STATE_DIR/orders.json` (empty without a state dir).
func defaultStatePath() string {
	if d := os.Getenv("FA_STATE_DIR"); d != "" {
		return filepath.Join(d, "orders.json")
	}
	return ""
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
