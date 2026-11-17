// Command broker-gateway executes approved order intents (spec 40).
//
// One execution path (D-121): the paper engine and the Kraken / lemon.markets / OANDA adapters all
// implement `VenueAdapter`, and `Router` sends an intent to one of them by the intent's own
// `TradingMode`. A `LIVE` intent that cannot be routed — no venue, no adapter, no credentials, no
// symbol — is **rejected with the reason** and never filled on the paper book. `ctl.exec.venues`
// reports per-venue readiness, which is what the mode switch consults before it will flip.
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
	"fmt"
	"io/fs"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/proto"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/catalog"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const service = "broker-gateway"

// Subjects not yet in go/pkg/bus (D-051, provisional): quotes feed the spread, cancels pull
// resting limit orders.
const (
	allMdQuotes     = "md.quote.>"
	allOrderCancels = "order.cancel.>"
	// mdResumeWithin: market-data consumers idle longer than this skip their backlog.
	mdResumeWithin  = 2 * time.Minute
	costsCalibrated = "exec.costs.calibrated"
	// How long the venues get to answer `ctl.exec.positions`, comfortably inside the risk
	// engine's own request timeout so a slow venue is reported rather than timing the caller out.
	venueBooksTimeout = 8 * time.Second
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
	marketData := flag.String("market-data", envOr("FA_MARKET_DATA", "real"), "market data the paper engine fills against: real (default; synthetic venues ignored), synthetic (ephemeral stacks) or all (D-103)")
	secretsPath := flag.String("secrets", envOr("FA_SECRETS", "config/secrets.toml"), "secrets file holding [venues.<id>] trading credentials (D-121)")
	venueSymbols := flag.String("venue-symbols", envOr("FA_VENUE_SYMBOLS", ""), "JSON file mapping venue id -> instrument id -> venue symbol for live orders; without it a live intent is rejected for want of a symbol")
	livePoll := flag.Duration("live-poll", 2*time.Second, "how often open orders at live venues are polled for fills (0 disables live venues entirely)")
	flag.Parse()
	log := bus.NewLogger(service) // FA_LOG_FORMAT=json|text (spec 60 §4)
	mdMode, err := bus.ParseMarketDataMode(*marketData)
	if err != nil {
		log.Error("market data mode", "err", err)
		os.Exit(1)
	}
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
	// The one execution path (D-121): the paper engine and the live venues behind one router, which
	// routes by the intent's own mode. Adapters are registered even without credentials, so an
	// intent for an unconfigured venue is rejected by name instead of vanishing.
	router := NewRouter(engine)
	router.Counters = counters
	router.VenueOf = engine.VenueOf
	if *livePoll > 0 {
		secrets, err := connector.LoadSecrets(*secretsPath)
		if err != nil {
			log.Error("secrets", "path", *secretsPath, "err", err)
			os.Exit(1)
		}
		symbols := map[string]map[string]string{}
		if *venueSymbols != "" {
			if symbols, err = loadVenueSymbols(*venueSymbols); err != nil {
				log.Error("venue symbols", "path", *venueSymbols, "err", err)
				os.Exit(1)
			}
		}
		for _, id := range []string{krakenVenue, lemonVenue, oandaVenue} {
			creds := LoadVenueCreds(secrets, id)
			lim := connector.NewLimiter(catalog.DefaultRateLimit)
			var v LiveVenue
			switch id {
			case krakenVenue:
				k := NewKraken(creds, lim)
				k.Symbols = symbols[id]
				v = k
			case lemonVenue:
				l := NewLemon(creds, lim)
				l.Symbols = symbols[id]
				v = l
			case oandaVenue:
				o := NewOanda(creds, lim)
				o.Symbols = symbols[id]
				v = o
			}
			router.Add(v)
			ready := "configured"
			if err := v.Ready(); err != nil {
				ready = err.Error()
			}
			// The credentials themselves are never logged, only whether each one is present.
			log.Info("venue adapter", "venue", id, "state", ready, "creds", creds.Redacted(),
				"symbols", len(symbols[id]))
		}
	}
	var store *StateStore
	if *statePath != "" {
		store = &StateStore{Path: *statePath, Router: router}
		switch st, err := store.Load(); {
		case err != nil:
			log.Error("orders state unreadable; refusing to start with unknown working orders", "path", *statePath, "err", err)
			os.Exit(1)
		case st != nil:
			if err := engine.Restore(st); err != nil {
				log.Error("orders state", "path", *statePath, "err", err)
				os.Exit(1)
			}
			if orphaned := router.Restore(st.Live); len(orphaned) > 0 {
				// Loud: an open order at a venue this process no longer talks to is nobody's, and
				// only a human can decide what to do about it.
				log.Error("open live orders belong to venues that are not registered; they are not being tracked",
					"venues", strings.Join(orphaned, ", "))
			}
			log.Info("orders state restored", "path", *statePath, "working", len(st.Working), "resting", len(st.Resting),
				"positions", len(st.Positions), "seen", len(st.Seen), "live", len(st.Live), "saved_at", st.SavedAt)
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
		// Anything that did not come through the router is the paper engine's own (a resting order
		// filling on the tick, an algo slice), so this is where those get their venue. Only the
		// router produces live events, and it has already stamped them.
		for _, it := range stamp(PaperVenue, items) {
			if err := pub.Publish(bus.OrderEvent(it.Account), it.Event); err != nil {
				log.Warn("publish order event", "err", err)
				continue
			}
			counters.Add("events."+it.Event.Kind.String(), 1)
		}
		counters.Set("resting", float64(engine.Resting()))
		counters.Set("working", float64(engine.Working()))
	}
	// Durable JetStream consumers (spec 21 §5): approvals and cancels always resume after a
	// restart (Execute / CancelResting are idempotent through the restored `Seen` set); market
	// data resumes within mdResumeWithin, quotes coalesce to latest-only when behind (Q2).
	js, err := jetstream.New(pub.NC)
	if err != nil {
		log.Error("jetstream", "err", err)
		os.Exit(1)
	}
	if _, err := bus.EnsureStreams(ctx, js); err != nil { // idempotent; the gateway may start first
		log.Error("ensure streams", "err", err)
		os.Exit(1)
	}
	consume := func(spec bus.LiveSpec, handle func(subject string, data []byte)) {
		lc, err := bus.EnsureLive(ctx, js, spec)
		if err != nil {
			log.Error("consumer", "durable", spec.Durable, "err", err)
			os.Exit(1)
		}
		go bus.WatchConsumerLag(ctx, lc.Consumer)
		go lc.Run(ctx, handle)
		go func() { // consumer stats into the heartbeat counters
			t := time.NewTicker(5 * time.Second)
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C:
					counters.Set("consumer."+spec.Durable+".coalesced", float64(lc.Coalesced))
					counters.Set("consumer."+spec.Durable+".rejected_schema", float64(lc.RejectedSchema))
				}
			}
		}()
	}
	md := func(durable, filter string) bus.LiveSpec {
		s := bus.MarketDataLive("MD", durable, filter)
		s.ResumeWithin = mdResumeWithin
		return s
	}
	consume(md("broker-gateway-md-trade", bus.AllMdTrades), func(subject string, data []byte) {
		if !mdMode.AdmitsSubject(subject) {
			counters.Add("md_skipped", 1) // other market-data mode: never priced, never filled (D-103)
			return
		}
		var e fav1.Envelope
		if proto.Unmarshal(data, &e) != nil || e.GetTrade() == nil {
			return
		}
		var at time.Time
		if e.EventTime != nil {
			at = e.EventTime.AsTime()
		}
		publish(engine.ObserveTrade(subject, e.GetTrade().GetPrice(), e.GetTrade().GetSize(), at))
	})
	consume(md("broker-gateway-md-quote", allMdQuotes), func(subject string, data []byte) {
		if !mdMode.AdmitsSubject(subject) {
			counters.Add("md_skipped", 1)
			return
		}
		var e fav1.Envelope
		if proto.Unmarshal(data, &e) == nil && e.GetQuote() != nil {
			engine.ObserveQuote(subject, e.GetQuote().GetBidPrice(), e.GetQuote().GetAskPrice())
			counters.Add("quotes", 1)
		}
	})
	consume(bus.DurableLive("ORDERS", "broker-gateway-order-approved", bus.AllOrderApproved), func(_ string, data []byte) {
		var in fav1.OrderIntent
		if err := proto.Unmarshal(data, &in); err != nil {
			counters.Add("errors.decode", 1)
			return
		}
		out := router.Execute(ctx, &in)
		if len(out) == 0 {
			counters.Add("intents.duplicate", 1)
			return
		}
		publish(out)
	})
	consume(bus.DurableLive("ORDERS", "broker-gateway-order-cancel", allOrderCancels), func(_ string, data []byte) {
		in, err := decodeCancel(data)
		if err != nil {
			counters.Add("errors.decode", 1)
			return
		}
		if c := router.Cancel(ctx, in.IntentId); len(c) > 0 {
			publish(c)
		} else {
			// A live cancel is accepted by the venue and shows up as an event on the next poll,
			// so "nothing published" is not the same as "unknown" there.
			counters.Add("cancels.unknown", 1)
		}
	})
	// `ctl.exec.venues` answers "could this platform trade real money, and where?" — the question
	// the mode switch asks before it will flip (D-121). The reply carries the per-venue readiness
	// as JSON in `message`; it never carries a credential, only whether one is present.
	if _, err := pub.NC.Subscribe(bus.Ctl("exec", "venues"), func(m *nats.Msg) {
		var req fav1.CtlRequest
		_ = proto.Unmarshal(m.Data, &req)
		status, possible := router.Ready()
		body, _ := json.Marshal(map[string]any{
			"venues":        status,
			"live_possible": possible,
			"live_polling":  *livePoll > 0,
		})
		reply := &fav1.CtlReply{Ok: possible, Message: string(body)}
		_, l := bus.Consumed(ctx, log, m)
		l.Info("ctl", "subject", m.Subject, "live_possible", possible)
		if b, err := proto.Marshal(reply); err == nil && m.Reply != "" {
			_ = m.Respond(b)
		}
	}); err != nil {
		log.Error("subscribe venue status", "err", err)
		os.Exit(1)
	}

	// `ctl.exec.positions` answers "what does each venue say it holds?" — the question the live
	// book is reconciled against (D-121). The venues are asked when the question is put, not on a
	// timer: a reconciliation pass needs the venue's view *at its own cut*, and a cached one would
	// make a stale number look like a difference.
	if _, err := pub.NC.Subscribe(bus.Ctl("exec", "positions"), func(m *nats.Msg) {
		var req fav1.CtlRequest
		_ = proto.Unmarshal(m.Data, &req)
		rctx, cancel := context.WithTimeout(ctx, venueBooksTimeout)
		defer cancel()
		books := router.Books(rctx)
		ok := false
		for _, b := range books {
			if b.Configured && b.Error == "" {
				ok = true
			}
		}
		body, _ := json.Marshal(map[string]any{"venues": books, "asked_at": time.Now().UTC()})
		reply := &fav1.CtlReply{Ok: ok, Message: string(body)}
		_, l := bus.Consumed(ctx, log, m)
		l.Info("ctl", "subject", m.Subject, "venues", len(books), "answered", ok)
		if b, err := proto.Marshal(reply); err == nil && m.Reply != "" {
			_ = m.Respond(b)
		}
	}); err != nil {
		log.Error("subscribe venue positions", "err", err)
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
				if *livePoll > 0 {
					publish(router.Poll(ctx))
				}
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
	log.Info("started", "mode", "paper", "market_data", string(mdMode), "costs", *costsPath, "calibrated", *calibPath, "orders_state", *statePath,
		"risk_heartbeat_timeout", *riskTimeout)
	<-ctx.Done()
	if store != nil {
		if _, err := store.SaveIfChanged(engine, true); err != nil {
			log.Warn("orders state not saved at shutdown", "err", err)
		}
	}
	log.Info("stopped")
}

// loadVenueSymbols reads `{"kraken": {"<instrument-uuid>": "XBTEUR"}}`.
//
// A file rather than a lookup in the reference database, because this process must not need
// Postgres to execute an order (spec 40: the execution path stays on the bus and its own state),
// and because the mapping between an internal instrument and a venue's symbol is a decision an
// operator should make explicitly before real money moves. An instrument that is not in the file
// cannot be traded live at all — the order is rejected for want of a symbol, which is the safe
// direction.
func loadVenueSymbols(path string) (map[string]map[string]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out map[string]map[string]string
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	for venue, m := range out {
		for instrument, symbol := range m {
			if instrument == "" || symbol == "" {
				return nil, fmt.Errorf("%s: %s has an empty instrument or symbol", path, venue)
			}
		}
	}
	return out, nil
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
