package bus

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/nats-io/nats.go/jetstream"
)

// Prometheus text exporter fed by the heartbeat Counters (spec 60 §4/§6.5, D-056). Hand-rolled
// text format, no client_golang dependency: the Go services only have gauges to report.

var processStart = time.Now()

// Heartbeat publication stats, the Go side of fa_metrics::heartbeat::record: rendered as
// fa_heartbeat_published_total{service} / fa_heartbeat_last_unix_seconds{service} so the
// HeartbeatStale alert covers Go services too (D-056 parity).
var hbStats struct {
	sync.Mutex
	published uint64
	last      time.Time
}

func heartbeatPublished(at time.Time) {
	hbStats.Lock()
	hbStats.published++
	hbStats.last = at
	hbStats.Unlock()
}

// MetricsBindDefault returns FA_METRICS_BIND when set, else the host default port of the service
// (same table as fa_metrics::default_bind in Rust). "off" disables the endpoint.
func MetricsBindDefault(service string) string {
	if v := strings.TrimSpace(os.Getenv("FA_METRICS_BIND")); v != "" {
		return v
	}
	switch service {
	case "ingestor":
		return "127.0.0.1:9105"
	case "broker-gateway":
		return "127.0.0.1:9106"
	case "evo-orchestrator":
		return "127.0.0.1:9107"
	}
	return "127.0.0.1:9100"
}

// ServeMetrics serves GET /metrics on bind until ctx is done. Disabled for bind "off"/"0"/"".
// Errors are logged, never fatal: the service runs without metrics.
func ServeMetrics(ctx context.Context, bind, service, version string, counters *Counters, log *slog.Logger) {
	bind = strings.TrimSpace(bind)
	if bind == "" || bind == "off" || bind == "0" || bind == "none" || bind == "false" {
		return
	}
	if log == nil {
		log = slog.Default()
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		_, _ = w.Write([]byte(RenderMetrics(service, version, counters)))
	})
	mux.HandleFunc("GET /", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("fa-metrics: see /metrics\n")) })
	ln, err := net.Listen("tcp", bind)
	if err != nil {
		log.Warn("metrics endpoint not started", "bind", bind, "err", err)
		return
	}
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	log.Info("metrics endpoint listening", "bind", bind)
	if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
		log.Warn("metrics server stopped", "err", err)
	}
}

// RenderMetrics renders the heartbeat counters, process metrics and consumer lag gauges.
func RenderMetrics(service, version string, counters *Counters) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# HELP fa_build_info Service name and version\n# TYPE fa_build_info gauge\nfa_build_info{service=%q,version=%q} 1\n", service, version)
	fmt.Fprintf(&b, "# HELP fa_process_start_time_seconds Process start, unix seconds\n# TYPE fa_process_start_time_seconds gauge\nfa_process_start_time_seconds %s\n", ff(float64(processStart.UnixNano())/1e9))
	fmt.Fprintf(&b, "# HELP fa_process_uptime_seconds Seconds since process start\n# TYPE fa_process_uptime_seconds gauge\nfa_process_uptime_seconds %s\n", ff(time.Since(processStart).Seconds()))
	if rss, vsz, ok := memoryBytes(); ok {
		fmt.Fprintf(&b, "# HELP fa_process_resident_memory_bytes Resident set size from /proc/self/statm\n# TYPE fa_process_resident_memory_bytes gauge\nfa_process_resident_memory_bytes %d\n", rss)
		fmt.Fprintf(&b, "# HELP fa_process_virtual_memory_bytes Virtual memory size from /proc/self/statm\n# TYPE fa_process_virtual_memory_bytes gauge\nfa_process_virtual_memory_bytes %d\n", vsz)
	}
	if counters != nil {
		snap := counters.Snapshot()
		keys := make([]string, 0, len(snap))
		for k := range snap {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteString("# HELP fa_heartbeat_counter Heartbeat counter values mirrored per service and key\n# TYPE fa_heartbeat_counter gauge\n")
		for _, k := range keys {
			fmt.Fprintf(&b, "fa_heartbeat_counter{service=%q,key=%q} %s\n", service, k, ff(snap[k]))
		}
	}
	hbStats.Lock()
	published, last := hbStats.published, hbStats.last
	hbStats.Unlock()
	if published > 0 {
		fmt.Fprintf(&b, "# HELP fa_heartbeat_published_total Heartbeats published by this process\n# TYPE fa_heartbeat_published_total counter\nfa_heartbeat_published_total{service=%q} %d\n", service, published)
		fmt.Fprintf(&b, "# HELP fa_heartbeat_last_unix_seconds Unix time of the last heartbeat published\n# TYPE fa_heartbeat_last_unix_seconds gauge\nfa_heartbeat_last_unix_seconds{service=%q} %s\n", service, ff(float64(last.UnixNano())/1e9))
	}
	renderConsumerLag(&b)
	return b.String()
}

func ff(v float64) string {
	switch {
	case v != v:
		return "NaN"
	case v > 1.7976931348623157e308:
		return "+Inf"
	case v < -1.7976931348623157e308:
		return "-Inf"
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}

// memoryBytes reads /proc/self/statm: (resident, virtual) bytes.
func memoryBytes() (rss, vsz uint64, ok bool) {
	data, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0, 0, false
	}
	return parseStatm(string(data), uint64(os.Getpagesize()))
}

func parseStatm(s string, page uint64) (rss, vsz uint64, ok bool) {
	f := strings.Fields(s)
	if len(f) < 2 {
		return 0, 0, false
	}
	size, err1 := strconv.ParseUint(f[0], 10, 64)
	res, err2 := strconv.ParseUint(f[1], 10, 64)
	if err1 != nil || err2 != nil {
		return 0, 0, false
	}
	return res * page, size * page, true
}

// ── consumer lag (spec 60 §6.5 "bus consumer lag") ──────────────────────

type consumerLag struct {
	pending, ackPending, redelivered, waiting float64
}

var (
	lagMu sync.Mutex
	lag   = map[[2]string]consumerLag{} // (stream, consumer) → last info
)

// ConsumerLagPoll is how often WatchConsumerLag refreshes a consumer's gauges.
const ConsumerLagPoll = 10 * time.Second

// WatchConsumerLag polls c.Info() until ctx is done and exports
// fa_bus_consumer_{num_pending,ack_pending,redelivered,waiting}{stream,consumer}.
// EnsureWorkQueue starts it for every consumer it creates. Ten consecutive failures drop the series.
func WatchConsumerLag(ctx context.Context, c jetstream.Consumer) {
	ci := c.CachedInfo()
	if ci == nil {
		return
	}
	key := [2]string{ci.Stream, ci.Name}
	lagMu.Lock()
	_, running := lag[key]
	if !running {
		lag[key] = consumerLag{pending: float64(ci.NumPending), ackPending: float64(ci.NumAckPending), redelivered: float64(ci.NumRedelivered), waiting: float64(ci.NumWaiting)}
	}
	lagMu.Unlock()
	if running {
		return
	}
	t := time.NewTicker(ConsumerLagPoll)
	defer t.Stop()
	failures := 0
	for {
		select {
		case <-ctx.Done():
			lagMu.Lock()
			delete(lag, key)
			lagMu.Unlock()
			return
		case <-t.C:
		}
		ictx, cancel := context.WithTimeout(ctx, 5*time.Second)
		info, err := c.Info(ictx)
		cancel()
		if err != nil {
			failures++
			if failures >= 10 {
				lagMu.Lock()
				delete(lag, key)
				lagMu.Unlock()
				return
			}
			continue
		}
		failures = 0
		lagMu.Lock()
		lag[key] = consumerLag{pending: float64(info.NumPending), ackPending: float64(info.NumAckPending), redelivered: float64(info.NumRedelivered), waiting: float64(info.NumWaiting)}
		lagMu.Unlock()
	}
}

func renderConsumerLag(b *strings.Builder) {
	lagMu.Lock()
	defer lagMu.Unlock()
	if len(lag) == 0 {
		return
	}
	keys := make([][2]string, 0, len(lag))
	for k := range lag {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i][0]+"/"+keys[i][1] < keys[j][0]+"/"+keys[j][1] })
	write := func(name, help string, pick func(consumerLag) float64) {
		fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s gauge\n", name, help, name)
		for _, k := range keys {
			fmt.Fprintf(b, "%s{stream=%q,consumer=%q} %s\n", name, k[0], k[1], ff(pick(lag[k])))
		}
	}
	write("fa_bus_consumer_num_pending", "JetStream messages not yet delivered to the consumer", func(l consumerLag) float64 { return l.pending })
	write("fa_bus_consumer_ack_pending", "JetStream messages delivered and awaiting ack", func(l consumerLag) float64 { return l.ackPending })
	write("fa_bus_consumer_redelivered", "JetStream messages redelivered to the consumer", func(l consumerLag) float64 { return l.redelivered })
	write("fa_bus_consumer_waiting", "Pull requests waiting on the consumer", func(l consumerLag) float64 { return l.waiting })
}
