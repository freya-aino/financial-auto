package bus

import (
	"strings"
	"testing"
	"time"
)

func TestRenderMetricsFromCounters(t *testing.T) {
	c := NewCounters()
	c.Add("events.ok", 2)
	c.Add("events.ok", 1)
	c.Set("lag", 0.5)
	out := RenderMetrics("svc-x", "0.1.0", c)
	for _, want := range []string{
		"# TYPE fa_build_info gauge\nfa_build_info{service=\"svc-x\",version=\"0.1.0\"} 1\n",
		"# TYPE fa_heartbeat_counter gauge\n",
		"fa_heartbeat_counter{service=\"svc-x\",key=\"events.ok\"} 3\n",
		"fa_heartbeat_counter{service=\"svc-x\",key=\"lag\"} 0.5\n",
		"fa_process_uptime_seconds ",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in\n%s", want, out)
		}
	}
	// keys render sorted and nothing is emitted for consumers we don't watch
	if strings.Index(out, "key=\"events.ok\"") > strings.Index(out, "key=\"lag\"") {
		t.Errorf("keys not sorted:\n%s", out)
	}
	if strings.Contains(out, "fa_bus_consumer") {
		t.Errorf("unexpected consumer gauges:\n%s", out)
	}
}

func TestConsumerLagRendersPerConsumer(t *testing.T) {
	lagMu.Lock()
	lag[[2]string{"JOBS_INGEST", "ingestor"}] = consumerLag{pending: 4, ackPending: 1}
	lagMu.Unlock()
	defer func() { lagMu.Lock(); delete(lag, [2]string{"JOBS_INGEST", "ingestor"}); lagMu.Unlock() }()
	out := RenderMetrics("ingestor", "x", nil)
	if !strings.Contains(out, "fa_bus_consumer_num_pending{stream=\"JOBS_INGEST\",consumer=\"ingestor\"} 4\n") {
		t.Errorf("missing pending gauge:\n%s", out)
	}
	if !strings.Contains(out, "fa_bus_consumer_ack_pending{stream=\"JOBS_INGEST\",consumer=\"ingestor\"} 1\n") {
		t.Errorf("missing ack gauge:\n%s", out)
	}
}

func TestParseStatmAndFloats(t *testing.T) {
	rss, vsz, ok := parseStatm("1000 250 100 1 0 50 0\n", 4096)
	if !ok || rss != 250*4096 || vsz != 1000*4096 {
		t.Fatalf("parseStatm = %d %d %v", rss, vsz, ok)
	}
	if _, _, ok := parseStatm("garbage", 4096); ok {
		t.Fatal("garbage parsed")
	}
	if ff(1) != "1" || ff(2.5) != "2.5" {
		t.Fatalf("ff: %s %s", ff(1), ff(2.5))
	}
}

func TestMetricsBindDefault(t *testing.T) {
	t.Setenv("FA_METRICS_BIND", "")
	if got := MetricsBindDefault("ingestor"); got != "127.0.0.1:9105" {
		t.Fatalf("ingestor default = %s", got)
	}
	if got := MetricsBindDefault("unknown"); got != "127.0.0.1:9100" {
		t.Fatalf("unknown default = %s", got)
	}
	t.Setenv("FA_METRICS_BIND", "0.0.0.0:9100")
	if got := MetricsBindDefault("ingestor"); got != "0.0.0.0:9100" {
		t.Fatalf("env override = %s", got)
	}
}

func TestRenderMetricsHeartbeatStats(t *testing.T) {
	heartbeatPublished(time.Unix(1_800_000_000, 500_000_000))
	out := RenderMetrics("svc-hb", "x", nil)
	if !strings.Contains(out, `fa_heartbeat_last_unix_seconds{service="svc-hb"} 1.8000000005e+09`) {
		t.Fatalf("last heartbeat missing:\n%s", out)
	}
	if !strings.Contains(out, `fa_heartbeat_published_total{service="svc-hb"} `) {
		t.Fatalf("published counter missing:\n%s", out)
	}
}
