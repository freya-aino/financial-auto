package bus

import (
	"bytes"
	"context"
	"os"
	"testing"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
)

// Mirrors crates/fa-bus/src/streams.rs tests.
func TestStreamTableMatchesSpec21(t *testing.T) {
	tb := Table()
	want := []string{"MD", "TEXT", "MEDIA", "MACRO", "LATENT", "SIGNAL", "ORDERS", "JOB_RESULTS", "DLQ"}
	if len(tb.Streams) != len(want) {
		t.Fatalf("streams = %d", len(tb.Streams))
	}
	by := map[string]StreamSpec{}
	for i, s := range tb.Streams {
		if s.Name != want[i] {
			t.Errorf("stream %d = %s, want %s", i, s.Name, want[i])
		}
		by[s.Name] = s
	}
	if by["MD"].MaxAgeHours != 24 || by["ORDERS"].MaxAgeHours != 365*24 || by["MACRO"].MaxAgeHours != 30*24 {
		t.Errorf("max ages: %+v", by)
	}
	if by["ORDERS"].Storage != "file" || by["ORDERS"].Config().Storage != jetstream.FileStorage {
		t.Errorf("ORDERS storage: %+v", by["ORDERS"])
	}
	for _, sp := range tb.Streams {
		if sp.MaxMiB < 1024 {
			t.Errorf("%s is not capped: %d MiB", sp.Name, sp.MaxMiB)
		}
	}
	if by["LATENT"].Config().MaxBytes != 24<<30 {
		t.Errorf("LATENT cap: %d", by["LATENT"].Config().MaxBytes)
	}
	if safeMaxBytes(-1, 100, 50) != 100 || safeMaxBytes(-1, 100, 150) != -1 || safeMaxBytes(200, -1, 150) != -1 {
		t.Error("safeMaxBytes must never purge")
	}
	if tb.WorkQueue != (WorkQueueSpec{AckWaitSecs: 60, MaxDeliver: 5}) {
		t.Errorf("work queue: %+v", tb.WorkQueue)
	}
	if c := by["MD"].Config(); c.MaxAge != 24*time.Hour || c.Retention != jetstream.LimitsPolicy {
		t.Errorf("MD config: %+v", c)
	}
	if WorkQueueName("evo") != "JOBS_EVO" || WorkQueueSubject("evo") != "jobs.evo.>" || WorkQueueName("media.asr") != "JOBS_MEDIA_ASR" {
		t.Error("work queue naming")
	}
	if c := WorkQueueConsumer("evo-worker"); c.AckWait != 60*time.Second || c.MaxDeliver != 5 {
		t.Errorf("consumer: %+v", c)
	}
}

// The embedded copy must be identical to the Rust original when the repo is available
// (the Docker test mount is go/ only, so it is skipped there; cargo test checks the same).
func TestStreamsFixtureMatchesRust(t *testing.T) {
	orig, err := os.ReadFile("../../../crates/fa-bus/streams.json")
	if err != nil {
		t.Skip("crates/fa-bus/streams.json not reachable:", err)
	}
	if !bytes.Equal(orig, streamsJSON) {
		t.Fatal("go/pkg/bus/streams.json differs from crates/fa-bus/streams.json; copy it over")
	}
}

func TestNoStreamSubjectsOverlap(t *testing.T) {
	type f struct{ stream, filter string }
	var filters []f
	for _, s := range Table().Streams {
		for _, sub := range s.Subjects {
			filters = append(filters, f{s.Name, sub})
		}
	}
	for _, k := range []string{"evo", "mcmc", "ingest", "train"} {
		filters = append(filters, f{WorkQueueName(k), WorkQueueSubject(k)})
	}
	for i, a := range filters {
		for _, b := range filters[i+1:] {
			if a.stream != b.stream && SubjectsOverlap(a.filter, b.filter) {
				t.Errorf("%s %q overlaps %s %q", a.stream, a.filter, b.stream, b.filter)
			}
		}
	}
	cases := map[string]string{
		"md.trade.sim.0190":            "MD",
		"news.item.hn":                 "TEXT",
		"risk.halt.portfolio":          "ORDERS",
		"jobs.result.evo.eval.decoder": "JOB_RESULTS",
		"jobs.evo.eval.decoder":        "JOBS_EVO",
		"jobs.mcmc.refit":              "JOBS_MCMC",
		"dlq.LATENT.latent.state.1m.x": "DLQ",
		"ops.alert":                    "",
		"jobs.progress.01J":            "",
		"svc.heartbeat.ingestor":       "",
	}
	for subject, want := range cases {
		got := ""
		for _, fl := range filters {
			if SubjectsOverlap(fl.filter, subject) {
				if got != "" {
					t.Errorf("%s captured by %s and %s", subject, got, fl.stream)
				}
				got = fl.stream
			}
		}
		if got != want {
			t.Errorf("%s → %q, want %q", subject, got, want)
		}
	}
	if SubjectsOverlap("md.>", "md") || !SubjectsOverlap("md.*.sim.>", "md.>") {
		t.Error("overlap predicate")
	}
}

// Against the compose NATS: FA_NATS_TEST=1 go test ./pkg/bus -run Live -v
func TestLiveEnsureStreams(t *testing.T) {
	if os.Getenv("FA_NATS_TEST") == "" {
		t.Skip("set FA_NATS_TEST=1")
	}
	url := os.Getenv("FA_NATS_URL")
	if url == "" {
		url = nats.DefaultURL
	}
	nc, err := nats.Connect(url)
	if err != nil {
		t.Fatal(err)
	}
	defer nc.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	first, err := EnsureStreamsConn(ctx, nc)
	if err != nil {
		t.Fatal(err)
	}
	second, err := EnsureStreamsConn(ctx, nc)
	if err != nil || len(second) != 0 {
		t.Fatalf("second ensure: changed=%v err=%v", second, err)
	}
	t.Logf("first run changed %v, second run changed nothing", first)
	js, _ := jetstream.New(nc)
	s, c, err := EnsureWorkQueue(ctx, js, "gobustest", "go-bus-test")
	if err != nil {
		t.Fatal(err)
	}
	if s.CachedInfo().Config.Retention != jetstream.WorkQueuePolicy || c.CachedInfo().Config.AckWait != 60*time.Second {
		t.Errorf("work queue: %+v %+v", s.CachedInfo().Config, c.CachedInfo().Config)
	}
	_ = js.DeleteStream(ctx, "JOBS_GOBUSTEST")
}
