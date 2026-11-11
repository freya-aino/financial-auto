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
	if tb.WorkQueue != (WorkQueueSpec{AckWaitSecs: 60, MaxDeliver: 5, MaxAckPending: 64}) {
		t.Errorf("work queue: %+v", tb.WorkQueue)
	}
	if c := by["MD"].Config(); c.MaxAge != 24*time.Hour || c.Retention != jetstream.LimitsPolicy {
		t.Errorf("MD config: %+v", c)
	}
	if WorkQueueName("evo") != "JOBS_EVO" || WorkQueueSubject("evo") != "jobs.evo.>" || WorkQueueName("media.asr") != "JOBS_MEDIA_ASR" {
		t.Error("work queue naming")
	}
	if c := WorkQueueConsumer("evo-worker"); c.AckWait != 60*time.Second || c.MaxDeliver != 5 || c.MaxAckPending != 64 {
		t.Errorf("consumer: %+v", c)
	}
}

func TestSafeMaxAgeNeverShrinks(t *testing.T) {
	h := time.Hour
	cases := []struct{ have, want, out time.Duration }{
		{24 * h, 48 * h, 48 * h}, // growing is applied
		{24 * h, 24 * h, 24 * h}, // unchanged
		{48 * h, 24 * h, 48 * h}, // shrinking is held
		{0, 24 * h, 0},           // a finite age on an unlimited stream would purge: held
		{24 * h, 0, 0},           // removing the limit is fine
	}
	for _, c := range cases {
		if got := safeMaxAge(c.have, c.want); got != c.out {
			t.Errorf("safeMaxAge(%v, %v) = %v, want %v", c.have, c.want, got, c.out)
		}
	}
}

func TestMergeStreamConfigHoldsPurgingLimits(t *testing.T) {
	md := Table().Streams[0].Config() // MD: 24 h, 4 GiB
	// live MD was given a longer age and more room by an operator: config must not shrink it
	live := md
	live.MaxAge = 72 * time.Hour
	live.MaxBytes = md.MaxBytes * 2
	live.Duplicates = 2 * time.Minute // unmanaged server-side field
	cfg, update, held := mergeStreamConfig(live, uint64(md.MaxBytes)+1, md)
	if update {
		t.Fatalf("nothing but held limits differ, yet an update was planned: %+v", cfg)
	}
	if len(held) != 2 || held[0].Limit != "max_bytes" || held[1].Limit != "max_age" || held[1].Have != int64(72*time.Hour) {
		t.Fatalf("held: %+v", held)
	}

	// description drift plus a shorter configured age: update the description, keep the age
	live.Description = "drifted"
	cfg, update, held = mergeStreamConfig(live, 0, md)
	if !update || cfg.Description != md.Description || cfg.MaxAge != 72*time.Hour || cfg.Duplicates != 2*time.Minute {
		t.Fatalf("update=%v cfg=%+v", update, cfg)
	}
	if cfg.MaxBytes != md.MaxBytes { // a smaller cap above usage is fine
		t.Errorf("max_bytes = %d", cfg.MaxBytes)
	}
	if len(held) != 1 || held[0].Limit != "max_age" {
		t.Errorf("held: %+v", held)
	}

	// a longer configured age is applied
	live = md
	live.MaxAge = 12 * time.Hour
	cfg, update, held = mergeStreamConfig(live, 0, md)
	if !update || cfg.MaxAge != 24*time.Hour || len(held) != 0 {
		t.Errorf("grow: update=%v age=%v held=%+v", update, cfg.MaxAge, held)
	}

	// identical config: no update
	if _, update, held = mergeStreamConfig(md, 0, md); update || len(held) != 0 {
		t.Errorf("identical: update=%v held=%+v", update, held)
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
