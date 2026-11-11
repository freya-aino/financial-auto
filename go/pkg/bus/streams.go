package bus

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
)

// streamsJSON is the JetStream stream table of spec 21 §5. It is a verbatim copy of
// crates/fa-bus/streams.json (the Rust side embeds the original and its tests assert the two
// files are identical, so the tables cannot drift).
//
//go:embed streams.json
var streamsJSON []byte

// StreamSpec is one limits-retention stream.
type StreamSpec struct {
	Name        string   `json:"name"`
	Subjects    []string `json:"subjects"`
	MaxAgeHours uint64   `json:"max_age_hours"`
	MaxMiB      uint64   `json:"max_mib"` // disk cap (spec 21 §10.1); 0 = unlimited
	Storage     string   `json:"storage"`
	Description string   `json:"description"`
}

// WorkQueueSpec holds the work-queue consumer tunables (spec 21 §7 jobs.ack_wait).
type WorkQueueSpec struct {
	AckWaitSecs uint64 `json:"ack_wait_secs"`
	MaxDeliver  int    `json:"max_deliver"`
	// MaxAckPending caps jobs in flight across all workers of a kind (spec 21 §7: each worker
	// pulls one job at a time, so this is the most workers that can run concurrently).
	MaxAckPending int `json:"max_ack_pending"`
}

// StreamTable is the parsed fixture.
type StreamTable struct {
	Streams   []StreamSpec  `json:"streams"`
	WorkQueue WorkQueueSpec `json:"work_queue"`
}

// Table parses the embedded fixture. It panics only if the fixture is malformed (tested).
func Table() StreamTable {
	var t StreamTable
	if err := json.Unmarshal(streamsJSON, &t); err != nil {
		panic("go/pkg/bus/streams.json: " + err.Error())
	}
	return t
}

// Config is the JetStream config this spec provisions.
func (s StreamSpec) Config() jetstream.StreamConfig {
	storage := jetstream.FileStorage
	if s.Storage == "memory" {
		storage = jetstream.MemoryStorage
	}
	return jetstream.StreamConfig{
		Name:        s.Name,
		Description: s.Description,
		Subjects:    s.Subjects,
		Retention:   jetstream.LimitsPolicy,
		MaxAge:      time.Duration(s.MaxAgeHours) * time.Hour,
		MaxBytes:    s.MaxBytes(),
		Storage:     storage,
	}
}

// MaxBytes is max_mib in bytes, -1 (unlimited) when absent.
func (s StreamSpec) MaxBytes() int64 {
	if s.MaxMiB == 0 {
		return -1
	}
	return int64(s.MaxMiB) * 1024 * 1024
}

// safeMaxBytes never applies a cap below the bytes a stream already holds (JetStream would
// purge the oldest messages at once); the current cap stays. Mirrors fa_bus::streams.
func safeMaxBytes(have, want int64, used uint64) int64 {
	if want > 0 && used > uint64(want) {
		return have
	}
	return want
}

// safeMaxAge never lowers a live stream's max_age by config: JetStream drops every message
// older than the new limit as soon as the update lands (D-075 incident: MD lost ~13.4k
// messages to a shrunk max_age). A longer age, or 0 (unlimited), is applied; a shorter one
// (including a finite age on an unlimited stream) keeps the current value, so shortening
// retention is an operator action (`nats stream edit`), never a side effect of a service start.
func safeMaxAge(have, want time.Duration) time.Duration {
	if want > 0 && (have == 0 || want < have) {
		return have
	}
	return want
}

// LimitHeld reports a configured limit EnsureStreams did not apply because it would purge.
type LimitHeld struct {
	Stream string
	Limit  string // "max_age" | "max_bytes"
	Have   int64  // current value (ns for max_age, bytes for max_bytes)
	Want   int64  // configured value
}

// OnLimitHeld, when set, is called for every held limit (services log it as a warning).
var OnLimitHeld func(LimitHeld)

// mergeStreamConfig overlays the managed fields of want onto the live config, holding back any
// limit change that would purge messages. It returns the config to send and whether an update
// is needed at all.
func mergeStreamConfig(have jetstream.StreamConfig, usedBytes uint64, want jetstream.StreamConfig) (jetstream.StreamConfig, bool, []LimitHeld) {
	var held []LimitHeld
	if mb := safeMaxBytes(have.MaxBytes, want.MaxBytes, usedBytes); mb != want.MaxBytes {
		held = append(held, LimitHeld{Stream: want.Name, Limit: "max_bytes", Have: have.MaxBytes, Want: want.MaxBytes})
		want.MaxBytes = mb
	}
	if ma := safeMaxAge(have.MaxAge, want.MaxAge); ma != want.MaxAge {
		held = append(held, LimitHeld{Stream: want.Name, Limit: "max_age", Have: int64(have.MaxAge), Want: int64(want.MaxAge)})
		want.MaxAge = ma
	}
	if configMatches(have, want) {
		return have, false, held
	}
	// Keep server-side fields we do not manage and overlay ours.
	have.Subjects = want.Subjects
	have.MaxAge = want.MaxAge
	have.MaxBytes = want.MaxBytes
	have.Description = want.Description
	have.Retention = want.Retention
	have.Storage = want.Storage
	return have, true, held
}

// WorkQueueName is the stream for a job kind: "evo" → "JOBS_EVO", "media.asr" → "JOBS_MEDIA_ASR".
func WorkQueueName(kind string) string {
	return "JOBS_" + strings.NewReplacer(".", "_", "-", "_").Replace(strings.ToUpper(kind))
}

// WorkQueueSubject is the filter a work queue captures: "jobs.<kind>.>".
func WorkQueueSubject(kind string) string {
	return "jobs." + kind + ".>"
}

// WorkQueueConsumer is a durable pull-consumer config with the spec's ack_wait / max_deliver.
func WorkQueueConsumer(durable string) jetstream.ConsumerConfig {
	wq := Table().WorkQueue
	return jetstream.ConsumerConfig{
		Durable:       durable,
		AckPolicy:     jetstream.AckExplicitPolicy,
		AckWait:       time.Duration(wq.AckWaitSecs) * time.Second,
		MaxDeliver:    wq.MaxDeliver,
		MaxAckPending: wq.MaxAckPending,
	}
}

// EnsureStreams creates every spec 21 §5 stream and updates the ones whose live config drifted
// (subjects, max age, max bytes, storage, description). Idempotent. A limit change that would
// purge is not applied: a max_bytes cap below a stream's current size (mirroring
// fa_bus::streams) and any max_age shorter than the live one (safeMaxAge); each is reported
// through OnLimitHeld. Returns the names created or updated.
func EnsureStreams(ctx context.Context, js jetstream.JetStream) ([]string, error) {
	var changed []string
	for _, spec := range Table().Streams {
		did, err := ensureStream(ctx, js, spec.Config())
		if err != nil {
			return changed, fmt.Errorf("stream %s: %w", spec.Name, err)
		}
		if did {
			changed = append(changed, spec.Name)
		}
	}
	return changed, nil
}

// EnsureStreamsConn is EnsureStreams over a plain connection (for services that only publish).
func EnsureStreamsConn(ctx context.Context, nc *nats.Conn) ([]string, error) {
	js, err := jetstream.New(nc)
	if err != nil {
		return nil, err
	}
	return EnsureStreams(ctx, js)
}

func ensureStream(ctx context.Context, js jetstream.JetStream, want jetstream.StreamConfig) (bool, error) {
	s, err := js.Stream(ctx, want.Name)
	if errors.Is(err, jetstream.ErrStreamNotFound) {
		_, err = js.CreateStream(ctx, want)
		return err == nil, err
	}
	if err != nil {
		return false, err
	}
	info := s.CachedInfo()
	cfg, update, held := mergeStreamConfig(info.Config, info.State.Bytes, want)
	if OnLimitHeld != nil {
		for _, h := range held {
			OnLimitHeld(h)
		}
	}
	if !update {
		return false, nil
	}
	_, err = js.UpdateStream(ctx, cfg)
	return err == nil, err
}

func configMatches(have, want jetstream.StreamConfig) bool {
	a, b := append([]string(nil), have.Subjects...), append([]string(nil), want.Subjects...)
	sort.Strings(a)
	sort.Strings(b)
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return have.MaxAge == want.MaxAge && have.MaxBytes == want.MaxBytes && have.Retention == want.Retention &&
		have.Storage == want.Storage && have.Description == want.Description
}

// EnsureWorkQueue creates the JOBS_<KIND> work-queue stream on jobs.<kind>.> and a durable
// pull consumer with the spec's ack_wait 60 s / max_deliver 5.
func EnsureWorkQueue(ctx context.Context, js jetstream.JetStream, kind, durable string) (jetstream.Stream, jetstream.Consumer, error) {
	s, err := js.CreateOrUpdateStream(ctx, jetstream.StreamConfig{
		Name:      WorkQueueName(kind),
		Subjects:  []string{WorkQueueSubject(kind)},
		Retention: jetstream.WorkQueuePolicy,
	})
	if err != nil {
		return nil, nil, err
	}
	c, err := s.CreateOrUpdateConsumer(ctx, WorkQueueConsumer(durable))
	if err != nil {
		return s, nil, err
	}
	go WatchConsumerLag(ctx, c) // fa_bus_consumer_num_pending{stream,consumer} for the owner's /metrics (D-056)
	return s, c, nil
}

// SubjectsOverlap reports whether some concrete subject matches both filters (NATS `*` / `>`).
// JetStream refuses a stream whose subjects overlap another stream's.
func SubjectsOverlap(a, b string) bool {
	at, bt := strings.Split(a, "."), strings.Split(b, ".")
	n := len(at)
	if len(bt) > n {
		n = len(bt)
	}
	for i := 0; i < n; i++ {
		if i >= len(at) || i >= len(bt) {
			return false
		}
		if at[i] == ">" || bt[i] == ">" {
			return true
		}
		if at[i] != "*" && bt[i] != "*" && at[i] != bt[i] {
			return false
		}
	}
	return true
}
