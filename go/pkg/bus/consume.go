package bus

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/nats-io/nats.go/jetstream"
)

// Durable live consumers with backpressure and quote coalescing (spec 21 §5, §6, §7, Q2);
// mirror of crates/fa-bus/src/consume.rs. MaxAckPending bounds delivered-but-unacked messages,
// so a slow consumer leaves the backlog in the stream; when more than BehindThreshold messages
// wait behind a batch, subjects of the coalesce policy (md.quote.*, md.book.* only — never
// trades, bars, latents, signals, order.* or risk.*) are reduced to their latest message per
// subject within the batch and the superseded ones are acked unprocessed.

// Coalescable are the subject prefixes that may be reduced to latest-only.
var Coalescable = []string{"md.quote.", "md.book."}

// MDMaxAckPending is spec 21 §7 consumer.max_ack_pending for market data.
const MDMaxAckPending = 10_000

// LiveSpec configures a live durable consumer.
type LiveSpec struct {
	Stream          string
	Durable         string
	FilterSubjects  []string
	MaxAckPending   int
	AckWait         time.Duration
	Batch           int
	BehindThreshold uint64
	Coalesce        []string // subject prefixes; each must contain md.quote. or md.book.
	// ResumeWithin: an existing consumer that acked nothing for longer than this skips its
	// backlog (recreated at new messages); 0 = always resume (e.g. order approvals).
	ResumeWithin time.Duration
}

// MarketDataLive returns the market-data defaults (10 000 in flight, batches of 500, coalesce
// quotes/books when more than 1 000 messages wait).
func MarketDataLive(stream, durable string, filters ...string) LiveSpec {
	return LiveSpec{Stream: stream, Durable: durable, FilterSubjects: filters, MaxAckPending: MDMaxAckPending,
		AckWait: 30 * time.Second, Batch: 500, BehindThreshold: 1_000, Coalesce: Coalescable}
}

// DurableLive returns the defaults for a non-market-data stream (orders, signals): 1 000 in
// flight, batches of 100, never coalesced.
func DurableLive(stream, durable string, filters ...string) LiveSpec {
	return LiveSpec{Stream: stream, Durable: durable, FilterSubjects: filters, MaxAckPending: 1_000,
		AckWait: 30 * time.Second, Batch: 100, BehindThreshold: 1_000}
}

// Validate refuses coalescing anything but quotes and books.
func (s LiveSpec) Validate() error {
	for _, p := range s.Coalesce {
		if !strings.Contains(p, "md.quote.") && !strings.Contains(p, "md.book.") {
			return fmt.Errorf("coalescing %s is not allowed (spec 21 Q2: md.quote/md.book only)", p)
		}
	}
	return nil
}

// Config is the durable, explicit-ack, deliver-new consumer config.
func (s LiveSpec) Config() jetstream.ConsumerConfig {
	return jetstream.ConsumerConfig{Durable: s.Durable, AckPolicy: jetstream.AckExplicitPolicy,
		DeliverPolicy: jetstream.DeliverNewPolicy, AckWait: s.AckWait, MaxAckPending: s.MaxAckPending,
		FilterSubjects: s.FilterSubjects}
}

func coalescable(prefixes []string, subject string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(subject, p) {
			return true
		}
	}
	return false
}

// Coalesce splits items into (kept, superseded): only the last item per coalescable subject is
// kept, everything else as is, in order.
func Coalesce[T any](items []T, prefixes []string, subject func(T) string) (kept, superseded []T) {
	last := map[string]int{}
	for i, it := range items {
		if s := subject(it); coalescable(prefixes, s) {
			last[s] = i
		}
	}
	for i, it := range items {
		s := subject(it)
		if coalescable(prefixes, s) && last[s] != i {
			superseded = append(superseded, it)
		} else {
			kept = append(kept, it)
		}
	}
	return kept, superseded
}

// LiveConsumer pulls batches from a durable consumer (see LiveSpec).
type LiveConsumer struct {
	Spec           LiveSpec
	Consumer       jetstream.Consumer
	Delivered      uint64
	Coalesced      uint64
	RejectedSchema uint64
	FetchErrors    uint64
}

// EnsureLive creates or updates the durable consumer.
func EnsureLive(ctx context.Context, js jetstream.JetStream, s LiveSpec) (*LiveConsumer, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if s.ResumeWithin > 0 {
		if old, err := js.Consumer(ctx, s.Stream, s.Durable); err == nil {
			if info, err := old.Info(ctx); err == nil && info.NumPending > 0 && !recentlyActive(info, s.ResumeWithin) {
				if err := js.DeleteConsumer(ctx, s.Stream, s.Durable); err != nil {
					return nil, err
				}
			}
		}
	}
	c, err := js.CreateOrUpdateConsumer(ctx, s.Stream, s.Config())
	if err != nil {
		return nil, err
	}
	return &LiveConsumer{Spec: s, Consumer: c}, nil
}

func recentlyActive(info *jetstream.ConsumerInfo, window time.Duration) bool {
	last := info.AckFloor.Last
	if last == nil {
		last = info.Delivered.Last
	}
	return last != nil && time.Since(*last) <= window
}

// Run pulls batches until ctx is done. Messages naming an unknown contract major are dropped
// (acked, counted in RejectedSchema); every other message is passed to handle and acked after
// it returns, so a crash mid-handling redelivers it (at least once; handlers must be
// idempotent). Fetch errors are retried after a second.
func (l *LiveConsumer) Run(ctx context.Context, handle func(subject string, data []byte)) {
	for ctx.Err() == nil {
		msgs, err := l.NextBatch(5 * time.Second)
		if err != nil && len(msgs) == 0 {
			l.FetchErrors++
			select {
			case <-ctx.Done():
			case <-time.After(time.Second):
			}
			continue
		}
		for _, m := range msgs {
			if CheckSchema(m.Headers()) != nil {
				l.RejectedSchema++
			} else {
				handle(m.Subject(), m.Data())
			}
			_ = m.Ack()
		}
	}
}

// NextBatch fetches up to one batch — the messages available now, or else the first one to
// arrive within wait — coalesces when behind and acks the superseded messages; the caller acks every returned message after handling it.
func (l *LiveConsumer) NextBatch(wait time.Duration) ([]jetstream.Msg, error) {
	n := l.Spec.Batch
	if l.Spec.MaxAckPending > 0 && n > l.Spec.MaxAckPending {
		n = l.Spec.MaxAckPending
	}
	if n <= 0 {
		n = 1
	}
	// what is there now, without waiting; else block for the first message, so an idle
	// consumer neither spins nor holds delivered messages until a full batch or the timeout
	b, err := l.Consumer.FetchNoWait(n)
	if err != nil {
		return nil, err
	}
	var msgs []jetstream.Msg
	for m := range b.Messages() {
		msgs = append(msgs, m)
	}
	if len(msgs) == 0 {
		if b, err = l.Consumer.Fetch(1, jetstream.FetchMaxWait(wait)); err != nil {
			return nil, err
		}
		for m := range b.Messages() {
			msgs = append(msgs, m)
		}
	}
	var behind uint64
	if len(msgs) > 0 {
		if md, err := msgs[len(msgs)-1].Metadata(); err == nil {
			behind = md.NumPending
		}
	}
	kept := msgs
	if behind > l.Spec.BehindThreshold {
		var sup []jetstream.Msg
		kept, sup = Coalesce(msgs, l.Spec.Coalesce, func(m jetstream.Msg) string { return m.Subject() })
		for _, m := range sup {
			_ = m.Ack()
		}
		l.Coalesced += uint64(len(sup))
	}
	l.Delivered += uint64(len(kept))
	return kept, b.Error()
}
