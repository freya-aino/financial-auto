package main

import (
	"testing"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/fey/financial-auto/go/pkg/archive"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// ulidNorm mimics a real connector: a fresh random ULID per call.
type ulidNorm struct{}

func (ulidNorm) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	return []*fav1.Envelope{{
		EventId: ulid.Make().String(), SourceId: raw.SourceID, InstrumentId: "i",
		EventTime: timestamppb.New(time.Unix(1_762_800_000, 0)), IngestTime: timestamppb.New(raw.ReceivedAt), PublishTime: timestamppb.New(raw.ReceivedAt),
		Payload: &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: float64(len(raw.Bytes)), Size: 1}},
	}}, nil, nil
}

func TestReplayIsDeterministicAndMatchesLiveFinalize(t *testing.T) {
	at := time.Date(2026, 11, 11, 2, 0, 0, 0, time.UTC)
	items := []archive.ReplayItem{
		{Raw: connector.RawMessage{SourceID: "yahoo", ReceivedAt: at, Bytes: []byte("abc")}, Ref: &fav1.RawRef{Key: "raw/yahoo/k.raw.gz", Offset: 0, Length: 3}},
		{Raw: connector.RawMessage{SourceID: "yahoo", ReceivedAt: at.Add(time.Second), Bytes: []byte("abcd"), TransportMeta: map[string]string{connector.MetaBackfill: "1"}}, Ref: &fav1.RawRef{Key: "raw/yahoo/k.raw.gz", Offset: 4, Length: 4}},
	}
	a, errs := ReplayNormalize(ulidNorm{}, items)
	b, _ := ReplayNormalize(ulidNorm{}, items)
	if errs != 0 || len(a) != 2 || Digest(a) != Digest(b) {
		t.Fatalf("replay not deterministic: %d errs, %s vs %s", errs, Digest(a), Digest(b))
	}
	if !a[1].Backfill || a[0].Backfill || a[0].RawMissing || a[0].SchemaVersion != 1 {
		t.Fatalf("flags: %+v", a)
	}
	// the live path: same raw, same ref → identical bytes
	live, _, _ := ulidNorm{}.Normalize(items[0].Raw)
	Finalize(live[0], items[0].Ref, true, false)
	if !proto.Equal(live[0], a[0]) {
		t.Fatalf("live %v != replay %v", live[0], a[0])
	}
	// archive configured but no ref → raw_missing
	miss, _, _ := ulidNorm{}.Normalize(items[0].Raw)
	Finalize(miss[0], nil, true, false)
	if !miss[0].RawMissing {
		t.Fatal("raw_missing not flagged")
	}
}

func TestRetryWaitBreaker(t *testing.T) {
	iv := 2 * time.Minute
	if w := retryWait(1, iv, nil); w != 5*time.Second {
		t.Fatalf("first retry %s", w)
	}
	if w := retryWait(3, iv, nil); w != 20*time.Second {
		t.Fatalf("third retry %s", w)
	}
	if w := retryWait(BreakerFailuresToOpen, 30*time.Second, nil); w != RetryMaxBackoff {
		t.Fatalf("open breaker %s", w)
	}
	if w := retryWait(2, 3*time.Second, func() float64 { return 1 }); w != 3*time.Second+600*time.Millisecond {
		t.Fatalf("jitter %s", w)
	}
}

func TestHealthProtoCarriesTheFrame(t *testing.T) {
	now := time.Date(2026, 11, 11, 2, 0, 0, 0, time.UTC)
	act := now.Add(-10 * time.Second)
	f := Frame{Source: "yahoo", Status: "degraded", Running: true, Mode: "poll", At: now, Reason: "r", LastActivityAt: &act, LimiterSaturation: 0.9, ErrorsWindow: 2}
	m := HealthProto(f)
	if m.GetStatus() != fav1.HealthStatus_HEALTH_STATUS_DEGRADED || m.GetLastActivityAt().AsTime() != act || m.GetLastEventAt() != nil || m.GetErrorsWindow() != 2 {
		t.Fatalf("proto %v", m)
	}
	g := GapProto(Gap{Source: "yahoo", InstrumentID: "i", Timeframe: "1m", From: act, To: now, MissingPeriods: 3, At: now})
	u := UnresolvedProto(UnresolvedEvent{Source: "yahoo", VenueSymbol: "X", Reason: "r", At: now})
	if g.GetMissingPeriods() != 3 || u.GetVenueSymbol() != "X" {
		t.Fatal("gap/unresolved")
	}
}
