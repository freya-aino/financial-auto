package connector

import (
	"testing"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

func bar(close float64, ingest time.Time) *fav1.Envelope {
	t := time.Date(2026, 11, 10, 15, 0, 0, 0, time.UTC)
	return &fav1.Envelope{SourceId: "yahoo", InstrumentId: "i1", EventTime: timestamppb.New(t),
		IngestTime: timestamppb.New(ingest), PublishTime: timestamppb.New(ingest),
		Payload: &fav1.Envelope_Bar{Bar: &fav1.Bar{Timeframe: "1m", Open: 1, High: 2, Low: 0.5, Close: close, Volume: 10}}}
}

func TestStableEventIDIgnoresIngestTimes(t *testing.T) {
	a := StableEventID(bar(1.5, time.Now()))
	b := StableEventID(bar(1.5, time.Now().Add(time.Minute)))
	if a != b {
		t.Fatalf("same fact, different ids: %s %s", a, b)
	}
	if c := StableEventID(bar(1.6, time.Now())); c == a {
		t.Fatal("different payload must change the id")
	}
	if len(a) != 26 {
		t.Fatalf("not a ULID: %q", a)
	}
}

func TestStableEventIDUsesNewsDedupKey(t *testing.T) {
	mk := func(title string) *fav1.Envelope {
		return &fav1.Envelope{SourceId: "google_news", EventTime: timestamppb.New(time.Unix(1_700_000_000, 0)),
			Payload: &fav1.Envelope_News{News: &fav1.NewsItem{Title: title, DedupKey: "u:abc"}}}
	}
	if StableEventID(mk("A")) != StableEventID(mk("A (updated)")) {
		t.Fatal("news id must follow the dedup key")
	}
}

func TestNormalizeURL(t *testing.T) {
	cases := map[string]string{
		"https://www.Example.com/a/b/?utm_source=x&b=2&a=1#frag": "example.com/a/b?a=1&b=2",
		"http://example.com/a/b?a=1&b=2&fbclid=zzz":              "example.com/a/b?a=1&b=2",
		"https://example.com:443/":                               "example.com",
	}
	for in, want := range cases {
		if got := NormalizeURL(in); got != want {
			t.Errorf("NormalizeURL(%q) = %q, want %q", in, got, want)
		}
	}
	at := time.Date(2026, 1, 1, 10, 5, 0, 0, time.UTC)
	if NewsDedupKey("https://www.example.com/x?utm_medium=y", "", "", at) != NewsDedupKey("http://example.com/x/", "", "", at) {
		t.Fatal("equivalent URLs must share a key")
	}
	if NewsDedupKey("", "Title", "Out", at) != NewsDedupKey("", " title ", "out", at.Add(20*time.Minute)) {
		t.Fatal("title fallback buckets by hour")
	}
}
