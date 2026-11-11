package connector

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/proto"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// StableEventID derives a deterministic ULID for a canonical envelope (spec 10
// §9: replaying the raw archive must reproduce byte-identical events; spec 21
// §3: Nats-Msg-Id = event_id lets JetStream drop re-polled duplicates).
//
// The ULID time part is the event time (ingest time when absent); the entropy
// is the first 10 bytes of sha256(source, instrument, natural key), where the
// natural key is the news dedup key when set, else the deterministic encoding
// of the payload. Ingest/publish times and the raw ref are excluded, so the
// same fact fetched twice (two polls, live + backfill) gets the same id.
func StableEventID(e *fav1.Envelope) string {
	h := sha256.New()
	h.Write([]byte(e.GetSourceId()))
	h.Write([]byte{0})
	h.Write([]byte(e.GetInstrumentId()))
	h.Write([]byte{0})
	if n := e.GetNews(); n != nil && n.GetDedupKey() != "" {
		h.Write([]byte("news:" + n.GetDedupKey()))
	} else {
		// Marshal a copy holding only the payload oneof and the event time.
		p := &fav1.Envelope{Payload: e.Payload, EventTime: e.EventTime}
		b, _ := proto.MarshalOptions{Deterministic: true}.Marshal(p)
		h.Write(b)
	}
	sum := h.Sum(nil)
	t := e.GetEventTime().AsTime()
	if e.GetEventTime() == nil {
		t = e.GetIngestTime().AsTime()
	}
	if t.Before(time.Unix(0, 0)) {
		t = time.Unix(0, 0)
	}
	id, err := ulid.New(ulid.Timestamp(t), bytes.NewReader(sum[:10]))
	if err != nil { // only for times beyond the ULID range
		id = ulid.MustNew(0, bytes.NewReader(sum[:10]))
	}
	return id.String()
}

// trackingParams are dropped by NormalizeURL.
var trackingParams = map[string]bool{"fbclid": true, "gclid": true, "mc_cid": true, "mc_eid": true, "ref": true, "ref_src": true, "cmpid": true, "ocid": true}

// NormalizeURL canonicalizes a URL for news de-duplication (spec 10 §6.5):
// lowercase scheme/host, no "www.", no fragment, no utm_*/tracking params,
// sorted query, no trailing slash, http == https.
func NormalizeURL(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Host == "" {
		return strings.ToLower(strings.TrimSpace(raw))
	}
	host := strings.TrimPrefix(strings.ToLower(u.Hostname()), "www.")
	if p := u.Port(); p != "" && p != "80" && p != "443" {
		host += ":" + p
	}
	q := u.Query()
	keys := make([]string, 0, len(q))
	for k := range q {
		lk := strings.ToLower(k)
		if strings.HasPrefix(lk, "utm_") || trackingParams[lk] {
			continue
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var qs []string
	for _, k := range keys {
		vs := append([]string(nil), q[k]...)
		sort.Strings(vs)
		for _, v := range vs {
			qs = append(qs, url.QueryEscape(k)+"="+url.QueryEscape(v))
		}
	}
	path := strings.TrimRight(u.EscapedPath(), "/")
	out := host + path
	if len(qs) > 0 {
		out += "?" + strings.Join(qs, "&")
	}
	return out
}

// NewsDedupKey is the cross-source news de-duplication key (spec 10 §6.5):
// "u:" + sha256(normalized URL)[:16] when a URL exists, else "t:" + hash of
// lowercased title, outlet and the publish time bucketed to an hour.
func NewsDedupKey(rawURL, title, outlet string, published time.Time) string {
	if strings.TrimSpace(rawURL) != "" {
		s := sha256.Sum256([]byte(NormalizeURL(rawURL)))
		return "u:" + hex.EncodeToString(s[:8])
	}
	bucket := published.UTC().Truncate(time.Hour).Unix()
	s := sha256.Sum256([]byte(strings.ToLower(strings.TrimSpace(title)) + "\x00" + strings.ToLower(strings.TrimSpace(outlet)) + "\x00" + time.Unix(bucket, 0).UTC().Format(time.RFC3339)))
	return "t:" + hex.EncodeToString(s[:8])
}
