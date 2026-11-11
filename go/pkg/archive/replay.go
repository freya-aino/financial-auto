package archive

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// ReadStore reads the archive back (S3Store or a test fake). Get returns the
// gunzipped object.
type ReadStore interface {
	Get(ctx context.Context, bucket, key string) ([]byte, error)
	List(ctx context.Context, bucket, prefix string) ([]string, error)
}

// List returns every key under prefix (paginated ListObjectsV2).
func (s *S3Store) List(ctx context.Context, bucket, prefix string) ([]string, error) {
	var keys []string
	p := s3.NewListObjectsV2Paginator(s.Client, &s3.ListObjectsV2Input{Bucket: aws.String(bucket), Prefix: aws.String(prefix)})
	for p.HasMorePages() {
		page, err := p.NextPage(ctx)
		if err != nil {
			return nil, err
		}
		for _, o := range page.Contents {
			keys = append(keys, aws.ToString(o.Key))
		}
	}
	return keys, nil
}

// ReplayItem is one archived payload rebuilt as it was received, with the ref
// the live pipeline handed out for it.
type ReplayItem struct {
	Raw connector.RawMessage
	Ref *fav1.RawRef
}

// ReplayStats summarizes a replay read.
type ReplayStats struct {
	Objects      int `json:"objects"`
	Messages     int `json:"messages"`
	MissingIndex int `json:"missing_index"` // objects written before index sidecars existed
	Errors       int `json:"errors"`
}

// HourPrefixes lists the `raw/<source>/<yyyy>/<mm>/<dd>/<hh>/` prefixes covering [from, to).
func HourPrefixes(source string, from, to time.Time) []string {
	var out []string
	for h := from.UTC().Truncate(time.Hour); h.Before(to); h = h.Add(time.Hour) {
		out = append(out, fmt.Sprintf("raw/%s/%04d/%02d/%02d/%02d/", source, h.Year(), h.Month(), h.Day(), h.Hour()))
	}
	return out
}

// Replay reads every archived payload of a source received in [from, to), in
// (received_at, key, offset) order, so normalizing the items reproduces the
// live envelopes (spec 10 §9). Objects lacking an index are counted, not read.
func Replay(ctx context.Context, store ReadStore, bucket, source string, from, to time.Time) ([]ReplayItem, ReplayStats, error) {
	if bucket == "" {
		bucket = Bucket
	}
	var st ReplayStats
	var items []ReplayItem
	for _, prefix := range HourPrefixes(source, from, to) {
		keys, err := store.List(ctx, bucket, prefix)
		if err != nil {
			return nil, st, fmt.Errorf("archive: list %s: %w", prefix, err)
		}
		have := map[string]bool{}
		for _, k := range keys {
			have[k] = true
		}
		sort.Strings(keys)
		for _, k := range keys {
			if !strings.HasSuffix(k, ".raw.gz") {
				continue
			}
			st.Objects++
			if !have[IndexKey(k)] {
				st.MissingIndex++
				continue
			}
			data, err := store.Get(ctx, bucket, k)
			if err != nil {
				st.Errors++
				continue
			}
			idxRaw, err := store.Get(ctx, bucket, IndexKey(k))
			if err != nil {
				st.Errors++
				continue
			}
			sc := bufio.NewScanner(bytes.NewReader(idxRaw))
			sc.Buffer(make([]byte, 0, 64<<10), 16<<20)
			for sc.Scan() {
				var e IndexEntry
				if json.Unmarshal(sc.Bytes(), &e) != nil || e.Offset+e.Length > uint64(len(data)) {
					st.Errors++
					continue
				}
				if e.ReceivedAt.Before(from) || !e.ReceivedAt.Before(to) {
					continue
				}
				payload := append([]byte(nil), data[e.Offset:e.Offset+e.Length]...)
				items = append(items, ReplayItem{
					Raw: connector.RawMessage{SourceID: source, ReceivedAt: e.ReceivedAt, TransportMeta: e.Meta, Bytes: payload, ContentType: e.ContentType},
					Ref: &fav1.RawRef{Key: k, Offset: e.Offset, Length: e.Length},
				})
			}
		}
	}
	sort.SliceStable(items, func(i, j int) bool {
		a, b := items[i], items[j]
		if !a.Raw.ReceivedAt.Equal(b.Raw.ReceivedAt) {
			return a.Raw.ReceivedAt.Before(b.Raw.ReceivedAt)
		}
		if a.Ref.Key != b.Ref.Key {
			return a.Ref.Key < b.Ref.Key
		}
		return a.Ref.Offset < b.Ref.Offset
	})
	st.Messages = len(items)
	return items, st, nil
}
