package archive

import (
	"bytes"
	"compress/gzip"
	"context"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/connector"
)

type memStore struct {
	mu   sync.Mutex
	objs map[string][]byte
	fail int
}

func (m *memStore) Put(_ context.Context, bucket, key string, body []byte, ct string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.fail > 0 {
		m.fail--
		return io.ErrUnexpectedEOF
	}
	if m.objs == nil {
		m.objs = map[string][]byte{}
	}
	m.objs[bucket+"/"+key] = body
	return nil
}

type fixedClock struct{ t time.Time }

func (f fixedClock) Now() time.Time { return f.t }

func gunzip(t *testing.T, b []byte) []byte {
	zr, err := gzip.NewReader(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	out, _ := io.ReadAll(zr)
	return out
}

func TestBatchesPerSourceMinuteAndRefs(t *testing.T) {
	st := &memStore{}
	now := time.Date(2026, 11, 10, 5, 7, 30, 0, time.UTC)
	w := NewWriter(st, WriterOptions{Clock: fixedClock{now}, Retries: 1})
	ctx := context.Background()
	r1, _ := w.Put(ctx, connector.RawMessage{SourceID: "yahoo", ReceivedAt: now, Bytes: []byte(`{"a":1}`)})
	r2, _ := w.Put(ctx, connector.RawMessage{SourceID: "yahoo", ReceivedAt: now.Add(time.Second), Bytes: []byte(`{"b":22}`)})
	r3, _ := w.Put(ctx, connector.RawMessage{SourceID: "hackernews", ReceivedAt: now, Bytes: []byte(`x`)})
	r4, _ := w.Put(ctx, connector.RawMessage{SourceID: "yahoo", ReceivedAt: now.Add(time.Minute), Bytes: []byte(`{"c":3}`)})
	if r1.Key != r2.Key || r1.Key == r3.Key || r1.Key == r4.Key {
		t.Fatalf("keys: %s %s %s %s", r1.Key, r2.Key, r3.Key, r4.Key)
	}
	if !strings.HasPrefix(r1.Key, "raw/yahoo/2026/11/10/05/") || !strings.HasSuffix(r1.Key, ".raw.gz") {
		t.Fatalf("key %s", r1.Key)
	}
	if r1.Offset != 0 || r1.Length != 7 || r2.Offset != 8 || r2.Length != 8 {
		t.Fatalf("refs %v %v", r1, r2)
	}
	// Rotating the first yahoo minute queued it; drain the queue by running briefly.
	if _, err := w.Put(ctx, connector.RawMessage{Bytes: []byte("no source")}); err == nil {
		t.Fatal("source id required")
	}
	rctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() { w.Run(rctx, func(err error) { t.Error(err) }); close(done) }()
	cancel()
	<-done
	st.mu.Lock()
	n := len(st.objs)
	obj := st.objs[Bucket+"/"+r1.Key]
	st.mu.Unlock()
	if n != 3 {
		t.Fatalf("objects uploaded: %d", n)
	}
	plain := gunzip(t, obj)
	if string(plain[r1.Offset:r1.Offset+r1.Length]) != `{"a":1}` || string(plain[r2.Offset:r2.Offset+r2.Length]) != `{"b":22}` {
		t.Fatalf("payload slice mismatch: %q", plain)
	}
	s := w.Stats()
	if s.Messages != 4 || s.Objects != 3 || s.Errors != 0 || s.Bytes == 0 {
		t.Fatalf("stats %+v", s)
	}
}

func TestUploadErrorsCounted(t *testing.T) {
	st := &memStore{fail: 5}
	w := NewWriter(st, WriterOptions{Retries: 1})
	w.Put(context.Background(), connector.RawMessage{SourceID: "s", Bytes: []byte("x")})
	if err := w.FlushSync(context.Background()); err == nil || w.Stats().Errors != 1 {
		t.Fatalf("expected one failed upload, got %v %+v", err, w.Stats())
	}
}

// Live test against compose Garage: FA_GARAGE_URL, FA_GARAGE_ACCESS_KEY_ID, FA_GARAGE_SECRET_ACCESS_KEY.
func TestLiveGaragePutGet(t *testing.T) {
	url, ak, sk := os.Getenv("FA_GARAGE_URL"), os.Getenv("FA_GARAGE_ACCESS_KEY_ID"), os.Getenv("FA_GARAGE_SECRET_ACCESS_KEY")
	if url == "" || ak == "" || sk == "" {
		t.Skip("FA_GARAGE_* not set")
	}
	st, err := NewS3(Options{Endpoint: url, AccessKey: ak, SecretKey: sk})
	if err != nil {
		t.Fatal(err)
	}
	w := NewWriter(st, WriterOptions{})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	ref, err := w.Put(ctx, connector.RawMessage{SourceID: "archive-test", ReceivedAt: time.Now(), Bytes: []byte(`{"live":true}`)})
	if err != nil {
		t.Fatal(err)
	}
	if err := w.FlushSync(ctx); err != nil {
		t.Fatal(err)
	}
	got, err := st.Get(ctx, Bucket, ref.Key)
	if err != nil {
		t.Fatal(err)
	}
	if string(got[ref.Offset:ref.Offset+ref.Length]) != `{"live":true}` {
		t.Fatalf("round trip: %q", got)
	}
	t.Logf("wrote and read back s3://%s/%s (%d bytes)", Bucket, ref.Key, len(got))
}
