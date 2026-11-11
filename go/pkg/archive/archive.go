// Package archive is the Garage (S3) raw-payload writer (spec 10 §4.2, spec 20 §5.5).
//
// Raw messages are appended to one object per (source, UTC minute):
//
//	raw/<source>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.raw.gz
//
// The object is the gzip of the concatenated payloads, each terminated by
// "\n"; the RawRef returned by Put points into the *decompressed* stream
// (offset, length), so an envelope's raw bytes are
// gunzip(object)[offset:offset+length]. Objects are uploaded when their
// minute ends, when they exceed MaxBatchBytes, and on Flush/shutdown
// (at-least-once, asynchronous — the ref is handed out before the upload).
//
// Next to every data object the writer stores an index object
// (`<ulid>.idx.jsonl.gz`, one IndexEntry per payload: offset, length,
// received_at, content type, transport meta), which is what Replay needs to
// rebuild the exact RawMessages and re-run the normalizers (spec 10 §9).
// Failed uploads go to a bounded local spool and are retried; when the spool
// is full Put returns ErrSpoolFull so the pipeline flags raw_missing (§8).
package archive

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/oklog/ulid/v2"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// Bucket is the raw-archive bucket (spec 20 §5.5).
const Bucket = "fa-raw"

// Object store subset used by the writer (S3 client or a test fake).
type ObjectStore interface {
	Put(ctx context.Context, bucket, key string, body []byte, contentType string) error
}

// S3Store is ObjectStore over aws-sdk-go-v2 with path-style addressing (Garage).
type S3Store struct{ Client *s3.Client }

// Options for the S3 client.
type Options struct {
	Endpoint  string // e.g. http://127.0.0.1:3900
	Region    string // Garage's configured region (default "garage")
	AccessKey string
	SecretKey string
}

// NewS3 builds an S3 client for Garage (static credentials, SigV4, path style).
func NewS3(o Options) (*S3Store, error) {
	if o.Endpoint == "" || o.AccessKey == "" || o.SecretKey == "" {
		return nil, errors.New("archive: endpoint, access key and secret key are required")
	}
	if o.Region == "" {
		o.Region = "garage"
	}
	client := s3.New(s3.Options{
		Region:       o.Region,
		BaseEndpoint: aws.String(o.Endpoint),
		UsePathStyle: true,
		Credentials:  credentials.NewStaticCredentialsProvider(o.AccessKey, o.SecretKey, ""),
		HTTPClient:   &http.Client{Timeout: 30 * time.Second},
		// Garage does not implement the newer default checksum behaviour.
		RequestChecksumCalculation: aws.RequestChecksumCalculationWhenRequired,
		ResponseChecksumValidation: aws.ResponseChecksumValidationWhenRequired,
	})
	return &S3Store{Client: client}, nil
}

func (s *S3Store) Put(ctx context.Context, bucket, key string, body []byte, contentType string) error {
	_, err := s.Client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(bucket), Key: aws.String(key), Body: bytes.NewReader(body),
		ContentType: aws.String(contentType), ContentLength: aws.Int64(int64(len(body))),
	})
	return err
}

// Get downloads and gunzips an object (for tests and replay tooling).
func (s *S3Store) Get(ctx context.Context, bucket, key string) ([]byte, error) {
	out, err := s.Client.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(bucket), Key: aws.String(key)})
	if err != nil {
		return nil, err
	}
	defer out.Body.Close()
	zr, err := gzip.NewReader(out.Body)
	if err != nil {
		return nil, err
	}
	return io.ReadAll(zr)
}

// Stats are the writer's cumulative counters.
type Stats struct {
	Messages   uint64 // payloads accepted by Put
	Objects    uint64 // objects uploaded
	Bytes      uint64 // compressed bytes uploaded
	Errors     uint64 // failed uploads (after retries; spooled when a spool is configured)
	Spooled    uint64 // batches written to the local spool
	Respooled  uint64 // spooled batches uploaded later
	Dropped    uint64 // payloads refused because the spool was full (raw_missing)
	SpoolBytes int64  // current spool size
}

type batch struct {
	key    string
	minute time.Time
	buf    bytes.Buffer
	n      int
	idx    []IndexEntry
}

// IndexEntry describes one payload inside a data object.
type IndexEntry struct {
	Offset      uint64            `json:"o"`
	Length      uint64            `json:"l"`
	ReceivedAt  time.Time         `json:"t"`
	ContentType string            `json:"ct,omitempty"`
	Meta        map[string]string `json:"m,omitempty"`
}

// IndexKey is the index object key for a data object key.
func IndexKey(dataKey string) string { return strings.TrimSuffix(dataKey, ".raw.gz") + ".idx.jsonl.gz" }

// ErrSpoolFull is returned by Put while uploads fail and the spool is full.
var ErrSpoolFull = errors.New("archive: garage unavailable and spool full (raw_missing)")

// Writer implements connector.RawArchive.
type Writer struct {
	store         ObjectStore
	bucket        string
	clock         connector.Clock
	maxBatchBytes int
	retries       int

	spoolDir      string
	spoolMaxBytes int64

	mu        sync.Mutex
	open      map[string]*batch // source → current minute batch
	pending   chan *batch
	stats     Stats
	wg        sync.WaitGroup
	spoolFull bool // last upload failed and the spool had no room
}

// WriterOptions tune batching; zero values take the defaults.
type WriterOptions struct {
	Bucket        string          // default Bucket
	MaxBatchBytes int             // upload early beyond this (default 8 MiB uncompressed)
	Clock         connector.Clock // default wall clock
	Retries       int             // upload attempts (default 3)
	Queue         int             // pending uploads before Put blocks (default 64)
	// SpoolDir keeps batches whose upload failed (spec 10 §8); empty disables.
	SpoolDir string
	// SpoolMaxBytes bounds the spool (default 256 MiB).
	SpoolMaxBytes int64
}

// NewWriter creates a writer; call Run in a goroutine.
func NewWriter(store ObjectStore, o WriterOptions) *Writer {
	if o.Bucket == "" {
		o.Bucket = Bucket
	}
	if o.MaxBatchBytes <= 0 {
		o.MaxBatchBytes = 8 << 20
	}
	if o.Clock == nil {
		o.Clock = connector.SystemClock{}
	}
	if o.Retries <= 0 {
		o.Retries = 3
	}
	if o.Queue <= 0 {
		o.Queue = 64
	}
	if o.SpoolMaxBytes <= 0 {
		o.SpoolMaxBytes = 256 << 20
	}
	w := &Writer{store: store, bucket: o.Bucket, clock: o.Clock, maxBatchBytes: o.MaxBatchBytes, retries: o.Retries,
		open: map[string]*batch{}, pending: make(chan *batch, o.Queue), spoolDir: o.SpoolDir, spoolMaxBytes: o.SpoolMaxBytes}
	if w.spoolDir != "" {
		_ = os.MkdirAll(w.spoolDir, 0o755)
		w.stats.SpoolBytes = spoolSize(w.spoolDir)
	}
	return w
}

// Key builds the object key for a source and minute.
func Key(source string, at time.Time, id ulid.ULID) string {
	at = at.UTC()
	return fmt.Sprintf("raw/%s/%04d/%02d/%02d/%02d/%s.raw.gz", source, at.Year(), at.Month(), at.Day(), at.Hour(), id.String())
}

// Put appends raw to the source's current batch and returns its reference.
func (w *Writer) Put(_ context.Context, raw connector.RawMessage) (*fav1.RawRef, error) {
	if raw.SourceID == "" {
		return nil, errors.New("archive: raw message without source id")
	}
	at := raw.ReceivedAt
	if at.IsZero() {
		at = w.clock.Now()
	}
	minute := at.UTC().Truncate(time.Minute)
	w.mu.Lock()
	if w.spoolFull {
		w.stats.Dropped++
		w.mu.Unlock()
		return nil, ErrSpoolFull
	}
	b := w.open[raw.SourceID]
	if b != nil && (!b.minute.Equal(minute) || b.buf.Len() >= w.maxBatchBytes) {
		w.rotateLocked(raw.SourceID)
		b = nil
	}
	if b == nil {
		b = &batch{key: Key(raw.SourceID, at, ulid.Make()), minute: minute}
		w.open[raw.SourceID] = b
	}
	ref := &fav1.RawRef{Key: b.key, Offset: uint64(b.buf.Len()), Length: uint64(len(raw.Bytes))}
	b.idx = append(b.idx, IndexEntry{Offset: ref.Offset, Length: ref.Length, ReceivedAt: at.UTC(), ContentType: raw.ContentType, Meta: raw.TransportMeta})
	b.buf.Write(raw.Bytes)
	b.buf.WriteByte('\n')
	b.n++
	w.stats.Messages++
	w.mu.Unlock()
	return ref, nil
}

// rotateLocked moves a source's open batch to the upload queue.
func (w *Writer) rotateLocked(source string) {
	b := w.open[source]
	if b == nil || b.n == 0 {
		return
	}
	delete(w.open, source)
	w.pending <- b // may block briefly when uploads lag; bounded by Queue
}

// Run uploads batches as minutes roll over; it returns after draining on ctx done.
func (w *Writer) Run(ctx context.Context, onErr func(error)) {
	w.wg.Add(1)
	go func() {
		defer w.wg.Done()
		for b := range w.pending {
			w.upload(context.Background(), b, onErr)
		}
	}()
	t := time.NewTicker(time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			w.Flush()
			close(w.pending)
			w.wg.Wait()
			return
		case <-t.C:
			if w.spoolDir != "" && w.clock.Now().Second()%30 == 0 {
				w.retrySpool(ctx, onErr)
			}
			now := w.clock.Now().UTC().Truncate(time.Minute)
			w.mu.Lock()
			for src, b := range w.open {
				if b.minute.Before(now) {
					w.rotateLocked(src)
				}
			}
			w.mu.Unlock()
		}
	}
}

// Flush queues every open batch for upload (does not wait for the uploads).
func (w *Writer) Flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	for src := range w.open {
		w.rotateLocked(src)
	}
}

// FlushSync uploads every open batch now (tests, tooling without Run).
func (w *Writer) FlushSync(ctx context.Context) error {
	w.mu.Lock()
	var bs []*batch
	for src, b := range w.open {
		if b.n > 0 {
			bs = append(bs, b)
		}
		delete(w.open, src)
	}
	w.mu.Unlock()
	var errs []error
	for _, b := range bs {
		w.upload(ctx, b, func(err error) { errs = append(errs, err) })
	}
	return errors.Join(errs...)
}

func gz(b []byte) []byte {
	var zb bytes.Buffer
	zw := gzip.NewWriter(&zb)
	zw.Write(b)
	zw.Close()
	return zb.Bytes()
}

func encodeIndex(idx []IndexEntry) []byte {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, e := range idx {
		_ = enc.Encode(e)
	}
	return buf.Bytes()
}

func (w *Writer) putRetry(ctx context.Context, key string, body []byte) error {
	var err error
	for attempt := 0; attempt < w.retries; attempt++ {
		if err = w.store.Put(ctx, w.bucket, key, body, "application/gzip"); err == nil {
			return nil
		}
		time.Sleep(time.Duration(attempt+1) * 500 * time.Millisecond)
	}
	return err
}

func (w *Writer) upload(ctx context.Context, b *batch, onErr func(error)) {
	data, index := gz(b.buf.Bytes()), gz(encodeIndex(b.idx))
	err := w.putRetry(ctx, b.key, data)
	if err == nil {
		err = w.putRetry(ctx, IndexKey(b.key), index)
	}
	w.mu.Lock()
	if err == nil {
		w.stats.Objects++
		w.stats.Bytes += uint64(len(data) + len(index))
		w.spoolFull = false
	} else {
		w.stats.Errors++
	}
	w.mu.Unlock()
	if err != nil {
		spooled := w.spool(b.key, data, index)
		if onErr != nil {
			onErr(fmt.Errorf("archive: put %s (spooled=%v): %w", b.key, spooled, err))
		}
	}
}

// spoolName flattens an object key into one file name.
func spoolName(key string) string { return strings.ReplaceAll(key, "/", "__") }

func spoolSize(dir string) int64 {
	var n int64
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if fi, err := e.Info(); err == nil {
			n += fi.Size()
		}
	}
	return n
}

// spool keeps a failed batch on local disk; reports false (and marks the
// writer spool-full) when there is no spool or no room.
func (w *Writer) spool(key string, data, index []byte) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	size := int64(len(data) + len(index))
	if w.spoolDir == "" || w.stats.SpoolBytes+size > w.spoolMaxBytes {
		w.spoolFull = w.spoolDir != ""
		return false
	}
	base := filepath.Join(w.spoolDir, spoolName(key))
	if os.WriteFile(base+".data", data, 0o644) != nil || os.WriteFile(base+".idx", index, 0o644) != nil {
		w.spoolFull = true
		return false
	}
	w.stats.Spooled++
	w.stats.SpoolBytes += size
	return true
}

// retrySpool uploads spooled batches (oldest first) until one fails.
func (w *Writer) retrySpool(ctx context.Context, onErr func(error)) {
	entries, err := os.ReadDir(w.spoolDir)
	if err != nil {
		return
	}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".data") {
			continue
		}
		base := filepath.Join(w.spoolDir, strings.TrimSuffix(name, ".data"))
		key := strings.ReplaceAll(strings.TrimSuffix(name, ".data"), "__", "/")
		data, err1 := os.ReadFile(base + ".data")
		index, err2 := os.ReadFile(base + ".idx")
		if err1 != nil || err2 != nil {
			continue
		}
		if err := w.store.Put(ctx, w.bucket, key, data, "application/gzip"); err != nil {
			return
		}
		if err := w.store.Put(ctx, w.bucket, IndexKey(key), index, "application/gzip"); err != nil {
			return
		}
		_ = os.Remove(base + ".data")
		_ = os.Remove(base + ".idx")
		w.mu.Lock()
		w.stats.Respooled++
		w.stats.Objects++
		w.stats.SpoolBytes -= int64(len(data) + len(index))
		w.spoolFull = false
		w.mu.Unlock()
		if onErr != nil {
			onErr(fmt.Errorf("archive: spooled %s uploaded", key)) // informational
		}
	}
}

// Stats returns a copy of the counters.
func (w *Writer) Stats() Stats {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.stats
}

var _ connector.RawArchive = (*Writer)(nil)
