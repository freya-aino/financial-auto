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
package archive

import (
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
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
	Messages uint64 // payloads accepted by Put
	Objects  uint64 // objects uploaded
	Bytes    uint64 // compressed bytes uploaded
	Errors   uint64 // failed uploads (whole batch lost after retries)
}

type batch struct {
	key    string
	minute time.Time
	buf    bytes.Buffer
	n      int
}

// Writer implements connector.RawArchive.
type Writer struct {
	store         ObjectStore
	bucket        string
	clock         connector.Clock
	maxBatchBytes int
	retries       int

	mu      sync.Mutex
	open    map[string]*batch // source → current minute batch
	pending chan *batch
	stats   Stats
	wg      sync.WaitGroup
}

// WriterOptions tune batching; zero values take the defaults.
type WriterOptions struct {
	Bucket        string          // default Bucket
	MaxBatchBytes int             // upload early beyond this (default 8 MiB uncompressed)
	Clock         connector.Clock // default wall clock
	Retries       int             // upload attempts (default 3)
	Queue         int             // pending uploads before Put blocks (default 64)
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
	return &Writer{store: store, bucket: o.Bucket, clock: o.Clock, maxBatchBytes: o.MaxBatchBytes, retries: o.Retries,
		open: map[string]*batch{}, pending: make(chan *batch, o.Queue)}
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

func (w *Writer) upload(ctx context.Context, b *batch, onErr func(error)) {
	var zb bytes.Buffer
	zw := gzip.NewWriter(&zb)
	zw.Write(b.buf.Bytes())
	zw.Close()
	var err error
	for attempt := 0; attempt < w.retries; attempt++ {
		if err = w.store.Put(ctx, w.bucket, b.key, zb.Bytes(), "application/gzip"); err == nil {
			break
		}
		time.Sleep(time.Duration(attempt+1) * 500 * time.Millisecond)
	}
	w.mu.Lock()
	if err == nil {
		w.stats.Objects++
		w.stats.Bytes += uint64(zb.Len())
	} else {
		w.stats.Errors++
	}
	w.mu.Unlock()
	if err != nil && onErr != nil {
		onErr(fmt.Errorf("archive: put %s: %w", b.key, err))
	}
}

// Stats returns a copy of the counters.
func (w *Writer) Stats() Stats {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.stats
}

var _ connector.RawArchive = (*Writer)(nil)
