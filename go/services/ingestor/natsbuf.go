package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
	"google.golang.org/protobuf/proto"

	"github.com/fey/financial-auto/go/pkg/bus"
)

// ErrBusBufferFull means NATS is unreachable and the disk buffer is at its
// limit: the pipeline stops the connector instead of losing envelopes
// silently (spec 10 §8 "bounded buffer; then stop connector").
var ErrBusBufferFull = errors.New("nats unavailable and the envelope disk buffer is full")

// EnvelopePublisher publishes canonical envelopes with a JetStream dedup id.
type EnvelopePublisher interface {
	PublishWithID(subject, msgID string, msg proto.Message) error
}

// NATSConn is the part of *nats.Conn the disk buffer uses.
type NATSConn interface {
	IsConnected() bool
	PublishMsg(*nats.Msg) error
	FlushTimeout(time.Duration) error
}

// DefaultBusBufferBytes bounds the envelope disk buffer (tunable `nats.buffer_bytes`).
const DefaultBusBufferBytes = 512 << 20

// BusBuffer spools envelopes to a local append-only file while NATS is
// unreachable and replays them in order once it is back (spec 10 §8).
//
// While the connection is up and the buffer is empty, PublishWithID publishes
// directly. When the connection is down, a publish fails, or older envelopes
// are still buffered (order is kept), the envelope is appended to the file
// instead. Drain republishes the file in batches, flushing after each one; a
// batch that cannot be flushed is kept and retried, and JetStream drops the
// envelopes of a partly delivered batch by their Nats-Msg-Id. A torn record at
// the end of the file (crash mid-write) is cut off on open.
type BusBuffer struct {
	NC       NATSConn
	Path     string
	MaxBytes int64
	Counters *bus.Counters
	// OnDrained runs (outside the lock) after a drain emptied the buffer.
	OnDrained func()
	// Batch is the number of records published per flush (default 500).
	Batch int

	mu      sync.Mutex
	f       *os.File
	size    int64
	records int64
	kick    chan struct{}
}

// OpenBusBuffer opens (or creates) dir/nats-envelopes.spool and recovers records left by a previous run.
func OpenBusBuffer(nc NATSConn, dir string, maxBytes int64, counters *bus.Counters) (*BusBuffer, error) {
	if maxBytes <= 0 {
		maxBytes = DefaultBusBufferBytes
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, fmt.Errorf("bus buffer: %w", err)
	}
	b := &BusBuffer{NC: nc, Path: filepath.Join(dir, "nats-envelopes.spool"), MaxBytes: maxBytes, Counters: counters, kick: make(chan struct{}, 1)}
	f, err := os.OpenFile(b.Path, os.O_RDWR|os.O_CREATE, 0o640)
	if err != nil {
		return nil, fmt.Errorf("bus buffer: %w", err)
	}
	n, good, err := scanRecords(f)
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("bus buffer: scan: %w", err)
	}
	if err := f.Truncate(good); err != nil { // cut a torn tail
		_ = f.Close()
		return nil, fmt.Errorf("bus buffer: truncate: %w", err)
	}
	if _, err := f.Seek(good, io.SeekStart); err != nil {
		_ = f.Close()
		return nil, err
	}
	b.f, b.size, b.records = f, good, n
	b.gauges()
	return b, nil
}

// Records returns the number of buffered envelopes.
func (b *BusBuffer) Records() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.records
}

// PublishWithID implements EnvelopePublisher.
func (b *BusBuffer) PublishWithID(subject, msgID string, msg proto.Message) error {
	data, err := proto.Marshal(msg)
	if err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.records == 0 && b.NC.IsConnected() {
		if err := b.NC.PublishMsg(natsMsg(subject, msgID, data)); err == nil {
			return nil
		}
	}
	return b.appendLocked(subject, msgID, data)
}

func natsMsg(subject, msgID string, data []byte) *nats.Msg {
	m := nats.NewMsg(subject)
	m.Header.Set(nats.MsgIdHdr, msgID)
	m.Data = data
	return m
}

func (b *BusBuffer) appendLocked(subject, msgID string, data []byte) error {
	rec := encodeRecord(subject, msgID, data)
	if b.size+int64(len(rec)) > b.MaxBytes {
		b.count("natsbuf.full", 1)
		return ErrBusBufferFull
	}
	if _, err := b.f.Write(rec); err != nil {
		return fmt.Errorf("bus buffer: write: %w", err)
	}
	b.size += int64(len(rec))
	b.records++
	b.count("natsbuf.spooled", 1)
	b.gauges()
	return nil
}

// Kick asks Run to drain now (nats reconnect handler).
func (b *BusBuffer) Kick() {
	select {
	case b.kick <- struct{}{}:
	default:
	}
}

// Run drains whenever kicked or every interval while records are buffered.
func (b *BusBuffer) Run(done <-chan struct{}, interval time.Duration, logf func(msg string, args ...any)) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-done:
			return
		case <-t.C:
		case <-b.kick:
		}
		if b.Records() == 0 || !b.NC.IsConnected() {
			continue
		}
		n, err := b.Drain()
		if err != nil && logf != nil {
			logf("bus buffer drain incomplete", "replayed", n, "left", b.Records(), "err", err)
		} else if n > 0 && logf != nil {
			logf("bus buffer drained", "replayed", n)
		}
	}
}

// Drain republishes every buffered record in order. It returns the number of
// records confirmed (flushed) and leaves the rest in the file.
func (b *BusBuffer) Drain() (int, error) {
	b.mu.Lock()
	n, emptied, err := b.drainLocked()
	b.mu.Unlock()
	if emptied && b.OnDrained != nil {
		b.OnDrained()
	}
	return n, err
}

func (b *BusBuffer) drainLocked() (int, bool, error) {
	if b.records == 0 {
		return 0, false, nil
	}
	batch := b.Batch
	if batch <= 0 {
		batch = 500
	}
	if _, err := b.f.Seek(0, io.SeekStart); err != nil {
		return 0, false, err
	}
	r := bufio.NewReaderSize(b.f, 1<<20)
	var committed int64 // file offset up to which records are confirmed
	var off int64
	done, inBatch := 0, 0
	var drainErr error
	for {
		subj, id, data, n, err := readRecord(r)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			drainErr = err
			break
		}
		if err := b.NC.PublishMsg(natsMsg(subj, id, data)); err != nil {
			drainErr = err
			break
		}
		off += n
		inBatch++
		if inBatch == batch {
			if err := b.NC.FlushTimeout(5 * time.Second); err != nil {
				drainErr = err
				break
			}
			committed, done, inBatch = off, done+inBatch, 0
		}
	}
	if drainErr == nil && inBatch > 0 {
		if err := b.NC.FlushTimeout(5 * time.Second); err != nil {
			drainErr = err
		} else {
			committed, done = off, done+inBatch
		}
	}
	b.count("natsbuf.replayed", float64(done))
	if err := b.compactLocked(committed, int64(done)); err != nil {
		return done, false, err
	}
	return done, b.records == 0, drainErr
}

// compactLocked drops the first `upto` bytes (`n` records) of the file.
func (b *BusBuffer) compactLocked(upto, n int64) error {
	defer b.gauges()
	if upto >= b.size {
		if err := b.f.Truncate(0); err != nil {
			return err
		}
		_, err := b.f.Seek(0, io.SeekStart)
		b.size, b.records = 0, 0
		return err
	}
	if upto == 0 {
		_, err := b.f.Seek(0, io.SeekEnd)
		return err
	}
	tmp := b.Path + ".tmp"
	out, err := os.OpenFile(tmp, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o640)
	if err != nil {
		return err
	}
	if _, err := b.f.Seek(upto, io.SeekStart); err != nil {
		_ = out.Close()
		return err
	}
	if _, err := io.Copy(out, b.f); err != nil {
		_ = out.Close()
		return err
	}
	if err := os.Rename(tmp, b.Path); err != nil {
		_ = out.Close()
		return err
	}
	_ = b.f.Close()
	b.f = out
	b.size -= upto
	b.records -= n
	_, err = b.f.Seek(0, io.SeekEnd)
	return err
}

// Close closes the file (buffered records stay on disk for the next run).
func (b *BusBuffer) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.f.Close()
}

func (b *BusBuffer) count(k string, v float64) {
	if b.Counters != nil && v != 0 {
		b.Counters.Add(k, v)
	}
}

func (b *BusBuffer) gauges() {
	if b.Counters != nil {
		b.Counters.Set("natsbuf.records", float64(b.records))
		b.Counters.Set("natsbuf.bytes", float64(b.size))
	}
}

// Record layout: uvarint len + subject, uvarint len + msg id, uvarint len + data.
func encodeRecord(subject, msgID string, data []byte) []byte {
	out := make([]byte, 0, len(subject)+len(msgID)+len(data)+3*binary.MaxVarintLen64)
	out = binary.AppendUvarint(out, uint64(len(subject)))
	out = append(out, subject...)
	out = binary.AppendUvarint(out, uint64(len(msgID)))
	out = append(out, msgID...)
	out = binary.AppendUvarint(out, uint64(len(data)))
	return append(out, data...)
}

// maxField guards against reading garbage lengths from a damaged file.
const maxField = 64 << 20

type countingReader struct {
	r *bufio.Reader
	n int64
}

func (c *countingReader) ReadByte() (byte, error) {
	b, err := c.r.ReadByte()
	if err == nil {
		c.n++
	}
	return b, err
}

func (c *countingReader) field() ([]byte, error) {
	l, err := binary.ReadUvarint(c)
	if err != nil {
		return nil, err
	}
	if l > maxField {
		return nil, fmt.Errorf("bus buffer: field length %d", l)
	}
	buf := make([]byte, l)
	k, err := io.ReadFull(c.r, buf)
	c.n += int64(k)
	if err != nil {
		return nil, io.ErrUnexpectedEOF
	}
	return buf, nil
}

// readRecord returns io.EOF at a clean end and io.ErrUnexpectedEOF for a torn record.
func readRecord(r *bufio.Reader) (subject, msgID string, data []byte, n int64, err error) {
	c := &countingReader{r: r}
	s, err := c.field()
	if err != nil {
		if errors.Is(err, io.EOF) && c.n == 0 {
			return "", "", nil, 0, io.EOF
		}
		return "", "", nil, c.n, io.ErrUnexpectedEOF
	}
	id, err := c.field()
	if err != nil {
		return "", "", nil, c.n, io.ErrUnexpectedEOF
	}
	d, err := c.field()
	if err != nil {
		return "", "", nil, c.n, io.ErrUnexpectedEOF
	}
	return string(s), string(id), d, c.n, nil
}

// scanRecords counts complete records and returns the offset after the last one.
func scanRecords(f *os.File) (int64, int64, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return 0, 0, err
	}
	r := bufio.NewReaderSize(f, 1<<20)
	var n, off int64
	for {
		_, _, _, k, err := readRecord(r)
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return n, off, nil
		}
		if err != nil {
			return n, off, nil
		}
		n++
		off += k
	}
}
