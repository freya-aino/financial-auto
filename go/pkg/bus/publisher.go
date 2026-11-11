package bus

import (
	"context"

	"github.com/nats-io/nats.go"
	"google.golang.org/protobuf/proto"
)

// Publisher publishes protobuf messages on NATS.
type Publisher struct {
	NC *nats.Conn
}

// Connect dials NATS with reconnects enabled.
func Connect(url, name string) (*Publisher, error) {
	nc, err := nats.Connect(url, nats.Name(name), nats.MaxReconnects(-1))
	if err != nil {
		return nil, err
	}
	return &Publisher{NC: nc}, nil
}

// setTrace adds a `traceparent` header (spec 60 §6.5) continuing the trace in ctx, or a new
// one; market data and heartbeats stay untraced (Traced).
func setTrace(m *nats.Msg, ctx context.Context) {
	if !Traced(m.Subject) {
		return
	}
	t, ok := TraceFrom(ctx)
	if ok {
		t = t.Child()
	} else {
		t = NewTrace()
	}
	if m.Header == nil {
		m.Header = nats.Header{}
	}
	m.Header.Set(TraceHeader, t.Header())
}

// Publish encodes and publishes msg on subject (new trace, see PublishCtx).
func (p *Publisher) Publish(subject string, msg proto.Message) error {
	return p.PublishCtx(context.Background(), subject, msg)
}

// PublishCtx encodes and publishes msg on subject, continuing the trace stored in ctx
// (WithTrace / Consumed).
func (p *Publisher) PublishCtx(ctx context.Context, subject string, msg proto.Message) error {
	b, err := proto.Marshal(msg)
	if err != nil {
		return err
	}
	if !Traced(subject) {
		return p.NC.Publish(subject, b)
	}
	m := nats.NewMsg(subject)
	m.Data = b
	setTrace(m, ctx)
	return p.NC.PublishMsg(m)
}

// PublishWithID sets Nats-Msg-Id so JetStream can de-duplicate (jobs, spec 21).
func (p *Publisher) PublishWithID(subject, msgID string, msg proto.Message) error {
	b, err := proto.Marshal(msg)
	if err != nil {
		return err
	}
	m := nats.NewMsg(subject)
	m.Header.Set(nats.MsgIdHdr, msgID)
	m.Data = b
	setTrace(m, context.Background())
	return p.NC.PublishMsg(m)
}

// Close drains the connection.
func (p *Publisher) Close() {
	if p.NC != nil {
		_ = p.NC.Drain()
	}
}
