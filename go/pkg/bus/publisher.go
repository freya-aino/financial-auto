package bus

import (
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

// Publish encodes and publishes msg on subject.
func (p *Publisher) Publish(subject string, msg proto.Message) error {
	b, err := proto.Marshal(msg)
	if err != nil {
		return err
	}
	return p.NC.Publish(subject, b)
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
	return p.NC.PublishMsg(m)
}

// Close drains the connection.
func (p *Publisher) Close() {
	if p.NC != nil {
		_ = p.NC.Drain()
	}
}
