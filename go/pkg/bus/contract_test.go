package bus

import (
	"testing"

	"github.com/nats-io/nats.go"
	"google.golang.org/protobuf/proto"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

func TestContractMajor(t *testing.T) {
	if CheckSchema(nil) != nil || CheckSchema(nats.Header{}) != nil {
		t.Fatal("absent header is v1")
	}
	for v, ok := range map[string]bool{"fa.v1": true, "1": true, "fa.v2": false, "v3": false, "fa.vX": false} {
		h := nats.Header{}
		h.Set(SchemaHeader, v)
		if (CheckSchema(h) == nil) != ok {
			t.Errorf("%s: %v", v, CheckSchema(h))
		}
	}
	b, _ := proto.Marshal(&fav1.Heartbeat{Service: "x"})
	m := &nats.Msg{Subject: "svc.heartbeat.x", Data: b, Header: nats.Header{}}
	var hb fav1.Heartbeat
	if err := Decode(m, &hb); err != nil || hb.Service != "x" {
		t.Fatal(err)
	}
	m.Header.Set(SchemaHeader, "fa.v2")
	if Decode(m, &hb) == nil {
		t.Fatal("v2 must be rejected")
	}
}
