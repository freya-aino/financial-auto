package bus

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/nats-io/nats.go"
	"google.golang.org/protobuf/proto"
)

// Runtime contract-version check (spec 21 §6, §8; mirror of crates/fa-bus/src/contract.rs).
// Every payload is package fa.v1 today. A breaking change ships as fa.v2 side by side and its
// producers set the header `Fa-Schema: fa.v2`; a message without the header is fa.v1, so v1
// producers add no bytes. Consumers reject majors they cannot decode instead of mis-decoding.

// SchemaHeader names the protobuf package of the payload.
const SchemaHeader = "Fa-Schema"

// ProtoMajor is the package major this build is generated from (go/pkg/bus/pb/fa/v1).
const ProtoMajor = 1

// ParseMajor reads "fa.v2", "v2" or "2".
func ParseMajor(v string) (int, bool) {
	v = strings.TrimPrefix(strings.TrimSpace(v), "fa.")
	v = strings.TrimPrefix(v, "v")
	n, err := strconv.Atoi(v)
	return n, err == nil
}

// CheckSchema returns nil for a message without the header or with a supported major.
func CheckSchema(h nats.Header) error {
	v := h.Get(SchemaHeader)
	if v == "" {
		return nil
	}
	n, ok := ParseMajor(v)
	if !ok {
		return fmt.Errorf("malformed %s header %q", SchemaHeader, v)
	}
	if n != ProtoMajor {
		return fmt.Errorf("unsupported contract major version %d (this build decodes %d)", n, ProtoMajor)
	}
	return nil
}

// Decode checks the contract version of m and unmarshals its payload into into.
func Decode(m *nats.Msg, into proto.Message) error {
	if err := CheckSchema(m.Header); err != nil {
		return err
	}
	return proto.Unmarshal(m.Data, into)
}
