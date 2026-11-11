package bus

import (
	"testing"

	"github.com/nats-io/nats.go/jetstream"
)

func TestCoalesceQuotesOnly(t *testing.T) {
	subs := []string{"md.quote.k.a", "md.trade.k.a", "md.quote.k.b", "md.quote.k.a", "order.event.paper",
		"md.book.k.a", "md.quote.k.a", "md.trade.k.a", "md.book.k.a"}
	idx := make([]int, len(subs))
	for i := range idx {
		idx[i] = i
	}
	kept, sup := Coalesce(idx, Coalescable, func(i int) string { return subs[i] })
	want := []int{1, 2, 4, 6, 7, 8}
	if len(kept) != len(want) || len(sup) != 3 {
		t.Fatalf("kept %v sup %v", kept, sup)
	}
	for i := range want {
		if kept[i] != want[i] {
			t.Fatalf("kept %v", kept)
		}
	}
	if (LiveSpec{Coalesce: []string{"order."}}).Validate() == nil || (LiveSpec{Coalesce: []string{"md."}}).Validate() == nil {
		t.Fatal("orders / all md must not coalesce")
	}
	c := MarketDataLive("MD", "x", "md.quote.>").Config()
	if c.MaxAckPending != 10_000 || c.AckPolicy != jetstream.AckExplicitPolicy || c.DeliverPolicy != jetstream.DeliverNewPolicy {
		t.Fatalf("%+v", c)
	}
}
