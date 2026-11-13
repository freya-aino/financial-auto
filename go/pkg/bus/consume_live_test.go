package bus

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
)

// Against the compose NATS: FA_NATS_TEST=1 go test ./pkg/bus -run LiveRun -v
// A handler interrupted before its ack (context cancelled mid-handling) gets the message again
// after the restart; handled ones are not redelivered; an unknown contract major is dropped; an
// idle consumer with ResumeWithin skips its backlog.
func TestLiveRunRedeliversAndSkipsStaleBacklog(t *testing.T) {
	if os.Getenv("FA_NATS_TEST") == "" {
		t.Skip("set FA_NATS_TEST=1")
	}
	url := os.Getenv("FA_NATS_URL")
	if url == "" {
		url = nats.DefaultURL
	}
	nc, err := nats.Connect(url)
	if err != nil {
		t.Fatal(err)
	}
	defer nc.Close()
	js, _ := jetstream.New(nc)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	n := time.Now().UnixNano() % 1_000_000_000
	name, root := fmt.Sprintf("FAGORUN%d", n), fmt.Sprintf("fagotest.run%d", n)
	if _, err := js.CreateStream(ctx, jetstream.StreamConfig{Name: name, Subjects: []string{root + ".>"}}); err != nil {
		t.Fatal(err)
	}
	defer js.DeleteStream(context.Background(), name)
	spec := DurableLive(name, "run", root+".>")
	spec.AckWait = 2 * time.Second
	lc, err := EnsureLive(ctx, js, spec)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		m := nats.NewMsg(root + ".order.event.paper")
		m.Data = []byte(fmt.Sprint(i))
		if i == 1 {
			m.Header.Set(SchemaHeader, "fa.v9")
		}
		if _, err := js.PublishMsg(ctx, m); err != nil {
			t.Fatal(err)
		}
	}
	// first run: handles 0, then blocks in the handler of 2 until cancelled (never acked)
	run1, stop1 := context.WithCancel(ctx)
	var seen []string
	go func() {
		lc.Run(run1, func(_ string, data []byte) {
			seen = append(seen, string(data))
			if string(data) == "2" {
				stop1()
				<-run1.Done()
				runtime.Goexit() // "crash": the handler never returns, so 2 is never acked
			}
		})

	}()
	<-run1.Done()
	time.Sleep(100 * time.Millisecond)
	if fmt.Sprint(seen) != "[0 2]" || lc.RejectedSchema != 1 {
		t.Fatalf("first run saw %v, rejected %d", seen, lc.RejectedSchema)
	}
	// restart: 2 comes back after ack_wait, then 3 and 4
	lc2, err := EnsureLive(ctx, js, spec)
	if err != nil {
		t.Fatal(err)
	}
	run2, stop2 := context.WithCancel(ctx)
	var after []string
	go lc2.Run(run2, func(_ string, data []byte) {
		after = append(after, string(data))
		if len(after) == 3 {
			stop2()
		}
	})
	<-run2.Done()
	time.Sleep(100 * time.Millisecond)
	if fmt.Sprint(after) != "[2 3 4]" {
		t.Fatalf("after restart %v, want [2 3 4]", after)
	}
	// stale backlog
	if _, err := js.Publish(ctx, root+".order.event.paper", []byte("5")); err != nil {
		t.Fatal(err)
	}
	time.Sleep(1200 * time.Millisecond)
	stale := spec
	stale.ResumeWithin = time.Millisecond
	lc3, err := EnsureLive(ctx, js, stale)
	if err != nil {
		t.Fatal(err)
	}
	info, err := lc3.Consumer.Info(ctx)
	if err != nil || info.NumPending != 0 {
		t.Fatalf("backlog not skipped: %+v %v", info, err)
	}
}
