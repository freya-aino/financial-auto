package tsdb

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"testing"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// Live round trip against a real VictoriaMetrics (D-038): three trades in one millisecond must be
// stored as ONE sample per series with the VWAP price and the summed size.
// Run: FA_VM_TEST=1 FA_VM_URL=http://victoriametrics:8428 go test ./pkg/tsdb -run Live
func TestLiveVWAPRoundTrip(t *testing.T) {
	if os.Getenv("FA_VM_TEST") != "1" {
		t.Skip("set FA_VM_TEST=1 (and FA_VM_URL) to run against VictoriaMetrics")
	}
	vm := os.Getenv("FA_VM_URL")
	if vm == "" {
		vm = "http://127.0.0.1:8428"
	}
	src := fmt.Sprintf("vwaptest%d", time.Now().UnixNano())
	at := time.Now().Add(-time.Minute).Truncate(time.Millisecond)
	w := New(vm, Options{})
	t.Cleanup(func() { // drop the test series again (only this run's unique src label)
		q := url.Values{"match[]": {fmt.Sprintf(`{src=%q}`, src)}}
		if res, err := http.Post(vm+"/api/v1/admin/tsdb/delete_series?"+q.Encode(), "", nil); err == nil {
			res.Body.Close()
		}
	})
	for _, tr := range []struct{ px, sz float64 }{{100, 1}, {103, 2}, {110, 0}} {
		w.AddEnvelope(&fav1.Envelope{SourceId: src, InstrumentId: "01900000-0000-7000-8000-0000000000ff",
			EventTime: timestamppb.New(at), Payload: &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: tr.px, Size: tr.sz}}}, "test")
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	if st := w.Stats(); st.Aggregated != 2 || st.Written != 3 {
		t.Fatalf("stats %+v", st)
	}
	query := func(metric string) [][]any {
		q := url.Values{"query": {fmt.Sprintf(`%s{src=%q}[5m]`, metric, src)}, "time": {fmt.Sprint(time.Now().Unix())}}
		var out struct {
			Data struct {
				Result []struct {
					Values [][]any `json:"values"`
				} `json:"result"`
			} `json:"data"`
		}
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) {
			_, _ = http.Get(vm + "/internal/force_flush")
			res, err := http.Get(vm + "/api/v1/query?" + q.Encode())
			if err == nil {
				_ = json.NewDecoder(res.Body).Decode(&out)
				res.Body.Close()
				if len(out.Data.Result) == 1 {
					return out.Data.Result[0].Values
				}
			}
			time.Sleep(500 * time.Millisecond)
		}
		t.Fatalf("%s for src=%s not found", metric, src)
		return nil
	}
	px, sz, n := query(MetricTradePrice), query(MetricTradeSize), query(MetricTradeCount)
	if len(px) != 1 || len(sz) != 1 || len(n) != 1 {
		t.Fatalf("want one sample each, got price %v size %v count %v", px, sz, n)
	}
	if px[0][1] != "102" || sz[0][1] != "3" || n[0][1] != "3" { // (100·1 + 103·2 + 110·0) / 3 = 102
		t.Fatalf("price %v size %v count %v, want 102 / 3 / 3", px[0], sz[0], n[0])
	}
}
