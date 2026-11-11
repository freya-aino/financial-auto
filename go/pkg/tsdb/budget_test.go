package tsdb

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func pt(metric, inst string, extra ...string) Point {
	l := map[string]string{"inst": inst, "venue": "v", "src": "s"}
	for i := 0; i+1 < len(extra); i += 2 {
		l[extra[i]] = extra[i+1]
	}
	return Point{Metric: metric, Labels: l, Value: 1, TsMs: 1}
}

func TestBudgetDemotesTicksKeepsBars(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	b := newSeriesBudget(4, time.Hour)
	// inst a: trade price + size + 1m bar close = 3 series
	for _, p := range []Point{pt(MetricTradePrice, "a"), pt(MetricTradeSize, "a"), pt(MetricBarClose, "a", "tf", "1m")} {
		if v := b.admit(p, now); v != admitted {
			t.Fatalf("%v: %v", p.Metric, v)
		}
	}
	if v := b.admit(pt(MetricTradePrice, "a"), now); v != admitted {
		t.Fatal("known series must stay admitted")
	}
	// inst b: the 4th series fits, the 5th demotes b
	if b.admit(pt(MetricTradePrice, "b"), now) != admitted {
		t.Fatal("4th series fits")
	}
	if v := b.admit(pt(MetricTradeSize, "b"), now); v != droppedNewDemotion {
		t.Fatalf("5th tick series: %v", v)
	}
	// b's already-admitted tick series is dropped too once demoted; a is untouched
	if v := b.admit(pt(MetricTradePrice, "b"), now); v != droppedDemoted {
		t.Fatalf("demoted instrument tick: %v", v)
	}
	if v := b.admit(pt(MetricTradePrice, "a"), now); v != admitted {
		t.Fatalf("other instrument: %v", v)
	}
	// a new bar series over budget is rejected, not demoted
	if v := b.admit(pt(MetricBarClose, "c", "tf", "1m"), now); v != rejectedBar {
		t.Fatalf("bar over budget: %v", v)
	}
	// after the window, idle series expire and free the budget for bars of b
	b.expire(now.Add(2 * time.Hour))
	if len(b.seen) != 0 {
		t.Fatalf("expired: %d", len(b.seen))
	}
	if v := b.admit(pt(MetricBarClose, "b", "tf", "1m"), now.Add(2*time.Hour)); v != admitted {
		t.Fatalf("bar of demoted instrument: %v", v)
	}
	if v := b.admit(pt(MetricQuoteBidPx, "b"), now.Add(2*time.Hour)); v != droppedDemoted {
		t.Fatalf("demotion persists: %v", v)
	}
}

func TestBudgetUsesVMCount(t *testing.T) {
	b := newSeriesBudget(10, time.Hour)
	b.vm = 10 // other writers already fill the budget
	if v := b.admit(pt(MetricTradePrice, "a"), time.Now()); v != droppedNewDemotion {
		t.Fatalf("%v", v)
	}
	if b := newSeriesBudget(-1, time.Hour); b.admit(pt(MetricTradePrice, "a"), time.Now()) != admitted {
		t.Fatal("disabled budget admits")
	}
}

func TestParseVMActive(t *testing.T) {
	body := "vm_cache_entries{type=\"storage/date_metric_ids\"} 9\nvm_cache_entries{type=\"storage/hour_metric_ids\"} 7045\nx 1\n"
	n, err := parseVMActive(bufio.NewScanner(strings.NewReader(body)))
	if err != nil || n != 7045 {
		t.Fatalf("%d %v", n, err)
	}
}

// End to end through the Writer: a fake VM reporting a nearly full budget makes the writer
// demote the second instrument's ticks while its bars and the first instrument keep flowing.
func TestWriterSeriesBudget(t *testing.T) {
	var mu sync.Mutex
	imported := map[string]int{} // metric|inst → samples
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			fmt.Fprintln(w, `vm_cache_entries{type="storage/hour_metric_ids"} 0`)
			return
		}
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		defer mu.Unlock()
		for _, ln := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			var l line
			if err := json.Unmarshal([]byte(ln), &l); err != nil {
				t.Errorf("bad line %q", ln)
				continue
			}
			imported[l.Metric["__name__"]+"|"+l.Metric["inst"]] += len(l.Values)
		}
	}))
	defer srv.Close()
	w := New(srv.URL, Options{FlushInterval: time.Hour, MaxActiveSeries: 5})
	w.Add(pt(MetricTradePrice, "a"), pt(MetricTradeSize, "a"), pt(MetricBarClose, "b", "tf", "1m"),
		pt(MetricBarOpen, "b", "tf", "1m"))
	w.Add(pt(MetricTradePrice, "b"), pt(MetricTradeSize, "b"), pt(MetricTradePrice, "b"))
	w.Add(pt(MetricBarHigh, "b", "tf", "1m"))
	if err := w.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	st := w.Stats()
	if st.Demoted != 1 || st.OverBudget != 2 || st.RejectedBars != 1 || st.ActiveSeries != 5 || st.Written != 5 {
		t.Fatalf("stats %+v", st)
	}
	if got := w.Demoted(); len(got) != 1 || got[0] != "b|v|s" {
		t.Fatalf("demoted %v", got)
	}
	mu.Lock()
	defer mu.Unlock()
	if imported[MetricTradePrice+"|b"] != 1 || imported[MetricTradeSize+"|b"] != 0 || imported[MetricBarClose+"|b"] != 1 {
		t.Fatalf("imported %v", imported)
	}
	if w.Promote() != 1 || len(w.Demoted()) != 0 {
		t.Fatal("promote")
	}
}
