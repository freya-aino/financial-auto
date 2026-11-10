package connector

import (
	"context"
	"testing"
	"time"
)

type sleepLimiter struct{ d time.Duration }

func (s sleepLimiter) Wait(context.Context) error { time.Sleep(s.d); return nil }

func TestMeteredLimiterSaturation(t *testing.T) {
	m := NewMeteredLimiter(sleepLimiter{20 * time.Millisecond})
	for range 3 {
		_ = m.Wait(context.Background())
	}
	st := m.Snapshot(true)
	if st.Requests != 3 || st.Total != 3 || st.Saturation <= 0.5 || st.Saturation > 1 {
		t.Fatalf("stats %+v", st)
	}
	st = m.Snapshot(false)
	if st.Requests != 0 || st.Total != 3 || st.Saturation != 0 {
		t.Fatalf("after reset %+v", st)
	}
	u := NewMeteredLimiter(nil)
	_ = u.Wait(context.Background())
	if s := u.Snapshot(false); s.Requests != 1 || s.Saturation != 0 {
		t.Fatalf("unlimited %+v", s)
	}
}

func TestMemBackfillCursors(t *testing.T) {
	c := NewMemBackfillCursors()
	ctx := context.Background()
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	if _, _, ok, _ := c.GetBackfill(ctx, "yahoo", "i", "1m", start); ok {
		t.Fatal("unexpected row")
	}
	if err := c.SetBackfill(ctx, "yahoo", "i", "1m", start, start.Add(time.Hour), start.Add(30*time.Minute), "running"); err != nil {
		t.Fatal(err)
	}
	cur, status, ok, err := c.GetBackfill(ctx, "yahoo", "i", "1m", start)
	if err != nil || !ok || status != "running" || !cur.Equal(start.Add(30*time.Minute)) {
		t.Fatalf("got %v %q %v %v", cur, status, ok, err)
	}
}
