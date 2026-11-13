package synthetic

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/fey/financial-auto/go/pkg/connector"
)

func newConn(t *testing.T, n, k int) *Connector {
	t.Helper()
	c := &Connector{Instruments: n, Factors: k, Seed: 7, Interval: 5 * time.Millisecond, NewsEvery: 3}
	if err := c.Init(context.Background(), connector.SourceConfig{}, connector.Deps{}); err != nil {
		t.Fatal(err)
	}
	return c
}

func TestStreamAndNormalize(t *testing.T) {
	c := newConn(t, 4, 2)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out := make(chan connector.RawMessage, 1)
	go func() { _ = c.Stream(ctx, nil, out) }()
	news := 0
	for i := 0; i < 3; i++ {
		envs, unresolved, err := c.Normalize(<-out)
		if err != nil || len(unresolved) != 0 {
			t.Fatalf("normalize: %v %v", unresolved, err)
		}
		for _, e := range envs {
			if e.EventTime == nil || e.PublishTime == nil || e.IngestTime == nil || e.InstrumentId == "" {
				t.Fatalf("incomplete envelope %v", e)
			}
			if q := e.GetQuote(); q != nil && !(q.BidPrice < q.AskPrice && q.BidPrice > 0) {
				t.Fatalf("crossed quote %v", q)
			}
			if e.GetNews() != nil {
				news++
			}
		}
		if want := 2 * 4; len(envs) < want {
			t.Fatalf("tick %d: %d envelopes, want ≥ %d", i, len(envs), want)
		}
	}
	if news != 1 {
		t.Fatalf("news every 3 ticks: got %d in 3", news)
	}
}

func TestDeterministicPerSeed(t *testing.T) {
	a, b := newConn(t, 3, 2), newConn(t, 3, 2)
	at := time.Unix(0, 0)
	for i := 0; i < 100; i++ {
		ta, tb := a.Step(at), b.Step(at)
		for k := range ta.Prices {
			if ta.Prices[k] != tb.Prices[k] {
				t.Fatalf("step %d instrument %d differs", i, k)
			}
		}
	}
}

// Instruments sharing a factor co-move far more than instruments on different factors.
func TestFactorStructureIsRecoverable(t *testing.T) {
	c := newConn(t, 6, 3) // instrument k loads on factor k%3: pairs (0,3) (1,4) (2,5)
	c.NewsEvery = 0
	prev := c.Step(time.Unix(0, 0)).Prices
	rets := make([][]float64, 6)
	for i := 0; i < 4000; i++ {
		p := c.Step(time.Unix(int64(i), 0)).Prices
		for k := range p {
			rets[k] = append(rets[k], math.Log(p[k]/prev[k]))
		}
		prev = p
	}
	same, cross := corr(rets[0], rets[3]), corr(rets[0], rets[1])
	if same < 0.6 || math.Abs(cross) > 0.4 {
		t.Fatalf("same-factor corr %.2f, cross-factor corr %.2f", same, cross)
	}
}

func corr(a, b []float64) float64 {
	n := float64(len(a))
	var ma, mb float64
	for i := range a {
		ma += a[i] / n
		mb += b[i] / n
	}
	var sab, saa, sbb float64
	for i := range a {
		sab += (a[i] - ma) * (b[i] - mb)
		saa += (a[i] - ma) * (a[i] - ma)
		sbb += (b[i] - mb) * (b[i] - mb)
	}
	return sab / math.Sqrt(saa*sbb)
}
