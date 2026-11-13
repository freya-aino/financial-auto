// Package synthetic is a local multi-instrument market generator for ephemeral test stacks (D-096).
// N instruments on venue `sim` are driven by K latent factors with regime switching (calm / volatile
// blocks, the same shape as the trainer's `generation::synthetic_bars`), so the whole pipeline —
// encoders, regions, strategies, risk, paper fills — gets structured traffic without any network.
// Each tick emits one Trade and one Quote per instrument; every NewsEvery ticks one NewsItem.
//
// Tunables (environment, the connector is not a catalog source):
//
//	FA_SYNTHETIC_INSTRUMENTS (8)  FA_SYNTHETIC_FACTORS (3)  FA_SYNTHETIC_SEED (1)
//	FA_SYNTHETIC_INTERVAL (1s)    FA_SYNTHETIC_NEWS_EVERY (60; 0 = no news)
package synthetic

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const (
	ID     = "synthetic"
	Venue  = "sim"
	Outlet = "synthetic"
	// RegimeBlock is the number of ticks between regime draws (trainer synthetic_bars uses 40 bars).
	RegimeBlock = 40
)

func init() {
	connector.Register(ID, func() connector.Connector { return FromEnv() })
}

// InstrumentID returns the fixed id of synthetic instrument k (UUID v7 layout, heartbeat is ...0001).
func InstrumentID(k int) string { return fmt.Sprintf("01900000-0000-7000-8000-%012d", 100+k) }

// Symbol returns the display symbol of synthetic instrument k.
func Symbol(k int) string { return fmt.Sprintf("SYN%02d", k) }

// Connector implements Connector, Streamer and Normalizer.
type Connector struct {
	Instruments int
	Factors     int
	Seed        uint64
	Interval    time.Duration
	NewsEvery   int

	deps    connector.Deps
	rng     *rand.Rand
	load    [][]float64 // [instrument][factor] loadings
	regime  []int       // per factor: 0 calm, 1 volatile
	price   []float64
	tickNum int
}

// FromEnv builds a connector from the FA_SYNTHETIC_* environment.
func FromEnv() *Connector {
	return &Connector{
		Instruments: envInt("FA_SYNTHETIC_INSTRUMENTS", 8),
		Factors:     envInt("FA_SYNTHETIC_FACTORS", 3),
		Seed:        uint64(envInt("FA_SYNTHETIC_SEED", 1)),
		Interval:    envDuration("FA_SYNTHETIC_INTERVAL", time.Second),
		NewsEvery:   envInt("FA_SYNTHETIC_NEWS_EVERY", 60),
	}
}

// Tick is the raw payload of one generator step (all instruments at once).
type Tick struct {
	At      time.Time `json:"at"`
	Prices  []float64 `json:"prices"`
	Sizes   []float64 `json:"sizes"`
	Buys    []bool    `json:"buys"`
	Spreads []float64 `json:"spreads"` // relative half-spread per instrument
	Regimes []int     `json:"regimes"`
	News    *NewsTick `json:"news,omitempty"`
}

// NewsTick is a templated headline about one instrument and its dominant factor's regime.
type NewsTick struct {
	Instrument int    `json:"instrument"`
	Title      string `json:"title"`
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Stream: true, Venue: Venue,
		Events: []connector.EventKind{connector.KindTrade, connector.KindQuote, connector.KindNews}}
}

func (c *Connector) Init(_ context.Context, _ connector.SourceConfig, deps connector.Deps) error {
	if c.Instruments < 1 || c.Factors < 1 {
		return fmt.Errorf("synthetic: need ≥ 1 instrument and factor, got %d / %d", c.Instruments, c.Factors)
	}
	if c.Interval <= 0 {
		c.Interval = time.Second
	}
	if deps.Clock == nil {
		deps.Clock = connector.SystemClock{}
	}
	c.deps = deps
	c.rng = rand.New(rand.NewPCG(c.Seed, c.Seed^0x9E3779B97F4A7C15))
	c.load = make([][]float64, c.Instruments)
	for k := range c.load {
		c.load[k] = make([]float64, c.Factors)
		// Each instrument loads mainly on one factor, weakly on the others: K recoverable clusters.
		for f := range c.load[k] {
			c.load[k][f] = 0.2 * c.rng.NormFloat64()
		}
		c.load[k][k%c.Factors] += 1
	}
	c.regime = make([]int, c.Factors)
	c.price = make([]float64, c.Instruments)
	for k := range c.price {
		c.price[k] = 50 + 10*float64(k)
	}
	return nil
}

// Step advances the generator by one tick and returns it. Exported for tests and replays.
func (c *Connector) Step(at time.Time) Tick {
	if c.tickNum%RegimeBlock == 0 {
		for f := range c.regime {
			c.regime[f] = c.rng.IntN(2)
		}
	}
	shock := make([]float64, c.Factors)
	for f := range shock {
		vol, drift := 0.001, 0.0003
		if c.regime[f] == 1 {
			vol, drift = 0.006, -0.0005
		}
		shock[f] = drift + vol*c.rng.NormFloat64()
	}
	t := Tick{At: at, Prices: make([]float64, c.Instruments), Sizes: make([]float64, c.Instruments),
		Buys: make([]bool, c.Instruments), Spreads: make([]float64, c.Instruments), Regimes: append([]int(nil), c.regime...)}
	for k := range c.price {
		r := 0.0005 * c.rng.NormFloat64() // idiosyncratic
		for f, l := range c.load[k] {
			r += l * shock[f]
		}
		c.price[k] *= math.Exp(r)
		dom := c.regime[k%c.Factors]
		size := (1 + 4*float64(dom)) * (0.5 + c.rng.Float64())
		t.Prices[k], t.Sizes[k], t.Buys[k] = c.price[k], size, r >= 0
		t.Spreads[k] = 0.0002 * (1 + 3*float64(dom))
	}
	if c.NewsEvery > 0 && c.tickNum%c.NewsEvery == c.NewsEvery-1 {
		k := c.rng.IntN(c.Instruments)
		mood := "steady trading"
		if c.regime[k%c.Factors] == 1 {
			mood = "volatility spikes"
		}
		t.News = &NewsTick{Instrument: k, Title: fmt.Sprintf("%s: %s in factor %d names", Symbol(k), mood, k%c.Factors)}
	}
	c.tickNum++
	return t
}

// Stream emits one Tick per interval.
func (c *Connector) Stream(ctx context.Context, _ []connector.Subscription, out chan<- connector.RawMessage) error {
	t := time.NewTicker(c.Interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			now := c.deps.Clock.Now()
			b, err := json.Marshal(c.Step(now))
			if err != nil {
				return err
			}
			msg := connector.RawMessage{SourceID: ID, ReceivedAt: now, Bytes: b, ContentType: "application/json"}
			select {
			case out <- msg:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
}

// Normalize turns one Tick into a Trade and a Quote per instrument (plus the optional news item).
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var tk Tick
	if err := json.Unmarshal(raw.Bytes, &tk); err != nil {
		return nil, nil, fmt.Errorf("synthetic: %w", err)
	}
	if len(tk.Sizes) != len(tk.Prices) || len(tk.Buys) != len(tk.Prices) || len(tk.Spreads) != len(tk.Prices) {
		return nil, nil, fmt.Errorf("synthetic: ragged tick (%d prices)", len(tk.Prices))
	}
	env := func(inst string, p any) *fav1.Envelope {
		e := &fav1.Envelope{EventId: ulid.Make().String(), SourceId: ID, InstrumentId: inst,
			EventTime: timestamppb.New(tk.At), PublishTime: timestamppb.New(tk.At),
			IngestTime: timestamppb.New(raw.ReceivedAt), SchemaVersion: 1}
		switch v := p.(type) {
		case *fav1.Trade:
			e.Payload = &fav1.Envelope_Trade{Trade: v}
		case *fav1.Quote:
			e.Payload = &fav1.Envelope_Quote{Quote: v}
		case *fav1.NewsItem:
			e.Payload = &fav1.Envelope_News{News: v}
		}
		return e
	}
	envs := make([]*fav1.Envelope, 0, 2*len(tk.Prices)+1)
	for k, p := range tk.Prices {
		side := fav1.Side_SIDE_SELL
		if tk.Buys[k] {
			side = fav1.Side_SIDE_BUY
		}
		inst := InstrumentID(k)
		half := p * tk.Spreads[k]
		envs = append(envs,
			env(inst, &fav1.Trade{Price: p, Size: tk.Sizes[k], Aggressor: side}),
			env(inst, &fav1.Quote{BidPrice: p - half, BidSize: 10 * tk.Sizes[k], AskPrice: p + half, AskSize: 10 * tk.Sizes[k]}))
	}
	if n := tk.News; n != nil {
		envs = append(envs, env(InstrumentID(n.Instrument), &fav1.NewsItem{Title: n.Title, Language: "en", Outlet: Outlet,
			EntitiesRaw: []string{Symbol(n.Instrument)}, DedupKey: fmt.Sprintf("synthetic:%d", tk.At.UnixNano())}))
	}
	return envs, nil, nil
}

func envInt(key string, def int) int {
	if n, err := strconv.Atoi(os.Getenv(key)); err == nil {
		return n
	}
	return def
}

func envDuration(key string, def time.Duration) time.Duration {
	if d, err := time.ParseDuration(os.Getenv(key)); err == nil {
		return d
	}
	return def
}
