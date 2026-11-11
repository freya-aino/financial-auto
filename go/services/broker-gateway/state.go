package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"google.golang.org/protobuf/proto"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// Restart safety (spec 40 §8 "broker disconnect mid-execution", D-076 open item): working algo
// orders, resting limit orders, the recently seen intent ids (idempotency) and the per-account
// net positions the fail-closed guard judges "risk-increasing" by are persisted to
// `$FA_STATE_DIR/orders.json` (atomic rename) whenever they change, at most once per second, and
// at shutdown. On start the file is loaded before any subscription: a TWAP interrupted by a
// restart resumes its schedule (slices that fell due while the gateway was down go out with the
// next tick, priced at the then-current market), a resting limit keeps resting, a redelivered
// intent is still a duplicate.

const stateVersion = 1

// maxSeen bounds the idempotency memory (and the persisted list).
const maxSeen = 50_000

type persistedWorking struct {
	Intent    []byte    `json:"intent"` // fa.v1.OrderIntent
	Algo      AlgoSpec  `json:"algo"`
	Token     string    `json:"token"`
	Arrival   float64   `json:"arrival"`
	Start     time.Time `json:"start"`
	Filled    float64   `json:"filled"`
	Seq       int       `json:"seq"`
	Cum       []float64 `json:"cum,omitempty"`
	Allowance float64   `json:"allowance,omitempty"`
	Limit     float64   `json:"limit,omitempty"`
}

type persistedResting struct {
	Intent  []byte  `json:"intent"`
	Arrival float64 `json:"arrival"`
	Token   string  `json:"token"`
}

// GatewayState is the persisted form (`orders.json`).
type GatewayState struct {
	Version   int                `json:"version"`
	SavedAt   time.Time          `json:"saved_at"`
	Working   []persistedWorking `json:"working"`
	Resting   []persistedResting `json:"resting"`
	Seen      []string           `json:"seen"`
	Positions map[string]float64 `json:"positions"`
	FillSeq   uint64             `json:"fill_seq"`
}

func posKey(account, instrument string) string { return account + "|" + instrument }

// markSeen records an intent id (caller holds p.mu), bounded to maxSeen.
func (p *PaperEngine) markSeen(id string) {
	p.seen[id] = true
	p.seenOrder = append(p.seenOrder, id)
	if over := len(p.seenOrder) - maxSeen; over > 0 {
		for _, old := range p.seenOrder[:over] {
			delete(p.seen, old)
		}
		p.seenOrder = append([]string(nil), p.seenOrder[over:]...)
	}
}

// touch marks the persisted state dirty (caller holds p.mu).
func (p *PaperEngine) touch() { p.rev++ }

// Revision changes whenever persisted state changed.
func (p *PaperEngine) Revision() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.rev
}

// Snapshot returns the persistable state.
func (p *PaperEngine) Snapshot() (*GatewayState, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	st := &GatewayState{Version: stateVersion, SavedAt: p.now().UTC(), Positions: map[string]float64{}, FillSeq: p.fillSeq}
	for _, w := range p.working {
		b, err := proto.Marshal(w.intent)
		if err != nil {
			return nil, err
		}
		st.Working = append(st.Working, persistedWorking{
			Intent: b, Algo: w.algo, Token: w.token, Arrival: w.arrival, Start: w.start, Filled: w.filled,
			Seq: w.seq, Cum: w.cum, Allowance: w.allowance, Limit: w.limit,
		})
	}
	for _, r := range p.resting {
		b, err := proto.Marshal(r.intent)
		if err != nil {
			return nil, err
		}
		st.Resting = append(st.Resting, persistedResting{Intent: b, Arrival: r.arrival, Token: r.token})
	}
	st.Seen = append(st.Seen, p.seenOrder...)
	for k, v := range p.positions {
		if v != 0 {
			st.Positions[k] = v
		}
	}
	return st, nil
}

// Restore replaces the order state with a loaded snapshot (call before any traffic).
func (p *PaperEngine) Restore(st *GatewayState) error {
	if st == nil {
		return nil
	}
	if st.Version != stateVersion {
		return fmt.Errorf("orders state version %d, want %d", st.Version, stateVersion)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, pw := range st.Working {
		var in fav1.OrderIntent
		if err := proto.Unmarshal(pw.Intent, &in); err != nil {
			return fmt.Errorf("working order: %w", err)
		}
		p.working[in.IntentId] = &working{
			intent: &in, algo: pw.Algo, token: pw.Token, arrival: pw.Arrival, start: pw.Start, filled: pw.Filled,
			seq: pw.Seq, cum: pw.Cum, allowance: pw.Allowance, limit: pw.Limit,
		}
		p.algoCount(pw.Algo.Name, "working", 1)
	}
	for _, pr := range st.Resting {
		var in fav1.OrderIntent
		if err := proto.Unmarshal(pr.Intent, &in); err != nil {
			return fmt.Errorf("resting order: %w", err)
		}
		p.resting[in.IntentId] = &resting{intent: &in, arrival: pr.Arrival, token: pr.Token}
	}
	for _, id := range st.Seen {
		p.markSeen(id)
	}
	for k, v := range st.Positions {
		p.positions[k] = v
	}
	if st.FillSeq > p.fillSeq {
		p.fillSeq = st.FillSeq
	}
	return nil
}

// StateStore writes and reads `orders.json`.
type StateStore struct {
	Path string
	mu   sync.Mutex
	last uint64
}

// Load reads the state file; (nil, nil) when it does not exist.
func (s *StateStore) Load() (*GatewayState, error) {
	b, err := os.ReadFile(s.Path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var st GatewayState
	if err := json.Unmarshal(b, &st); err != nil {
		return nil, fmt.Errorf("%s: %w", s.Path, err)
	}
	return &st, nil
}

// SaveIfChanged persists the engine state when its revision moved (or force).
func (s *StateStore) SaveIfChanged(p *PaperEngine, force bool) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rev := p.Revision()
	if !force && rev == s.last {
		return false, nil
	}
	st, err := p.Snapshot()
	if err != nil {
		return false, err
	}
	b, err := json.Marshal(st)
	if err != nil {
		return false, err
	}
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o755); err != nil {
		return false, err
	}
	if err := writeAtomic(s.Path, b); err != nil {
		return false, err
	}
	s.last = rev
	return true, nil
}

// RiskGuard is the fail-closed rule of spec 40 §8 ("risk service crash → broker-gateway rejects
// all new risk-increasing intents if no risk heartbeat within T"): it tracks the last
// `svc.heartbeat.risk-engine`; while that is older than Timeout (or none was ever seen), an
// intent that would increase the account's absolute position in the instrument — counting the
// filled position plus every working / resting order's remainder — is rejected. Reductions (and
// closing to zero) still execute, so a flatten issued before the crash completes.
type RiskGuard struct {
	mu      sync.Mutex
	last    time.Time
	Timeout time.Duration
}

func (g *RiskGuard) Beat(at time.Time) {
	g.mu.Lock()
	if at.After(g.last) {
		g.last = at
	}
	g.mu.Unlock()
}

// Age of the last heartbeat (math.Inf when none was seen).
func (g *RiskGuard) Age(now time.Time) float64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.last.IsZero() {
		return math.Inf(1)
	}
	return now.Sub(g.last).Seconds()
}

// Alive reports whether the risk-engine heartbeat is fresh; a zero Timeout disables the guard.
func (g *RiskGuard) Alive(now time.Time) bool {
	if g == nil || g.Timeout <= 0 {
		return true
	}
	return g.Age(now) <= g.Timeout.Seconds()
}

// projected returns the account's position in the instrument after every working / resting
// order completes (caller holds p.mu).
func (p *PaperEngine) projected(account, instrument string) float64 {
	pos := p.positions[posKey(account, instrument)]
	for _, w := range p.working {
		if w.intent.AccountId == account && w.intent.InstrumentId == instrument {
			pos += signedQty(w.intent.Side, w.remaining())
		}
	}
	for _, r := range p.resting {
		if r.intent.AccountId == account && r.intent.InstrumentId == instrument {
			pos += signedQty(r.intent.Side, r.intent.Quantity)
		}
	}
	return pos
}

func signedQty(s fav1.Side, q float64) float64 {
	if s == fav1.Side_SIDE_SELL {
		return -q
	}
	return q
}

// increasesRisk: the order moves the projected position away from zero (or through it).
func increasesRisk(pos, signed float64) bool {
	after := pos + signed
	eps := 1e-12 * math.Max(1, math.Abs(pos))
	if math.Abs(after) > math.Abs(pos)+eps {
		return true
	}
	return after != 0 && pos != 0 && math.Signbit(after) != math.Signbit(pos) && math.Abs(after) > eps
}
