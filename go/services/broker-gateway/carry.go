package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/fey/financial-auto/go/pkg/bus"
)

// Paper carry (spec 40 §6.5 shorts, spec 41 §6.3): the venue charges what the backtest charges
// (`fa_backtest::cost::ClassCosts::carry`) — `funding_bps_day` on any open position, plus
// `borrow_bps_day` on shorts — of the schedule of the position's (venue, class) in `costs.yaml`.
//
// Accrual is time-weighted: every scheduler tick adds |qty · last| · dt / 1 d per open
// (account, instrument) position to the running funding and (shorts only) borrow
// notional-days. At the end of every `CarryPeriod` (UTC grid, 1 h) each non-zero accrual
// becomes one CarryCharge, published as JSON on `exec.carry.<account>` and kept for
// `carryRepublish` periods: every period re-announces the recent charges, so a booking
// consumer that was briefly down still books them (the risk-engine books each reference once,
// `ledger.cash_movement` is unique on (book, venue, kind, ref)). Amounts are in the
// instrument's price (quote) currency, negative = paid; the booking side converts to EUR as of
// `period_end`. Accruals and the recent charges persist in `orders.json`, so a restart neither
// drops nor repeats a period; time the gateway was down is accrued on the first tick after it
// (at most `carryMaxGap`), priced at the then-current mark.

// CarrySubjectPrefix + account is the subject of carry charges.
const CarrySubjectPrefix = "exec.carry."

const (
	carryRepublish = 6
	carryMaxGap    = 7 * 24 * time.Hour
)

// CarryCharge is one funding or borrow charge for one (account, instrument) and period.
type CarryCharge struct {
	Account      string    `json:"account"`
	InstrumentID string    `json:"instrument_id"`
	Kind         string    `json:"kind"`   // funding | borrow
	Amount       float64   `json:"amount"` // quote currency, negative = paid
	Ccy          string    `json:"ccy"`    // empty = the instrument's quote currency
	BpsDay       float64   `json:"bps_day"`
	NotionalDays float64   `json:"notional_days"` // Σ |qty·price| · days accrued
	Venue        string    `json:"venue"`
	Class        string    `json:"class"`
	PeriodStart  time.Time `json:"period_start"`
	PeriodEnd    time.Time `json:"period_end"`
	// Reference is the idempotency key of the booking (`ledger.cash_movement.ref`).
	Reference string `json:"reference"`
}

type carryAccrual struct {
	Funding float64 `json:"funding"` // Σ notional · days · funding_bps_day (bps·EUR·d)
	Borrow  float64 `json:"borrow"`
	FundND  float64 `json:"fund_nd"` // notional-days behind Funding
	BorrND  float64 `json:"borr_nd"`
	FundBps float64 `json:"fund_bps"` // last rate applied (report)
	BorrBps float64 `json:"borr_bps"`
	Venue   string  `json:"venue"`
	Class   string  `json:"class"`
}

// carryState is the persisted carry book.
type carryState struct {
	LastAt      time.Time                `json:"last_at"`
	PeriodStart time.Time                `json:"period_start"`
	Accruals    map[string]*carryAccrual `json:"accruals"` // posKey → accrual
	Recent      [][]CarryCharge          `json:"recent"`   // newest last, ≤ carryRepublish periods
}

func newCarryState() *carryState { return &carryState{Accruals: map[string]*carryAccrual{}} }

// accrueCarry adds the carry of every open position since the last call (caller holds p.mu)
// and closes the period when `now` crossed its end. Returns the charges to publish (the closed
// period's plus the recent ones, re-announced), nil when no period closed.
func (p *PaperEngine) accrueCarry(now time.Time) []CarryCharge {
	if p.CarryPeriod <= 0 {
		return nil
	}
	c := p.carry
	if c.LastAt.IsZero() || now.Before(c.LastAt) {
		c.LastAt = now
		if c.PeriodStart.IsZero() {
			c.PeriodStart = now.Truncate(p.CarryPeriod)
		}
		return nil
	}
	var out []CarryCharge
	closedN := 0
	for !now.Before(c.PeriodStart.Add(p.CarryPeriod)) {
		end := c.PeriodStart.Add(p.CarryPeriod)
		p.accrueUntil(end)
		c.Recent = append(c.Recent, p.closeCarryPeriod(end))
		closedN++
		c.PeriodStart = end
	}
	p.accrueUntil(now)
	if closedN == 0 {
		return nil
	}
	if len(c.Recent) > carryRepublish+closedN {
		c.Recent = c.Recent[len(c.Recent)-carryRepublish-closedN:]
	}
	// the closed periods plus the older recent ones, re-announced (idempotent downstream)
	for _, per := range c.Recent {
		out = append(out, per...)
	}
	if len(c.Recent) > carryRepublish {
		c.Recent = c.Recent[len(c.Recent)-carryRepublish:]
	}
	return out
}

// accrueUntil integrates every open position from LastAt to t at the current marks.
func (p *PaperEngine) accrueUntil(t time.Time) {
	c := p.carry
	dt := t.Sub(c.LastAt)
	if dt <= 0 {
		return
	}
	if dt > carryMaxGap {
		dt = carryMaxGap
	}
	days := dt.Seconds() / 86_400
	keys := make([]string, 0, len(p.positions))
	for k := range p.positions {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		qty := p.positions[k]
		if qty == 0 || math.IsNaN(qty) {
			continue
		}
		_, inst, ok := strings.Cut(k, "|")
		if !ok {
			continue
		}
		m, ok := p.markets[bus.SubjectToken(inst)]
		if !ok || m.last <= 0 {
			continue // no mark: nothing to price the carry at (counted)
		}
		class := p.Costs.ClassOf(inst)
		s := p.Costs.Schedule(m.venue, class)
		if s.FundingBpsDay == 0 && (qty > 0 || s.BorrowBpsDay == 0) {
			continue
		}
		a := c.Accruals[k]
		if a == nil {
			a = &carryAccrual{}
			c.Accruals[k] = a
		}
		nd := math.Abs(qty*m.last) * days
		a.Venue, a.Class = m.venue, class
		if s.FundingBpsDay != 0 {
			a.Funding += nd * s.FundingBpsDay
			a.FundND += nd
			a.FundBps = s.FundingBpsDay
		}
		if qty < 0 && s.BorrowBpsDay != 0 {
			a.Borrow += nd * s.BorrowBpsDay
			a.BorrND += nd
			a.BorrBps = s.BorrowBpsDay
		}
	}
	c.LastAt = t
	p.touch()
}

// closeCarryPeriod turns the accruals into charges for the period ending at `end`.
func (p *PaperEngine) closeCarryPeriod(end time.Time) []CarryCharge {
	c := p.carry
	start := end.Add(-p.CarryPeriod)
	keys := make([]string, 0, len(c.Accruals))
	for k := range c.Accruals {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var out []CarryCharge
	for _, k := range keys {
		a := c.Accruals[k]
		account, inst, _ := strings.Cut(k, "|")
		mk := func(kind string, bpsND, nd, bps float64) {
			if bpsND == 0 {
				return
			}
			out = append(out, CarryCharge{
				Account: account, InstrumentID: inst, Kind: kind, Amount: -bpsND / 10_000,
				BpsDay: bps, NotionalDays: nd, Venue: a.Venue, Class: a.Class,
				PeriodStart: start.UTC(), PeriodEnd: end.UTC(),
				Reference: fmt.Sprintf("carry:%s:%s:%d", inst, kind, end.Unix()),
			})
		}
		mk("funding", a.Funding, a.FundND, a.FundBps)
		mk("borrow", a.Borrow, a.BorrND, a.BorrBps)
	}
	c.Accruals = map[string]*carryAccrual{}
	p.touch()
	if p.Counters != nil {
		p.Counters.Add("carry.charges", float64(len(out)))
		for _, ch := range out {
			p.Counters.Add("carry."+ch.Kind+"_paid", -ch.Amount)
		}
	}
	return out
}

// TickCarry accrues carry up to now; it returns the charges to publish when a period closed.
func (p *PaperEngine) TickCarry() []CarryCharge {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.accrueCarry(p.now())
}
