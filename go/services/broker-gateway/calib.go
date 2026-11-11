package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"go.yaml.in/yaml/v3"
)

// Live slippage recalibration (spec 41 §6.3: "realised live slippage recalibrates it
// periodically"). Every taker fill contributes a sample per (venue, asset class):
//
//	x = 1e4 · σ_day · sqrt(notional / ADV)       (the impact regressor, bps per unit k)
//	y = realised_bps − half_spread_bps            (realised = signed fill vs the PARENT arrival)
//
// k̂ = Σxy / Σx² (least squares through the origin), shrunk toward the prior k of the base
// costs file with weight n/(n+n0) and clamped to [lo, hi] × prior. Samples whose child was
// priced at the parent arrival itself (market orders, single-shot crosses) are, in paper, the
// model echoing itself (y = k_in_use · x exactly) and are excluded from the fit
// (`n_echo`); only algo children carry information (market drift between arrival and child).
// A live venue adapter would feed real fills through the same path, all informative.

type calibKey struct{ Venue, Class string }

type slipSample struct{ x, y float64 }

type slipStats struct {
	samples  []slipSample
	n, nEcho int
	bpsEWMA  float64
	modelEW  float64
}

// Calibrator collects samples and fits k.
type Calibrator struct {
	mu         sync.Mutex
	stats      map[calibKey]*slipStats
	MaxSamples int     // rolling window per key
	MinSamples int     // informative samples needed for a fit
	N0         float64 // shrinkage pseudo-count toward the prior
	Lo, Hi     float64 // clamp, multiples of the prior k
}

func NewCalibrator() *Calibrator {
	return &Calibrator{stats: map[calibKey]*slipStats{}, MaxSamples: 500, MinSamples: 30, N0: 50, Lo: 0.5, Hi: 2}
}

const slipAlpha = 0.05 // EWMA weight of the slippage gauges

// observeSlippage records one fill (caller holds p.mu): gauges for every fill, a calibration
// sample for informative taker fills.
func (p *PaperEngine) observeSlippage(ci CostInputs, res CostResult, arrival, bps float64, algo string) {
	if ci.Price <= 0 || math.IsNaN(bps) || math.IsInf(bps, 0) {
		return
	}
	modelBps := (res.HalfSpread + res.Impact) / ci.Price * 10_000
	key := calibKey{Venue: ci.Venue, Class: ci.Class}
	if p.Counters != nil {
		pre := fmt.Sprintf("slippage.%s.%s.", metricTok(key.Venue), metricTok(key.Class))
		snap := p.Counters.Snapshot()
		ew := func(name string, v float64) {
			if _, ok := snap[name]; !ok {
				p.Counters.Set(name, v)
				return
			}
			p.Counters.Set(name, (1-slipAlpha)*snap[name]+slipAlpha*v)
		}
		ew(pre+"bps_mean", bps)
		ew(pre+"model_bps_mean", modelBps)
		p.Counters.Add(pre+"n", 1)
		ew("exec_slippage_bps."+metricTok(algo), bps)
		k := p.Costs.DefaultK
		if s := p.Costs.Schedule(ci.Venue, ci.Class); s.ImpactK != nil {
			k = *s.ImpactK
		}
		p.Counters.Set(fmt.Sprintf("costs.k.%s.%s", metricTok(key.Venue), metricTok(key.Class)), k)
	}
	if p.Calib == nil || ci.Passive {
		return
	}
	sigma, adv := ci.Sigma, ci.ADV
	if sigma <= 0 {
		sigma = p.Costs.DefaultSigma
	}
	if adv <= 0 {
		adv = p.Costs.DefaultADV
	}
	x := 1e4 * sigma * math.Sqrt(math.Abs(ci.Price*ci.Qty)/adv)
	y := bps - res.HalfSpread/ci.Price*10_000
	echo := math.Abs(ci.Price-arrival) <= 1e-12*arrival
	p.Calib.add(key, slipSample{x: x, y: y}, echo, bps, modelBps)
}

func metricTok(s string) string {
	if s == "" {
		return "unknown"
	}
	return strings.NewReplacer(".", "_", " ", "_").Replace(s)
}

func (c *Calibrator) add(k calibKey, s slipSample, echo bool, bps, modelBps float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	st := c.stats[k]
	if st == nil {
		st = &slipStats{bpsEWMA: bps, modelEW: modelBps}
		c.stats[k] = st
	}
	st.n++
	st.bpsEWMA = (1-slipAlpha)*st.bpsEWMA + slipAlpha*bps
	st.modelEW = (1-slipAlpha)*st.modelEW + slipAlpha*modelBps
	if echo {
		st.nEcho++
		return
	}
	if math.IsNaN(s.x) || math.IsInf(s.x, 0) || s.x <= 0 || math.IsNaN(s.y) || math.IsInf(s.y, 0) {
		return
	}
	st.samples = append(st.samples, s)
	if len(st.samples) > c.MaxSamples {
		st.samples = st.samples[len(st.samples)-c.MaxSamples:]
	}
}

// CalibEntry is the fit for one (venue, class).
type CalibEntry struct {
	Venue           string  `json:"venue"`
	Class           string  `json:"class"`
	Fills           int     `json:"fills"`
	Informative     int     `json:"informative"` // samples in the window used for the fit
	Echo            int     `json:"echo"`        // model-echo fills excluded from the fit
	KPrior          float64 `json:"k_prior"`
	KFit            float64 `json:"k_fit"`
	Weight          float64 `json:"weight"`
	KNew            float64 `json:"k_new"`
	Clamped         bool    `json:"clamped"`
	RealisedBpsMean float64 `json:"realised_bps_ewma"`
	ModelBpsMean    float64 `json:"model_bps_ewma"`
	Status          string  `json:"status"` // calibrated | insufficient | prior_zero
}

// CalibReport is published on `exec.costs.calibrated`.
type CalibReport struct {
	At      time.Time    `json:"at"`
	Applied bool         `json:"applied"`
	Path    string       `json:"path,omitempty"`
	Error   string       `json:"error,omitempty"`
	Entries []CalibEntry `json:"entries"`
}

// FitK is the shrunk, clamped least-squares impact coefficient.
func (c *Calibrator) FitK(samples []slipSample, prior float64) (kFit, w, kNew float64, clamped bool) {
	var sxy, sxx float64
	for _, s := range samples {
		sxy += s.x * s.y
		sxx += s.x * s.x
	}
	if sxx > 0 {
		kFit = sxy / sxx
	}
	n := float64(len(samples))
	w = n / (n + c.N0)
	kNew = w*kFit + (1-w)*prior
	lo, hi := c.Lo*prior, c.Hi*prior
	if kNew < lo {
		kNew, clamped = lo, true
	} else if kNew > hi {
		kNew, clamped = hi, true
	}
	return
}

// Calibrate fits every key against the base (prior) model and returns the report and the
// calibrated model (base + per (venue, class) impact_k overrides); model is nil when nothing
// was calibrated.
func (c *Calibrator) Calibrate(base *CostModel, now time.Time) (CalibReport, *CostModel) {
	c.mu.Lock()
	keys := make([]calibKey, 0, len(c.stats))
	for k := range c.stats {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i].Venue < keys[j].Venue || (keys[i].Venue == keys[j].Venue && keys[i].Class < keys[j].Class)
	})
	rep := CalibReport{At: now.UTC(), Entries: []CalibEntry{}}
	var out *CostModel
	for _, k := range keys {
		st := c.stats[k]
		sched := base.Schedule(k.Venue, k.Class)
		prior := base.DefaultK
		if sched.ImpactK != nil {
			prior = *sched.ImpactK
		}
		e := CalibEntry{Venue: k.Venue, Class: k.Class, Fills: st.n, Informative: len(st.samples), Echo: st.nEcho,
			KPrior: prior, KNew: prior, RealisedBpsMean: st.bpsEWMA, ModelBpsMean: st.modelEW}
		switch {
		case prior <= 0:
			e.Status = "prior_zero" // a costless schedule (e.g. the synthetic sim venue) stays costless
		case len(st.samples) < c.MinSamples || k.Venue == "":
			e.Status = "insufficient"
		default:
			e.KFit, e.Weight, e.KNew, e.Clamped = c.FitK(st.samples, prior)
			e.Status = "calibrated"
			if out == nil {
				out = base.Clone()
			}
			kk := e.KNew
			s := sched
			s.ImpactK = &kk
			v := out.Venues[k.Venue]
			if v.Classes == nil {
				v.Classes = map[string]ClassCosts{}
			}
			v.Classes[k.Class] = s
			if out.Venues == nil {
				out.Venues = map[string]VenueCosts{}
			}
			out.Venues[k.Venue] = v
		}
		rep.Entries = append(rep.Entries, e)
	}
	c.mu.Unlock()
	return rep, out
}

// Clone deep-copies a model.
func (m *CostModel) Clone() *CostModel {
	b, _ := json.Marshal(m)
	var c CostModel
	_ = json.Unmarshal(b, &c)
	return &c
}

// WriteCalibrated writes the calibrated model as a full costs YAML plus `<path>.report.json`
// (atomically via rename).
func WriteCalibrated(path string, m *CostModel, rep CalibReport) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	y, err := yaml.Marshal(m)
	if err != nil {
		return err
	}
	head := fmt.Sprintf("# Calibrated by broker-gateway at %s from live fills (spec 41 §6.3).\n"+
		"# Base model + per (venue, class) impact_k overrides; see %s.report.json. Do not edit.\n",
		rep.At.Format(time.RFC3339), filepath.Base(path))
	if err := writeAtomic(path, append([]byte(head), y...)); err != nil {
		return err
	}
	j, _ := json.MarshalIndent(rep, "", "  ")
	return writeAtomic(path+".report.json", j)
}

func writeAtomic(path string, b []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// SetCosts swaps the engine's cost model (calibration applied in memory).
func (p *PaperEngine) SetCosts(m *CostModel) {
	p.mu.Lock()
	p.Costs = m
	p.mu.Unlock()
}
