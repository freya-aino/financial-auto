package main

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// Spec 34 §6.2 step 1 (seed) and §6.1 RNN prior: source→target pairs proposed by the MCMC
// graph's cross-asset priors (`latent.crossprior.<ts>`, D-069, ranked by predictive gain /
// uncertainty) become `nn_forecast` genomes that read the source's latents (`src`) and forecast
// the target's series (`inst`); evolution then selects and mutates them like any genome.

// DefaultPairTemplate is the genome a pair is built from when no `nn_forecast` genome is in the
// population (small GRU prior; the pair genes are appended).
const DefaultPairTemplate = "nn_forecast:h=5;lb=32;width=16;depth=2;gru=1;steps=300;lr=0.003"

// PairProposal is one cross-asset pair from the latest `latent.crossprior.<ts>`.
type PairProposal struct {
	Timescale string
	Source    string
	Target    string
	Score     float64
	Gain      float64
	LeadLag   uint32
	AsOf      time.Time
}

// PairBook keeps the latest proposals per timescale (thread-safe).
type PairBook struct {
	mu   sync.Mutex
	byTS map[string][]PairProposal
	seen int
}

func NewPairBook() *PairBook { return &PairBook{byTS: map[string][]PairProposal{}} }

// Update replaces the proposals of one timescale.
func (b *PairBook) Update(m *fav1.CrossAssetPriors) {
	var ps []PairProposal
	at := time.Now()
	if m.GetAsOf() != nil {
		at = m.GetAsOf().AsTime()
	}
	for _, p := range m.GetProposals() {
		if p.GetSourceInstrumentId() == "" || p.GetTargetInstrumentId() == "" || p.GetSourceInstrumentId() == p.GetTargetInstrumentId() {
			continue
		}
		ps = append(ps, PairProposal{Timescale: m.GetTimescale(), Source: p.GetSourceInstrumentId(), Target: p.GetTargetInstrumentId(),
			Score: p.GetScore(), Gain: p.GetPredictiveGain(), LeadLag: p.GetLeadLag(), AsOf: at})
	}
	sort.SliceStable(ps, func(i, j int) bool { return ps[i].Score > ps[j].Score })
	b.mu.Lock()
	defer b.mu.Unlock()
	b.byTS[m.GetTimescale()] = ps
	b.seen++
}

// Top returns up to n proposals over all timescales, best score first.
func (b *PairBook) Top(n int) []PairProposal {
	b.mu.Lock()
	defer b.mu.Unlock()
	var all []PairProposal
	for _, ps := range b.byTS {
		all = append(all, ps...)
	}
	sort.SliceStable(all, func(i, j int) bool {
		if all[i].Score != all[j].Score {
			return all[i].Score > all[j].Score
		}
		return all[i].Source+all[i].Target < all[j].Source+all[j].Target
	})
	if n >= 0 && len(all) > n {
		all = all[:n]
	}
	return all
}

// Stats returns (messages seen, proposals held).
func (b *PairBook) Stats() (int, int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := 0
	for _, ps := range b.byTS {
		n += len(ps)
	}
	return b.seen, n
}

// genomeFamily is the family token of a kv genome (`nn_forecast:…` → `nn_forecast`).
func genomeFamily(g string) string {
	f, _, _ := strings.Cut(strings.TrimSpace(g), ":")
	return strings.ToLower(f)
}

// isNeural reports whether a genome string names a trained (candle) family.
func isNeural(g string) bool {
	return strings.HasPrefix(genomeFamily(g), "nn_") || strings.HasPrefix(genomeFamily(g), "neural_")
}

// withGenes replaces (or adds) genes of a kv genome; an empty value removes the gene.
func withGenes(g string, genes map[string]string) string {
	fam, rest, _ := strings.Cut(strings.TrimSpace(g), ":")
	var kept []string
	for _, kv := range strings.FieldsFunc(rest, func(r rune) bool { return r == ';' || r == ' ' }) {
		k, _, _ := strings.Cut(kv, "=")
		if _, replaced := genes[strings.TrimSpace(k)]; replaced {
			continue
		}
		kept = append(kept, kv)
	}
	keys := make([]string, 0, len(genes))
	for k := range genes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if v := genes[k]; v != "" {
			kept = append(kept, k+"="+v)
		}
	}
	if len(kept) == 0 {
		return fam
	}
	return fam + ":" + strings.Join(kept, ";")
}

// PairGenomes turns proposals into cross-asset `nn_forecast` genomes, built from the first
// `nn_forecast` genome of `population` (else DefaultPairTemplate), skipping genomes already present.
func PairGenomes(props []PairProposal, population []string) []string {
	template := DefaultPairTemplate
	present := map[string]bool{}
	for _, g := range population {
		present[g] = true
		if genomeFamily(g) == "nn_forecast" && template == DefaultPairTemplate {
			template = g
		}
	}
	var out []string
	for _, p := range props {
		genes := map[string]string{"inst": p.Target, "src": p.Source}
		if p.Timescale != "" {
			genes["ts"] = p.Timescale
		}
		g := withGenes(template, genes)
		if present[g] {
			continue
		}
		present[g] = true
		out = append(out, g)
	}
	return out
}

// weightsRefOf extracts the Garage ref ("<bucket>/<key>") of a neural eval result's bundle
// (`neural.weights {persisted, bucket, key}`, spec 34 §6.5); "" when none was persisted.
func weightsRefOf(r *fav1.JobResult) string {
	if r == nil || r.Status != fav1.JobResult_STATUS_OK || len(r.Result) == 0 {
		return ""
	}
	var v struct {
		Neural *struct {
			Weights struct {
				Persisted bool   `json:"persisted"`
				Bucket    string `json:"bucket"`
				Key       string `json:"key"`
			} `json:"weights"`
		} `json:"neural"`
	}
	if err := json.Unmarshal(r.Result, &v); err != nil || v.Neural == nil || !v.Neural.Weights.Persisted || v.Neural.Weights.Key == "" {
		return ""
	}
	return fmt.Sprintf("%s/%s", v.Neural.Weights.Bucket, v.Neural.Weights.Key)
}

// lineageOf extracts `lineage [{child, parents, parents_as_sent}]` from a vary result (child →
// parents as sent, then canonical).
func lineageOf(r *fav1.JobResult) map[string][]string {
	var v struct {
		Lineage []struct {
			Child         string   `json:"child"`
			Parents       []string `json:"parents"`
			ParentsAsSent []string `json:"parents_as_sent"`
		} `json:"lineage"`
	}
	out := map[string][]string{}
	if err := json.Unmarshal(r.Result, &v); err != nil {
		return out
	}
	for _, l := range v.Lineage {
		// the worker canonicalises genome strings; parents_as_sent are this orchestrator's ids
		out[l.Child] = append(append([]string(nil), l.ParentsAsSent...), l.Parents...)
	}
	return out
}

// Budget is the per-loop GPU budget accounting of a generation (spec 34 §6.4, §8 "evaluation
// cost exceeds budget": truncate rungs, record partial results).
type Budget struct {
	TotalMs     uint64 `json:"total_budget_ms"`       // per loop; 0 = unlimited
	LoopUsedMs  uint64 `json:"loop_used_ms"`          // reported used_gpu_ms of this loop's jobs
	UsedMs      uint64 `json:"used_ms"`               // all loops
	PlannedMs   uint64 `json:"planned_ms"`            // TotalBudget(Plan) of this loop
	Truncated   bool   `json:"truncated"`             // a rung of this loop was skipped
	Truncations int    `json:"truncations,omitempty"` // over the whole generation
}
