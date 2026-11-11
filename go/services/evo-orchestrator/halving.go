package main

import (
	"math"
	"sort"
	"strings"
)

// Candidate is a genome with its latest evaluation. Objectives/Directions come from the
// worker's JobResult (proto maps, else the result JSON); they are nil for failed jobs and
// for workers that only report the scalar `score` (D-025).
type Candidate struct {
	ID         string // genome string as sent in Job.genome_id
	Score      float64
	Objectives map[string]float64
	Directions map[string]string
}

// hasObjectives reports whether NSGA-II ranking is possible for this candidate.
func (c Candidate) hasObjectives() bool {
	return len(c.Objectives) > 0 && len(RankedKeys(c.Directions)) > 0
}

// rankedKeys is the sorted union of ranked objective names over the candidates.
func rankedKeys(cands []Candidate) []string {
	set := map[string]string{}
	for _, c := range cands {
		for k, d := range c.Directions {
			if _, ok := ParseDirection(d); ok {
				set[k] = d
			}
		}
	}
	return RankedKeys(set)
}

// byID orders candidates by ID so ranking does not depend on result arrival order.
func byID(cands []Candidate) []Candidate {
	s := append([]Candidate(nil), cands...)
	sort.SliceStable(s, func(i, j int) bool { return s[i].ID < s[j].ID })
	return s
}

// Niche is the selection niche of a genome (spec 34 §6.3 "fronts are computed per family for
// selection", Q6): the decoder output family its genome family produces (baseline and neural
// genomes of one output family compete), "" for anything else (bare ids, strategies, allocators).
func Niche(id string) string {
	fam, _, ok := strings.Cut(id, ":")
	if !ok {
		return ""
	}
	switch strings.ToLower(strings.TrimSpace(fam)) {
	case "forecast", "quantile", "rnn_prior", "nn_forecast", "nn_quantile", "nn_rnn", "neural_forecast":
		return "rnn_prior"
	case "retrieval", "analog", "nn_retrieval", "nn_metric", "neural_retrieval":
		return "retrieval"
	case "nn_chart", "neural_chart":
		return "visual.chart"
	case "nn_future", "neural_future":
		return "visual.future"
	case "nn_frame", "neural_frame":
		return "visual.frame"
	case "nn_news", "neural_news":
		return "text.news"
	case "nn_summary", "neural_summary":
		return "text.summary"
	case "nn_narrative", "neural_narrative":
		return "text.narrative"
	}
	return ""
}

// Rank orders candidates best first. Candidates with objectives are ranked **per niche**
// ([Niche]): by non-dominated front and, inside a front, by **hypervolume contribution** (spec
// 34 §6.2 step 4, the scalarised rung score: objectives min–max normalised over the niche,
// `HVContributions` over the objectives the whole front defines), then crowding distance, then
// the scalar score, then ID; the niches' orders are interleaved by relative position (the i-th
// of n at (i+½)/n, ties by niche name), so every prefix keeps each family's share. Candidates
// lacking objectives (failed jobs, scalar-only workers) come after all ranked ones. Without any
// objectives it falls back to the scalar score (descending, ties by ID).
func Rank(cands []Candidate) []Candidate {
	s := byID(cands)
	var rest []Candidate
	niches := map[string][]Candidate{}
	for _, c := range s {
		if c.hasObjectives() {
			n := Niche(c.ID)
			niches[n] = append(niches[n], c)
		} else {
			rest = append(rest, c)
		}
	}
	if len(niches) == 0 {
		sort.SliceStable(rest, func(i, j int) bool {
			if rest[i].Score != rest[j].Score {
				return rest[i].Score > rest[j].Score
			}
			return rest[i].ID < rest[j].ID
		})
		return rest
	}
	type slot struct {
		pos   float64
		niche string
		c     Candidate
	}
	var slots []slot
	for name, members := range niches {
		ordered := rankNiche(members)
		for i, c := range ordered {
			slots = append(slots, slot{(float64(i) + 0.5) / float64(len(ordered)), name, c})
		}
	}
	sort.SliceStable(slots, func(i, j int) bool {
		if slots[i].pos != slots[j].pos {
			return slots[i].pos < slots[j].pos
		}
		if slots[i].niche != slots[j].niche {
			return slots[i].niche < slots[j].niche
		}
		return slots[i].c.ID < slots[j].c.ID
	})
	out := make([]Candidate, 0, len(cands))
	for _, sl := range slots {
		out = append(out, sl.c)
	}
	// unranked candidates last; a finite score still orders them among themselves
	sort.SliceStable(rest, func(i, j int) bool { return rest[i].Score > rest[j].Score })
	return append(out, rest...)
}

// rankNiche orders candidates that all carry objectives (one niche, sorted by ID): fronts, then
// hypervolume contribution, crowding distance, score.
func rankNiche(ranked []Candidate) []Candidate {
	objs := minimisedVectors(ranked)
	norm := NormaliseObjectives(objs)
	out := make([]Candidate, 0, len(ranked))
	for _, front := range NonDominatedSort(objs) {
		hv := HVContributions(FrontPoints(objs, norm, front))
		cd := CrowdingDistance(objs, front)
		idx := make([]int, len(front))
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool {
			if ha, hb := hv[idx[a]], hv[idx[b]]; ha != hb {
				return ha > hb
			}
			if ca, cb := cd[idx[a]], cd[idx[b]]; ca != cb {
				return ca > cb
			}
			return ranked[front[idx[a]]].Score > ranked[front[idx[b]]].Score
		})
		for _, i := range idx {
			out = append(out, ranked[front[i]])
		}
	}
	return out
}

// minimisedVectors are the candidates' minimisation vectors over their ranked keys.
func minimisedVectors(ranked []Candidate) [][]float64 {
	keys := rankedKeys(ranked)
	objs := make([][]float64, len(ranked))
	for i, c := range ranked {
		objs[i] = Minimised(c.Objectives, c.Directions, keys)
	}
	return objs
}

// SelectElites is the loop-end survivor selection (spec 34 §6.2 step 6): per niche ([Niche]),
// ceil(k·n_f/n) of its n_f candidates with objectives chosen by NSGA-II (≤ 3 objectives) or
// NSGA-III (≥ 4, reference directions); the result follows Rank order and is cut at k.
// Scalar-only candidates fill remaining places by score; candidates that failed (no objectives,
// score −∞) are never selected.
func SelectElites(cands []Candidate, k int) []Candidate {
	if k <= 0 {
		return nil
	}
	niches := map[string][]Candidate{}
	total := 0
	for _, c := range byID(cands) {
		if c.hasObjectives() {
			niches[Niche(c.ID)] = append(niches[Niche(c.ID)], c)
			total++
		}
	}
	chosen := map[string]bool{}
	for _, members := range niches {
		kf := min(len(members), (k*len(members)+total-1)/total)
		for _, i := range SelectSurvivors(minimisedVectors(members), kf) {
			chosen[members[i].ID] = true
		}
	}
	out := make([]Candidate, 0, k)
	order := Rank(cands)
	for _, c := range order {
		if len(out) < k && c.hasObjectives() && chosen[c.ID] {
			out = append(out, c)
		}
	}
	for _, c := range order {
		if len(out) >= k {
			break
		}
		if !c.hasObjectives() && c.Score != math.Inf(-1) {
			out = append(out, c)
		}
	}
	return out
}

// Halve keeps the best ceil(n/eta) candidates by Rank.
func Halve(cands []Candidate, eta int) []Candidate {
	if eta < 2 {
		eta = 2
	}
	sorted := Rank(cands)
	keep := (len(sorted) + eta - 1) / eta
	return sorted[:keep]
}

// FrontMember is one Pareto-front entry of a generation summary.
type FrontMember struct {
	Genome     string             `json:"genome"`
	Rank       int                `json:"rank"`
	Crowding   float64            `json:"crowding"` // +Inf encoded as a large number for JSON
	Objectives map[string]float64 `json:"objectives"`
	Score      float64            `json:"score"`
	// HV is the member's exclusive hypervolume contribution inside its front (objectives
	// normalised over the ranked set, reference 1.1; the rung score of spec 34 §6.2 step 4).
	HV float64 `json:"hv"`
}

// Pareto computes the non-dominated fronts of the candidates that carry objectives.
// Returns the ranked objective keys, the members of every front (front 0 first) and the
// number of candidates without objectives.
func Pareto(cands []Candidate) (keys []string, fronts [][]FrontMember, unranked int) {
	var ranked []Candidate
	for _, c := range byID(cands) {
		if c.hasObjectives() {
			ranked = append(ranked, c)
		} else {
			unranked++
		}
	}
	if len(ranked) == 0 {
		return nil, nil, unranked
	}
	keys = rankedKeys(ranked)
	objs := minimisedVectors(ranked)
	_, _, hv := HVRank(objs)
	for r, front := range NonDominatedSort(objs) {
		cd := CrowdingDistance(objs, front)
		members := make([]FrontMember, len(front))
		for i, idx := range front {
			d := cd[i]
			if math.IsInf(d, 1) {
				d = math.MaxFloat64 // JSON has no Inf
			}
			members[i] = FrontMember{Genome: ranked[idx].ID, Rank: r, Crowding: d, Objectives: ranked[idx].Objectives, Score: ranked[idx].Score, HV: hv[idx]}
		}
		fronts = append(fronts, members)
	}
	return keys, fronts, unranked
}

// Rung is one successive-halving stage.
type Rung struct {
	Index      int
	Candidates int
	BudgetMs   uint64 // per-candidate GPU budget
}

// Plan lays out rungs: candidates shrink by eta, per-candidate budget grows by
// eta, until one candidate remains or the budget would exceed maxBudget.
func Plan(n, eta int, minBudget, maxBudget uint64) []Rung {
	if eta < 2 {
		eta = 2
	}
	var rungs []Rung
	b := minBudget
	for i := 0; n >= 1 && b <= maxBudget; i++ {
		rungs = append(rungs, Rung{Index: i, Candidates: n, BudgetMs: b})
		if n == 1 {
			break
		}
		n = (n + eta - 1) / eta
		b *= uint64(eta)
	}
	return rungs
}

// TotalBudget is the GPU-ms a plan consumes (budget-driven sizing, spec 34).
func TotalBudget(rungs []Rung) uint64 {
	var t uint64
	for _, r := range rungs {
		t += uint64(r.Candidates) * r.BudgetMs
	}
	return t
}
