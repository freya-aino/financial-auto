package main

import (
	"math"
	"sort"
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

// Rank orders candidates best first. When at least one candidate carries objectives the
// order is NSGA-II over the minimised objective vectors: by non-dominated front, inside a
// front by crowding distance (descending, ties by ID), so any prefix of the order is the
// NSGA2Select survivor set of that size. Candidates lacking objectives (failed jobs,
// scalar-only workers) come after all ranked ones. Without any objectives it falls back to
// the scalar score (descending, ties by ID).
func Rank(cands []Candidate) []Candidate {
	s := byID(cands)
	var ranked, rest []Candidate
	for _, c := range s {
		if c.hasObjectives() {
			ranked = append(ranked, c)
		} else {
			rest = append(rest, c)
		}
	}
	if len(ranked) == 0 {
		sort.SliceStable(rest, func(i, j int) bool {
			if rest[i].Score != rest[j].Score {
				return rest[i].Score > rest[j].Score
			}
			return rest[i].ID < rest[j].ID
		})
		return rest
	}
	keys := rankedKeys(ranked)
	objs := make([][]float64, len(ranked))
	for i, c := range ranked {
		objs[i] = Minimised(c.Objectives, c.Directions, keys)
	}
	out := make([]Candidate, 0, len(cands))
	for _, front := range NonDominatedSort(objs) {
		cd := CrowdingDistance(objs, front)
		idx := make([]int, len(front))
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool { return cd[idx[a]] > cd[idx[b]] })
		for _, i := range idx {
			out = append(out, ranked[front[i]])
		}
	}
	// unranked candidates last; a finite score still orders them among themselves
	sort.SliceStable(rest, func(i, j int) bool { return rest[i].Score > rest[j].Score })
	return append(out, rest...)
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
	objs := make([][]float64, len(ranked))
	for i, c := range ranked {
		objs[i] = Minimised(c.Objectives, c.Directions, keys)
	}
	for r, front := range NonDominatedSort(objs) {
		cd := CrowdingDistance(objs, front)
		members := make([]FrontMember, len(front))
		for i, idx := range front {
			d := cd[i]
			if math.IsInf(d, 1) {
				d = math.MaxFloat64 // JSON has no Inf
			}
			members[i] = FrontMember{Genome: ranked[idx].ID, Rank: r, Crowding: d, Objectives: ranked[idx].Objectives, Score: ranked[idx].Score}
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
