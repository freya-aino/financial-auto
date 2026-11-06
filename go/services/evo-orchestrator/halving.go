package main

import (
	"sort"
)

// Candidate is a genome with its latest scalar score. Pareto (NSGA) ranking
// replaces the scalar once multi-objective FitnessRecords exist (spec 34).
type Candidate struct {
	ID    string
	Score float64
}

// Halve keeps the best ceil(n/eta) candidates (ties broken by ID for determinism).
func Halve(cands []Candidate, eta int) []Candidate {
	if eta < 2 {
		eta = 2
	}
	sorted := append([]Candidate(nil), cands...)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Score != sorted[j].Score {
			return sorted[i].Score > sorted[j].Score
		}
		return sorted[i].ID < sorted[j].ID
	})
	keep := (len(sorted) + eta - 1) / eta
	return sorted[:keep]
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
