package main

import (
	"reflect"
	"testing"
)

func ids(cs []Candidate) []string {
	out := make([]string, len(cs))
	for i, c := range cs {
		out[i] = c.ID
	}
	return out
}

func TestHalveKeepsBestByScalar(t *testing.T) {
	got := Halve([]Candidate{{ID: "a", Score: 1}, {ID: "b", Score: 3}, {ID: "c", Score: 2}, {ID: "d", Score: 3}, {ID: "e", Score: 0}}, 2)
	if want := []string{"b", "d", "c"}; !reflect.DeepEqual(ids(got), want) {
		t.Fatalf("got %v want %v", ids(got), want)
	}
}

var dirs = map[string]string{"crps": "min", "crps_skill": "max", "compute_cost": "min", "n_eval": "count"}

func cand(id string, crps, skill, cost float64) Candidate {
	return Candidate{ID: id, Score: skill, Objectives: map[string]float64{"crps": crps, "crps_skill": skill, "compute_cost": cost, "n_eval": 10}, Directions: dirs}
}

func TestHalveUsesParetoRankOverObjectives(t *testing.T) {
	// b has the best scalar (skill) but is dominated by nobody; d is dominated by a and b;
	// e failed (no objectives). Scalar-only ranking would keep b, a; NSGA keeps the front.
	cands := []Candidate{
		cand("a", 0.10, 0.30, 1.0), // front 0 (cheapest)
		cand("b", 0.05, 0.60, 3.0), // front 0 (best crps / skill)
		cand("c", 0.08, 0.40, 2.0), // front 0 (middle)
		cand("d", 0.12, 0.20, 3.5), // dominated by a, b, c
		{ID: "e", Score: -1e308},   // failed, no objectives
		cand("f", 0.20, 0.10, 4.0), // dominated by everyone
	}
	ranked := Rank(cands)
	// a and b are both front extremes (+Inf crowding); the tie falls back to score, so b (0.60) leads.
	if got := ids(ranked); !reflect.DeepEqual(got[:3], []string{"b", "a", "c"}) || got[5] != "e" {
		t.Fatalf("rank = %v", got)
	}
	if ranked[3].ID != "d" && ranked[3].ID != "f" {
		t.Fatalf("rank = %v", ids(ranked))
	}
	kept := Halve(cands, 2) // ceil(6/2) = 3
	if got := ids(kept); !reflect.DeepEqual(got, []string{"b", "a", "c"}) {
		t.Fatalf("kept %v", got)
	}
	// Arrival order must not change the result.
	rev := []Candidate{cands[5], cands[4], cands[3], cands[2], cands[1], cands[0]}
	if !reflect.DeepEqual(ids(Halve(rev, 2)), ids(kept)) {
		t.Fatalf("order dependent: %v vs %v", ids(Halve(rev, 2)), ids(kept))
	}
	keys, fronts, unranked := Pareto(cands)
	if !reflect.DeepEqual(keys, []string{"compute_cost", "crps", "crps_skill"}) || unranked != 1 {
		t.Fatalf("keys %v unranked %d", keys, unranked)
	}
	if len(fronts) != 3 || len(fronts[0]) != 3 || fronts[1][0].Genome != "d" || fronts[2][0].Genome != "f" {
		t.Fatalf("fronts %+v", fronts)
	}
}

func TestCrowdingBreaksLastFront(t *testing.T) {
	// Four non-dominated points; keeping 3 must keep both extremes (infinite crowding).
	cands := []Candidate{cand("w", 0.0, 0.0, 3.0), cand("x", 0.1, 0.1, 2.0), cand("y", 0.11, 0.11, 1.9), cand("z", 0.3, 0.3, 0.0)}
	for i := range cands {
		delete(cands[i].Objectives, "crps_skill") // two objectives: crps (min) vs compute_cost (min)
	}
	kept := ids(Halve(cands, 2)) // ceil(4/2) = 2
	if !(contains(kept, "w") && contains(kept, "z")) {
		t.Fatalf("extremes not kept: %v", kept)
	}
}

func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

func TestPlan(t *testing.T) {
	rungs := Plan(27, 3, 100, 10_000)
	if len(rungs) != 4 {
		t.Fatalf("rungs %v", rungs)
	}
	if rungs[3].Candidates != 1 || rungs[3].BudgetMs != 2700 {
		t.Fatalf("last rung %+v", rungs[3])
	}
	if TotalBudget(rungs) != 27*100+9*300+3*900+1*2700 {
		t.Fatalf("total %d", TotalBudget(rungs))
	}
}

func TestPlanStopsAtMaxBudget(t *testing.T) {
	rungs := Plan(100, 2, 1000, 3000)
	if len(rungs) != 2 || rungs[1].BudgetMs != 2000 {
		t.Fatalf("rungs %v", rungs)
	}
}

func TestRankBreaksCrowdingTiesByScore(t *testing.T) {
	dirs := map[string]string{"a": "max", "b": "min"}
	// Mutually non-dominated pair: both are extremes of the single front, so both have +Inf crowding.
	cands := []Candidate{
		{ID: "alpha", Score: 0.1, Objectives: map[string]float64{"a": 1, "b": 1}, Directions: dirs},
		{ID: "zeta", Score: 0.9, Objectives: map[string]float64{"a": 2, "b": 2}, Directions: dirs},
	}
	got := Rank(cands)
	if got[0].ID != "zeta" {
		t.Fatalf("equal crowding must fall back to score, got order %v, %v", got[0].ID, got[1].ID)
	}
}
