package main

import "testing"

func TestHalveKeepsBest(t *testing.T) {
	got := Halve([]Candidate{{"a", 1}, {"b", 3}, {"c", 2}, {"d", 3}, {"e", 0}}, 2)
	want := []string{"b", "d", "c"}
	if len(got) != len(want) {
		t.Fatalf("len %d", len(got))
	}
	for i, w := range want {
		if got[i].ID != w {
			t.Fatalf("got %v", got)
		}
	}
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
