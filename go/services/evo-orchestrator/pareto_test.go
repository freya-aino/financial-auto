package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestParetoMatchesSharedFixture pins HVRank / HVContributions / NSGA3Select to the Rust
// reference (crates/fa-evolution/src/pareto.rs writes the `pareto` cases).
func TestParetoMatchesSharedFixture(t *testing.T) {
	raw, err := os.ReadFile(filepath.FromSlash(fixturePath))
	if err != nil {
		t.Fatalf("shared fixture missing: %v", err)
	}
	var fx struct {
		Pareto []struct {
			Name       string    `json:"name"`
			Objectives [][]fxNum `json:"objectives"`
			HVOrder    []int     `json:"hv_order"`
			HVRank     []int     `json:"hv_rank"`
			HV         []float64 `json:"hv"`
			NSGA3      []struct {
				K        int   `json:"k"`
				Expected []int `json:"expected"`
			} `json:"nsga3"`
		} `json:"pareto"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatal(err)
	}
	if len(fx.Pareto) < 4 {
		t.Fatalf("expected ≥ 4 pareto cases, got %d", len(fx.Pareto))
	}
	for _, c := range fx.Pareto {
		objs := make([][]float64, len(c.Objectives))
		for i, r := range c.Objectives {
			objs[i] = toF(r)
		}
		order, rank, hv := HVRank(objs)
		if !reflect.DeepEqual(order, c.HVOrder) {
			t.Errorf("%s: hv order %v, want %v", c.Name, order, c.HVOrder)
		}
		if !reflect.DeepEqual(rank, c.HVRank) {
			t.Errorf("%s: ranks %v, want %v", c.Name, rank, c.HVRank)
		}
		for i := range hv {
			if math.Abs(hv[i]-c.HV[i]) > 1e-12 {
				t.Errorf("%s: hv %v, want %v", c.Name, hv, c.HV)
				break
			}
		}
		for _, s := range c.NSGA3 {
			got := NSGA3Select(objs, s.K)
			if len(got) == 0 && len(s.Expected) == 0 {
				continue
			}
			if !reflect.DeepEqual(got, s.Expected) {
				t.Errorf("%s: nsga3 k=%d got %v, want %v", c.Name, s.K, got, s.Expected)
			}
		}
	}
}

func TestSplitMixReferenceValues(t *testing.T) {
	r := splitMix64{s: 0}
	if a, b := r.next(), r.next(); a != 0xe220a8397b1dcdaf || b != 0x6e789e6aa1b965f4 {
		t.Fatalf("splitmix64 %x %x", a, b)
	}
}

// Elites at loop end: NSGA-III with ≥ 4 objectives keeps one member per reference direction
// instead of three near-duplicates; failed genomes are never selected.
func TestSelectElitesUsesNSGA3ForManyObjectives(t *testing.T) {
	dirs := map[string]string{"a": "min", "b": "min", "c": "min", "d": "min"}
	mk := func(id string, a, b, c, d float64) Candidate {
		return Candidate{ID: id, Objectives: map[string]float64{"a": a, "b": b, "c": c, "d": d}, Directions: dirs}
	}
	cands := []Candidate{
		mk("g1", 1, 0, 0, 0), mk("g2", 0.99, 0.01, 0, 0), mk("g3", 0.98, 0.02, 0, 0),
		mk("g4", 0, 1, 0, 0), mk("g5", 0, 0, 1, 0), mk("g6", 0, 0, 0, 1),
		{ID: "failed", Score: math.Inf(-1)},
	}
	el := SelectElites(cands, 4)
	ids := map[string]bool{}
	for _, c := range el {
		ids[c.ID] = true
	}
	if len(el) != 4 || !ids["g4"] || !ids["g5"] || !ids["g6"] || ids["failed"] {
		t.Fatalf("elites %v", ids)
	}
	if got := SelectElites(cands, 10); len(got) != 6 {
		t.Fatalf("all evaluated genomes, never the failed one: %d", len(got))
	}
}
