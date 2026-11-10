package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// fixturePath is crates/fa-evolution/testdata/nsga_fixture.json, shared with the Rust tests.
// Run Go tests with the repository root mounted (see README) so the file is reachable.
const fixturePath = "../../../crates/fa-evolution/testdata/nsga_fixture.json"

type fxNum float64

func (n *fxNum) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		switch s {
		case "inf":
			*n = fxNum(math.Inf(1))
		case "-inf":
			*n = fxNum(math.Inf(-1))
		}
		return nil
	}
	var f float64
	if err := json.Unmarshal(b, &f); err != nil {
		return err
	}
	*n = fxNum(f)
	return nil
}

func toF(xs []fxNum) []float64 {
	out := make([]float64, len(xs))
	for i, x := range xs {
		out[i] = float64(x)
	}
	return out
}

func close(a, b float64) bool {
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		return a == b
	}
	return math.Abs(a-b) < 1e-9
}

func TestNSGAMatchesSharedFixture(t *testing.T) {
	raw, err := os.ReadFile(filepath.FromSlash(fixturePath))
	if err != nil {
		t.Fatalf("shared fixture missing (mount the repo root, not only go/): %v", err)
	}
	var fx struct {
		Cases []struct {
			Name       string    `json:"name"`
			Objectives [][]fxNum `json:"objectives"`
			Fronts     [][]int   `json:"fronts"`
			Crowding   []struct {
				Front    []int   `json:"front"`
				Distance []fxNum `json:"distance"`
			} `json:"crowding"`
			Select []struct {
				K        int   `json:"k"`
				Expected []int `json:"expected"`
			} `json:"select"`
		} `json:"cases"`
		Named struct {
			Directions map[string]string    `json:"directions"`
			Candidates []map[string]float64 `json:"candidates"`
			Keys       []string             `json:"keys"`
			Minimised  [][]fxNum            `json:"minimised"`
			Fronts     [][]int              `json:"fronts"`
		} `json:"named"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatal(err)
	}
	if len(fx.Cases) < 5 {
		t.Fatalf("fixture has %d cases", len(fx.Cases))
	}
	for _, c := range fx.Cases {
		objs := make([][]float64, len(c.Objectives))
		for i, r := range c.Objectives {
			objs[i] = toF(r)
		}
		if got := NonDominatedSort(objs); !reflect.DeepEqual(got, c.Fronts) {
			t.Errorf("%s: fronts %v != %v", c.Name, got, c.Fronts)
		}
		for _, cr := range c.Crowding {
			got := CrowdingDistance(objs, cr.Front)
			want := toF(cr.Distance)
			if len(got) != len(want) {
				t.Fatalf("%s: crowding len %d != %d", c.Name, len(got), len(want))
			}
			for i := range got {
				if !close(got[i], want[i]) {
					t.Errorf("%s: crowding %v != %v", c.Name, got, want)
					break
				}
			}
		}
		for _, s := range c.Select {
			if got := NSGA2Select(objs, s.K); !reflect.DeepEqual(got, s.Expected) {
				t.Errorf("%s: select k=%d %v != %v", c.Name, s.K, got, s.Expected)
			}
		}
	}
	n := fx.Named
	if keys := RankedKeys(n.Directions); !reflect.DeepEqual(keys, n.Keys) {
		t.Fatalf("ranked keys %v != %v", keys, n.Keys)
	}
	vecs := make([][]float64, len(n.Candidates))
	for i, c := range n.Candidates {
		vecs[i] = Minimised(c, n.Directions, n.Keys)
		want := toF(n.Minimised[i])
		for j := range want {
			if !close(vecs[i][j], want[j]) {
				t.Errorf("named %d: %v != %v", i, vecs[i], want)
				break
			}
		}
	}
	if got := NonDominatedSort(vecs); !reflect.DeepEqual(got, n.Fronts) {
		t.Errorf("named fronts %v != %v", got, n.Fronts)
	}
}
