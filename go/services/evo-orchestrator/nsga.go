package main

import (
	"math"
	"sort"
)

// NSGA-II (Deb et al. 2002) over minimised objective vectors. This is a line-by-line port of
// crates/fa-evolution/src/lib.rs (non_dominated_sort, crowding_distance, nsga2_select) and
// crates/fa-evolution/src/objectives.rs (minimised); both test suites read
// go/services/evo-orchestrator/testdata/nsga_fixture.json so the two cannot drift. Sorts are stable in
// both languages, so ties resolve identically.

// Dominates reports whether a dominates b (all objectives minimised).
func Dominates(a, b []float64) bool {
	n := min(len(a), len(b))
	all, any := true, false
	for i := 0; i < n; i++ {
		if a[i] > b[i] {
			all = false
		}
		if a[i] < b[i] {
			any = true
		}
	}
	return all && any
}

// NonDominatedSort returns fronts of indices (front 0 = Pareto front).
func NonDominatedSort(objs [][]float64) [][]int {
	n := len(objs)
	dominatedBy := make([][]int, n)
	count := make([]int, n)
	fronts := [][]int{{}}
	for p := 0; p < n; p++ {
		for q := 0; q < n; q++ {
			if Dominates(objs[p], objs[q]) {
				dominatedBy[p] = append(dominatedBy[p], q)
			} else if Dominates(objs[q], objs[p]) {
				count[p]++
			}
		}
		if count[p] == 0 {
			fronts[0] = append(fronts[0], p)
		}
	}
	for i := 0; len(fronts[i]) > 0; i++ {
		var next []int
		for _, p := range fronts[i] {
			for _, q := range dominatedBy[p] {
				count[q]--
				if count[q] == 0 {
					next = append(next, q)
				}
			}
		}
		fronts = append(fronts, next)
	}
	return fronts[:len(fronts)-1]
}

// CrowdingDistance of the members of one front (same order as front).
func CrowdingDistance(objs [][]float64, front []int) []float64 {
	n := len(front)
	d := make([]float64, n)
	if n <= 2 {
		for i := range d {
			d[i] = math.Inf(1)
		}
		return d
	}
	m := len(objs[front[0]])
	for k := 0; k < m; k++ {
		order := make([]int, n)
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return objs[front[order[a]]][k] < objs[front[order[b]]][k] })
		lo := objs[front[order[0]]][k]
		hi := objs[front[order[n-1]]][k]
		d[order[0]] = math.Inf(1)
		d[order[n-1]] = math.Inf(1)
		// A non-finite span would make every interior distance NaN; the extremes are already
		// infinite, so skip it (same rule as the Rust reference).
		if hi > lo && !math.IsInf(hi-lo, 0) {
			for w := 1; w < n-1; w++ {
				d[order[w]] += (objs[front[order[w+1]]][k] - objs[front[order[w-1]]][k]) / (hi - lo)
			}
		}
	}
	return d
}

// NSGA2Select fills by fronts and breaks the last front by crowding distance (descending,
// stable). Returns indices into objs, best first.
func NSGA2Select(objs [][]float64, k int) []int {
	out := make([]int, 0, k)
	for _, front := range NonDominatedSort(objs) {
		if len(out)+len(front) <= k {
			out = append(out, front...)
		} else {
			cd := CrowdingDistance(objs, front)
			idx := make([]int, len(front))
			for i := range idx {
				idx[i] = i
			}
			sort.SliceStable(idx, func(a, b int) bool { return cd[idx[a]] > cd[idx[b]] })
			for _, i := range idx[:k-len(out)] {
				out = append(out, front[i])
			}
		}
		if len(out) >= k {
			break
		}
	}
	return out
}

// ParseDirection maps a direction string to +1 (minimise) / -1 (maximise); ok=false when unknown.
func ParseDirection(s string) (sign float64, ok bool) {
	switch s {
	case "min", "minimise", "minimize", "lower":
		return 1, true
	case "max", "maximise", "maximize", "higher":
		return -1, true
	}
	return 0, false
}

// RankedKeys returns the sorted objective names that have a known direction.
func RankedKeys(directions map[string]string) []string {
	keys := make([]string, 0, len(directions))
	for k, d := range directions {
		if _, ok := ParseDirection(d); ok {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	return keys
}

// Minimised builds one candidate's minimisation vector over keys: max objectives are negated,
// a missing or non-finite value becomes +Inf (worst).
func Minimised(objectives map[string]float64, directions map[string]string, keys []string) []float64 {
	v := make([]float64, len(keys))
	for i, k := range keys {
		sign, ok := ParseDirection(directions[k])
		if !ok {
			sign = 1
		}
		x, has := objectives[k]
		if !has || math.IsNaN(x) || math.IsInf(x, 0) {
			v[i] = math.Inf(1)
			continue
		}
		v[i] = sign * x
	}
	return v
}
