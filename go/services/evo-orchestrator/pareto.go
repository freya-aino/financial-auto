package main

import (
	"math"
	"sort"
)

// Hypervolume rung scores and NSGA-III selection (spec 34 §6.2 steps 4 and 6, D-085): a port of
// crates/fa-evolution/src/pareto.rs, pinned to it by testdata/nsga_fixture.json ("hypervolume"
// and "nsga3" cases). Objective vectors are minimised; see the Rust module for the definitions.

const (
	hvRef         = 1.1  // reference coordinate of the normalised hypervolume
	hvSamples     = 8192 // Monte-Carlo samples for three or more objectives
	hvSeed        = uint64(0x5eed0f4b1d)
	maxDirections = 5000 // cap on Das–Dennis directions
)

// splitMix64 is SplitMix64, identical to fa_evolution::pareto::SplitMix64.
type splitMix64 struct{ s uint64 }

func (r *splitMix64) next() uint64 {
	r.s += 0x9e3779b97f4a7c15
	z := r.s
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (r *splitMix64) uniform() float64 {
	return float64(r.next()>>11) * (1.0 / float64(uint64(1)<<53))
}

func finite(x float64) bool { return !math.IsInf(x, 0) && !math.IsNaN(x) }

// NormaliseObjectives min–max normalises every objective over objs: finite values to [0, 1], a
// constant objective to 0, a missing / non-finite value to hvRef.
func NormaliseObjectives(objs [][]float64) [][]float64 {
	if len(objs) == 0 {
		return nil
	}
	d := len(objs[0])
	lo := make([]float64, d)
	hi := make([]float64, d)
	for k := range lo {
		lo[k], hi[k] = math.Inf(1), math.Inf(-1)
	}
	for _, o := range objs {
		for k := 0; k < d; k++ {
			if finite(o[k]) {
				lo[k] = math.Min(lo[k], o[k])
				hi[k] = math.Max(hi[k], o[k])
			}
		}
	}
	out := make([][]float64, len(objs))
	for i, o := range objs {
		v := make([]float64, d)
		for k := 0; k < d; k++ {
			switch {
			case !finite(o[k]):
				v[k] = hvRef
			case hi[k] > lo[k]:
				v[k] = (o[k] - lo[k]) / (hi[k] - lo[k])
			}
		}
		out[i] = v
	}
	return out
}

func weaklyDominates(a, b []float64) bool {
	for i := range a {
		if a[i] > b[i] {
			return false
		}
	}
	return true
}

// HVContributions is the exclusive hypervolume of each (normalised) point of one front.
func HVContributions(pts [][]float64) []float64 {
	n := len(pts)
	c := make([]float64, n)
	if n == 0 {
		return c
	}
	d := len(pts[0])
	switch {
	case d == 0:
		return c
	case d == 1:
		order := make([]int, n)
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return pts[order[a]][0] < pts[order[b]][0] })
		next := hvRef
		if n > 1 {
			next = pts[order[1]][0]
		}
		c[order[0]] = math.Max(math.Min(next, hvRef)-pts[order[0]][0], 0)
		return c
	case d == 2:
		order := make([]int, n)
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool {
			pa, pb := pts[order[a]], pts[order[b]]
			if pa[0] != pb[0] {
				return pa[0] < pb[0]
			}
			return pa[1] < pb[1]
		})
		for pos, i := range order {
			x, y := pts[i][0], pts[i][1]
			if x >= hvRef || y >= hvRef {
				continue
			}
			dominated := false
			for j := 0; j < n; j++ {
				if j != i && weaklyDominates(pts[j], pts[i]) {
					dominated = true
					break
				}
			}
			if dominated {
				continue
			}
			right := hvRef
			for _, j := range order[pos+1:] {
				if pts[j][0] > x {
					right = pts[j][0]
					break
				}
			}
			up := hvRef
			for q := pos - 1; q >= 0; q-- {
				if v := pts[order[q]][1]; v > y {
					up = v
					break
				}
			}
			c[i] = (math.Min(right, hvRef) - x) * (math.Min(up, hvRef) - y)
		}
		return c
	}
	rng := splitMix64{s: hvSeed}
	hits := make([]int, n)
	s := make([]float64, d)
	for t := 0; t < hvSamples; t++ {
		for k := range s {
			s[k] = rng.uniform() * hvRef
		}
		owner, count := -1, 0
		for i, p := range pts {
			if weaklyDominates(p, s) {
				count++
				owner = i
				if count > 1 {
					break
				}
			}
		}
		if count == 1 {
			hits[owner]++
		}
	}
	vol := math.Pow(hvRef, float64(d))
	for i, h := range hits {
		c[i] = float64(h) / float64(hvSamples) * vol
	}
	return c
}

// FrontPoints are the normalised points of one front over the objectives every member defines
// (finite raw value), so a family lacking an objective still owns volume (spec 34 Q6).
func FrontPoints(objs, norm [][]float64, front []int) [][]float64 {
	if len(front) == 0 {
		return nil
	}
	var dims []int
	for k := range objs[front[0]] {
		all := true
		for _, i := range front {
			if !finite(objs[i][k]) {
				all = false
				break
			}
		}
		if all {
			dims = append(dims, k)
		}
	}
	pts := make([][]float64, len(front))
	for j, i := range front {
		v := make([]float64, len(dims))
		for m, k := range dims {
			v[m] = norm[i][k]
		}
		pts[j] = v
	}
	return pts
}

// HVRank orders candidates by front, inside a front by hypervolume contribution (descending,
// normalised over all objs), then crowding distance, then input order. Returns the order and,
// per candidate, the front rank and the contribution.
func HVRank(objs [][]float64) (order []int, rank []int, hv []float64) {
	norm := NormaliseObjectives(objs)
	rank = make([]int, len(objs))
	hv = make([]float64, len(objs))
	for r, front := range NonDominatedSort(objs) {
		c := HVContributions(FrontPoints(objs, norm, front))
		cd := CrowdingDistance(objs, front)
		idx := make([]int, len(front))
		for j := range idx {
			idx[j] = j
			rank[front[j]] = r
			hv[front[j]] = c[j]
		}
		sort.SliceStable(idx, func(a, b int) bool {
			if c[idx[a]] != c[idx[b]] {
				return c[idx[a]] > c[idx[b]]
			}
			return cd[idx[a]] > cd[idx[b]]
		})
		for _, j := range idx {
			order = append(order, front[j])
		}
	}
	return order, rank, hv
}

// ReferenceDirections are the Das–Dennis directions with h divisions on the d-simplex (first
// coordinate descending, as in Rust).
func ReferenceDirections(d, h int) [][]float64 {
	var out [][]float64
	if d == 0 || h == 0 {
		return out
	}
	cur := make([]int, 0, d)
	var rec func(left int)
	rec = func(left int) {
		if len(cur)+1 == d {
			w := make([]float64, d)
			for i, x := range cur {
				w[i] = float64(x) / float64(h)
			}
			w[d-1] = float64(left) / float64(h)
			out = append(out, w)
			return
		}
		for x := left; x >= 0; x-- {
			cur = append(cur, x)
			rec(left - x)
			cur = cur[:len(cur)-1]
		}
	}
	rec(h)
	return out
}

func binom(n, k int) int {
	if k > n-k {
		k = n - k
	}
	r := 1
	for i := 0; i < k; i++ {
		r = r * (n - i) / (i + 1)
	}
	return r
}

// DivisionsFor is the smallest h ≥ 1 with at least k directions in d objectives (capped).
func DivisionsFor(d, k int) int {
	h := 1
	for binom(h+d-1, d-1) < k && binom(h+d, d-1) <= maxDirections {
		h++
	}
	return h
}

// NSGA3Select selects k indices: fronts that fit, then reference-direction niching on the split
// front (normalised by the ideal point and the nadir of the fronts taken; closest member first).
func NSGA3Select(objs [][]float64, k int) []int {
	out := make([]int, 0, k)
	var last []int
	for _, front := range NonDominatedSort(objs) {
		if len(out)+len(front) <= k {
			out = append(out, front...)
			if len(out) == k {
				return out
			}
		} else {
			last = front
			break
		}
	}
	if last == nil {
		return out
	}
	d := len(objs[last[0]])
	if d == 0 {
		return append(out, last[:k-len(out)]...)
	}
	ideal := make([]float64, d)
	nadir := make([]float64, d)
	for m := range ideal {
		ideal[m], nadir[m] = math.Inf(1), math.Inf(-1)
	}
	pool := append(append([]int(nil), out...), last...)
	for _, i := range pool {
		for m := 0; m < d; m++ {
			if x := objs[i][m]; finite(x) {
				ideal[m] = math.Min(ideal[m], x)
				nadir[m] = math.Max(nadir[m], x)
			}
		}
	}
	norm := func(i int) []float64 {
		v := make([]float64, d)
		for m := 0; m < d; m++ {
			x := objs[i][m]
			switch {
			case !finite(x):
				v[m] = hvRef
			case nadir[m] > ideal[m]:
				v[m] = (x - ideal[m]) / (nadir[m] - ideal[m])
			}
		}
		return v
	}
	dirs := ReferenceDirections(d, DivisionsFor(d, k))
	units := make([][]float64, len(dirs))
	for j, w := range dirs {
		n := 0.0
		for _, x := range w {
			n += x * x
		}
		n = math.Sqrt(n)
		u := make([]float64, d)
		for m, x := range w {
			u[m] = x / n
		}
		units[j] = u
	}
	associate := func(p []float64) (int, float64) {
		best, bd := 0, math.Inf(1)
		for j, u := range units {
			t := 0.0
			for m := range p {
				t += p[m] * u[m]
			}
			d2 := 0.0
			for m := range p {
				e := p[m] - t*u[m]
				d2 += e * e
			}
			dist := math.Sqrt(math.Max(d2, 0))
			if dist < bd {
				best, bd = j, dist
			}
		}
		return best, bd
	}
	niche := make([]int, len(units))
	for _, i := range out {
		j, _ := associate(norm(i))
		niche[j]++
	}
	type cand struct {
		i, j int
		dist float64
	}
	cands := make([]cand, len(last))
	for p, i := range last {
		j, dist := associate(norm(i))
		cands[p] = cand{i, j, dist}
	}
	excluded := make([]bool, len(units))
	for len(out) < k && len(cands) > 0 {
		jBest := -1
		for j := range units {
			if excluded[j] {
				continue
			}
			has := false
			for _, c := range cands {
				if c.j == j {
					has = true
					break
				}
			}
			if has && (jBest < 0 || niche[j] < niche[jBest]) {
				jBest = j
			}
		}
		if jBest < 0 {
			break
		}
		pos := -1
		for p, c := range cands {
			if c.j == jBest && (pos < 0 || c.dist < cands[pos].dist) {
				pos = p
			}
		}
		if pos < 0 {
			excluded[jBest] = true
			continue
		}
		out = append(out, cands[pos].i)
		cands = append(cands[:pos], cands[pos+1:]...)
		niche[jBest]++
	}
	return out
}

// SelectSurvivors is spec 34 §6.2 step 6: NSGA-II for ≤ 3 objectives, NSGA-III for ≥ 4.
func SelectSurvivors(objs [][]float64, k int) []int {
	if len(objs) == 0 || len(objs[0]) <= 3 {
		return NSGA2Select(objs, k)
	}
	return NSGA3Select(objs, k)
}
