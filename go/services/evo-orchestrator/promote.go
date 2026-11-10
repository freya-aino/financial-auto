package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ArtifactsBucket holds eval reports (evo-worker) and promoted strategy populations.
const ArtifactsBucket = "fa-artifacts"

// StrategyEvalKind is the strategy evaluation job kind (evo-worker `evo.eval.strategy`).
const StrategyEvalKind = "evo.eval.strategy"

// StrategiesKey is the Garage key of a generation's promoted population file.
func StrategiesKey(generationID string) string { return "strategies/" + generationID + ".toml" }

// StrategiesTOML renders a `strategies.toml` population file from the final front. Genome
// strings of strategy generations are TOML inline tables (`fa_strategies::genome`), so the file
// is `instruments = []` plus `strategy = [ {…}, {…} ]`, which `StrategyFile::parse` reads.
// Entries that are not inline tables (e.g. a decoder kv genome) are skipped.
func StrategiesTOML(generationID string, kind string, front []FrontMember, at time.Time) (string, int) {
	var b strings.Builder
	fmt.Fprintf(&b, "# Promoted strategy population: final Pareto front of generation %s (%s).\n", generationID, kind)
	fmt.Fprintf(&b, "# Written by evo-orchestrator at %s (spec 35 §6.4). Candidates still need the\n", at.UTC().Format(time.RFC3339))
	b.WriteString("# locked-holdout evaluation and the live paper shadow before trading (spec 41 §6.4, spec 40).\n")
	b.WriteString("instruments = []\n\nstrategy = [\n")
	n := 0
	for _, m := range front {
		g := strings.TrimSpace(m.Genome)
		if !strings.HasPrefix(g, "{") || !strings.HasSuffix(g, "}") {
			continue
		}
		fmt.Fprintf(&b, "  # rank %d, score %.6g\n  %s,\n", m.Rank, m.Score, g)
		n++
	}
	b.WriteString("]\n")
	return b.String(), n
}

// ObjectPutter is the subset of archive.S3Store the promotion needs.
type ObjectPutter interface {
	Put(ctx context.Context, bucket, key string, body []byte, contentType string) error
}

// AllocatorEvalKind is the allocator evaluation job kind (evo-worker `evo.eval.allocator`).
const AllocatorEvalKind = "evo.eval.allocator"

// AllocatorsKey is the Garage key of an allocator generation's promoted front.
func AllocatorsKey(generationID string) string { return "allocators/" + generationID + ".toml" }

// AllocatorSelection is the `{name, params}` payload risk-engine accepts under
// `ctl.risk.limits {set: {allocator: …}}` (fa_allocator::genome::AllocatorSpec).
type AllocatorSelection struct {
	Name   string             `json:"name"`
	Params map[string]float64 `json:"params"`
}

// ParseAllocatorGenome splits `alloc:name=risk_parity;cap=0.4;…` into its selection payload.
func ParseAllocatorGenome(genome string) (AllocatorSelection, bool) {
	body, ok := strings.CutPrefix(strings.TrimSpace(genome), "alloc:")
	if !ok {
		return AllocatorSelection{}, false
	}
	sel := AllocatorSelection{Name: "equal_weight", Params: map[string]float64{}}
	for _, part := range strings.Split(body, ";") {
		k, v, found := strings.Cut(strings.TrimSpace(part), "=")
		if !found {
			continue
		}
		if k = strings.TrimSpace(k); k == "name" {
			sel.Name = strings.TrimSpace(v)
		} else if f, err := strconv.ParseFloat(strings.TrimSpace(v), 64); err == nil {
			sel.Params[k] = f
		}
	}
	return sel, true
}

func tomlFloat(v float64) string {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return "0.0"
	}
	s := strconv.FormatFloat(v, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

func tomlInline(sel AllocatorSelection) string {
	keys := make([]string, 0, len(sel.Params))
	for k := range sel.Params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = k + " = " + tomlFloat(sel.Params[k])
	}
	return "{ " + strings.Join(parts, ", ") + " }"
}

// AllocatorsTOML renders the final front of an allocator generation: `[active]` = the rank-0
// member with the best score (deflated Sharpe) and one `[[allocator]]` table per front member,
// each with its genome text, `name` and `params` (the ctl.risk.limits payload). Entries that are
// not `alloc:` genomes are skipped.
func AllocatorsTOML(generationID, kind string, front []FrontMember, at time.Time) (string, int) {
	type entry struct {
		m   FrontMember
		sel AllocatorSelection
	}
	var es []entry
	for _, m := range front {
		if sel, ok := ParseAllocatorGenome(m.Genome); ok {
			es = append(es, entry{m, sel})
		}
	}
	if len(es) == 0 {
		return "", 0
	}
	best := 0
	for i, e := range es {
		if e.m.Rank < es[best].m.Rank || (e.m.Rank == es[best].m.Rank && e.m.Score > es[best].m.Score) {
			best = i
		}
	}
	payload, _ := json.Marshal(map[string]any{"allocator": es[best].sel})
	var b strings.Builder
	fmt.Fprintf(&b, "# Promoted allocator population: final Pareto front of generation %s (%s).\n", generationID, kind)
	fmt.Fprintf(&b, "# Written by evo-orchestrator at %s (spec 40 §6.2 allocator tournament). Paper only until\n", at.UTC().Format(time.RFC3339))
	b.WriteString("# selected: risk-engine takes one through ctl.risk.limits {set: {allocator: {name, params}}}, e.g.\n")
	fmt.Fprintf(&b, "#   set=%s\n\n", payload)
	fmt.Fprintf(&b, "[active]\ngenome = %q\nname = %q\nparams = %s\n\n", es[best].m.Genome, es[best].sel.Name, tomlInline(es[best].sel))
	for _, e := range es {
		fmt.Fprintf(&b, "[[allocator]]\nrank = %d\nscore = %s\ngenome = %q\nname = %q\nparams = %s\n\n",
			e.m.Rank, tomlFloat(e.m.Score), e.m.Genome, e.sel.Name, tomlInline(e.sel))
	}
	return b.String(), len(es)
}

// PromoteStrategies is PromoteFronts (kept for callers of the strategy-only name).
func PromoteStrategies(store ObjectPutter, onErr func(error)) Promote {
	return PromoteFronts(store, onErr)
}

// PromoteFronts returns the Promote hook: the final front of a strategy generation becomes
// `fa-artifacts/strategies/<generation>.toml`, of an allocator generation
// `fa-artifacts/allocators/<generation>.toml`; other kinds are not archived (empty key).
func PromoteFronts(store ObjectPutter, onErr func(error)) Promote {
	return func(g *Generation, front []FrontMember) (string, error) {
		var body, key, what string
		var n int
		switch g.Kind {
		case StrategyEvalKind:
			body, n = StrategiesTOML(g.ID, g.Kind, front, time.Now())
			key, what = StrategiesKey(g.ID), "strategy"
		case AllocatorEvalKind:
			body, n = AllocatorsTOML(g.ID, g.Kind, front, time.Now())
			key, what = AllocatorsKey(g.ID), "allocator"
		default:
			return "", nil
		}
		if n == 0 {
			return "", fmt.Errorf("no %s genome in the front", what)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		if err := store.Put(ctx, ArtifactsBucket, key, []byte(body), "application/toml"); err != nil {
			if onErr != nil {
				onErr(err)
			}
			return "", err
		}
		return ArtifactsBucket + "/" + key, nil
	}
}
