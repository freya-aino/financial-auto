package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// neuralResult is an OK eval result of a neural genome with a persisted bundle.
func neuralResult(j *fav1.Job, used uint64) *fav1.JobResult {
	r := evalResult(j)
	r.UsedGpuMs = used
	r.Result = fmt.Appendf(nil, `{"score": 0.1, "neural": {"weights": {"persisted": true, "bucket": "fa-models", "key": "decoders/%s.safetensors"}}}`, j.JobId)
	return r
}

type envelope struct {
	Genome        string   `json:"genome"`
	WarmStart     string   `json:"warm_start"`
	StepsFraction *float64 `json:"train_steps_fraction"`
}

func envOf(t *testing.T, j *fav1.Job) envelope {
	var e envelope
	if err := json.Unmarshal(j.Spec, &e); err != nil {
		t.Fatal(err)
	}
	return e
}

func TestNeuralRungsWarmStartAndScaleSteps(t *testing.T) {
	var summaries []Summary
	g, _ := NewGenerationWith(GenerationConfig{Kind: "evo.eval.decoder", Eta: 2, MinBudgetMs: 100, MaxBudgetMs: 1000, MaxGenerations: 2,
		Genomes:   []string{"nn_forecast:q=0.5", "nn_forecast:q=0.2", "forecast:q=0.3", "nn_news:q=0.4"},
		OnSummary: func(s Summary) { summaries = append(summaries, s) }})
	s := &sink{}
	_ = g.Start(s.pub)
	// rungs 4@100, 2@200, 1@400: fractions 1/4, 1/2, 1; no warm start at rung 0
	for _, j := range s.jobs[:4] {
		e := envOf(t, j)
		if isNeural(j.GenomeId) != (e.StepsFraction != nil) || e.WarmStart != "" {
			t.Fatalf("rung 0 envelope %s", j.Spec)
		}
		if e.StepsFraction != nil && math.Abs(*e.StepsFraction-0.25) > 1e-12 {
			t.Fatalf("fraction %v", *e.StepsFraction)
		}
	}
	refOf := map[string]string{}     // genome -> latest bundle ref
	parentRef := map[string]string{} // child -> its parent's bundle at variation time
	var varyChildren []string
	for i := 0; i < len(s.jobs); i++ {
		j := s.jobs[i]
		var r *fav1.JobResult
		switch j.Kind {
		case "evo.vary.decoder":
			var v struct {
				Parents []struct {
					Genome string `json:"genome"`
				} `json:"parents"`
			}
			_ = json.Unmarshal(j.Spec, &v)
			var lineage []map[string]any
			for k, p := range v.Parents {
				c := fmt.Sprintf("%s;child=%d", p.Genome, k)
				varyChildren = append(varyChildren, c)
				parentRef[c] = refOf[p.Genome]
				lineage = append(lineage, map[string]any{"child": c, "parents": []string{"nope", p.Genome}})
			}
			b, _ := json.Marshal(map[string]any{"children": varyChildren, "lineage": lineage})
			r = &fav1.JobResult{JobId: j.JobId, Status: fav1.JobResult_STATUS_OK, Result: b, UsedGpuMs: 5}
		default:
			e := envOf(t, j)
			if want := refOf[j.GenomeId]; isNeural(j.GenomeId) && want != "" && e.WarmStart != want {
				t.Fatalf("%s: warm start %q, want own previous bundle %q", j.GenomeId, e.WarmStart, want)
			}
			if strings.Contains(j.GenomeId, ";child=") && isNeural(j.GenomeId) && refOf[j.GenomeId] == "" {
				if e.WarmStart == "" || e.WarmStart != parentRef[j.GenomeId] {
					t.Fatalf("child %s warm start %q, parent bundle %q", j.GenomeId, e.WarmStart, parentRef[j.GenomeId])
				}
			}
			if isNeural(j.GenomeId) {
				r = neuralResult(j, 10)
				refOf[j.GenomeId] = "fa-models/decoders/" + j.JobId + ".safetensors"
			} else {
				r = evalResult(j)
				r.UsedGpuMs = 10
			}
		}
		done, err := g.OnResult(r, s.pub)
		if err != nil {
			t.Fatal(err)
		}
		if done {
			break
		}
	}
	if len(varyChildren) == 0 {
		t.Fatal("no variation")
	}
	warmed := 0
	for _, sm := range summaries {
		warmed += sm.WarmStarted
		if sm.Budget == nil {
			t.Fatal("summary without budget")
		}
	}
	last := summaries[len(summaries)-1]
	if warmed == 0 || last.Budget.UsedMs == 0 || last.Budget.UsedMs < last.Budget.LoopUsedMs {
		t.Fatalf("warm %d budget %+v", warmed, last.Budget)
	}
}

func TestBudgetTruncatesRungs(t *testing.T) {
	var summaries []Summary
	// 4@100 + 2@200 + 1@400 = 1200 planned; total 500: after rung 0 (4×100 used) rung 1 would need 400 → 800 > 550
	g, _ := NewGenerationWith(GenerationConfig{Kind: "evo.eval.decoder", Eta: 2, MinBudgetMs: 100, MaxBudgetMs: 1000, MaxGenerations: 1,
		TotalBudgetMs: 500, Genomes: []string{"forecast:q=0.5", "forecast:q=0.2", "forecast:q=0.3", "forecast:q=0.4"},
		OnSummary: func(s Summary) { summaries = append(summaries, s) }})
	s := &sink{}
	_ = g.Start(s.pub)
	var done bool
	for _, j := range s.jobs {
		r := evalResult(j)
		r.UsedGpuMs = 100
		done, _ = g.OnResult(r, s.pub)
	}
	if !done || len(s.jobs) != 4 || len(summaries) != 1 {
		t.Fatalf("done=%v jobs=%d summaries=%d", done, len(s.jobs), len(summaries))
	}
	b := summaries[0].Budget
	if !summaries[0].Final || !b.Truncated || b.Truncations != 1 || b.LoopUsedMs != 400 || b.PlannedMs != 1200 || !strings.Contains(summaries[0].Note, "truncated") {
		t.Fatalf("summary %+v budget %+v", summaries[0], b)
	}
	if f := g.Final(); len(f) != 2 || f[0].ID != "forecast:q=0.2" {
		t.Fatalf("partial results kept: %v", f)
	}
}

func TestMCMCPairsSeedTheRNNPrior(t *testing.T) {
	book := NewPairBook()
	book.Update(&fav1.CrossAssetPriors{Timescale: "1m", Proposals: []*fav1.CrossAssetPair{
		{SourceInstrumentId: "s1", TargetInstrumentId: "t1", Score: 2},
		{SourceInstrumentId: "s2", TargetInstrumentId: "t2", Score: 5},
		{SourceInstrumentId: "x", TargetInstrumentId: "x", Score: 9}, // self pair dropped
	}})
	if seen, held := book.Stats(); seen != 1 || held != 2 {
		t.Fatalf("stats %d %d", seen, held)
	}
	top := book.Top(1)
	if len(top) != 1 || top[0].Source != "s2" {
		t.Fatalf("top %v", top)
	}
	pop := []string{"forecast:q=0.1", "nn_forecast:h=3;lb=16;q=0.2;inst=old"}
	pairs := PairGenomes(book.Top(5), pop)
	if len(pairs) != 2 || pairs[0] != "nn_forecast:h=3;lb=16;q=0.2;inst=t2;src=s2;ts=1m" {
		t.Fatalf("pairs %v", pairs)
	}
	if again := PairGenomes(book.Top(5), append(pop, pairs...)); len(again) != 0 {
		t.Fatalf("duplicates %v", again)
	}
	if d := PairGenomes(book.Top(1), nil); len(d) != 1 || !strings.HasPrefix(d[0], "nn_forecast:h=5;lb=32") || !strings.HasSuffix(d[0], ";inst=t2;src=s2;ts=1m") {
		t.Fatalf("default template %v", d)
	}
	var summaries []Summary
	g, err := NewGenerationWith(GenerationConfig{Kind: "evo.eval.decoder", Eta: 2, MinBudgetMs: 100, MaxBudgetMs: 100, MaxGenerations: 1,
		Genomes: pop, PairProposals: book.Top, MaxPairs: 1, OnSummary: func(s Summary) { summaries = append(summaries, s) }})
	if err != nil {
		t.Fatal(err)
	}
	s := &sink{}
	_ = g.Start(s.pub)
	if len(s.jobs) != 3 || !strings.Contains(s.jobs[2].GenomeId, "src=s2") {
		t.Fatalf("jobs %v", s.jobs)
	}
	for _, j := range s.jobs {
		_, _ = g.OnResult(evalResult(j), s.pub)
	}
	if len(summaries) == 0 || summaries[0].PairsSeeded != 1 {
		t.Fatalf("summaries %+v", summaries)
	}
	// strategy generations never get pairs
	sg, _ := NewGenerationWith(GenerationConfig{Kind: "evo.eval.strategy", Eta: 2, MinBudgetMs: 100, MaxBudgetMs: 100,
		Genomes: []string{"a"}, PairProposals: book.Top, MaxPairs: 3})
	if sg.popSize != 1 {
		t.Fatalf("strategy population %d", sg.popSize)
	}
}

func TestWithGenesAndHelpers(t *testing.T) {
	if got := withGenes("nn_forecast", map[string]string{"src": "a", "inst": ""}); got != "nn_forecast:src=a" {
		t.Fatal(got)
	}
	if got := withGenes("nn_forecast:h=3;src=b;lb=4", map[string]string{"src": "", "ts": "1h"}); got != "nn_forecast:h=3;lb=4;ts=1h" {
		t.Fatal(got)
	}
	if !isNeural("nn_chart:res=8") || isNeural("forecast:h=1") || genomeFamily(" NN_News:x=1") != "nn_news" {
		t.Fatal("family helpers")
	}
	if weightsRefOf(&fav1.JobResult{Status: fav1.JobResult_STATUS_OK, Result: []byte(`{"neural":{"weights":{"persisted":false,"key":"k"}}}`)}) != "" {
		t.Fatal("unpersisted bundle used")
	}
	if l := lineageOf(&fav1.JobResult{Result: []byte(`{"lineage":[{"child":"c","parents":["p1","p2"]}]}`)}); len(l["c"]) != 2 {
		t.Fatal(l)
	}
	if l := lineageOf(&fav1.JobResult{Result: []byte(`{"lineage":[{"child":"c","parents":["nn:ts=1s;h=5"],"parents_as_sent":["nn:h=5"]}]}`)}); l["c"][0] != "nn:h=5" {
		t.Fatal(l)
	}
}
