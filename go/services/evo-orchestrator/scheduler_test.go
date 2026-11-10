package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

type sink struct{ jobs []*fav1.Job }

func (s *sink) pub(subject, _ string, j *fav1.Job) error {
	if !strings.HasPrefix(subject, "jobs.evo.") {
		return fmt.Errorf("unexpected subject %s", subject)
	}
	s.jobs = append(s.jobs, j)
	return nil
}

// score genomes by index so g3 wins.
func result(j *fav1.Job) *fav1.JobResult {
	var n int
	fmt.Sscanf(j.GenomeId, "g%d", &n)
	return &fav1.JobResult{JobId: j.JobId, Status: fav1.JobResult_STATUS_OK, Result: fmt.Appendf(nil, `{"score": %d}`, n)}
}

func TestGenerationRunsAllRungs(t *testing.T) {
	g, err := NewGeneration("evo.eval.decoder", "snap0", []string{"g0", "g1", "g2", "g3"}, 2, 100, 1000, 7)
	if err != nil {
		t.Fatal(err)
	}
	s := &sink{}
	if err := g.Start(s.pub); err != nil {
		t.Fatal(err)
	}
	done := false
	for i := 0; i < len(s.jobs) && !done; i++ {
		j := s.jobs[i]
		if j.Kind != "evo.eval.decoder" || j.SnapshotId != "snap0" || j.ParentJobId != g.ID {
			t.Fatalf("bad job %v", j)
		}
		var env struct {
			Genome     string    `json:"genome"`
			Generation int       `json:"generation"`
			Rung       int       `json:"rung"`
			DataWindow *struct{} `json:"data_window"`
		}
		if err := json.Unmarshal(j.Spec, &env); err != nil || env.Genome != j.GenomeId || env.Generation != 0 || env.DataWindow != nil {
			t.Fatalf("eval envelope %s (%v)", j.Spec, err)
		}
		if done, err = g.OnResult(result(j), s.pub); err != nil {
			t.Fatal(err)
		}
	}
	if !done {
		t.Fatal("generation did not finish")
	}
	// rungs: 4@100, 2@200, 1@400 -> 7 jobs, no variation job with max_generations = 1
	if len(s.jobs) != 7 {
		t.Fatalf("jobs submitted = %d", len(s.jobs))
	}
	// elites = ceil(4/2) = 2, best first
	if f := g.Final(); len(f) != 2 || f[0].ID != "g3" || f[1].ID != "g2" {
		t.Fatalf("final %v", f)
	}
	if len(g.Summaries) != 3 || !g.Summaries[2].Final || g.Summaries[0].Rung != 0 || len(g.Summaries[0].Survivors) != 2 {
		t.Fatalf("summaries %+v", g.Summaries)
	}
}

func TestPreemptedJobIsRequeued(t *testing.T) {
	g, _ := NewGeneration("evo.eval.decoder", "snap0", []string{"g0", "g1"}, 2, 100, 100, 1)
	s := &sink{}
	_ = g.Start(s.pub)
	j := s.jobs[0]
	done, err := g.OnResult(&fav1.JobResult{JobId: j.JobId, Status: fav1.JobResult_STATUS_PREEMPTED, CheckpointRef: "ck1"}, s.pub)
	if err != nil || done {
		t.Fatalf("done=%v err=%v", done, err)
	}
	if len(s.jobs) != 3 || s.jobs[2].CheckpointRef != "ck1" || s.jobs[2].JobId != j.JobId {
		t.Fatalf("expected requeue with checkpoint, got %v", s.jobs)
	}
}

func TestFailedResultScoresLowest(t *testing.T) {
	if ScoreResult(&fav1.JobResult{Status: fav1.JobResult_STATUS_FAILED}) > -1e300 {
		t.Fatal("failed job must score -inf")
	}
	c := CandidateOf("g", &fav1.JobResult{Status: fav1.JobResult_STATUS_CANCELLED, Result: []byte(`{"score": 9, "objectives": {"crps": 1}, "directions": {"crps": "min"}}`)})
	if c.Objectives != nil || !math.IsInf(c.Score, -1) {
		t.Fatalf("cancelled job must carry no objectives: %+v", c)
	}
}

func TestCandidateOfPrefersProtoMaps(t *testing.T) {
	r := &fav1.JobResult{Status: fav1.JobResult_STATUS_OK, Result: []byte(`{"score": 0.5, "objectives": {"crps": 2}, "directions": {"crps": "min"}}`),
		Objectives: map[string]float64{"crps": 1, "crps_skill": 0.5}, Directions: map[string]string{"crps": "min", "crps_skill": "max"}}
	c := CandidateOf("g", r)
	if c.Objectives["crps"] != 1 || c.Score != 0.5 || !c.hasObjectives() {
		t.Fatalf("%+v", c)
	}
	r.Objectives, r.Directions = nil, nil
	if c := CandidateOf("g", r); c.Objectives["crps"] != 2 {
		t.Fatalf("json fallback: %+v", c)
	}
	if VaryKind("evo.eval.decoder") != "evo.vary.decoder" || VaryKind("evo.strategy") != "evo.strategy.vary" {
		t.Fatal("vary kind")
	}
}

// evalResult reports objectives from the genome string "<family>:q=<crps>" so the test
// controls the front; children keep their parents' family.
func evalResult(j *fav1.Job) *fav1.JobResult {
	var crps float64
	fmt.Sscanf(j.GenomeId[strings.Index(j.GenomeId, "q=")+2:], "%f", &crps)
	return &fav1.JobResult{JobId: j.JobId, Status: fav1.JobResult_STATUS_OK,
		Result:     fmt.Appendf(nil, `{"score": %f}`, 1-crps),
		Objectives: map[string]float64{"crps": crps, "crps_skill": 1 - crps, "compute_cost": float64(len(j.GenomeId))},
		Directions: map[string]string{"crps": "min", "crps_skill": "max", "compute_cost": "min"}}
}

func TestGenerationLoopsThroughVariation(t *testing.T) {
	var summaries []Summary
	g, err := NewGenerationWith(GenerationConfig{
		Kind: "evo.eval.decoder", SnapshotID: "s", Eta: 2, MinBudgetMs: 100, MaxBudgetMs: 1000, Seed: 3, MaxGenerations: 2, DataWindowHours: 6,
		Genomes:   []string{"forecast:q=0.5", "forecast:q=0.2", "retrieval:q=0.9", "forecast:q=0.3", "retrieval:q=0.4", "forecast:q=0.7"},
		OnSummary: func(s Summary) { summaries = append(summaries, s) },
	})
	if err != nil {
		t.Fatal(err)
	}
	s := &sink{}
	if err := g.Start(s.pub); err != nil {
		t.Fatal(err)
	}
	var spec struct {
		Genome     string `json:"genome"`
		DataWindow struct {
			Hours int `json:"hours"`
		} `json:"data_window"`
	}
	if err := json.Unmarshal(s.jobs[0].Spec, &spec); err != nil || spec.DataWindow.Hours != 6 || spec.Genome != s.jobs[0].GenomeId {
		t.Fatalf("eval spec %s (%v)", s.jobs[0].Spec, err)
	}
	done := false
	varyJobs := 0
	for i := 0; i < len(s.jobs) && !done; i++ {
		j := s.jobs[i]
		if j.Kind == "evo.vary.decoder" {
			varyJobs++
			var v struct {
				Parents []struct {
					Genome     string             `json:"genome"`
					Objectives map[string]float64 `json:"objectives"`
				} `json:"parents"`
				NChildren int    `json:"n_children"`
				Seed      uint64 `json:"seed"`
			}
			if err := json.Unmarshal(j.Spec, &v); err != nil {
				t.Fatal(err)
			}
			// elites = ceil(6/2) = 3, children = 3
			if len(v.Parents) != 3 || v.NChildren != 3 || v.Parents[0].Objectives["crps"] == 0 {
				t.Fatalf("vary spec %s", j.Spec)
			}
			children := []string{}
			for k, p := range v.Parents {
				children = append(children, fmt.Sprintf("%s;child=%d", p.Genome, k))
			}
			b, _ := json.Marshal(map[string]any{"children": children})
			done, err = g.OnResult(&fav1.JobResult{JobId: j.JobId, Status: fav1.JobResult_STATUS_OK, Result: b}, s.pub)
		} else {
			done, err = g.OnResult(evalResult(j), s.pub)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if !done || varyJobs != 1 {
		t.Fatalf("done=%v vary=%d", done, varyJobs)
	}
	// loop 0: 6+3+2+1 = 12 evals, 1 vary; loop 1: 6+3+2+1 = 12 evals
	if len(s.jobs) != 25 {
		t.Fatalf("jobs = %d", len(s.jobs))
	}
	gen, maxG, _, _ := g.Index()
	if gen != 1 || maxG != 2 {
		t.Fatalf("index %d/%d", gen, maxG)
	}
	f := g.Final()
	if len(f) != 3 || f[0].Objectives["crps"] > f[len(f)-1].Objectives["crps"] && f[0].Objectives["compute_cost"] > f[len(f)-1].Objectives["compute_cost"] {
		t.Fatalf("final %v", f)
	}
	// 4 rung summaries per loop
	if len(summaries) != 8 || summaries[3].VaryJobID == "" || summaries[3].NextPopSize != 6 || summaries[7].VaryJobID != "" || !summaries[7].Final {
		t.Fatalf("summaries %d: %+v", len(summaries), summaries)
	}
	if len(summaries[0].ParetoFront) == 0 || summaries[0].Keys[0] != "compute_cost" || summaries[0].Generation != 0 || summaries[4].Generation != 1 {
		t.Fatalf("summary 0: %+v", summaries[0])
	}
}

func TestCancelledVaryJobEndsGeneration(t *testing.T) {
	g, _ := NewGenerationWith(GenerationConfig{Kind: "evo.eval.decoder", Eta: 2, MinBudgetMs: 100, MaxBudgetMs: 100, MaxGenerations: 3,
		Genomes: []string{"forecast:q=0.5", "forecast:q=0.2"}})
	s := &sink{}
	_ = g.Start(s.pub)
	for _, j := range s.jobs[:2] {
		if done, _ := g.OnResult(evalResult(j), s.pub); done {
			t.Fatal("finished early")
		}
	}
	vary := s.jobs[2]
	if vary.Kind != "evo.vary.decoder" || !g.Owns(vary.JobId) || g.GenomeOf(vary.JobId) != "" {
		t.Fatalf("expected vary job, got %v", vary)
	}
	done, err := g.OnResult(&fav1.JobResult{JobId: vary.JobId, Status: fav1.JobResult_STATUS_CANCELLED, Error: "cancelled by operator"}, s.pub)
	if !done || err == nil {
		t.Fatalf("done=%v err=%v", done, err)
	}
	if f := g.Final(); len(f) != 1 || f[0].ID != "forecast:q=0.2" {
		t.Fatalf("final %v", f)
	}
}

// statsResult is an OK eval result carrying worker-side generation_stats.
func statsResult(j *fav1.Job, n int, dsr, pbo float64) *fav1.JobResult {
	r := evalResult(j)
	r.Result = fmt.Appendf(nil, `{"score": 0.1, "generation_stats": {"n_trials": %d, "deflated_sharpe": %g, "pbo": %g, "pbo_trials": %d, "pbo_blocks": 16}}`, n, dsr, pbo, n)
	return r
}

func TestValidationStatsRideAlongInSummaries(t *testing.T) {
	var summaries []Summary
	g, _ := NewGenerationWith(GenerationConfig{Kind: "evo.eval.decoder", Eta: 2, MinBudgetMs: 100, MaxBudgetMs: 100, MaxGenerations: 1,
		Genomes: []string{"forecast:q=0.5", "forecast:q=0.2"}, OnSummary: func(s Summary) { summaries = append(summaries, s) }})
	s := &sink{}
	_ = g.Start(s.pub)
	if done, _ := g.OnResult(statsResult(s.jobs[0], 1, 0.6, 0.5), s.pub); done {
		t.Fatal("early")
	}
	// a result without stats keeps the last seen ones
	done, _ := g.OnResult(evalResult(s.jobs[1]), s.pub)
	if !done || len(summaries) != 1 || summaries[0].Validation == nil || summaries[0].Validation.NTrials != 1 || *summaries[0].Validation.PBO != 0.5 {
		t.Fatalf("summaries %+v", summaries)
	}
	if summaries[0].State != "done" || summaries[0].ArtifactKey != "" {
		t.Fatalf("state %q artifact %q", summaries[0].State, summaries[0].ArtifactKey)
	}
	if validationOf(&fav1.JobResult{Status: fav1.JobResult_STATUS_FAILED, Result: []byte(`{"generation_stats":{"n_trials":3}}`)}) != nil {
		t.Fatal("failed results carry no stats")
	}
}

func TestPauseDefersRungAndResumeSubmitsIt(t *testing.T) {
	g, _ := NewGenerationWith(GenerationConfig{Kind: "evo.eval.decoder", Eta: 2, MinBudgetMs: 100, MaxBudgetMs: 1000, MaxGenerations: 1,
		Genomes: []string{"forecast:q=0.5", "forecast:q=0.2", "forecast:q=0.9", "forecast:q=0.1"}})
	s := &sink{}
	_ = g.Start(s.pub)
	if !g.Pause() || g.Pause() || !g.Paused() {
		t.Fatal("pause should succeed once")
	}
	for _, j := range s.jobs[:4] {
		if done, err := g.OnResult(evalResult(j), s.pub); done || err != nil {
			t.Fatalf("done=%v err=%v", done, err)
		}
	}
	if len(s.jobs) != 4 {
		t.Fatalf("paused generation submitted rung 1: %d jobs", len(s.jobs))
	}
	_, gens, rung, _ := g.Index()
	if rung != 1 || gens != 1 {
		t.Fatalf("rung %d", rung)
	}
	ok, err := g.Resume(s.pub)
	if !ok || err != nil || len(s.jobs) != 6 {
		t.Fatalf("resume ok=%v err=%v jobs=%d", ok, err, len(s.jobs))
	}
	if ok, _ := g.Resume(s.pub); ok {
		t.Fatal("double resume")
	}
	var env struct {
		Rung int `json:"rung"`
	}
	_ = json.Unmarshal(s.jobs[5].Spec, &env)
	if env.Rung != 1 {
		t.Fatalf("deferred rung envelope %s", s.jobs[5].Spec)
	}
}

func TestStopCancelsPendingAndPromotesNothing(t *testing.T) {
	promoted := 0
	g, _ := NewGenerationWith(GenerationConfig{Kind: StrategyEvalKind, Eta: 2, MinBudgetMs: 100, MaxBudgetMs: 1000, MaxGenerations: 2,
		Genomes: []string{"s:q=0.5", "s:q=0.2", "s:q=0.9"},
		Promote: func(*Generation, []FrontMember) (string, error) { promoted++; return "k", nil }})
	s := &sink{}
	_ = g.Start(s.pub)
	_, _ = g.OnResult(evalResult(s.jobs[0]), s.pub)
	cancelled := g.Stop()
	if len(cancelled) != 2 || cancelled[s.jobs[1].JobId] != StrategyEvalKind {
		t.Fatalf("cancelled %v", cancelled)
	}
	if len(g.Stop()) != 0 {
		t.Fatal("stop twice")
	}
	if f := g.Final(); len(f) != 1 || f[0].ID != "s:q=0.5" {
		t.Fatalf("final %v", f)
	}
	last := g.Summaries[len(g.Summaries)-1]
	if last.State != "stopped" || !last.Final || !strings.Contains(last.Note, "2 pending") || promoted != 0 {
		t.Fatalf("summary %+v promoted=%d", last, promoted)
	}
	// results arriving later for cancelled jobs are ignored
	if done, _ := g.OnResult(evalResult(s.jobs[1]), s.pub); !done {
		t.Fatal("stopped generation stays done")
	}
}

func TestStrategyGenerationPromotesFinalFront(t *testing.T) {
	var gotFront []FrontMember
	var summaries []Summary
	genomes := []string{
		`{ id = "gp-a", timescale = "1s", signal = "(mul 2 analog_forecast)" }`,
		`{ id = "gp-b", timescale = "1s", signal = "(neg analog_forecast)" }`,
	}
	g, _ := NewGenerationWith(GenerationConfig{Kind: StrategyEvalKind, Eta: 2, MinBudgetMs: 100, MaxBudgetMs: 100, MaxGenerations: 1,
		Genomes: genomes, OnSummary: func(s Summary) { summaries = append(summaries, s) },
		Promote: func(g *Generation, front []FrontMember) (string, error) {
			gotFront = front
			return ArtifactsBucket + "/" + StrategiesKey(g.ID), nil
		}})
	s := &sink{}
	_ = g.Start(s.pub)
	for i, j := range s.jobs {
		r := &fav1.JobResult{JobId: j.JobId, Status: fav1.JobResult_STATUS_OK, Result: []byte(`{"score":0}`),
			Objectives: map[string]float64{"sharpe": float64(i), "turnover": float64(i)},
			Directions: map[string]string{"sharpe": "max", "turnover": "min"}}
		_, _ = g.OnResult(r, s.pub)
	}
	last := summaries[len(summaries)-1]
	if last.ArtifactKey != "fa-artifacts/strategies/"+g.ID+".toml" || len(gotFront) != 2 {
		t.Fatalf("artifact %q front %d", last.ArtifactKey, len(gotFront))
	}
	body, n := StrategiesTOML(g.ID, g.Kind, append(gotFront, FrontMember{Genome: "forecast:h=3"}), time.Unix(0, 0))
	if n != 2 || !strings.Contains(body, "instruments = []") || !strings.Contains(body, genomes[0]+",") || strings.Contains(body, "forecast:h=3") {
		t.Fatalf("toml:\n%s", body)
	}
	if VaryKind(StrategyEvalKind) != "evo.vary.strategy" {
		t.Fatal("vary kind")
	}
	// decoder generations are never archived by the strategy hook
	hook := PromoteStrategies(nil, nil)
	if key, err := hook(&Generation{Kind: "evo.eval.decoder"}, gotFront); key != "" || err != nil {
		t.Fatalf("decoder promote %q %v", key, err)
	}
}
