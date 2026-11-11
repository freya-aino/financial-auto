package main

import (
	"encoding/json"
	"testing"
)

func TestFrontJobOnlyForFinalDecoderSummaries(t *testing.T) {
	s := Summary{GenerationID: "g1", Kind: "evo.eval.decoder", SnapshotID: "s1", Generation: 2, Final: true, State: "done",
		Keys: []string{"crps"}, Fronts: [][]FrontMember{{{Genome: "forecast:h=3", Rank: 0, HV: 0.2, Score: 0.1}}, {{Genome: "nn_news:tw=5", Rank: 1}}}}
	j := FrontJob(s)
	if j == nil || j.Kind != FrontKind || j.SnapshotId != "s1" || j.ParentJobId != "g1" {
		t.Fatalf("front job %+v", j)
	}
	var spec map[string]any
	if err := json.Unmarshal(j.Spec, &spec); err != nil {
		t.Fatal(err)
	}
	if spec["generation_id"] != "g1" || spec["final"] != true || len(spec["fronts"].([]any)) != 2 {
		t.Fatalf("spec %v", spec)
	}
	s.Final = false
	if FrontJob(s) != nil {
		t.Fatal("rung summaries are not stored")
	}
	s.Final, s.Kind = true, "evo.eval.strategy"
	if FrontJob(s) != nil {
		t.Fatal("strategy fronts are archived by promote.go")
	}
}

func TestSnapshotWatchAndReevalJob(t *testing.T) {
	w := NewSnapshotWatch()
	if w.Observe("1m", "a") || w.Observe("1m", "a") || w.Observe("1s", "x") {
		t.Fatal("first sightings and repeats are no swap")
	}
	if !w.Observe("1m", "b") || w.Observe("1m", "b") {
		t.Fatal("a new snapshot id is a swap, once")
	}
	j := ReevalArgs(map[string]string{"snapshot_id": "b", "timescale": "1m", "k": "2", "data_window_hours": "3", "dry_run": "1"})
	var spec map[string]any
	_ = json.Unmarshal(j.Spec, &spec)
	if j.Kind != ReevalKind || spec["timescale"] != "1m" || spec["k"].(float64) != 2 || spec["dry_run"] != true ||
		spec["data_window"].(map[string]any)["hours"].(float64) != 3 {
		t.Fatalf("reeval spec %v", spec)
	}
	t.Setenv("FA_EVO_REEVAL_DELAY_MIN", "0")
	if ReevalDelay() != 0 {
		t.Fatal("0 disables")
	}
}
