package main

import (
	"fmt"
	"testing"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

type sink struct{ jobs []*fav1.Job }

func (s *sink) pub(subject, _ string, j *fav1.Job) error {
	if subject != "jobs.evo.eval.decoder" {
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
		if done, err = g.OnResult(result(j), s.pub); err != nil {
			t.Fatal(err)
		}
	}
	if !done {
		t.Fatal("generation did not finish")
	}
	// rungs: 4@100, 2@200, 1@400 -> 7 jobs
	if len(s.jobs) != 7 {
		t.Fatalf("jobs submitted = %d", len(s.jobs))
	}
	if f := g.Final(); len(f) != 1 || f[0].ID != "g3" {
		t.Fatalf("final %v", f)
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
}
