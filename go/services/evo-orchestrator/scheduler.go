package main

import (
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// PublishJob sends a job on a subject with a dedup message id.
type PublishJob func(subject, msgID string, job *fav1.Job) error

// ScoreResult extracts a scalar score from a JobResult. Provisional encoding:
// result bytes are JSON {"score": <float>} until FitnessRecord is defined (spec 34).
func ScoreResult(r *fav1.JobResult) float64 {
	if r.Status != fav1.JobResult_STATUS_OK {
		return math.Inf(-1)
	}
	var v struct {
		Score float64 `json:"score"`
	}
	if err := json.Unmarshal(r.Result, &v); err != nil {
		return math.Inf(-1)
	}
	return v.Score
}

// Generation runs one successive-halving generation over a genome population.
type Generation struct {
	mu         sync.Mutex
	ID         string
	Kind       string // e.g. "evo.eval.decoder"
	SnapshotID string
	Seed       uint64
	Priority   uint32
	eta        int
	rungs      []Rung
	rung       int
	pending    map[string]*fav1.Job // job id -> job
	scores     []Candidate
	alive      []string
	final      []Candidate
}

// NewGeneration plans rungs for the given genomes.
func NewGeneration(kind, snapshotID string, genomes []string, eta int, minBudget, maxBudget uint64, seed uint64) (*Generation, error) {
	rungs := Plan(len(genomes), eta, minBudget, maxBudget)
	if len(rungs) == 0 {
		return nil, fmt.Errorf("empty plan for %d genomes", len(genomes))
	}
	return &Generation{
		ID: ulid.Make().String(), Kind: kind, SnapshotID: snapshotID, Seed: seed,
		Priority: uint32(EvolutionEval), eta: max(eta, 2), rungs: rungs, pending: map[string]*fav1.Job{},
		alive: append([]string(nil), genomes...),
	}, nil
}

// Start submits the first rung.
func (g *Generation) Start(pub PublishJob) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.submitRungLocked(pub)
}

func (g *Generation) submitRungLocked(pub PublishJob) error {
	r := g.rungs[g.rung]
	g.scores = nil
	for i, genome := range g.alive {
		job := &fav1.Job{
			JobId: ulid.Make().String(), Kind: g.Kind, ParentJobId: g.ID, Priority: g.Priority,
			SnapshotId: g.SnapshotID, GenomeId: genome, BudgetGpuMs: r.BudgetMs,
			Seed:     g.Seed + uint64(g.rung)*1_000_003 + uint64(i),
			Deadline: timestamppb.New(time.Now().Add(time.Duration(r.BudgetMs)*time.Millisecond*10 + time.Minute)),
		}
		g.pending[job.JobId] = job
		if err := pub(bus.Jobs(g.Kind), job.JobId, job); err != nil {
			return err
		}
	}
	return nil
}

// Owns reports whether a job id belongs to this generation.
func (g *Generation) Owns(jobID string) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	_, ok := g.pending[jobID]
	return ok
}

// OnResult records a result; preempted jobs are re-queued from their checkpoint.
// When a rung completes, survivors advance. Returns true when the generation is done.
func (g *Generation) OnResult(r *fav1.JobResult, pub PublishJob) (bool, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	job, ok := g.pending[r.JobId]
	if !ok {
		return g.final != nil, nil
	}
	if r.Status == fav1.JobResult_STATUS_PREEMPTED {
		job.CheckpointRef = r.CheckpointRef
		return false, pub(bus.Jobs(g.Kind), job.JobId+"-"+r.CheckpointRef, job)
	}
	delete(g.pending, r.JobId)
	g.scores = append(g.scores, Candidate{ID: job.GenomeId, Score: ScoreResult(r)})
	if len(g.pending) > 0 {
		return false, nil
	}
	if g.rung == len(g.rungs)-1 {
		g.final = Halve(g.scores, max(len(g.scores), 2)) // single best
		return true, nil
	}
	survivors := Halve(g.scores, g.eta)
	g.alive = g.alive[:0]
	for _, s := range survivors {
		g.alive = append(g.alive, s.ID)
	}
	g.rung++
	return false, g.submitRungLocked(pub)
}

// Final returns the winner(s) once done.
func (g *Generation) Final() []Candidate {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.final
}
