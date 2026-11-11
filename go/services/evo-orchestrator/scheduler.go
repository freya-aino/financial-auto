package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// PublishJob sends a job on a subject with a dedup message id.
type PublishJob func(subject, msgID string, job *fav1.Job) error

// VaryKind derives the variation job kind from an evaluation kind:
// evo.eval.decoder → evo.vary.decoder (same for strategies later).
func VaryKind(evalKind string) string {
	if strings.Contains(evalKind, ".eval.") {
		return strings.Replace(evalKind, ".eval.", ".vary.", 1)
	}
	return evalKind + ".vary"
}

// ScoreResult extracts the provisional scalar score (D-025) from a JobResult:
// JSON {"score": <float>} in result; -Inf for anything but STATUS_OK.
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

// CandidateOf builds the candidate for a finished job: objectives/directions from the
// proto maps when the worker filled them, else from the result JSON (D-044 payload);
// failed jobs carry no objectives and score -Inf.
func CandidateOf(genome string, r *fav1.JobResult) Candidate {
	c := Candidate{ID: genome, Score: ScoreResult(r)}
	if r.Status != fav1.JobResult_STATUS_OK {
		return c
	}
	if len(r.Objectives) > 0 && len(r.Directions) > 0 {
		c.Objectives, c.Directions = r.Objectives, r.Directions
		return c
	}
	var v struct {
		Objectives map[string]float64 `json:"objectives"`
		Directions map[string]string  `json:"directions"`
	}
	if err := json.Unmarshal(r.Result, &v); err == nil && len(v.Objectives) > 0 && len(v.Directions) > 0 {
		c.Objectives, c.Directions = v.Objectives, v.Directions
	}
	return c
}

// ValidationStats are the per-generation multiple-testing statistics the worker computes over
// the trial registry after every evaluation (`generation_stats` in the eval result JSON,
// spec 41 §6.5): deflated Sharpe of the best trial and PBO over the trial set. The summary
// carries the latest one seen in this loop.
type ValidationStats struct {
	NTrials           int      `json:"n_trials"`
	SharpeMean        *float64 `json:"sharpe_mean,omitempty"`
	SharpeVar         *float64 `json:"sharpe_var,omitempty"`
	BestCandidate     string   `json:"best_candidate,omitempty"`
	BestSharpe        *float64 `json:"best_sharpe,omitempty"`
	ExpectedMaxSharpe *float64 `json:"expected_max_sharpe,omitempty"`
	DeflatedSharpe    *float64 `json:"deflated_sharpe,omitempty"`
	PBO               *float64 `json:"pbo,omitempty"`
	PBOTrials         int      `json:"pbo_trials"`
	PBOBlocks         int      `json:"pbo_blocks"`
}

// validationOf extracts `generation_stats` from an eval result JSON (nil when absent/null).
func validationOf(r *fav1.JobResult) *ValidationStats {
	if r == nil || r.Status != fav1.JobResult_STATUS_OK || len(r.Result) == 0 {
		return nil
	}
	var v struct {
		Stats *ValidationStats `json:"generation_stats"`
	}
	if err := json.Unmarshal(r.Result, &v); err != nil {
		return nil
	}
	return v.Stats
}

// Summary is the per-rung / per-generation Pareto report published as JSON
// (`generation.summary`) on jobs.result.evo.generation and mirrored on heartbeat counters.
type Summary struct {
	Type         string          `json:"type"` // "generation.summary"
	GenerationID string          `json:"generation_id"`
	Kind         string          `json:"kind"`
	SnapshotID   string          `json:"snapshot_id,omitempty"` // encoder snapshot the evaluations read
	Generation   int             `json:"generation"`            // 0-based loop index
	MaxGens      int             `json:"max_generations"`
	Rung         int             `json:"rung"`
	Rungs        int             `json:"rungs"`
	Final        bool            `json:"final"` // last rung of this generation
	Candidates   int             `json:"candidates"`
	Unranked     int             `json:"unranked"` // failed / scalar-only candidates
	Keys         []string        `json:"objective_keys"`
	FrontSizes   []int           `json:"front_sizes"`
	ParetoFront  []FrontMember   `json:"pareto_front"`
	Survivors    []string        `json:"survivors"` // next rung, or the elites when final
	Fronts       [][]FrontMember `json:"fronts,omitempty"`
	NextPopSize  int             `json:"next_population,omitempty"`
	VaryJobID    string          `json:"vary_job_id,omitempty"`
	// Latest trial-registry statistics of this generation (spec 41 §6.5), when a worker has a registry.
	Validation *ValidationStats `json:"validation,omitempty"`
	// Per-loop GPU budget accounting (spec 34 §6.4).
	Budget *Budget `json:"budget,omitempty"`
	// Cross-asset pair genomes (latent.crossprior) seeded into this loop's population.
	PairsSeeded int `json:"pairs_seeded,omitempty"`
	// Genomes of this rung that were sent a warm-start bundle (neural decoders).
	WarmStarted int `json:"warm_started,omitempty"`
	// Garage key of the promoted population file (`fa-artifacts/strategies/<generation>.toml`),
	// set on the final summary of a strategy generation (spec 35 §6.4).
	ArtifactKey string `json:"artifact_key,omitempty"`
	// running | paused | stopped | done
	State     string    `json:"state,omitempty"`
	At        time.Time `json:"at"`
	ElapsedMs int64     `json:"elapsed_ms"`
	Note      string    `json:"note,omitempty"`
}

// Promote writes the final Pareto front of a generation somewhere durable (Garage) and returns
// the artifact key; see PromoteStrategies in main.go.
type Promote func(g *Generation, front []FrontMember) (string, error)

// GenerationConfig holds the knobs of ctl.evo.start.
type GenerationConfig struct {
	Kind            string
	SnapshotID      string
	Genomes         []string
	Eta             int
	MinBudgetMs     uint64
	MaxBudgetMs     uint64
	Seed            uint64
	MaxGenerations  int    // loops of halving + variation; 1 = single halving (default)
	TotalBudgetMs   uint64 // >0: worker sizes the next population by spec 34 §6.4 (plan_halving)
	DataWindowHours int    // >0: evaluation window forwarded in Job.spec
	// OnSummary receives every rung summary (may be nil).
	OnSummary func(Summary)
	// Promote is called with the final front when the generation ends (may be nil).
	Promote Promote
	// PairProposals returns up to n MCMC cross-asset pairs (latent.crossprior) to seed as
	// `nn_forecast` src→target genomes at the start of every loop (decoder kinds only; may be nil).
	PairProposals func(n int) []PairProposal
	// MaxPairs caps the pair genomes added per loop (spec 34 §7 rnn_pairs_from_mcmc).
	MaxPairs int
}

// Generation runs successive halving over a population and, when MaxGenerations > 1,
// loops: elites + `evo.vary.<kind>` children form the next population.
type Generation struct {
	mu         sync.Mutex
	ID         string
	Kind       string // e.g. "evo.eval.decoder"
	SnapshotID string
	Seed       uint64
	Priority   uint32
	cfg        GenerationConfig
	eta        int
	popSize    int
	genIndex   int
	rungs      []Rung
	rung       int
	pending    map[string]*fav1.Job // job id -> job (eval or vary)
	scores     []Candidate          // current rung
	best       map[string]Candidate // genome -> latest result in this loop
	alive      []string
	elites     []Candidate
	final      []Candidate
	varyJobID  string
	started    time.Time
	loopStart  time.Time
	done       bool
	paused     bool
	// a rung whose submission was deferred while paused (resume submits it)
	deferred   bool
	stopped    bool
	validation *ValidationStats
	// weights: genome → Garage ref of its latest trained bundle (neural decoders); warm: genome →
	// the bundle its next evaluation warm-starts from (own previous rung, or its parent's).
	weights     map[string]string
	warm        map[string]string
	budget      Budget
	pairsSeeded int
	rungWarm    int // eval jobs of the current rung sent with a warm-start bundle
	Summaries   []Summary
}

// NewGeneration plans rungs for the given genomes.
func NewGeneration(kind, snapshotID string, genomes []string, eta int, minBudget, maxBudget uint64, seed uint64) (*Generation, error) {
	return NewGenerationWith(GenerationConfig{
		Kind: kind, SnapshotID: snapshotID, Genomes: genomes, Eta: eta, MinBudgetMs: minBudget, MaxBudgetMs: maxBudget, Seed: seed,
	})
}

// NewGenerationWith plans the first halving from a full config.
func NewGenerationWith(c GenerationConfig) (*Generation, error) {
	pairs := 0
	if c.PairProposals != nil && c.MaxPairs > 0 && strings.HasSuffix(c.Kind, ".decoder") {
		extra := PairGenomes(c.PairProposals(c.MaxPairs), c.Genomes)
		c.Genomes = append(append([]string(nil), c.Genomes...), extra...)
		pairs = len(extra)
	}
	rungs := Plan(len(c.Genomes), c.Eta, c.MinBudgetMs, c.MaxBudgetMs)
	if len(rungs) == 0 {
		return nil, fmt.Errorf("invalid: empty plan for %d genomes", len(c.Genomes))
	}
	if c.MaxGenerations < 1 {
		c.MaxGenerations = 1
	}
	now := time.Now()
	return &Generation{
		ID: ulid.Make().String(), Kind: c.Kind, SnapshotID: c.SnapshotID, Seed: c.Seed,
		Priority: uint32(EvolutionEval), cfg: c, eta: max(c.Eta, 2), popSize: len(c.Genomes), rungs: rungs,
		pending: map[string]*fav1.Job{}, best: map[string]Candidate{},
		alive: append([]string(nil), c.Genomes...), started: now, loopStart: now,
		weights: map[string]string{}, warm: map[string]string{}, pairsSeeded: pairs,
		budget: Budget{TotalMs: c.TotalBudgetMs, PlannedMs: TotalBudget(rungs)},
	}, nil
}

// Start submits the first rung.
func (g *Generation) Start(pub PublishJob) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.submitRungLocked(pub)
}

// evalSpec is the eval job envelope: the genome, the loop index and rung (the worker records
// them in the trial registry) and the data window when configured. Workers treat a JSON object
// without a `family` key as this envelope (D-049); strategy genomes are TOML inline tables and
// so never parse as JSON, which keeps the two apart.
func (g *Generation) evalSpec(genome string) []byte {
	spec := map[string]any{"genome": genome, "generation": g.genIndex, "rung": g.rung}
	if g.cfg.DataWindowHours > 0 {
		spec["data_window"] = map[string]any{"hours": g.cfg.DataWindowHours}
	}
	if strings.HasSuffix(g.Kind, ".decoder") && isNeural(genome) {
		// spec 34 §6.2 step 3: children inherit their parent's weights, a survivor continues from
		// its previous rung's bundle (rung checkpoint)
		if ref := g.warm[genome]; ref != "" {
			spec["warm_start"] = ref
		}
		// step 4: rung r of R trains η^-(R−1−r) of the genome's SGD steps (full fidelity last)
		if n := len(g.rungs); n > 1 {
			spec["train_steps_fraction"] = math.Pow(float64(g.eta), -float64(n-1-g.rung))
		}
	}
	b, _ := json.Marshal(spec)
	return b
}

// overBudgetLocked reports whether running the next rung (survivors × its per-candidate budget)
// would take this loop beyond its total budget by more than 10 % (spec 34 §9 criterion 3).
func (g *Generation) overBudgetLocked(survivors int) bool {
	if g.cfg.TotalBudgetMs == 0 || g.rung+1 >= len(g.rungs) {
		return false
	}
	next := uint64(survivors) * g.rungs[g.rung+1].BudgetMs
	return float64(g.budget.LoopUsedMs+next) > 1.1*float64(g.cfg.TotalBudgetMs)
}

func (g *Generation) submitRungLocked(pub PublishJob) error {
	if g.paused {
		g.deferred = true
		return nil
	}
	g.deferred = false
	r := g.rungs[g.rung]
	g.scores = nil
	g.rungWarm = 0
	for i, genome := range g.alive {
		if strings.HasSuffix(g.Kind, ".decoder") && isNeural(genome) && g.warm[genome] != "" {
			g.rungWarm++
		}
		job := &fav1.Job{
			JobId: ulid.Make().String(), Kind: g.Kind, ParentJobId: g.ID, Priority: g.Priority,
			SnapshotId: g.SnapshotID, GenomeId: genome, BudgetGpuMs: r.BudgetMs, Spec: g.evalSpec(genome),
			Seed:     g.Seed + uint64(g.genIndex)*1_000_000_007 + uint64(g.rung)*1_000_003 + uint64(i),
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

// GenomeOf returns the genome of a pending job ("" when unknown).
func (g *Generation) GenomeOf(jobID string) string {
	g.mu.Lock()
	defer g.mu.Unlock()
	if j, ok := g.pending[jobID]; ok {
		return j.GenomeId
	}
	return ""
}

// Index returns (loop index, max generations, rung, rungs).
func (g *Generation) Index() (gen, maxGens, rung, rungs int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.genIndex, g.cfg.MaxGenerations, g.rung, len(g.rungs)
}

// Pause stops dispatching new rungs (and the next population); running jobs finish and are
// recorded. Returns false when the generation is already paused or done.
func (g *Generation) Pause() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.done || g.paused {
		return false
	}
	g.paused = true
	s := g.stateSummaryLocked("paused by operator")
	g.emit(s)
	return true
}

// Resume continues a paused generation, submitting the rung that was deferred meanwhile.
func (g *Generation) Resume(pub PublishJob) (bool, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.done || !g.paused {
		return false, nil
	}
	g.paused = false
	s := g.stateSummaryLocked("resumed by operator")
	g.emit(s)
	if g.deferred {
		return true, g.submitRungLocked(pub)
	}
	return true, nil
}

// Stop cancels every pending job (eval or vary) and ends the generation. Returns the cancelled
// job ids with their kinds so the caller can publish STATUS_CANCELLED results and release leases.
func (g *Generation) Stop() (cancelled map[string]string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	cancelled = map[string]string{}
	if g.done {
		return cancelled
	}
	for id, j := range g.pending {
		cancelled[id] = j.Kind
		delete(g.pending, id)
	}
	g.stopped = true
	g.done = true
	g.varyJobID = ""
	// elites = best of what was evaluated so far (so Final() has something for the operator)
	all := make([]Candidate, 0, len(g.best))
	for _, c := range g.best {
		all = append(all, c)
	}
	g.final = g.final[:0]
	for _, c := range Rank(all) {
		if c.hasObjectives() || c.Score != math.Inf(-1) {
			g.final = append(g.final, c)
		}
	}
	s := g.summaryLocked(true)
	s.Note = fmt.Sprintf("stopped by operator; %d pending job(s) cancelled", len(cancelled))
	s.State = "stopped"
	for _, c := range g.final {
		s.Survivors = append(s.Survivors, c.ID)
	}
	g.emit(s)
	return cancelled
}

// Paused reports the pause flag.
func (g *Generation) Paused() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.paused
}

// stateSummaryLocked is a summary of the current rung with a note (pause/resume), no ranking change.
func (g *Generation) stateSummaryLocked(note string) Summary {
	s := g.summaryLocked(false)
	s.Note = note
	return s
}

func (g *Generation) stateLocked() string {
	switch {
	case g.stopped:
		return "stopped"
	case g.done:
		return "done"
	case g.paused:
		return "paused"
	}
	return "running"
}

// OnResult records a result; preempted jobs are re-queued from their checkpoint.
// When a rung completes, survivors advance; when the last rung completes the loop either
// ends (MaxGenerations reached) or asks a worker for the next population. Returns true
// when the generation is done.
func (g *Generation) OnResult(r *fav1.JobResult, pub PublishJob) (bool, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	job, ok := g.pending[r.JobId]
	if !ok {
		return g.done, nil
	}
	g.budget.LoopUsedMs += r.UsedGpuMs
	g.budget.UsedMs += r.UsedGpuMs
	if r.Status == fav1.JobResult_STATUS_PREEMPTED {
		job.CheckpointRef = r.CheckpointRef
		return false, pub(bus.Jobs(job.Kind), job.JobId+"-"+r.CheckpointRef, job)
	}
	delete(g.pending, r.JobId)
	if r.JobId == g.varyJobID {
		return g.onVaryLocked(r, pub)
	}
	c := CandidateOf(job.GenomeId, r)
	if ref := weightsRefOf(r); ref != "" {
		g.weights[c.ID] = ref
		g.warm[c.ID] = ref // the next rung continues from this bundle
	}
	g.scores = append(g.scores, c)
	g.best[c.ID] = c
	if v := validationOf(r); v != nil {
		g.validation = v
	}
	if len(g.pending) > 0 {
		return false, nil
	}
	final := g.rung == len(g.rungs)-1
	truncated := false
	if !final && g.overBudgetLocked((len(g.scores)+g.eta-1)/g.eta) {
		// the next rung would overrun the loop budget: end the loop here with partial results
		final, truncated = true, true
		g.budget.Truncated = true
		g.budget.Truncations++
	}
	s := g.summaryLocked(final)
	if truncated {
		s.Note = fmt.Sprintf("budget: %d of %d ms used, rungs %d..%d truncated", g.budget.LoopUsedMs, g.cfg.TotalBudgetMs, g.rung+1, len(g.rungs)-1)
	}
	if !final {
		survivors := Halve(g.scores, g.eta)
		g.alive = g.alive[:0]
		for _, sv := range survivors {
			g.alive = append(g.alive, sv.ID)
		}
		s.Survivors = append([]string(nil), g.alive...)
		g.emit(s)
		g.rung++
		return false, g.submitRungLocked(pub)
	}
	// Loop end: elites over every candidate evaluated in this loop (latest objectives).
	all := make([]Candidate, 0, len(g.best))
	for _, c := range g.best {
		all = append(all, c)
	}
	keep := max(1, (g.popSize+g.eta-1)/g.eta)
	// NSGA-II (≤ 3 objectives) / NSGA-III (≥ 4) survivor selection, best first; a failed
	// genome is never carried over (spec 34 §6.2 step 6)
	g.elites = append(g.elites[:0], SelectElites(all, keep)...)
	g.final = append([]Candidate(nil), g.elites...)
	for _, e := range g.elites {
		s.Survivors = append(s.Survivors, e.ID)
	}
	if g.genIndex+1 >= g.cfg.MaxGenerations || len(g.elites) == 0 {
		if len(g.elites) == 0 {
			s.Note = "no surviving genome; loop stopped"
		}
		g.done = true
		s.State = "done"
		// Promotion hand-off (spec 35 §6.4): the final front becomes a population artifact.
		if g.cfg.Promote != nil && len(s.ParetoFront) > 0 {
			if key, err := g.cfg.Promote(g, s.ParetoFront); err != nil {
				s.Note = strings.TrimSpace(s.Note + " promote: " + err.Error())
			} else {
				s.ArtifactKey = key
			}
		}
		g.emit(s)
		return true, nil
	}
	// Ask a worker for the next population (spec 34 §6.2 step 2, §6.5).
	parents := make([]map[string]any, len(g.elites))
	for i, e := range g.elites {
		parents[i] = map[string]any{"genome": e.ID, "objectives": e.Objectives, "directions": e.Directions, "score": e.Score}
	}
	spec := map[string]any{
		"protocol":   "evo.vary.decoder/v0",
		"parents":    parents,
		"n_children": max(0, g.popSize-g.pairsSeeded-len(g.elites)), // pairs are re-proposed, not bred into the size
		"seed":       g.Seed + uint64(g.genIndex+1)*7_919,
		"generation": g.genIndex + 1,
	}
	if g.cfg.TotalBudgetMs > 0 {
		// Worker sizes the population with fa_evolution::plan_halving (spec 34 §6.4).
		spec["n_children"] = 0
		spec["budget"] = map[string]any{
			"total_gpu_ms": g.cfg.TotalBudgetMs, "min_budget_gpu_ms": g.cfg.MinBudgetMs, "eta": g.eta,
			"rungs": len(Plan(g.popSize, g.eta, g.cfg.MinBudgetMs, g.cfg.MaxBudgetMs)),
		}
	}
	b, _ := json.Marshal(spec)
	kind := VaryKind(g.Kind)
	vj := &fav1.Job{
		JobId: ulid.Make().String(), Kind: kind, ParentJobId: g.ID, Priority: g.Priority, SnapshotId: g.SnapshotID,
		Spec: b, BudgetGpuMs: g.cfg.MinBudgetMs, Seed: g.Seed + uint64(g.genIndex+1)*7_919,
		Deadline: timestamppb.New(time.Now().Add(2 * time.Minute)),
	}
	g.varyJobID = vj.JobId
	g.pending[vj.JobId] = vj
	s.VaryJobID = vj.JobId
	g.emit(s)
	return false, pub(bus.Jobs(kind), vj.JobId, vj)
}

func (g *Generation) onVaryLocked(r *fav1.JobResult, pub PublishJob) (bool, error) {
	g.varyJobID = ""
	var v struct {
		Children []string `json:"children"`
	}
	if r.Status != fav1.JobResult_STATUS_OK {
		g.done = true
		return true, fmt.Errorf("variation job failed (%s): %s; generation %s stops after loop %d", r.Status, r.Error, g.ID, g.genIndex)
	}
	if err := json.Unmarshal(r.Result, &v); err != nil || len(v.Children) == 0 {
		g.done = true
		return true, fmt.Errorf("variation result unusable (%v, %d children); generation %s stops after loop %d", err, len(v.Children), g.ID, g.genIndex)
	}
	g.alive = g.alive[:0]
	warm := map[string]string{}
	weights := map[string]string{}
	for _, e := range g.elites {
		g.alive = append(g.alive, e.ID)
		if ref := g.weights[e.ID]; ref != "" {
			warm[e.ID], weights[e.ID] = ref, ref // elites are re-fine-tuned from their own bundle (§6.6)
		}
	}
	// children warm-start from the first parent that has a bundle (§6.2 step 3)
	lineage := lineageOf(r)
	for _, c := range v.Children {
		for _, p := range lineage[c] {
			if ref := g.weights[p]; ref != "" {
				warm[c] = ref
				break
			}
		}
	}
	g.alive = append(g.alive, v.Children...)
	// §6.2 step 1: MCMC-proposed cross-asset pairs join every loop's population
	g.pairsSeeded = 0
	if g.cfg.PairProposals != nil && g.cfg.MaxPairs > 0 && strings.HasSuffix(g.Kind, ".decoder") {
		extra := PairGenomes(g.cfg.PairProposals(g.cfg.MaxPairs), g.alive)
		g.alive = append(g.alive, extra...)
		g.pairsSeeded = len(extra)
	}
	g.warm, g.weights = warm, weights
	g.popSize = len(g.alive)
	g.genIndex++
	g.rung = 0
	g.best = map[string]Candidate{}
	g.loopStart = time.Now()
	g.rungs = Plan(len(g.alive), g.eta, g.cfg.MinBudgetMs, g.cfg.MaxBudgetMs)
	g.budget.LoopUsedMs, g.budget.Truncated, g.budget.PlannedMs = 0, false, TotalBudget(g.rungs)
	return false, g.submitRungLocked(pub)
}

func (g *Generation) summaryLocked(final bool) Summary {
	cands := g.scores
	if final {
		// the loop's front is over every genome evaluated in it, at its latest rung
		cands = make([]Candidate, 0, len(g.best))
		for _, c := range g.best {
			cands = append(cands, c)
		}
	}
	keys, fronts, unranked := Pareto(cands)
	s := Summary{
		Type: "generation.summary", GenerationID: g.ID, Kind: g.Kind, SnapshotID: g.SnapshotID, Generation: g.genIndex, MaxGens: g.cfg.MaxGenerations,
		Rung: g.rung, Rungs: len(g.rungs), Final: final, Candidates: len(cands), Unranked: unranked, Keys: keys,
		Fronts: fronts, Validation: g.validation, State: g.stateLocked(), At: time.Now(), ElapsedMs: time.Since(g.loopStart).Milliseconds(),
	}
	if len(fronts) > 0 {
		s.ParetoFront = fronts[0]
	}
	b := g.budget
	s.Budget = &b
	if g.rung == 0 {
		s.PairsSeeded = g.pairsSeeded
	}
	s.WarmStarted = g.rungWarm
	for _, f := range fronts {
		s.FrontSizes = append(s.FrontSizes, len(f))
	}
	if final && g.genIndex+1 < g.cfg.MaxGenerations {
		s.NextPopSize = g.popSize
	}
	return s
}

func (g *Generation) emit(s Summary) {
	g.Summaries = append(g.Summaries, s)
	if g.cfg.OnSummary != nil {
		g.cfg.OnSummary(s)
	}
}

// Final returns the elites (best first) once done.
func (g *Generation) Final() []Candidate {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.final
}
