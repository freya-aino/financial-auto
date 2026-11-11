package main

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Decoder-population bookkeeping jobs (spec 34 §5, §8, D-085), both served by evo-worker:
//
//   - evo.front.decoder: at every loop end of a decoder generation the final summary's ranked
//     candidates become a Helix ParetoFront (fire and forget; no generation waits for it).
//   - evo.reeval.decoder: the elite decoder cards are re-evaluated frozen on data read through a
//     newer encoder snapshot; their cards gain transfer records that decay feedback weights.
//     Sent by ctl.evo.reeval, and automatically FA_EVO_REEVAL_DELAY_MIN minutes (default 120,
//     0 disables) after latent.lifecycle.<ts> shows a new snapshot for a timescale, with a data
//     window of that length (only the new snapshot's latents).

const (
	FrontKind  = "evo.front.decoder"
	ReevalKind = "evo.reeval.decoder"
)

// frontEntry is one ranked candidate of a FrontJob spec.
type frontEntry struct {
	Genome   string  `json:"genome"`
	Rank     int     `json:"rank"`
	Crowding float64 `json:"crowding"`
	HV       float64 `json:"hv"`
	Score    float64 `json:"score"`
}

// FrontJob is the evo.front.decoder job of a final (loop-end) decoder summary; nil otherwise.
func FrontJob(s Summary) *fav1.Job {
	if !s.Final || !strings.HasSuffix(s.Kind, ".decoder") || len(s.Fronts) == 0 {
		return nil
	}
	fronts := make([][]frontEntry, len(s.Fronts))
	for i, f := range s.Fronts {
		for _, m := range f {
			fronts[i] = append(fronts[i], frontEntry{Genome: m.Genome, Rank: m.Rank, Crowding: m.Crowding, HV: m.HV, Score: m.Score})
		}
	}
	spec, _ := json.Marshal(map[string]any{
		"protocol": "evo.front.decoder/v0", "generation_id": s.GenerationID, "generation": s.Generation, "kind": s.Kind,
		"snapshot_id": s.SnapshotID, "keys": s.Keys, "fronts": fronts, "final": s.State == "done" || s.State == "stopped",
	})
	return &fav1.Job{
		JobId: ulid.Make().String(), Kind: FrontKind, ParentJobId: s.GenerationID, SnapshotId: s.SnapshotID, Spec: spec,
		BudgetGpuMs: 60_000, Deadline: timestamppb.New(time.Now().Add(10 * time.Minute)),
	}
}

// ReevalJob is one evo.reeval.decoder job (zero values are left to the worker's defaults).
func ReevalJob(snapshotID, timescale string, k, hours int, dryRun bool) *fav1.Job {
	spec := map[string]any{"protocol": "evo.reeval.decoder/v0", "snapshot_id": snapshotID, "dry_run": dryRun}
	if timescale != "" {
		spec["timescale"] = timescale
	}
	if k > 0 {
		spec["k"] = k
	}
	if hours > 0 {
		spec["data_window"] = map[string]any{"hours": hours}
	}
	b, _ := json.Marshal(spec)
	return &fav1.Job{
		JobId: ulid.Make().String(), Kind: ReevalKind, SnapshotId: snapshotID, Spec: b, BudgetGpuMs: 600_000,
		Deadline: timestamppb.New(time.Now().Add(30 * time.Minute)),
	}
}

// ReevalArgs reads ctl.evo.reeval args {snapshot_id?, timescale?, k?, data_window_hours?, dry_run?}.
func ReevalArgs(args map[string]string) *fav1.Job {
	k, _ := strconv.Atoi(args["k"])
	h, _ := strconv.Atoi(args["data_window_hours"])
	dry := args["dry_run"] == "true" || args["dry_run"] == "1"
	return ReevalJob(args["snapshot_id"], args["timescale"], k, h, dry)
}

// SnapshotWatch remembers the snapshot each timescale serves (from latent.lifecycle.<ts>) and
// reports a swap: a snapshot id different from the one seen before for that timescale. The
// first observation of a timescale only records it.
type SnapshotWatch struct {
	mu   sync.Mutex
	seen map[string]string
}

func NewSnapshotWatch() *SnapshotWatch { return &SnapshotWatch{seen: map[string]string{}} }

// Observe records (timescale, snapshot) and returns true on a swap.
func (w *SnapshotWatch) Observe(timescale, snapshot string) bool {
	if timescale == "" || snapshot == "" {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	prev, ok := w.seen[timescale]
	w.seen[timescale] = snapshot
	return ok && prev != snapshot
}

// ReevalDelay is FA_EVO_REEVAL_DELAY_MIN (default 120 minutes; 0 disables the automatic
// re-evaluation after a snapshot swap).
func ReevalDelay() time.Duration {
	m := 120
	if v, err := strconv.Atoi(os.Getenv("FA_EVO_REEVAL_DELAY_MIN")); err == nil && v >= 0 {
		m = v
	}
	return time.Duration(m) * time.Minute
}
