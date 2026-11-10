// Command evo-orchestrator schedules decoder/strategy populations onto
// evo-workers (spec 34) and hosts the GPU arbiter (spec 60, D-006).
//
// Control subjects (request/reply, CtlRequest/CtlReply):
//
//	ctl.gpu.acquire  args: lease_id, class, mem_mb
//	ctl.gpu.release  args: lease_id
//	ctl.evo.start    args: kind, snapshot_id, genomes (newline list, D-044), eta, min_budget_ms,
//	                 max_budget_ms, max_generations (default -max-generations / 1),
//	                 total_budget_ms (0 = fixed population), data_window_hours (0 = worker default)
//	ctl.evo.cancel   args: job_id — drops the job from its generation (scored as failed), releases
//	                 the GPU lease named after the job id and publishes JobResult{CANCELLED, error
//	                 "cancelled by operator"} on jobs.result.<kind> so views update.
//
// Error replies carry a prefix control-api maps to HTTP: `not_found:` (404), `conflict:` (409).
//
// Ranking: NSGA-II (non-dominated sort + crowding distance) over JobResult.objectives /
// directions (proto maps, else the result JSON), falling back to the scalar `score`. After
// every rung a `generation.summary` JSON (Pareto front, front sizes, survivors) is published as
// JobResult{job_id: <generation id>, result: <json>} on jobs.result.evo.generation.
//
// Heartbeat counters: gpu.used_mb, gpu.active, gpu.queued, gpu.capacity_mb (from -gpu-mem-mb /
// FA_GPU_MEM_MB, default 24000; D-031), generations.started|done|active, generations.loops,
// jobs.cancelled, vary.jobs, vary.failed, pareto.front_size, pareto.fronts, pareto.candidates,
// pareto.unranked, pareto.generation, pareto.rung.
//
// Notifications (CtlRequest, no reply): ctl.gpu.preempt / ctl.gpu.granted with arg lease_id.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/proto"

	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

const service = "evo-orchestrator"

// GenerationSummaryKind is the `jobs.result.<kind>` suffix carrying generation.summary JSON.
const GenerationSummaryKind = "evo.generation"

func main() {
	natsURL := flag.String("nats", envOr("FA_NATS_URL", "nats://127.0.0.1:4222"), "NATS URL")
	gpuMem := flag.Int("gpu-mem-mb", envInt("FA_GPU_MEM_MB", 24_000), "GPU memory managed by the arbiter (MB), reported as gpu.capacity_mb")
	maxGens := flag.Int("max-generations", envInt("FA_EVO_MAX_GENERATIONS", 1), "default max_generations for ctl.evo.start (halving + variation loops)")
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil)).With("svc", service)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pub, err := bus.Connect(*natsURL, service)
	if err != nil {
		log.Error("nats connect", "err", err)
		os.Exit(1)
	}
	defer pub.Close()

	arb := NewArbiter(*gpuMem)
	counters := bus.NewCounters()
	counters.Set("gpu.capacity_mb", float64(*gpuMem))
	go bus.RunHeartbeat(ctx, pub, service, ulid.Make().String(), "0.1.0", 5*time.Second, counters, nil)
	go func() { // keep arbiter gauges fresh for heartbeats
		t := time.NewTicker(time.Second)
		defer t.Stop()
		for {
			used, active, queued := arb.Stats()
			counters.Set("gpu.used_mb", float64(used))
			counters.Set("gpu.active", float64(active))
			counters.Set("gpu.queued", float64(queued))
			select {
			case <-ctx.Done():
				return
			case <-t.C:
			}
		}
	}()

	notify := func(verb, leaseID string) {
		_ = pub.Publish(bus.Ctl("gpu", verb), &fav1.CtlRequest{Verb: verb, Args: map[string]string{"lease_id": leaseID}})
	}
	publishJob := func(subject, msgID string, job *fav1.Job) error { return pub.PublishWithID(subject, msgID, job) }

	var gensMu sync.Mutex
	gens := map[string]*Generation{}

	// Per-rung Pareto summaries: heartbeat gauges + generation.summary on jobs.result.evo.generation.
	onSummary := func(s Summary) {
		counters.Set("pareto.front_size", float64(len(s.ParetoFront)))
		counters.Set("pareto.fronts", float64(len(s.FrontSizes)))
		counters.Set("pareto.candidates", float64(s.Candidates))
		counters.Set("pareto.unranked", float64(s.Unranked))
		counters.Set("pareto.generation", float64(s.Generation))
		counters.Set("pareto.rung", float64(s.Rung))
		if s.VaryJobID != "" {
			counters.Add("vary.jobs", 1)
			counters.Add("generations.loops", 1)
		}
		b, err := json.Marshal(s)
		if err != nil {
			return
		}
		res := &fav1.JobResult{JobId: s.GenerationID, Status: fav1.JobResult_STATUS_OK, Result: b, WorkerId: service}
		if err := pub.Publish(bus.JobResult(GenerationSummaryKind), res); err != nil {
			log.Warn("summary publish", "err", err)
		}
		log.Info("generation.summary", "id", s.GenerationID, "gen", s.Generation, "rung", s.Rung, "final", s.Final,
			"candidates", s.Candidates, "front", len(s.ParetoFront), "fronts", s.FrontSizes, "survivors", s.Survivors)
	}
	finish := func(id string, g *Generation) {
		log.Info("generation done", "id", id, "final", g.Final())
		counters.Add("generations.done", 1)
		delete(gens, id)
		counters.Set("generations.active", float64(len(gens)))
	}

	handle := func(subject string, fn func(*fav1.CtlRequest) (string, error)) {
		_, err := pub.NC.Subscribe(subject, func(m *nats.Msg) {
			var req fav1.CtlRequest
			reply := &fav1.CtlReply{}
			if err := proto.Unmarshal(m.Data, &req); err != nil {
				reply.Message = "decode: " + err.Error()
			} else if msg, err := fn(&req); err != nil {
				reply.Message = err.Error()
			} else {
				reply.Ok, reply.Message = true, msg
			}
			if b, err := proto.Marshal(reply); err == nil && m.Reply != "" {
				_ = m.Respond(b)
			}
		})
		if err != nil {
			log.Error("subscribe", "subject", subject, "err", err)
			os.Exit(1)
		}
	}

	handle(bus.Ctl("gpu", "acquire"), func(r *fav1.CtlRequest) (string, error) {
		class, err := ParseClass(r.Args["class"])
		if err != nil {
			return "", err
		}
		mem, err := strconv.Atoi(r.Args["mem_mb"])
		if err != nil {
			return "", fmt.Errorf("mem_mb: %w", err)
		}
		d, err := arb.Request(r.Args["lease_id"], class, mem)
		if err != nil {
			return "", err
		}
		for _, id := range d.Preempt {
			notify("preempt", id)
		}
		if d.Granted {
			return "granted", nil
		}
		return "queued", nil
	})
	handle(bus.Ctl("gpu", "release"), func(r *fav1.CtlRequest) (string, error) {
		for _, id := range arb.Release(r.Args["lease_id"]) {
			notify("granted", id)
		}
		return "released", nil
	})
	handle(bus.Ctl("evo", "start"), func(r *fav1.CtlRequest) (string, error) {
		// Genomes are newline-separated (D-044): kv genomes contain ';' and JSON genomes contain ','.
		var genomes []string
		for _, g := range strings.Split(r.Args["genomes"], "\n") {
			if g = strings.TrimSpace(g); g != "" {
				genomes = append(genomes, g)
			}
		}
		eta, _ := strconv.Atoi(r.Args["eta"])
		minB, _ := strconv.ParseUint(r.Args["min_budget_ms"], 10, 64)
		maxB, _ := strconv.ParseUint(r.Args["max_budget_ms"], 10, 64)
		if minB == 0 {
			minB = 60_000
		}
		if maxB == 0 {
			maxB = 16 * minB
		}
		mg := *maxGens
		if v, err := strconv.Atoi(r.Args["max_generations"]); err == nil && v > 0 {
			mg = v
		}
		totalB, _ := strconv.ParseUint(r.Args["total_budget_ms"], 10, 64)
		window, _ := strconv.Atoi(r.Args["data_window_hours"])
		g, err := NewGenerationWith(GenerationConfig{
			Kind: r.Args["kind"], SnapshotID: r.Args["snapshot_id"], Genomes: genomes, Eta: eta, MinBudgetMs: minB, MaxBudgetMs: maxB,
			Seed: uint64(time.Now().UnixNano()), MaxGenerations: mg, TotalBudgetMs: totalB, DataWindowHours: window, OnSummary: onSummary,
		})
		if err != nil {
			return "", err
		}
		gensMu.Lock()
		gens[g.ID] = g
		counters.Set("generations.active", float64(len(gens)))
		gensMu.Unlock()
		counters.Add("generations.started", 1)
		log.Info("generation start", "id", g.ID, "kind", g.Kind, "genomes", len(genomes), "eta", g.eta, "max_generations", mg, "min_budget_ms", minB)
		return g.ID, g.Start(publishJob)
	})

	handle(bus.Ctl("evo", "cancel"), func(r *fav1.CtlRequest) (string, error) {
		jobID := r.Args["job_id"]
		if jobID == "" {
			return "", fmt.Errorf("invalid: job_id is required")
		}
		gensMu.Lock()
		defer gensMu.Unlock()
		for _, g := range gens {
			if !g.Owns(jobID) {
				continue
			}
			res := &fav1.JobResult{
				JobId: jobID, Status: fav1.JobResult_STATUS_CANCELLED, Error: "cancelled by operator", WorkerId: service,
			}
			kind := g.Kind
			if g.GenomeOf(jobID) == "" {
				kind = VaryKind(g.Kind) // the pending variation job
			}
			done, err := g.OnResult(res, publishJob)
			if err != nil {
				log.Warn("generation after cancel", "id", g.ID, "err", err)
				if kind != g.Kind {
					counters.Add("vary.failed", 1)
				}
			}
			if done {
				finish(g.ID, g)
			}
			// Lease ids follow job ids (convention until workers report their lease ids).
			for _, id := range arb.Release(jobID) {
				notify("granted", id)
			}
			if err := pub.Publish(bus.JobResult(kind), res); err != nil {
				return "", err
			}
			counters.Add("jobs.cancelled", 1)
			return "cancelled " + jobID, nil
		}
		return "", fmt.Errorf("not_found: job %q is not pending in any generation", jobID)
	})

	if _, err := pub.NC.Subscribe(bus.AllJobResults, func(m *nats.Msg) {
		var res fav1.JobResult
		if err := proto.Unmarshal(m.Data, &res); err != nil {
			counters.Add("errors.decode", 1)
			return
		}
		gensMu.Lock()
		defer gensMu.Unlock()
		for id, g := range gens {
			if !g.Owns(res.JobId) {
				continue
			}
			isVary := g.GenomeOf(res.JobId) == ""
			done, err := g.OnResult(&res, publishJob)
			if err != nil {
				log.Warn("generation", "id", id, "err", err)
				if isVary {
					counters.Add("vary.failed", 1)
				}
			}
			if done {
				finish(id, g)
			}
			return
		}
	}); err != nil {
		log.Error("subscribe results", "err", err)
		os.Exit(1)
	}

	log.Info("started", "gpu_mem_mb", *gpuMem, "max_generations", *maxGens)
	<-ctx.Done()
	log.Info("stopped")
}

func envInt(k string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return def
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
