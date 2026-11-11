// Command evo-orchestrator schedules decoder/strategy populations onto
// evo-workers (spec 34) and hosts the GPU arbiter (spec 60, D-006).
//
// Control subjects (request/reply, CtlRequest/CtlReply):
//
//	ctl.gpu.acquire  args: lease_id, class, mem_mb
//	ctl.gpu.release  args: lease_id
//	ctl.evo.start    args: kind, snapshot_id, genomes (newline list, D-044), eta, min_budget_ms,
//	                 max_budget_ms, max_generations (default -max-generations / 1),
//	                 total_budget_ms (0 = fixed population; also the per-loop budget whose overrun
//	                 truncates rungs), data_window_hours (0 = worker default), mcmc_pairs (cross-asset
//	                 nn_forecast src→target genomes seeded per loop from latent.crossprior.*; default
//	                 -mcmc-pairs, decoder kinds only)
//	ctl.evo.cancel   args: job_id — drops the job from its generation (scored as failed), releases
//	                 the GPU lease named after the job id and publishes JobResult{CANCELLED, error
//	                 "cancelled by operator"} on jobs.result.<kind> so views update.
//	ctl.evo.pause    args: generation_id — stop dispatching new rungs (running jobs finish)
//	ctl.evo.resume   args: generation_id — continue; a rung deferred while paused is submitted
//	ctl.evo.stop     args: generation_id — cancel every pending job (STATUS_CANCELLED results,
//	                 leases released) and end the generation with a final summary
//
// The eval kind names the family (`evo.eval.<family>`); the variation kind is derived
// (`evo.vary.<family>`), so decoder and strategy generations run through the same loop. Eval
// results may carry `generation_stats` (trial registry, spec 41 §6.5); the latest one is
// published in every summary as `validation`. When a strategy (allocator) generation ends, its
// Pareto front is written to Garage as `fa-artifacts/strategies/<generation>.toml`
// (`fa-artifacts/allocators/<generation>.toml`; `-garage`, secrets from `-secrets`/FA_SECRETS)
// and announced as `artifact_key`.
//
// Error replies carry a prefix control-api maps to HTTP: `not_found:` (404), `conflict:` (409).
//
// Neural decoders (spec 34 §6.2): eval envelopes carry `warm_start` ("<bucket>/<key>" of the
// genome's previous-rung bundle, or of its parent's for a child) and `train_steps_fraction`
// (η^-(R−1−r): rung r of R trains that share of the genome's SGD steps).
//
// Ranking: NSGA-II (non-dominated sort + crowding distance) over JobResult.objectives /
// directions (proto maps, else the result JSON), falling back to the scalar `score`. After
// every rung a `generation.summary` JSON (Pareto front, front sizes, survivors) is published as
// JobResult{job_id: <generation id>, result: <json>} on jobs.result.evo.generation.
//
// Heartbeat counters: gpu.used_mb, gpu.active, gpu.queued, gpu.capacity_mb (from -gpu-mem-mb /
// FA_GPU_MEM_MB, default 24000; D-031), generations.started|done|active, generations.loops,
// jobs.cancelled, vary.jobs, vary.failed, pareto.front_size, pareto.fronts, pareto.candidates,
// pareto.unranked, pareto.generation, pareto.rung, budget.used_ms, budget.loop_used_ms,
// budget.truncations, warm_start.jobs, crossprior.messages, crossprior.proposals, pairs.seeded.
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

	"github.com/fey/financial-auto/go/pkg/archive"
	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const service = "evo-orchestrator"

// GenerationSummaryKind is the `jobs.result.<kind>` suffix carrying generation.summary JSON.
const GenerationSummaryKind = "evo.generation"

func main() {
	natsURL := flag.String("nats", envOr("FA_NATS_URL", "nats://127.0.0.1:4222"), "NATS URL")
	gpuMem := flag.Int("gpu-mem-mb", envInt("FA_GPU_MEM_MB", 24_000), "GPU memory managed by the arbiter (MB), reported as gpu.capacity_mb")
	maxGens := flag.Int("max-generations", envInt("FA_EVO_MAX_GENERATIONS", 1), "default max_generations for ctl.evo.start (halving + variation loops)")
	garageURL := flag.String("garage", envOr("FA_GARAGE_URL", "http://127.0.0.1:3900"), "Garage S3 endpoint for promoted strategy populations (empty disables)")
	secretsPath := flag.String("secrets", envOr("FA_SECRETS", "config/secrets.toml"), "secrets file with [garage] access_key_id / secret_access_key (D-020)")
	mcmcPairs := flag.Int("mcmc-pairs", envInt("FA_EVO_MCMC_PAIRS", 4), "default cross-asset pair genomes per decoder loop from latent.crossprior.* (0 disables)")
	metricsBind := flag.String("metrics", bus.MetricsBindDefault(service), "Prometheus GET /metrics bind (FA_METRICS_BIND; off disables)")
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
	go bus.ServeMetrics(ctx, *metricsBind, service, "0.1.0", counters, log)
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

	// MCMC cross-asset pair proposals (D-069) for the RNN-prior seed (spec 34 §6.2 step 1).
	pairBook := NewPairBook()
	if _, err := pub.NC.Subscribe("latent.crossprior.>", func(m *nats.Msg) {
		var x fav1.CrossAssetPriors
		if err := proto.Unmarshal(m.Data, &x); err != nil {
			counters.Add("errors.decode", 1)
			return
		}
		pairBook.Update(&x)
		seen, held := pairBook.Stats()
		counters.Set("crossprior.messages", float64(seen))
		counters.Set("crossprior.proposals", float64(held))
	}); err != nil {
		log.Warn("subscribe latent.crossprior", "err", err)
	}

	var gensMu sync.Mutex
	gens := map[string]*Generation{}

	// Promotion hand-off (spec 35 §6.4): final strategy fronts → Garage population files.
	var promote Promote
	if *garageURL != "" {
		secrets, err := connector.LoadSecrets(*secretsPath)
		if err != nil {
			log.Warn("secrets", "path", *secretsPath, "err", err)
		} else if st, err := archive.NewS3(archive.Options{Endpoint: *garageURL, AccessKey: secrets.Get("garage", "access_key_id"), SecretKey: secrets.Get("garage", "secret_access_key")}); err != nil {
			log.Warn("garage client unavailable; strategy fronts are not archived", "err", err)
		} else {
			promote = PromoteFronts(st, func(err error) { log.Warn("promote", "err", err) })
			log.Info("promoted strategy / allocator fronts go to Garage", "endpoint", *garageURL, "bucket", ArtifactsBucket)
		}
	}

	// Per-rung Pareto summaries: heartbeat gauges + generation.summary on jobs.result.evo.generation.
	onSummary := func(s Summary) {
		counters.Set("pareto.front_size", float64(len(s.ParetoFront)))
		counters.Set("pareto.fronts", float64(len(s.FrontSizes)))
		counters.Set("pareto.candidates", float64(s.Candidates))
		counters.Set("pareto.unranked", float64(s.Unranked))
		counters.Set("pareto.generation", float64(s.Generation))
		counters.Set("pareto.rung", float64(s.Rung))
		if s.Budget != nil {
			counters.Set("budget.used_ms", float64(s.Budget.UsedMs))
			counters.Set("budget.loop_used_ms", float64(s.Budget.LoopUsedMs))
			counters.Set("budget.truncations", float64(s.Budget.Truncations))
		}
		counters.Add("warm_start.jobs", float64(s.WarmStarted))
		counters.Add("pairs.seeded", float64(s.PairsSeeded))
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
		if s.Validation != nil {
			counters.Set("validation.n_trials", float64(s.Validation.NTrials))
			if s.Validation.DeflatedSharpe != nil {
				counters.Set("validation.deflated_sharpe", *s.Validation.DeflatedSharpe)
			}
			if s.Validation.PBO != nil {
				counters.Set("validation.pbo", *s.Validation.PBO)
			}
		}
		if s.ArtifactKey != "" {
			counters.Add("promotions.artifacts", 1)
		}
		log.Info("generation.summary", "id", s.GenerationID, "gen", s.Generation, "rung", s.Rung, "final", s.Final, "state", s.State,
			"candidates", s.Candidates, "front", len(s.ParetoFront), "fronts", s.FrontSizes, "survivors", s.Survivors, "artifact", s.ArtifactKey)
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
		pairs := *mcmcPairs
		if v, err := strconv.Atoi(r.Args["mcmc_pairs"]); err == nil && v >= 0 {
			pairs = v
		}
		g, err := NewGenerationWith(GenerationConfig{
			Kind: r.Args["kind"], SnapshotID: r.Args["snapshot_id"], Genomes: genomes, Eta: eta, MinBudgetMs: minB, MaxBudgetMs: maxB,
			Seed: uint64(time.Now().UnixNano()), MaxGenerations: mg, TotalBudgetMs: totalB, DataWindowHours: window, OnSummary: onSummary,
			Promote: promote, PairProposals: pairBook.Top, MaxPairs: pairs,
		})
		if err != nil {
			return "", err
		}
		gensMu.Lock()
		gens[g.ID] = g
		counters.Set("generations.active", float64(len(gens)))
		gensMu.Unlock()
		counters.Add("generations.started", 1)
		log.Info("generation start", "id", g.ID, "kind", g.Kind, "genomes", len(genomes), "pairs", g.pairsSeeded, "eta", g.eta, "max_generations", mg, "min_budget_ms", minB)
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

	// Operator control of a running generation (spec 50 §4.6 `POST /api/evolution/runs/{id}/{action}`).
	byGeneration := func(r *fav1.CtlRequest) (*Generation, error) {
		id := r.Args["generation_id"]
		if id == "" {
			return nil, fmt.Errorf("invalid: generation_id is required")
		}
		g, ok := gens[id]
		if !ok {
			return nil, fmt.Errorf("not_found: generation %q is not active", id)
		}
		return g, nil
	}
	handle(bus.Ctl("evo", "pause"), func(r *fav1.CtlRequest) (string, error) {
		gensMu.Lock()
		defer gensMu.Unlock()
		g, err := byGeneration(r)
		if err != nil {
			return "", err
		}
		if !g.Pause() {
			return "", fmt.Errorf("conflict: generation %s is already paused or done", g.ID)
		}
		counters.Add("generations.paused", 1)
		log.Info("generation paused", "id", g.ID)
		return "paused " + g.ID, nil
	})
	handle(bus.Ctl("evo", "resume"), func(r *fav1.CtlRequest) (string, error) {
		gensMu.Lock()
		defer gensMu.Unlock()
		g, err := byGeneration(r)
		if err != nil {
			return "", err
		}
		ok, err := g.Resume(publishJob)
		if err != nil {
			return "", err
		}
		if !ok {
			return "", fmt.Errorf("conflict: generation %s is not paused", g.ID)
		}
		log.Info("generation resumed", "id", g.ID)
		return "resumed " + g.ID, nil
	})
	handle(bus.Ctl("evo", "stop"), func(r *fav1.CtlRequest) (string, error) {
		gensMu.Lock()
		defer gensMu.Unlock()
		g, err := byGeneration(r)
		if err != nil {
			return "", err
		}
		cancelled := g.Stop()
		for jobID, kind := range cancelled {
			res := &fav1.JobResult{JobId: jobID, Status: fav1.JobResult_STATUS_CANCELLED, Error: "cancelled by operator (generation stopped)", WorkerId: service}
			if err := pub.Publish(bus.JobResult(kind), res); err != nil {
				log.Warn("cancel publish", "job", jobID, "err", err)
			}
			for _, id := range arb.Release(jobID) {
				notify("granted", id)
			}
			counters.Add("jobs.cancelled", 1)
		}
		counters.Add("generations.stopped", 1)
		finish(g.ID, g)
		log.Info("generation stopped", "id", g.ID, "cancelled", len(cancelled))
		return fmt.Sprintf("stopped %s (%d job(s) cancelled)", g.ID, len(cancelled)), nil
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
