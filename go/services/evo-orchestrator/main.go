// Command evo-orchestrator schedules decoder/strategy populations onto
// evo-workers (spec 34) and hosts the GPU arbiter (spec 60, D-006).
//
// Control subjects (request/reply, CtlRequest/CtlReply):
//
//	ctl.gpu.acquire  args: lease_id, class, mem_mb
//	ctl.gpu.release  args: lease_id
//	ctl.evo.start    args: kind, snapshot_id, genomes (comma list), eta, min_budget_ms, max_budget_ms
//	ctl.evo.cancel   args: job_id — drops the job from its generation (scored as failed), releases
//	                 the GPU lease named after the job id and publishes JobResult{FAILED, error
//	                 "cancelled by operator"} on jobs.result.<kind> so views update.
//
// Error replies carry a prefix control-api maps to HTTP: `not_found:` (404), `conflict:` (409).
//
// Heartbeat counters: gpu.used_mb, gpu.active, gpu.queued, gpu.capacity_mb (from -gpu-mem-mb /
// FA_GPU_MEM_MB, default 24000; D-031), generations.started|done, jobs.cancelled.
//
// Notifications (CtlRequest, no reply): ctl.gpu.preempt / ctl.gpu.granted with arg lease_id.
package main

import (
	"context"
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

func main() {
	natsURL := flag.String("nats", envOr("FA_NATS_URL", "nats://127.0.0.1:4222"), "NATS URL")
	gpuMem := flag.Int("gpu-mem-mb", envInt("FA_GPU_MEM_MB", 24_000), "GPU memory managed by the arbiter (MB), reported as gpu.capacity_mb")
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
		genomes := strings.Split(r.Args["genomes"], ",")
		eta, _ := strconv.Atoi(r.Args["eta"])
		minB, _ := strconv.ParseUint(r.Args["min_budget_ms"], 10, 64)
		maxB, _ := strconv.ParseUint(r.Args["max_budget_ms"], 10, 64)
		if minB == 0 {
			minB = 60_000
		}
		if maxB == 0 {
			maxB = 16 * minB
		}
		g, err := NewGeneration(r.Args["kind"], r.Args["snapshot_id"], genomes, eta, minB, maxB, uint64(time.Now().UnixNano()))
		if err != nil {
			return "", err
		}
		gensMu.Lock()
		gens[g.ID] = g
		gensMu.Unlock()
		counters.Add("generations.started", 1)
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
				JobId: jobID, Status: fav1.JobResult_STATUS_FAILED, Error: "cancelled by operator", WorkerId: service,
			}
			done, err := g.OnResult(res, publishJob)
			if err != nil {
				log.Warn("generation after cancel", "id", g.ID, "err", err)
			}
			if done {
				log.Info("generation done", "id", g.ID, "final", g.Final())
				counters.Add("generations.done", 1)
				delete(gens, g.ID)
			}
			// Lease ids follow job ids (convention until workers report their lease ids).
			for _, id := range arb.Release(jobID) {
				notify("granted", id)
			}
			if err := pub.Publish(bus.JobResult(g.Kind), res); err != nil {
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
			done, err := g.OnResult(&res, publishJob)
			if err != nil {
				log.Warn("generation", "id", id, "err", err)
			}
			if done {
				log.Info("generation done", "id", id, "final", g.Final())
				counters.Add("generations.done", 1)
				delete(gens, id)
			}
			return
		}
	}); err != nil {
		log.Error("subscribe results", "err", err)
		os.Exit(1)
	}

	log.Info("started", "gpu_mem_mb", *gpuMem)
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
