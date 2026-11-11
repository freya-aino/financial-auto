package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/fey/financial-auto/go/pkg/archive"
	"github.com/fey/financial-auto/go/pkg/bus"
	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// ReplayPrefix is prepended to the original subject when a replay publishes
// (core NATS only: no JetStream stream captures `replay.>`, so live consumers
// and the stores never see replayed events twice).
const ReplayPrefix = "replay."

// ReplayResult is the JSON reply of `ctl.ingest.replay` (spec 10 §9).
type ReplayResult struct {
	Source    string              `json:"source"`
	Start     time.Time           `json:"start"`
	End       time.Time           `json:"end"`
	Mode      string              `json:"mode"`
	Read      archive.ReplayStats `json:"read"`
	Envelopes int                 `json:"envelopes"`
	Published int                 `json:"published"`
	Errors    int                 `json:"normalize_errors"`
	// Digest is sha256 over the deterministic encodings of every envelope in
	// order: two replays of the same range must agree (byte-identical events).
	Digest  string   `json:"digest"`
	FirstID string   `json:"first_event_id,omitempty"`
	LastID  string   `json:"last_event_id,omitempty"`
	Samples []string `json:"subjects,omitempty"`
}

// ReplayNormalize re-runs a connector's normalizer over archived payloads,
// applying the live pipeline's Finalize, and returns the envelopes in order.
func ReplayNormalize(norm connector.Normalizer, items []archive.ReplayItem) ([]*fav1.Envelope, int) {
	var out []*fav1.Envelope
	errs := 0
	for _, it := range items {
		envs, _, err := norm.Normalize(it.Raw)
		if err != nil {
			errs++
			continue
		}
		backfill := it.Raw.TransportMeta[connector.MetaBackfill] == "1"
		gapFill := it.Raw.TransportMeta[connector.MetaGapFill] == "1"
		for _, e := range envs {
			e.GapFilled = e.GapFilled || gapFill
			Finalize(e, it.Ref, true, backfill)
			out = append(out, e)
		}
	}
	return out, errs
}

// Digest hashes envelopes deterministically.
func Digest(envs []*fav1.Envelope) string {
	h := sha256.New()
	opt := proto.MarshalOptions{Deterministic: true}
	for _, e := range envs {
		b, _ := opt.Marshal(e)
		h.Write(b)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Replayer serves `ctl.ingest.replay {source, start, end, mode=verify|publish}`.
type Replayer struct {
	Sup   *Supervisor
	Store archive.ReadStore
	Pub   *bus.Publisher
}

func (r *Replayer) Replay(ctx context.Context, args map[string]string) (*ReplayResult, error) {
	if r.Store == nil {
		return nil, fmt.Errorf("%w: replay needs the Garage raw archive (-garage)", ErrConflict)
	}
	source := strings.TrimSpace(args["source"])
	src, err := r.Sup.Resolve(source)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	end, start := now, now.Add(-time.Hour)
	if v := args["end"]; v != "" {
		if end, err = time.Parse(time.RFC3339, v); err != nil {
			return nil, fmt.Errorf("%w: end: %v", ErrInvalid, err)
		}
	}
	if v := args["start"]; v != "" {
		if start, err = time.Parse(time.RFC3339, v); err != nil {
			return nil, fmt.Errorf("%w: start: %v", ErrInvalid, err)
		}
	}
	if !start.Before(end) || end.Sub(start) > 7*24*time.Hour {
		return nil, fmt.Errorf("%w: need start < end and a range of at most 7 days", ErrInvalid)
	}
	mode := args["mode"]
	if mode == "" {
		mode = "verify"
	}
	if mode != "verify" && mode != "publish" {
		return nil, fmt.Errorf("%w: mode must be verify or publish", ErrInvalid)
	}
	c, err := connector.NewFor(source, src.Adapter)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNotFound, err)
	}
	cfg := connector.SourceConfig{Source: src}
	if r.Sup.Secrets != nil {
		cfg.Secrets = r.Sup.Secrets(source)
	}
	deps := r.Sup.deps(src)
	deps.Archive = connector.NopArchive{} // never re-archive replayed payloads
	// Fresh in-memory cursors: normalizers that skip already-seen items by the persisted live
	// watermark would otherwise drop the whole replayed range.
	deps.Cursors = connector.NewMemCursors()
	if err := c.Init(ctx, cfg, deps); err != nil {
		return nil, fmt.Errorf("init %s: %w", source, err)
	}
	norm, ok := c.(connector.Normalizer)
	if !ok {
		return nil, fmt.Errorf("%w: %s has no normalizer", ErrConflict, source)
	}
	items, st, err := archive.Replay(ctx, r.Store, "", source, start.UTC(), end.UTC())
	if err != nil {
		return nil, err
	}
	envs, nerr := ReplayNormalize(norm, items)
	res := &ReplayResult{Source: source, Start: start.UTC(), End: end.UTC(), Mode: mode, Read: st, Envelopes: len(envs), Errors: nerr, Digest: Digest(envs)}
	if len(envs) > 0 {
		res.FirstID, res.LastID = envs[0].EventId, envs[len(envs)-1].EventId
	}
	venue := c.Capabilities().Venue
	seen := map[string]bool{}
	for _, e := range envs {
		subj, err := bus.SubjectFor(e, venue)
		if err != nil {
			continue
		}
		if !seen[subj] && len(res.Samples) < 5 {
			seen[subj] = true
			res.Samples = append(res.Samples, subj)
		}
		if mode == "publish" && r.Pub != nil {
			if err := r.Pub.PublishWithID(ReplayPrefix+subj, e.EventId, e); err == nil {
				res.Published++
			}
		}
	}
	return res, nil
}
