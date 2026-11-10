package main

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// ArtifactsBucket holds eval reports (evo-worker) and promoted strategy populations.
const ArtifactsBucket = "fa-artifacts"

// StrategyEvalKind is the strategy evaluation job kind (evo-worker `evo.eval.strategy`).
const StrategyEvalKind = "evo.eval.strategy"

// StrategiesKey is the Garage key of a generation's promoted population file.
func StrategiesKey(generationID string) string { return "strategies/" + generationID + ".toml" }

// StrategiesTOML renders a `strategies.toml` population file from the final front. Genome
// strings of strategy generations are TOML inline tables (`fa_strategies::genome`), so the file
// is `instruments = []` plus `strategy = [ {…}, {…} ]`, which `StrategyFile::parse` reads.
// Entries that are not inline tables (e.g. a decoder kv genome) are skipped.
func StrategiesTOML(generationID string, kind string, front []FrontMember, at time.Time) (string, int) {
	var b strings.Builder
	fmt.Fprintf(&b, "# Promoted strategy population: final Pareto front of generation %s (%s).\n", generationID, kind)
	fmt.Fprintf(&b, "# Written by evo-orchestrator at %s (spec 35 §6.4). Candidates still need the\n", at.UTC().Format(time.RFC3339))
	b.WriteString("# locked-holdout evaluation and the live paper shadow before trading (spec 41 §6.4, spec 40).\n")
	b.WriteString("instruments = []\n\nstrategy = [\n")
	n := 0
	for _, m := range front {
		g := strings.TrimSpace(m.Genome)
		if !strings.HasPrefix(g, "{") || !strings.HasSuffix(g, "}") {
			continue
		}
		fmt.Fprintf(&b, "  # rank %d, score %.6g\n  %s,\n", m.Rank, m.Score, g)
		n++
	}
	b.WriteString("]\n")
	return b.String(), n
}

// ObjectPutter is the subset of archive.S3Store the promotion needs.
type ObjectPutter interface {
	Put(ctx context.Context, bucket, key string, body []byte, contentType string) error
}

// PromoteStrategies returns the Promote hook: strategy generations' fronts become
// `fa-artifacts/strategies/<generation>.toml`; other kinds are not archived (empty key).
func PromoteStrategies(store ObjectPutter, onErr func(error)) Promote {
	return func(g *Generation, front []FrontMember) (string, error) {
		if g.Kind != StrategyEvalKind {
			return "", nil
		}
		body, n := StrategiesTOML(g.ID, g.Kind, front, time.Now())
		if n == 0 {
			return "", fmt.Errorf("no strategy genome in the front")
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		key := StrategiesKey(g.ID)
		if err := store.Put(ctx, ArtifactsBucket, key, []byte(body), "application/toml"); err != nil {
			if onErr != nil {
				onErr(err)
			}
			return "", err
		}
		return ArtifactsBucket + "/" + key, nil
	}
}
