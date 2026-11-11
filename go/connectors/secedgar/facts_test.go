package secedgar

import (
	"context"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/fey/financial-auto/go/pkg/bus"
	"github.com/fey/financial-auto/go/pkg/connector"
)

// Fixture: Apple companyfacts trimmed to 2 dei shares-outstanding facts, 4
// NetIncomeLoss, 2 Assets, 2 EarningsPerShareDiluted (all whitelisted) and 2
// AccountsPayableCurrent (not whitelisted).
func factsRaws(t *testing.T, c *Connector) []connector.RawMessage {
	t.Helper()
	raws, err := c.Poll(context.Background(), connector.PollRequest{})
	if err != nil {
		t.Fatal(err)
	}
	var out []connector.RawMessage
	for _, r := range raws {
		if r.TransportMeta[connector.MetaEndpoint] == endpointFacts {
			out = append(out, r)
		}
	}
	return out
}

func TestCompanyFactsNormalize(t *testing.T) {
	cur := connector.NewMemCursors()
	c, _, _ := newTest(t, cur, map[string]string{"facts_interval": "6h"}, now)
	raws := factsRaws(t, c)
	if len(raws) != 1 {
		t.Fatalf("facts raws: %d", len(raws))
	}
	envs, unres, err := c.Normalize(raws[0])
	if err != nil || len(unres) != 0 || len(envs) != 10 {
		t.Fatalf("normalize: %v %v %d", err, unres, len(envs))
	}
	found := false
	for _, e := range envs {
		f := e.GetFundamental()
		if f == nil || f.IssuerCik != "0000320193" || e.InstrumentId != "" || !e.PublishTimeEstimated {
			t.Fatalf("envelope: %v", e)
		}
		if subj, err := bus.SubjectFor(e, ""); err != nil || subj != "filing.fact.sec_edgar" {
			t.Fatalf("subject %q %v", subj, err)
		}
		if f.Field == "us-gaap:AccountsPayableCurrent" {
			t.Fatal("non-whitelisted concept emitted")
		}
		if f.Field == "us-gaap:NetIncomeLoss" && f.Period == "2025-09-28..2026-09-26" {
			found = true
			// filed 2026-11-03 → 22:00 ET = 2026-11-04T03:00Z
			if f.Value != 132175000000 || f.Unit != "USD" || f.Form != "10-K" || f.FiscalYear != "2026" || f.FiscalPeriod != "FY" ||
				f.Accession != "0000320193-26-000026" || f.Vintage.AsTime() != time.Date(2026, 11, 4, 3, 0, 0, 0, time.UTC) ||
				e.PublishTime.AsTime() != f.Vintage.AsTime() || e.EventTime.AsTime() != time.Date(2026, 9, 26, 0, 0, 0, 0, time.UTC) ||
				f.PeriodStart.AsTime() != time.Date(2025, 9, 28, 0, 0, 0, 0, time.UTC) {
				t.Fatalf("fact: %v", f)
			}
		}
		if f.Field == "us-gaap:Assets" && (f.PeriodStart != nil || f.Period != f.PeriodEnd.AsTime().Format(time.DateOnly)) {
			t.Fatalf("instant fact: %v", f)
		}
	}
	if !found {
		t.Fatal("FY2026 net income missing")
	}
	// cursor = newest vintage; a re-poll within the interval fetches nothing
	newest := time.Date(2026, 11, 4, 3, 0, 0, 0, time.UTC)
	if got, ok, _ := cur.GetCursor(context.Background(), ID, "", "facts:0000320193"); !ok || got != newest {
		t.Fatalf("cursor %v %v", got, ok)
	}
	if again := factsRaws(t, c); len(again) != 0 {
		t.Fatalf("not due yet: %d", len(again))
	}
	// After a restart only facts at or after the cursor vintage are re-emitted (DB-idempotent).
	c2, _, _ := newTest(t, cur, map[string]string{"facts_interval": "6h"}, now)
	envs2, _, _ := c2.Normalize(raws[0])
	for _, e := range envs2 {
		if e.GetFundamental().Vintage.AsTime().Before(newest) {
			t.Fatalf("old vintage re-emitted: %v", e)
		}
	}
	if len(envs2) == 0 || len(envs2) >= len(envs) {
		t.Fatalf("after restart: %d of %d", len(envs2), len(envs))
	}
}

func TestCompanyFactsConfig(t *testing.T) {
	c, _, _ := newTest(t, nil, map[string]string{"facts_interval": "1h", "facts_concepts": "us-gaap:Assets"}, now)
	raws := factsRaws(t, c)
	envs, _, err := c.Normalize(raws[0])
	if err != nil || len(envs) != 2 {
		t.Fatalf("whitelist: %v %d", err, len(envs))
	}
	if off, _, _ := newTest(t, nil, nil, now); len(factsRaws(t, off)) != 0 {
		t.Fatal("facts_interval off must not fetch")
	}
	bad := &Connector{}
	if err := bad.initFacts(context.Background(), map[string]string{"facts_concepts": "Assets"}); err == nil {
		t.Fatal("concept without taxonomy must fail")
	}
	// deterministic payloads
	a, _, _ := newTest(t, nil, map[string]string{"facts_interval": "1h"}, now)
	b, _, _ := newTest(t, nil, map[string]string{"facts_interval": "1h"}, now)
	ra, rb := factsRaws(t, a), factsRaws(t, b)
	ea, _, _ := a.Normalize(ra[0])
	eb, _, _ := b.Normalize(rb[0])
	for i := range ea {
		ea[i].EventId, eb[i].EventId = "", ""
		if !proto.Equal(ea[i], eb[i]) {
			t.Fatalf("envelope %d differs", i)
		}
	}
}
