// XBRL company facts → FundamentalFact envelopes (wh.fundamental, spec 20 §5.1).
//
// Keyless: the same data.sec.gov host and User-Agent rules as the submissions
// API. Per configured CIK the connector fetches
//
//	https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
//
// once at start and then every FactsInterval (default 6 h, `extra.facts_interval`;
// "off" disables) from inside Poll, so the shared rate limiter applies. Only a
// whitelist of concepts is emitted (`extra.facts_concepts`, comma list of
// "<taxonomy>:<concept>"; default DefaultConcepts). Every (concept, unit, fact)
// becomes one FundamentalFact:
//
//   - field = "<taxonomy>:<concept>", unit = the XBRL unit key;
//   - period = "<start>..<end>" for duration facts, "<end>" for instant facts
//     (the fact's own period: `fy`/`fp` belong to the reporting filing, which
//     also restates earlier years, so they are kept separately as
//     fiscal_year/fiscal_period);
//   - vintage = publish_time = the `filed` date at 22:00 ET (EDGAR's last
//     acceptance on a filing date, so the estimate is never earlier than the
//     real availability; publish_time_estimated is set); event_time = period end.
//
// Revisions are new vintages (a later filing reporting the same period), never
// overwrites; wh.fundamental is idempotent on its natural key. Live de-dup: a
// per-CIK cursor (ingest.backfill_cursor, scope "facts:<cik>") = newest vintage
// published; facts with an older vintage are skipped (the first run, without a
// cursor, emits the whole history of the whitelisted concepts).
package secedgar

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const (
	DefaultFactsInterval = 6 * time.Hour
	endpointFacts        = "companyfacts"
	factsMaxBody         = 96 << 20 // the largest filers' companyfacts run to tens of MiB
)

// DefaultConcepts is the default whitelist of reported facts.
var DefaultConcepts = []string{
	"us-gaap:Revenues",
	"us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
	"us-gaap:CostOfRevenue",
	"us-gaap:GrossProfit",
	"us-gaap:OperatingIncomeLoss",
	"us-gaap:NetIncomeLoss",
	"us-gaap:EarningsPerShareBasic",
	"us-gaap:EarningsPerShareDiluted",
	"us-gaap:Assets",
	"us-gaap:Liabilities",
	"us-gaap:StockholdersEquity",
	"us-gaap:CashAndCashEquivalentsAtCarryingValue",
	"us-gaap:LongTermDebt",
	"us-gaap:NetCashProvidedByUsedInOperatingActivities",
	"us-gaap:CommonStockSharesOutstanding",
	"us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
	"dei:EntityCommonStockSharesOutstanding",
}

func factsScope(cik int64) string { return "facts:" + PadCIK(cik) }

func (c *Connector) factsURL(cik int64) string {
	return fmt.Sprintf("%s/api/xbrl/companyfacts/CIK%s.json", c.BaseURL, PadCIK(cik))
}

// initFacts reads the facts settings and cursors; called from Init.
func (c *Connector) initFacts(ctx context.Context, extra map[string]string) error {
	if c.FactsInterval == 0 {
		switch s := strings.TrimSpace(extra["facts_interval"]); s {
		case "":
			c.FactsInterval = DefaultFactsInterval
		case "off", "0", "false", "none":
			c.FactsInterval = -1
		default:
			d, err := time.ParseDuration(s)
			if err != nil || d <= 0 {
				return fmt.Errorf("secedgar: bad facts_interval %q", s)
			}
			c.FactsInterval = d
		}
	}
	if c.Concepts == nil {
		list := DefaultConcepts
		if s := strings.TrimSpace(extra["facts_concepts"]); s != "" {
			list = strings.Split(s, ",")
		}
		c.Concepts = map[string]bool{}
		for _, k := range list {
			k = strings.TrimSpace(k)
			if k == "" {
				continue
			}
			if !strings.Contains(k, ":") {
				return fmt.Errorf("secedgar: concept %q must be <taxonomy>:<concept>", k)
			}
			c.Concepts[k] = true
		}
	}
	c.factsDue, c.factsFloor = map[int64]time.Time{}, map[int64]time.Time{}
	if c.deps.Cursors != nil {
		for _, n := range c.ciks {
			t, ok, err := c.deps.Cursors.GetCursor(ctx, ID, "", factsScope(n))
			if err != nil {
				return fmt.Errorf("secedgar: facts cursor: %w", err)
			}
			if ok {
				c.factsFloor[n] = t
			}
		}
	}
	return nil
}

// pollFacts fetches the companyfacts documents that are due.
func (c *Connector) pollFacts(ctx context.Context) ([]connector.RawMessage, error) {
	if c.FactsInterval <= 0 {
		return nil, nil
	}
	now := c.deps.Clock.Now()
	var out []connector.RawMessage
	var firstErr error
	for _, n := range c.ciks {
		c.mu.Lock()
		due := c.factsDue[n]
		c.mu.Unlock()
		if !due.IsZero() && now.Before(due) {
			continue
		}
		u := c.factsURL(n)
		b, err := c.factsFetch().Get(ctx, u)
		if err != nil {
			if ctx.Err() != nil {
				return out, ctx.Err()
			}
			if firstErr == nil {
				firstErr = fmt.Errorf("secedgar facts %s: %w", PadCIK(n), err)
			}
			// a filer without XBRL answers 404: retry at the normal cadence, not every poll
			c.mu.Lock()
			c.factsDue[n] = now.Add(c.FactsInterval)
			c.mu.Unlock()
			continue
		}
		c.mu.Lock()
		c.factsDue[n] = now.Add(c.FactsInterval)
		c.mu.Unlock()
		out = append(out, c.raw(endpointFacts, u, n, b))
	}
	return out, firstErr
}

// factsFetch is the shared fetcher with a larger body cap (same limiter and UA).
func (c *Connector) factsFetch() *connector.Fetcher {
	f := *c.Fetch
	if f.MaxBody < factsMaxBody {
		f.MaxBody = factsMaxBody
	}
	if f.Client != nil && f.Client.Timeout > 0 && f.Client.Timeout < 90*time.Second {
		cl := *f.Client
		cl.Timeout = 90 * time.Second
		f.Client = &cl
	}
	return &f
}

type xbrlFact struct {
	Start string          `json:"start"`
	End   string          `json:"end"`
	Val   json.RawMessage `json:"val"`
	Accn  string          `json:"accn"`
	FY    json.RawMessage `json:"fy"`
	FP    string          `json:"fp"`
	Form  string          `json:"form"`
	Filed string          `json:"filed"`
}

type companyFacts struct {
	CIK        json.RawMessage `json:"cik"`
	EntityName string          `json:"entityName"`
	Facts      map[string]map[string]struct {
		Units map[string][]xbrlFact `json:"units"`
	} `json:"facts"`
}

// filedVintage is the conservative availability time of a filing date.
func filedVintage(filed string) (time.Time, bool) {
	d, err := time.ParseInLocation(time.DateOnly, filed, et)
	if err != nil {
		return time.Time{}, false
	}
	return d.Add(22 * time.Hour).UTC(), true
}

func day(s string) (time.Time, bool) {
	t, err := time.Parse(time.DateOnly, s)
	return t.UTC(), err == nil
}

// fyString renders `fy` (a number, or null) as text.
func fyString(raw json.RawMessage) string {
	s := strings.Trim(strings.TrimSpace(string(raw)), `"`)
	if s == "null" {
		return ""
	}
	return s
}

// normalizeFacts maps a companyfacts document to FundamentalFact envelopes.
func (c *Connector) normalizeFacts(raw connector.RawMessage) ([]*fav1.Envelope, error) {
	var doc companyFacts
	if err := json.Unmarshal(raw.Bytes, &doc); err != nil {
		return nil, fmt.Errorf("secedgar: decode companyfacts: %w", err)
	}
	cik, err := ParseCIK(strings.Trim(string(doc.CIK), `"`))
	if err != nil {
		if cik, err = ParseCIK(raw.TransportMeta[metaCIK]); err != nil {
			return nil, err
		}
	}
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"
	c.mu.Lock()
	floor := c.factsFloor[cik]
	c.mu.Unlock()
	if backfill {
		floor = time.Time{}
	}
	// deterministic order: taxonomy, concept, unit, then as listed
	type key struct{ field, unit string }
	var keys []key
	for tax, concepts := range doc.Facts {
		for name, cf := range concepts {
			field := tax + ":" + name
			if !c.Concepts[field] {
				continue
			}
			for unit := range cf.Units {
				keys = append(keys, key{field, unit})
			}
		}
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].field != keys[j].field {
			return keys[i].field < keys[j].field
		}
		return keys[i].unit < keys[j].unit
	})
	var envs []*fav1.Envelope
	newest := floor
	for _, k := range keys {
		tax, name, _ := strings.Cut(k.field, ":")
		for _, f := range doc.Facts[tax][name].Units[k.unit] {
			v, err := strconv.ParseFloat(strings.TrimSpace(string(f.Val)), 64)
			if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
				continue
			}
			vintage, ok := filedVintage(f.Filed)
			end, ok2 := day(f.End)
			if !ok || !ok2 {
				continue
			}
			if !floor.IsZero() && vintage.Before(floor) {
				continue
			}
			ff := &fav1.FundamentalFact{
				IssuerCik: PadCIK(cik), Field: k.field, Unit: k.unit, Period: f.End,
				PeriodEnd: timestamppb.New(end), Value: v, Vintage: timestamppb.New(vintage),
				Form: f.Form, Accession: f.Accn, FiscalYear: fyString(f.FY), FiscalPeriod: f.FP,
			}
			if f.Start != "" {
				if start, ok := day(f.Start); ok {
					ff.PeriodStart = timestamppb.New(start)
					ff.Period = f.Start + ".." + f.End
				}
			}
			envs = append(envs, &fav1.Envelope{
				EventId:              ulid.Make().String(),
				SourceId:             ID,
				EventTime:            timestamppb.New(end),
				PublishTime:          timestamppb.New(vintage),
				IngestTime:           timestamppb.New(raw.ReceivedAt),
				SchemaVersion:        1,
				PublishTimeEstimated: true,
				Payload:              &fav1.Envelope_Fundamental{Fundamental: ff},
			})
			if vintage.After(newest) {
				newest = vintage
			}
		}
	}
	if !backfill && newest.After(floor) {
		c.mu.Lock()
		c.factsFloor[cik] = newest
		c.mu.Unlock()
		if c.deps.Cursors != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = c.deps.Cursors.SetCursor(ctx, ID, "", factsScope(cik), newest)
			cancel()
		}
	}
	return envs, nil
}
