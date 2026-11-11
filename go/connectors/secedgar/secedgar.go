// Package secedgar is the SEC EDGAR filings connector for catalog id "sec_edgar" (specs 10, 11).
//
// It polls the submissions API per configured CIK
//
//	https://data.sec.gov/submissions/CIK##########.json
//
// and maps every entry of `filings.recent` to a FilingEvent on
// `filing.event.sec_edgar`: issuer_cik (10 digits, zero padded), issuer_lei
// (top-level `lei` when present), form_type, accession, filing_time =
// `acceptanceDateTime` (UTC: the `filingDate` roll-over at 17:30 ET is
// consistent with the trailing Z), and document_refs = the primary document
// URL `https://www.sec.gov/Archives/edgar/data/<cik>/<accession without
// dashes>/<primaryDocument>` (the filing index page when no primary document
// is listed). Envelope event_time = publish_time = acceptance time. Without an
// acceptance time, event_time = filingDate 00:00 ET and publish_time =
// filingDate 22:00 ET (EDGAR's last acceptance on that filing date, so the
// estimate is never earlier than the real availability); ET is taken as UTC−5.
// Filings carry no instrument_id (spec 10 §5.1). XBRL company facts
// (wh.fundamental) are polled from the same host, see facts.go.
//
// CIKs come from `config.extra.ciks` (comma list). A backfill job may name a
// CIK as its instrument; the source has no venue, so the ingestor does not
// resolve it into a ref instrument (cursor scope `<tf>|<cik>`). SEC fair access
// requires a declared User-Agent: `config.extra.user_agent` or the secret
// `[sources.sec_edgar] user_agent` (FA_SEC_EDGAR_USER_AGENT), default
// "financial-auto ingestor"; the ingestor's generic UA is answered with 403.
//
// Live de-dup: an in-memory accession set plus a per-CIK cursor
// (ingest.backfill_cursor, scope "cik:<cik>") = newest acceptance time
// published; filings at or before it are skipped after a restart. Without a
// cursor (first run) only filings accepted within Lookback (7 days, config
// `extra.lookback`) of the poll are emitted, so the ~1000 historical entries
// are not replayed live — use Backfill for history.
//
// Backfill: the same endpoint filtered to [From, To) on acceptance time; the
// `recent` block holds roughly the last 1000 filings, older ones live in the
// `filings.files[]` pages (`https://data.sec.gov/submissions/<name>`) which
// are fetched when their [filingFrom, filingTo] overlaps the range.
//
// Verified live 2026-11-11: CIK0000320193 → 200 (1008 recent filings, one
// older page) with UA "financial-auto ingestor"; 403 with the default UA.
package secedgar

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/oklog/ulid/v2"
	"google.golang.org/protobuf/types/known/timestamppb"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/connector"
)

const (
	ID               = "sec_edgar"
	DefaultBaseURL   = "https://data.sec.gov"
	DefaultArchive   = "https://www.sec.gov/Archives/edgar/data"
	DefaultUserAgent = "financial-auto ingestor"
	DefaultLookback  = 7 * 24 * time.Hour
	metaCIK          = "cik"
	metaLEI          = "lei"
	metaFrom         = "from"
	metaTo           = "to"
)

// DefaultCIKs apply when config.extra.ciks is empty (Apple, Microsoft).
var DefaultCIKs = []string{"320193", "789019"}

// et is US Eastern standard time; used only for estimates without acceptance times.
var et = time.FixedZone("ET", -5*3600)

func init() { connector.Register(ID, func() connector.Connector { return &Connector{} }) }

// Connector implements Connector, Poller, Backfiller and Normalizer.
type Connector struct {
	BaseURL    string
	ArchiveURL string
	Fetch      *connector.Fetcher
	UserAgent  string
	Lookback   time.Duration
	// FactsInterval is the companyfacts cadence (facts.go); < 0 disables.
	FactsInterval time.Duration
	// Concepts is the "<taxonomy>:<concept>" whitelist of emitted facts.
	Concepts map[string]bool

	cfg  connector.SourceConfig
	deps connector.Deps
	ciks []int64
	seen *connector.Seen

	mu sync.Mutex
	// floor per CIK is the cursor loaded at Init; cursor is the newest acceptance published.
	floor, cursor map[int64]time.Time
	// factsDue is when each CIK's companyfacts is next fetched; factsFloor the newest vintage published.
	factsDue, factsFloor map[int64]time.Time
}

func (c *Connector) ID() string { return ID }

func (c *Connector) Capabilities() connector.Capabilities {
	return connector.Capabilities{Poll: true, Backfill: true, Events: []connector.EventKind{connector.KindFiling, connector.KindFundamental}}
}

// ParseCIK accepts "320193", "0000320193" or "CIK0000320193".
func ParseCIK(s string) (int64, error) {
	s = strings.TrimPrefix(strings.ToUpper(strings.TrimSpace(s)), "CIK")
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("secedgar: bad CIK %q", s)
	}
	return n, nil
}

// PadCIK renders a CIK as EDGAR's 10-digit form.
func PadCIK(n int64) string { return fmt.Sprintf("%010d", n) }

func scope(cik int64) string { return "cik:" + PadCIK(cik) }

func (c *Connector) Init(ctx context.Context, cfg connector.SourceConfig, deps connector.Deps) error {
	if deps.Clock == nil {
		deps.Clock = connector.SystemClock{}
	}
	c.cfg, c.deps = cfg, deps
	if c.BaseURL == "" {
		c.BaseURL = DefaultBaseURL
	}
	if c.ArchiveURL == "" {
		c.ArchiveURL = DefaultArchive
	}
	p := cfg.Source.Params()
	if c.UserAgent == "" {
		c.UserAgent = p.Extra["user_agent"]
	}
	if c.UserAgent == "" {
		c.UserAgent = cfg.Secrets["user_agent"]
	}
	if c.UserAgent == "" {
		c.UserAgent = DefaultUserAgent
	}
	if c.Fetch == nil {
		c.Fetch = connector.NewFetcher(deps.Limiter)
	}
	ua := c.UserAgent
	prev := c.Fetch.Prepare
	c.Fetch.Prepare = func(r *http.Request) {
		if prev != nil {
			prev(r)
		}
		r.Header.Set("User-Agent", ua)
	}
	if c.Lookback == 0 {
		if d, err := time.ParseDuration(p.Extra["lookback"]); err == nil && d > 0 {
			c.Lookback = d
		} else {
			c.Lookback = DefaultLookback
		}
	}
	list := DefaultCIKs
	if s := strings.TrimSpace(p.Extra["ciks"]); s != "" {
		list = strings.Split(s, ",")
	}
	c.ciks = nil
	for _, s := range list {
		if strings.TrimSpace(s) == "" {
			continue
		}
		n, err := ParseCIK(s)
		if err != nil {
			return err
		}
		c.ciks = append(c.ciks, n)
	}
	c.seen = connector.NewSeen(50_000)
	c.floor, c.cursor = map[int64]time.Time{}, map[int64]time.Time{}
	if deps.Cursors != nil {
		for _, n := range c.ciks {
			t, ok, err := deps.Cursors.GetCursor(ctx, ID, "", scope(n))
			if err != nil {
				return fmt.Errorf("secedgar: cursor: %w", err)
			}
			if ok {
				c.floor[n], c.cursor[n] = t, t
			}
		}
	}
	return c.initFacts(ctx, p.Extra)
}

func (c *Connector) Schedule() connector.PollSchedule {
	return connector.PollSchedule{Interval: 10 * time.Minute}
}

// CIKs lists the configured CIKs.
func (c *Connector) CIKs() []int64 { return append([]int64(nil), c.ciks...) }

func (c *Connector) submissionsURL(cik int64) string {
	return fmt.Sprintf("%s/submissions/CIK%s.json", c.BaseURL, PadCIK(cik))
}

func (c *Connector) raw(endpoint, u string, cik int64, b []byte) connector.RawMessage {
	return connector.RawMessage{
		SourceID: ID, ReceivedAt: c.deps.Clock.Now(), Bytes: b, ContentType: "application/json",
		TransportMeta: map[string]string{connector.MetaEndpoint: endpoint, connector.MetaURL: u, metaCIK: PadCIK(cik)},
	}
}

// Poll fetches the submissions document of every configured CIK.
func (c *Connector) Poll(ctx context.Context, _ connector.PollRequest) ([]connector.RawMessage, error) {
	var out []connector.RawMessage
	var firstErr error
	for _, n := range c.ciks {
		u := c.submissionsURL(n)
		b, err := c.Fetch.Get(ctx, u)
		if err != nil {
			if ctx.Err() != nil {
				return out, ctx.Err()
			}
			if firstErr == nil {
				firstErr = fmt.Errorf("secedgar %s: %w", PadCIK(n), err)
			}
			continue
		}
		out = append(out, c.raw("submissions", u, n, b))
	}
	facts, err := c.pollFacts(ctx)
	out = append(out, facts...)
	if err != nil && firstErr == nil {
		firstErr = err
	}
	if len(out) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// Backfill emits filings accepted in [From, To) for req.VenueSymbol (a CIK)
// or, when empty, every configured CIK.
func (c *Connector) Backfill(ctx context.Context, req connector.BackfillRequest, out chan<- connector.RawMessage) error {
	to := req.To
	if to.IsZero() {
		to = c.deps.Clock.Now()
	}
	from := req.From
	if cur, err := time.Parse(time.RFC3339, req.Cursor); req.Cursor != "" && err == nil && cur.After(from) {
		from = cur
	}
	ciks := c.ciks
	if req.VenueSymbol != "" {
		n, err := ParseCIK(req.VenueSymbol)
		if err != nil {
			return err
		}
		ciks = []int64{n}
	}
	send := func(m connector.RawMessage) error {
		m.TransportMeta[connector.MetaBackfill] = "1"
		m.TransportMeta[metaFrom] = from.UTC().Format(time.RFC3339Nano)
		m.TransportMeta[metaTo] = to.UTC().Format(time.RFC3339Nano)
		select {
		case out <- m:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	for _, n := range ciks {
		u := c.submissionsURL(n)
		b, err := c.Fetch.Get(ctx, u)
		if err != nil {
			return fmt.Errorf("secedgar backfill %s: %w", PadCIK(n), err)
		}
		var sub submissions
		if err := json.Unmarshal(b, &sub); err != nil {
			return fmt.Errorf("secedgar backfill %s: decode: %w", PadCIK(n), err)
		}
		if err := send(c.raw("submissions", u, n, b)); err != nil {
			return err
		}
		for _, f := range sub.Filings.Files {
			// Pages list filing dates; widen by a day for the acceptance/filing-date offset.
			pf, err1 := time.Parse(time.DateOnly, f.FilingFrom)
			pt, err2 := time.Parse(time.DateOnly, f.FilingTo)
			if err1 == nil && err2 == nil && (pt.Add(48*time.Hour).Before(from) || !pf.Add(-24*time.Hour).Before(to)) {
				continue
			}
			pu := c.BaseURL + "/submissions/" + f.Name
			pb, err := c.Fetch.Get(ctx, pu)
			if err != nil {
				return fmt.Errorf("secedgar backfill %s: %w", f.Name, err)
			}
			m := c.raw("page", pu, n, pb)
			if sub.LEI != nil {
				m.TransportMeta[metaLEI] = *sub.LEI
			}
			if err := send(m); err != nil {
				return err
			}
		}
	}
	return nil
}

type recent struct {
	AccessionNumber    []string `json:"accessionNumber"`
	FilingDate         []string `json:"filingDate"`
	AcceptanceDateTime []string `json:"acceptanceDateTime"`
	Form               []string `json:"form"`
	PrimaryDocument    []string `json:"primaryDocument"`
}

type submissions struct {
	CIK     string  `json:"cik"`
	LEI     *string `json:"lei"`
	Filings struct {
		Recent recent `json:"recent"`
		Files  []struct {
			Name       string `json:"name"`
			FilingFrom string `json:"filingFrom"`
			FilingTo   string `json:"filingTo"`
		} `json:"files"`
	} `json:"filings"`
}

func at(s []string, i int) string {
	if i < len(s) {
		return s[i]
	}
	return ""
}

// times returns (event_time, publish_time) for one filing.
func times(acceptance, filingDate string) (time.Time, time.Time, bool) {
	if t, err := time.Parse(time.RFC3339, acceptance); err == nil {
		return t.UTC(), t.UTC(), true
	}
	d, err := time.ParseInLocation(time.DateOnly, filingDate, et)
	if err != nil {
		return time.Time{}, time.Time{}, false
	}
	return d.UTC(), d.Add(22 * time.Hour).UTC(), true
}

// Normalize maps a submissions document (or an older page) to FilingEvents.
func (c *Connector) Normalize(raw connector.RawMessage) ([]*fav1.Envelope, []connector.Unresolved, error) {
	var (
		r   recent
		lei string
		cik int64
		err error
	)
	if raw.TransportMeta[connector.MetaEndpoint] == endpointFacts {
		envs, err := c.normalizeFacts(raw)
		return envs, nil, err
	}
	if raw.TransportMeta[connector.MetaEndpoint] == "page" {
		if err := json.Unmarshal(raw.Bytes, &r); err != nil {
			return nil, nil, fmt.Errorf("secedgar: decode page: %w", err)
		}
		lei = raw.TransportMeta[metaLEI]
		cik, err = ParseCIK(raw.TransportMeta[metaCIK])
	} else {
		var sub submissions
		if err := json.Unmarshal(raw.Bytes, &sub); err != nil {
			return nil, nil, fmt.Errorf("secedgar: decode: %w", err)
		}
		r = sub.Filings.Recent
		if sub.LEI != nil {
			lei = *sub.LEI
		}
		cik, err = ParseCIK(sub.CIK)
		if err != nil {
			cik, err = ParseCIK(raw.TransportMeta[metaCIK])
		}
	}
	if err != nil {
		return nil, nil, err
	}
	backfill := raw.TransportMeta[connector.MetaBackfill] == "1"
	var from, to time.Time
	if backfill {
		from, _ = time.Parse(time.RFC3339Nano, raw.TransportMeta[metaFrom])
		to, _ = time.Parse(time.RFC3339Nano, raw.TransportMeta[metaTo])
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	floor := c.floor[cik]
	if floor.IsZero() {
		floor = raw.ReceivedAt.Add(-c.Lookback) // first run: only recent filings live
	}
	newest := c.cursor[cik]
	var envs []*fav1.Envelope
	for i, acc := range r.AccessionNumber {
		if acc == "" {
			continue
		}
		evt, pub, ok := times(at(r.AcceptanceDateTime, i), at(r.FilingDate, i))
		if !ok {
			continue
		}
		if backfill {
			if (!from.IsZero() && pub.Before(from)) || (!to.IsZero() && !pub.Before(to)) {
				continue
			}
		} else {
			if !pub.After(floor) {
				c.seen.Add(acc)
				continue
			}
			if !c.seen.Add(acc) {
				continue
			}
		}
		envs = append(envs, &fav1.Envelope{
			EventId:       ulid.Make().String(),
			SourceId:      ID,
			EventTime:     timestamppb.New(evt),
			PublishTime:   timestamppb.New(pub),
			IngestTime:    timestamppb.New(raw.ReceivedAt),
			SchemaVersion: 1,
			Payload: &fav1.Envelope_Filing{Filing: &fav1.FilingEvent{
				IssuerLei:    lei,
				IssuerCik:    PadCIK(cik),
				FormType:     at(r.Form, i),
				FilingTime:   timestamppb.New(pub),
				Accession:    acc,
				DocumentRefs: []string{c.docURL(cik, acc, at(r.PrimaryDocument, i))},
			}},
		})
		if !backfill && pub.After(newest) {
			newest = pub
		}
	}
	if !backfill && newest.After(c.cursor[cik]) {
		c.cursor[cik] = newest
		if c.deps.Cursors != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = c.deps.Cursors.SetCursor(ctx, ID, "", scope(cik), newest)
			cancel()
		}
	}
	return envs, nil, nil
}

// docURL is the primary document URL, or the filing index when none is listed.
func (c *Connector) docURL(cik int64, accession, primary string) string {
	dir := fmt.Sprintf("%s/%d/%s", c.ArchiveURL, cik, strings.ReplaceAll(accession, "-", ""))
	if primary == "" {
		return dir + "/" + accession + "-index.htm"
	}
	return dir + "/" + primary
}
