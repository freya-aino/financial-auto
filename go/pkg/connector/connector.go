// Package connector defines the ingestion connector contract (spec 10 §4.1).
package connector

import (
	"context"
	"errors"
	"time"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"github.com/fey/financial-auto/go/pkg/catalog"
)

// ErrNotImplemented is returned by stub connectors.
var ErrNotImplemented = errors.New("connector: not implemented")

// Connector is implemented by every source adapter. ID matches the catalog id.
type Connector interface {
	ID() string
	Capabilities() Capabilities
	Init(ctx context.Context, cfg SourceConfig, deps Deps) error
}

// Streamer pushes raw messages from a live feed until ctx is done.
type Streamer interface {
	Stream(ctx context.Context, subs []Subscription, out chan<- RawMessage) error
}

// Poller fetches on a schedule.
type Poller interface {
	Poll(ctx context.Context, req PollRequest) ([]RawMessage, error)
	Schedule() PollSchedule
}

// Backfiller loads history for a time range.
type Backfiller interface {
	Backfill(ctx context.Context, req BackfillRequest, out chan<- RawMessage) error
}

// Normalizer maps raw payloads to canonical envelopes.
type Normalizer interface {
	Normalize(raw RawMessage) ([]*fav1.Envelope, []Unresolved, error)
}

// DiscoveryHook finds new targets (channels, feeds, keywords). No v1 implementation.
type DiscoveryHook interface {
	Discover(ctx context.Context, query string) ([]SourceTarget, error)
}

// EventKind names a canonical event type a connector can emit.
type EventKind string

const (
	KindTrade   EventKind = "trade"
	KindQuote   EventKind = "quote"
	KindBar     EventKind = "bar"
	KindBook    EventKind = "book"
	KindNews    EventKind = "news"
	KindMedia   EventKind = "media"
	KindMacro   EventKind = "macro"
	KindOnchain EventKind = "onchain"
	KindFiling  EventKind = "filing"
	// KindFundamental is a reported XBRL fact (FundamentalFact).
	KindFundamental EventKind = "fundamental"
)

// Capabilities declares what a connector supports.
type Capabilities struct {
	Stream    bool
	Poll      bool
	Backfill  bool
	Execution bool
	Events    []EventKind
	// Venue is used as the <venue> subject token for market-data events.
	Venue string
}

// SourceConfig is the catalog entry plus secrets and free-form params.
type SourceConfig struct {
	Source  catalog.Source
	Secrets map[string]string
	Params  map[string]string
}

// RawMessage is an unparsed payload as received (spec 10 §4.1).
type RawMessage struct {
	SourceID      string
	ReceivedAt    time.Time
	TransportMeta map[string]string
	Bytes         []byte
	ContentType   string
}

// Unresolved records a venue symbol that could not be mapped to an instrument (quarantine).
type Unresolved struct {
	SourceID    string
	VenueSymbol string
	Reason      string
}

// Subscription selects what a streamer should subscribe to.
type Subscription struct {
	VenueSymbol  string
	InstrumentID string
	Kinds        []EventKind
}

// PollRequest parameterizes one poll.
type PollRequest struct {
	Targets []string
	Since   time.Time
}

// PollSchedule says how often to poll.
type PollSchedule struct {
	Interval time.Duration
}

// BackfillRequest asks for history.
type BackfillRequest struct {
	VenueSymbol  string
	InstrumentID string
	Timeframe    string
	From, To     time.Time
	Cursor       string
}

// SourceTarget is a discovered feed target.
type SourceTarget struct {
	SourceID string
	Target   string
	Reason   string
}

// Deps are injected into connectors.
type Deps struct {
	Limiter  RateLimiter
	Resolver SymbolResolver
	Archive  RawArchive
	Clock    Clock
	// Cursors persists progress; nil means connectors start from "now" every run.
	Cursors CursorStore
}

// Meta keys connectors put in RawMessage.TransportMeta.
const (
	MetaEndpoint = "endpoint" // which API shape the payload has
	MetaSymbol   = "symbol"   // venue symbol the payload is about
	MetaTf       = "tf"       // bar timeframe
	MetaBackfill = "backfill" // "1" when produced by Backfill (no live de-dup)
	MetaURL      = "url"      // request URL, for the raw archive
)

// RateLimiter blocks until a request may proceed.
type RateLimiter interface {
	Wait(ctx context.Context) error
}

// SymbolResolver maps venue symbols to internal instrument UUIDs (spec 12).
type SymbolResolver interface {
	Resolve(sourceID, venueSymbol string) (instrumentID string, ok bool)
}

// RawArchive writes raw payloads to Garage and returns a reference.
type RawArchive interface {
	Put(ctx context.Context, raw RawMessage) (*fav1.RawRef, error)
}

// Clock is injectable for tests.
type Clock interface {
	Now() time.Time
}
