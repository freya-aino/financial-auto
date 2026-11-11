package connector

import (
	"fmt"
	"sort"
	"sync"
)

// Factory creates a fresh connector instance.
type Factory func() Connector

var (
	regMu    sync.RWMutex
	registry = map[string]Factory{}
)

// Register adds a factory; connectors call it from init(). Panics on duplicates.
func Register(id string, f Factory) {
	regMu.Lock()
	defer regMu.Unlock()
	if _, dup := registry[id]; dup {
		panic(fmt.Sprintf("connector %q registered twice", id))
	}
	registry[id] = f
}

// New instantiates a registered connector.
func New(id string) (Connector, error) {
	regMu.RLock()
	f, ok := registry[id]
	regMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("connector %q not registered", id)
	}
	return f(), nil
}

// Registered lists ids in sorted order.
func Registered() []string {
	regMu.RLock()
	defer regMu.RUnlock()
	ids := make([]string, 0, len(registry))
	for id := range registry {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// AdapterFactory creates a generic connector serving the catalog entry `id`
// (spec 11 §6.3: rss, sdmx).
type AdapterFactory func(id string) Connector

var adapters = map[string]AdapterFactory{}

// RegisterAdapter adds a generic adapter; packages call it from init(). Panics on duplicates.
func RegisterAdapter(name string, f AdapterFactory) {
	regMu.Lock()
	defer regMu.Unlock()
	if _, dup := adapters[name]; dup {
		panic(fmt.Sprintf("adapter %q registered twice", name))
	}
	adapters[name] = f
}

// NewFor instantiates the connector for a catalog entry: the one registered
// under its id, else its `adapter:` bound to the id.
func NewFor(id, adapter string) (Connector, error) {
	if c, err := New(id); err == nil {
		return c, nil
	}
	if adapter == "" {
		return nil, fmt.Errorf("connector %q not registered", id)
	}
	regMu.RLock()
	f, ok := adapters[adapter]
	regMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("connector %q not registered and adapter %q unknown", id, adapter)
	}
	return f(id), nil
}

// HasConnector reports whether id has a registered connector or names a known adapter.
func HasConnector(id, adapter string) bool {
	regMu.RLock()
	defer regMu.RUnlock()
	if _, ok := registry[id]; ok {
		return true
	}
	_, ok := adapters[adapter]
	return adapter != "" && ok
}

// Adapters lists registered adapter names in sorted order.
func Adapters() []string {
	regMu.RLock()
	defer regMu.RUnlock()
	out := make([]string, 0, len(adapters))
	for n := range adapters {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
