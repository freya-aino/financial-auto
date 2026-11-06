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
