package connector

import (
	"context"
	"testing"
)

type fake struct{}

func (fake) ID() string                                     { return "fake" }
func (fake) Capabilities() Capabilities                     { return Capabilities{} }
func (fake) Init(context.Context, SourceConfig, Deps) error { return nil }

func TestRegistry(t *testing.T) {
	Register("fake", func() Connector { return fake{} })
	c, err := New("fake")
	if err != nil || c.ID() != "fake" {
		t.Fatalf("New: %v %v", c, err)
	}
	if _, err := New("missing"); err == nil {
		t.Fatal("expected error for unregistered id")
	}
}

func TestStaticResolver(t *testing.T) {
	r := NewStaticResolver()
	r.Add("lemon", "AAPL", "id-1")
	if id, ok := r.Resolve("lemon", "AAPL"); !ok || id != "id-1" {
		t.Fatalf("resolve = %q %v", id, ok)
	}
	if _, ok := r.Resolve("oanda", "AAPL"); ok {
		t.Fatal("resolver must be per source")
	}
}
