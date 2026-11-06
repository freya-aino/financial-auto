package main

import (
	"slices"
	"testing"
)

func mustReq(t *testing.T, a *Arbiter, id string, c Class, mem int) Decision {
	t.Helper()
	d, err := a.Request(id, c, mem)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestGrantWhenFree(t *testing.T) {
	a := NewArbiter(100)
	if d := mustReq(t, a, "train", GenerationTraining, 60); !d.Granted {
		t.Fatal("expected grant")
	}
	if d := mustReq(t, a, "eval", EvolutionEval, 40); !d.Granted {
		t.Fatal("expected grant")
	}
	if u, act, q := a.Stats(); u != 100 || act != 2 || q != 0 {
		t.Fatalf("stats %d %d %d", u, act, q)
	}
}

func TestHigherPriorityPreemptsLowestFirst(t *testing.T) {
	a := NewArbiter(100)
	mustReq(t, a, "train", GenerationTraining, 50)
	mustReq(t, a, "eval", EvolutionEval, 50)
	d := mustReq(t, a, "live", LiveInference, 40)
	if d.Granted || !d.Queued || !slices.Equal(d.Preempt, []string{"train"}) {
		t.Fatalf("decision %+v", d)
	}
	if got := a.Release("train"); !slices.Equal(got, []string{"live"}) {
		t.Fatalf("release granted %v", got)
	}
}

func TestEqualOrLowerPriorityNeverPreempts(t *testing.T) {
	a := NewArbiter(100)
	mustReq(t, a, "eval1", EvolutionEval, 80)
	d := mustReq(t, a, "eval2", EvolutionEval, 40)
	if d.Granted || len(d.Preempt) != 0 || !d.Queued {
		t.Fatalf("decision %+v", d)
	}
	d = mustReq(t, a, "train", GenerationTraining, 10)
	if d.Granted {
		t.Fatal("lower priority must not overtake a queued higher-priority request")
	}
	if got := a.Release("eval1"); !slices.Equal(got, []string{"eval2", "train"}) {
		t.Fatalf("release granted %v", got)
	}
}

func TestRejectsOversizedRequest(t *testing.T) {
	a := NewArbiter(100)
	if _, err := a.Request("big", LiveInference, 101); err == nil {
		t.Fatal("expected error")
	}
}

func TestParseClass(t *testing.T) {
	c, err := ParseClass("online_adapter")
	if err != nil || c != OnlineAdapter || c.String() != "online_adapter" {
		t.Fatalf("%v %v", c, err)
	}
}
