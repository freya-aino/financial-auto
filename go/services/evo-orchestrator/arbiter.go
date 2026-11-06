package main

import (
	"fmt"
	"sort"
	"sync"
)

// Class is a GPU priority class (spec 60). Lower value = higher priority.
type Class int

const (
	LiveInference Class = iota
	OnlineAdapter
	EvolutionEval
	GenerationTraining
)

var classNames = map[string]Class{
	"live_inference": LiveInference, "online_adapter": OnlineAdapter,
	"evolution_eval": EvolutionEval, "generation_training": GenerationTraining,
}

func (c Class) String() string {
	for n, v := range classNames {
		if v == c {
			return n
		}
	}
	return fmt.Sprintf("class(%d)", int(c))
}

// ParseClass accepts the snake_case class names.
func ParseClass(s string) (Class, error) {
	if c, ok := classNames[s]; ok {
		return c, nil
	}
	return 0, fmt.Errorf("unknown gpu class %q", s)
}

type lease struct {
	id    string
	class Class
	memMB int
	seq   uint64
}

// Decision is the arbiter's answer to a request.
type Decision struct {
	Granted bool
	Queued  bool
	// Preempt lists active leases that must checkpoint and Release so the request can run.
	Preempt []string
}

// Arbiter is the in-memory GPU lease manager (D-006). It enforces strict
// priority: a queued request is never overtaken by a lower-priority one.
type Arbiter struct {
	mu         sync.Mutex
	capacityMB int
	seq        uint64
	active     map[string]*lease
	preempting map[string]bool
	queue      []*lease
}

func NewArbiter(capacityMB int) *Arbiter {
	return &Arbiter{capacityMB: capacityMB, active: map[string]*lease{}, preempting: map[string]bool{}}
}

func (a *Arbiter) usedLocked() int {
	u := 0
	for _, l := range a.active {
		u += l.memMB
	}
	return u
}

// Request asks for memMB of GPU memory at a priority class.
func (a *Arbiter) Request(id string, class Class, memMB int) (Decision, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if memMB <= 0 || memMB > a.capacityMB {
		return Decision{}, fmt.Errorf("lease %s: memory %d MB outside (0, %d]", id, memMB, a.capacityMB)
	}
	if _, ok := a.active[id]; ok {
		return Decision{Granted: true}, nil
	}
	for _, q := range a.queue {
		if q.id == id {
			return Decision{Queued: true}, nil
		}
	}
	a.seq++
	l := &lease{id: id, class: class, memMB: memMB, seq: a.seq}

	free := a.capacityMB - a.usedLocked()
	if memMB <= free && !a.higherOrEqualQueuedLocked(class) {
		a.active[id] = l
		return Decision{Granted: true}, nil
	}

	// Preempt lower-priority leases: lowest priority first, newest first.
	var victims []*lease
	for _, v := range a.active {
		if v.class > class && !a.preempting[v.id] {
			victims = append(victims, v)
		}
	}
	sort.Slice(victims, func(i, j int) bool {
		if victims[i].class != victims[j].class {
			return victims[i].class > victims[j].class
		}
		return victims[i].seq > victims[j].seq
	})
	pendingFree := free
	for id := range a.preempting {
		pendingFree += a.active[id].memMB
	}
	var preempt []string
	for _, v := range victims {
		if pendingFree >= memMB {
			break
		}
		pendingFree += v.memMB
		preempt = append(preempt, v.id)
	}
	if pendingFree >= memMB {
		for _, id := range preempt {
			a.preempting[id] = true
		}
	} else {
		preempt = nil
	}
	a.enqueueLocked(l)
	return Decision{Queued: true, Preempt: preempt}, nil
}

func (a *Arbiter) higherOrEqualQueuedLocked(c Class) bool {
	for _, q := range a.queue {
		if q.class <= c {
			return true
		}
	}
	return false
}

func (a *Arbiter) enqueueLocked(l *lease) {
	a.queue = append(a.queue, l)
	sort.SliceStable(a.queue, func(i, j int) bool {
		if a.queue[i].class != a.queue[j].class {
			return a.queue[i].class < a.queue[j].class
		}
		return a.queue[i].seq < a.queue[j].seq
	})
}

// Release frees a lease (or drops a queued request) and returns the ids
// granted from the queue as a result, in priority order.
func (a *Arbiter) Release(id string) []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.active, id)
	delete(a.preempting, id)
	for i, q := range a.queue {
		if q.id == id {
			a.queue = append(a.queue[:i], a.queue[i+1:]...)
			break
		}
	}
	var granted []string
	for len(a.queue) > 0 {
		head := a.queue[0]
		if head.memMB > a.capacityMB-a.usedLocked() {
			break // strict priority: nothing overtakes the head
		}
		a.active[head.id] = head
		a.queue = a.queue[1:]
		granted = append(granted, head.id)
	}
	return granted
}

// Stats reports used memory, active and queued counts.
func (a *Arbiter) Stats() (usedMB, active, queued int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.usedLocked(), len(a.active), len(a.queue)
}
