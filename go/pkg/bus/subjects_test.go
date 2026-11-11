package bus

import (
	"testing"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
)

// Expected strings mirror crates/fa-bus/src/subjects.rs tests.
func TestSubjectsMatchTaxonomy(t *testing.T) {
	inst := "01900000-0000-7000-8000-00000000AbCd"
	tok := "0190000000007000800000000000abcd"
	if got := SubjectToken(inst); got != tok {
		t.Fatalf("token = %s", got)
	}
	cases := map[string]string{
		MdTrade("sim", inst):                      "md.trade.sim." + tok,
		MdBar("1h", "lemon", inst):                "md.bar.1h.lemon." + tok,
		Heartbeat("ingestor"):                     "svc.heartbeat.ingestor",
		JobResult("evo.eval.decoder"):             "jobs.result.evo.eval.decoder",
		OrderEvent("paper"):                       "order.event.paper",
		Ctl("gpu", "acquire"):                     "ctl.gpu.acquire",
		LatentState("1m", "snap1"):                "latent.state.1m.snap1",
		MacroObs("fred", "CPIAUCSL"):              "macro.obs.fred.CPIAUCSL",
		LatentMembership("1m", tok):               "latent.membership.1m." + tok,
		LatentLifecycle("1h"):                     "latent.lifecycle.1h",
		LatentMcmc("1m", 2):                       "latent.mcmc.1m.2",
		LatentHier("1m", "1h"):                    "latent.hier.1m.1h",
		LatentCrossPrior("1d"):                    "latent.crossprior.1d",
		DeadLetter("LATENT", "latent.state.1m.x"): "dlq.LATENT.latent.state.1m.x",
		OpsAlert: "ops.alert",
	}
	for got, want := range cases {
		if got != want {
			t.Errorf("got %s, want %s", got, want)
		}
	}
}

func TestCounters(t *testing.T) {
	c := NewCounters()
	c.Add("events.heartbeat", 2)
	c.Add("events.heartbeat", 1)
	if c.Snapshot()["events.heartbeat"] != 3 {
		t.Fatal("counter mismatch")
	}
	hb := BuildHeartbeat("x", "i", "v", c)
	if hb.Counters["events.heartbeat"] != 3 || hb.Service != "x" {
		t.Fatal("heartbeat mismatch")
	}
}

func TestSubjectForRoutesPayloads(t *testing.T) {
	env := &fav1.Envelope{SourceId: "heartbeat", InstrumentId: "01900000-0000-7000-8000-000000000001",
		Payload: &fav1.Envelope_Trade{Trade: &fav1.Trade{Price: 1}}}
	s, err := SubjectFor(env, "sim")
	if err != nil || s != "md.trade.sim.01900000000070008000000000000001" {
		t.Fatalf("got %q %v", s, err)
	}
	env.Payload = &fav1.Envelope_News{News: &fav1.NewsItem{}}
	if s, _ := SubjectFor(env, "sim"); s != "news.item.heartbeat" {
		t.Fatalf("news routed to %q", s)
	}
	if _, err := SubjectFor(&fav1.Envelope{}, "x"); err == nil {
		t.Fatal("expected error for empty payload")
	}
}
