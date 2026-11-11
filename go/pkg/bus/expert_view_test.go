package bus

import (
	"os"
	"testing"

	fav1 "github.com/fey/financial-auto/go/pkg/bus/pb/fa/v1"
	"google.golang.org/protobuf/proto"
)

// Spec 32 §9.3: the ExpertView written by model-runtime (Rust, bins/model-runtime/src/expert_view.rs
// test fixture) decodes in Go with every §4.2 field, and re-encodes to an equal message.
func TestExpertViewRoundTripsFromRust(t *testing.T) {
	b, err := os.ReadFile("testdata/expert_view.bin")
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}
	var v fav1.ExpertView
	if err := proto.Unmarshal(b, &v); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if v.SnapshotId != "s" || v.Timescale != "1m" || v.InstrumentSet != "tok" || v.T == nil {
		t.Fatalf("header: %+v", &v)
	}
	if len(v.Experts) != 2 || len(v.ExpertMask) != 2 || v.McmcState == nil || len(v.McmcState.RegionIds) != 2 {
		t.Fatalf("experts / mask / mcmc: %+v", &v)
	}
	e := v.Experts[1]
	if e.ExpertId != 1 || len(e.Mu) != 3 || len(e.Logvar) != 3 || e.Gate <= 0 || e.Region == nil || e.NextStep == nil || len(e.Codes) != 1 {
		t.Fatalf("expert 1: %+v", e)
	}
	if !v.HasPriorScore || v.PriorScore != 1.5 || v.EncoderFamily != "hier_vae_flow" {
		t.Fatalf("prior: %+v", &v)
	}
	again, err := proto.Marshal(&v)
	if err != nil {
		t.Fatal(err)
	}
	var w fav1.ExpertView
	if err := proto.Unmarshal(again, &w); err != nil || !proto.Equal(&v, &w) {
		t.Fatalf("re-encode differs: %v", err)
	}
	if LatentExpert("1m", "tok") != "latent.expert.1m.tok" || LatentExpertHealth != "latent.expert.health" {
		t.Fatal("subjects")
	}
}
