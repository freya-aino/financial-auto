package catalog

import "testing"

func TestRepoCatalogIsValid(t *testing.T) {
	c, err := Load("../../../config/sources.yaml")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(c.Sources) < 70 {
		t.Fatalf("expected the full catalog, got %d sources", len(c.Sources))
	}
	for _, id := range []string{"yahoo", "lemon", "oanda", "hackernews", "google_news"} {
		s, ok := c.ByID(id)
		if !ok {
			t.Fatalf("first-wave source %q missing", id)
		}
		if s.Status != "stub" {
			t.Errorf("%s: status = %q, want stub", id, s.Status)
		}
	}
}

func TestValidateRejectsDuplicatesAndBadStatus(t *testing.T) {
	_, err := Parse([]byte(`
version: 1
sources:
  - {id: a, name: A, status: planned, auth: none}
  - {id: a, name: A2, status: bogus, auth: none}
`))
	if err == nil {
		t.Fatal("expected validation error")
	}
}
