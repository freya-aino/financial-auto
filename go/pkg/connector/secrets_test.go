package connector

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSecretsFileAndEnvPrecedence(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "secrets.toml")
	os.WriteFile(p, []byte(`# comment
[garage]
access_key_id = "GK1"
secret_access_key = "s3cr3t" # trailing comment

[sources.lemon]
api_key = "lemon-key"

[sources.oanda]
api_key = "oanda-key"
account_id = '101-004-1'
`), 0o600)
	s, err := LoadSecrets(p)
	if err != nil {
		t.Fatal(err)
	}
	// The developer's shell may carry real overrides; neutralize them.
	for _, k := range []string{"FA_GARAGE_ACCESS_KEY_ID", "FA_GARAGE_SECRET_ACCESS_KEY", "FA_LEMON_API_KEY", "FA_OANDA_API_KEY", "FA_OANDA_ACCOUNT_ID"} {
		t.Setenv(k, "")
	}
	if s.Get("garage", "access_key_id") != "GK1" || s.Get("garage", "secret_access_key") != "s3cr3t" {
		t.Fatalf("garage: %+v", s.m)
	}
	if got := s.ForSource("oanda"); got["api_key"] != "oanda-key" || got["account_id"] != "101-004-1" {
		t.Fatalf("oanda: %v", got)
	}
	t.Setenv("FA_LEMON_API_KEY", "env-key")
	if got := s.ForSource("lemon")["api_key"]; got != "env-key" {
		t.Fatalf("env must override file, got %q", got)
	}
	if s.Get("sources.lemon", "api_key") != "env-key" {
		t.Fatal("Get must honour env override")
	}
	cfg := SourceConfig{Secrets: s.ForSource("oanda")}
	cfg.Source.ID = "oanda"
	if cfg.Secret("api_key") != "oanda-key" {
		t.Fatal("Secret from map")
	}
	t.Setenv("FA_OANDA_ACCOUNT_ID", "env-acct")
	if (SourceConfig{Source: cfg.Source}).Secret("account_id") != "env-acct" {
		t.Fatal("Secret from env")
	}
	if EnvName("sources.google_news", "api_key") != "FA_GOOGLE_NEWS_API_KEY" {
		t.Fatal(EnvName("sources.google_news", "api_key"))
	}
	// Missing file is fine; malformed is not.
	if s2, err := LoadSecrets(filepath.Join(dir, "nope.toml")); err != nil || s2.Get("garage", "x") != "" {
		t.Fatalf("missing file: %v", err)
	}
	os.WriteFile(p, []byte("[x]\nnot a pair\n"), 0o600)
	if _, err := LoadSecrets(p); err == nil {
		t.Fatal("expected parse error")
	}
}

func TestSeenEvicts(t *testing.T) {
	s := NewSeen(3)
	for _, id := range []string{"a", "b", "c"} {
		if !s.Add(id) {
			t.Fatal("new id reported as seen")
		}
	}
	if s.Add("a") {
		t.Fatal("dup reported as new")
	}
	s.Add("d")
	if s.Has("a") || !s.Has("d") || s.Len() != 3 {
		t.Fatalf("eviction failed: len=%d", s.Len())
	}
}
