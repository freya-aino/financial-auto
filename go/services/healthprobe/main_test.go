package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRun(t *testing.T) {
	ok := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("fa_up 1\n")) }))
	defer ok.Close()
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { http.Error(w, "no", 503) }))
	defer bad.Close()
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { time.Sleep(300 * time.Millisecond) }))
	defer slow.Close()

	if err := run([]string{ok.URL}, time.Second); err != nil {
		t.Fatalf("healthy: %v", err)
	}
	if err := run([]string{ok.URL, bad.URL}, time.Second); err == nil {
		t.Fatal("503 must be unhealthy")
	}
	if err := run([]string{slow.URL}, 50*time.Millisecond); err == nil {
		t.Fatal("timeout must be unhealthy")
	}
	if err := run([]string{"http://127.0.0.1:1/metrics"}, time.Second); err == nil {
		t.Fatal("refused must be unhealthy")
	}
}
