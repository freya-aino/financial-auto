// Command healthprobe is the compose healthcheck of the Go service images (spec 60 §6.4/§8).
// The images are distroless (no shell, curl or wget), so compose runs
//
//	healthcheck: { test: ["CMD", "healthprobe", "http://127.0.0.1:9100/metrics"] }
//
// It GETs each URL argument (default the metrics endpoint) and exits 0 only when every one answers
// 2xx within the timeout. Answering /metrics proves the process is up and its HTTP loop is not hung.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

func probe(ctx context.Context, url string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(res.Body, 1<<20))
	if res.StatusCode/100 != 2 {
		return fmt.Errorf("%s: %s", url, res.Status)
	}
	return nil
}

func run(args []string, timeout time.Duration) error {
	if len(args) == 0 {
		args = []string{"http://127.0.0.1:9100/metrics"}
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	for _, u := range args {
		if err := probe(ctx, u); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	timeout := flag.Duration("timeout", 3*time.Second, "overall timeout")
	flag.Parse()
	if err := run(flag.Args(), *timeout); err != nil {
		fmt.Fprintln(os.Stderr, "unhealthy:", err)
		os.Exit(1)
	}
}
