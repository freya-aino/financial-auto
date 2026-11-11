package bus

import (
	"io"
	"log/slog"
	"os"
	"strings"
)

// NewLogger returns the service logger (spec 60 §4): `FA_LOG_FORMAT=json` selects slog's
// JSON handler (one object per line), anything else the text handler. Output goes to stderr
// and is kept by Docker's json-file log driver (`docker compose logs`). `FA_LOG_LEVEL`
// (`debug|info|warn|error`, default info) sets the level. Every record carries `svc`.
func NewLogger(service string) *slog.Logger {
	return newLogger(os.Stderr, service, os.Getenv("FA_LOG_FORMAT"), os.Getenv("FA_LOG_LEVEL"))
}

func newLogger(w io.Writer, service, format, level string) *slog.Logger {
	var lv slog.Level
	if err := lv.UnmarshalText([]byte(strings.TrimSpace(level))); err != nil || strings.TrimSpace(level) == "" {
		lv = slog.LevelInfo
	}
	opts := &slog.HandlerOptions{Level: lv}
	var h slog.Handler
	if strings.EqualFold(strings.TrimSpace(format), "json") {
		h = slog.NewJSONHandler(w, opts)
	} else {
		h = slog.NewTextHandler(w, opts)
	}
	return slog.New(h).With("svc", service)
}
