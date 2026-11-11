# fa-metrics

Dependency-free Prometheus text exporter shared by every Rust service (spec 60 §4/§6.5, D-056).

- Process-wide registry: counters, gauges, histograms with fixed buckets, labels; `render()` emits
  the Prometheus text format (`text/plain; version=0.0.4`).
- `serve(bind)` / `serve_from_env(service, version)`: `GET /metrics` on a small axum router.
  `FA_METRICS_BIND` overrides the bind (`off` disables); the host default is `127.0.0.1:91xx` per
  service (`default_bind`), compose sets `0.0.0.0:9100` and publishes the `91xx` port on the host.
- `heartbeat::record` mirrors a service's heartbeat counters into `fa_heartbeat_counter{service,key}`;
  `fa_bus::Bus::publish` calls it for every `svc.heartbeat.*` publish, so nothing is instrumented twice.
- `process`: uptime, start time, RSS/VSZ from `/proc/self/statm`, `fa_build_info{service,version}`.
- `http::layer()`: tower layer recording `fa_http_requests_total{method,path,status}` and
  `fa_http_request_duration_seconds{method,path}` (path = matched axum route, else the raw path with id-like segments collapsed to `{id}`).

Re-exported as `fa_bus::metrics` so services need no extra dependency.

## Governing specs
- [Ops reference (ex spec 60)](../../docker/README.md#reference-spec-60-ops-retired-2026-11-11) §4, §6.5
