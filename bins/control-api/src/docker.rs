//! Minimal Docker Engine API client over a unix socket (spec 50 §4.1 service control, D-060).
//!
//! Hand-written HTTP/1.1 on a tokio `UnixStream`: one request per connection with
//! `Connection: close`, the reply read to EOF and framed by `Content-Length` or chunked
//! transfer-encoding. Only what the supervisor needs: list the containers of one compose project
//! (filtered by the `com.docker.compose.project` label), inspect, start, stop, restart.
//!
//! Socket resolution (`Docker::from_env`): `FA_DOCKER_SOCK` (a path or `unix://` URL), else
//! `DOCKER_HOST` when it is a `unix://` URL, else the first existing of `/var/run/docker.sock` and
//! `$XDG_RUNTIME_DIR/docker.sock` (rootless). A missing or refusing socket is
//! [`DockerError::Unreachable`], which the routes turn into `501 no_supervisor`.
use std::{collections::BTreeMap, fmt, path::PathBuf, time::Duration};

use serde::Deserialize;
use serde_json::Value;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
};

/// Label compose puts on every container of a project.
pub const PROJECT_LABEL: &str = "com.docker.compose.project";
/// Label naming the compose service a container belongs to.
pub const SERVICE_LABEL: &str = "com.docker.compose.service";
/// Default compose project (`name:` in `compose.yaml`).
pub const DEFAULT_PROJECT: &str = "financial-auto";
/// Grace period handed to `stop` / `restart` before the daemon kills the process.
pub const STOP_TIMEOUT_SECS: u32 = 10;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
/// Covers a stop (grace period + kill) with margin.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(45);

#[derive(Debug)]
pub enum DockerError {
    /// No socket, or the daemon does not answer: no supervisor available.
    Unreachable(String),
    /// The daemon answered with an error status (`404` unknown container, `500`, ...).
    Api { status: u16, message: String },
    /// Malformed HTTP or JSON from the daemon.
    Protocol(String),
}

impl fmt::Display for DockerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreachable(m) => write!(f, "docker unreachable: {m}"),
            Self::Api { status, message } => write!(f, "docker api {status}: {message}"),
            Self::Protocol(m) => write!(f, "docker protocol error: {m}"),
        }
    }
}

impl std::error::Error for DockerError {}

/// One row of `GET /containers/json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ContainerSummary {
    #[serde(rename = "Id")]
    pub id: String,
    #[serde(rename = "Names", default)]
    pub names: Vec<String>,
    #[serde(rename = "Image", default)]
    pub image: String,
    /// `created|running|paused|restarting|removing|exited|dead`.
    #[serde(rename = "State", default)]
    pub state: String,
    /// Human text, e.g. `Up 3 hours (healthy)`.
    #[serde(rename = "Status", default)]
    pub status: String,
    #[serde(rename = "Labels", default)]
    pub labels: BTreeMap<String, String>,
}

impl ContainerSummary {
    /// Container name without the leading slash.
    pub fn name(&self) -> &str {
        self.names.first().map_or(self.id.as_str(), |n| n.trim_start_matches('/'))
    }

    pub fn service(&self) -> Option<&str> {
        self.labels.get(SERVICE_LABEL).map(String::as_str)
    }
}

/// The fields of `GET /containers/{id}/json` the API exposes.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContainerDetail {
    pub id: String,
    pub name: String,
    pub image: String,
    pub state: String,
    pub running: bool,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub exit_code: Option<i64>,
    pub restart_count: i64,
    /// `starting|healthy|unhealthy`, or null without a healthcheck.
    pub health: Option<String>,
}

impl ContainerDetail {
    fn from_json(v: &Value) -> Result<Self, DockerError> {
        let id = v["Id"].as_str().ok_or_else(|| DockerError::Protocol("inspect: missing Id".into()))?;
        let st = &v["State"];
        let none_if_zero = |s: &str| (!s.is_empty() && !s.starts_with("0001-")).then(|| s.to_owned());
        Ok(Self {
            id: id.to_owned(),
            name: v["Name"].as_str().unwrap_or(id).trim_start_matches('/').to_owned(),
            image: v["Config"]["Image"].as_str().unwrap_or_default().to_owned(),
            state: st["Status"].as_str().unwrap_or("unknown").to_owned(),
            running: st["Running"].as_bool().unwrap_or(false),
            started_at: st["StartedAt"].as_str().and_then(none_if_zero),
            finished_at: st["FinishedAt"].as_str().and_then(none_if_zero),
            exit_code: st["ExitCode"].as_i64(),
            restart_count: v["RestartCount"].as_i64().unwrap_or(0),
            health: st["Health"]["Status"].as_str().map(str::to_owned),
        })
    }
}

/// Outcome of a start/stop/restart call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applied {
    /// 204: the daemon did it.
    Done,
    /// 304: the container was already in the requested state.
    AlreadyInState,
}

#[derive(Debug, Clone)]
pub struct Docker {
    socket: Option<PathBuf>,
    project: String,
}

impl Docker {
    /// Client for `socket` and compose project `project`.
    pub fn at(socket: impl Into<PathBuf>, project: impl Into<String>) -> Self {
        Self { socket: Some(socket.into()), project: project.into() }
    }

    /// Resolves the socket from the environment (see the module docs); `None` means no candidate exists.
    pub fn from_env() -> Self {
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let project = get("FA_COMPOSE_PROJECT").unwrap_or_else(|| DEFAULT_PROJECT.to_owned());
        match Self::resolve_socket(get("FA_DOCKER_SOCK"), get("DOCKER_HOST"), get("XDG_RUNTIME_DIR")) {
            Some(socket) => Self::at(socket, project),
            None => Self { socket: None, project },
        }
    }

    /// Socket path from the explicit setting, `DOCKER_HOST` (`unix://` only), then the defaults that exist.
    pub fn resolve_socket(
        fa_sock: Option<String>,
        docker_host: Option<String>,
        xdg_runtime: Option<String>,
    ) -> Option<PathBuf> {
        if let Some(s) = fa_sock {
            return Some(PathBuf::from(s.strip_prefix("unix://").unwrap_or(&s)));
        }
        if let Some(path) = docker_host.as_deref().and_then(|h| h.strip_prefix("unix://")) {
            return Some(PathBuf::from(path));
        }
        let mut candidates = vec![PathBuf::from("/var/run/docker.sock")];
        if let Some(dir) = xdg_runtime {
            candidates.push(PathBuf::from(dir).join("docker.sock"));
        }
        candidates.into_iter().find(|p| p.exists())
    }

    pub fn socket(&self) -> Option<&std::path::Path> {
        self.socket.as_deref()
    }

    pub fn project(&self) -> &str {
        &self.project
    }

    /// All containers (running or not) of the compose project.
    pub async fn list_containers(&self) -> Result<Vec<ContainerSummary>, DockerError> {
        let filters = serde_json::json!({ "label": [format!("{PROJECT_LABEL}={}", self.project)] }).to_string();
        let path = format!("/containers/json?all=1&filters={}", percent_encode(&filters));
        let (_, body) = self.request("GET", &path).await?;
        serde_json::from_slice(&body).map_err(|e| DockerError::Protocol(format!("containers/json: {e}")))
    }

    pub async fn inspect(&self, id: &str) -> Result<ContainerDetail, DockerError> {
        let (_, body) = self.request("GET", &format!("/containers/{}/json", percent_encode(id))).await?;
        let v: Value = serde_json::from_slice(&body).map_err(|e| DockerError::Protocol(format!("inspect: {e}")))?;
        ContainerDetail::from_json(&v)
    }

    pub async fn start(&self, id: &str) -> Result<Applied, DockerError> {
        self.action(id, "start", None).await
    }

    pub async fn stop(&self, id: &str, timeout_secs: u32) -> Result<Applied, DockerError> {
        self.action(id, "stop", Some(timeout_secs)).await
    }

    pub async fn restart(&self, id: &str, timeout_secs: u32) -> Result<Applied, DockerError> {
        self.action(id, "restart", Some(timeout_secs)).await
    }

    async fn action(&self, id: &str, verb: &str, timeout_secs: Option<u32>) -> Result<Applied, DockerError> {
        let mut path = format!("/containers/{}/{verb}", percent_encode(id));
        if let Some(t) = timeout_secs {
            path.push_str(&format!("?t={t}"));
        }
        let (status, _) = self.request("POST", &path).await?;
        Ok(if status == 304 { Applied::AlreadyInState } else { Applied::Done })
    }

    /// One request on a fresh connection; 2xx/304 pass through, anything else is `Api`.
    async fn request(&self, method: &str, path: &str) -> Result<(u16, Vec<u8>), DockerError> {
        let socket = self.socket.as_ref().ok_or_else(|| DockerError::Unreachable("no docker socket found".into()))?;
        let unreachable = |e: std::io::Error| DockerError::Unreachable(format!("{}: {e}", socket.display()));
        let mut stream = tokio::time::timeout(CONNECT_TIMEOUT, UnixStream::connect(socket))
            .await
            .map_err(|_| DockerError::Unreachable(format!("{}: connect timed out", socket.display())))?
            .map_err(unreachable)?;
        let req = format!("{method} {path} HTTP/1.1\r\nHost: docker\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
        let exchange = async {
            stream.write_all(req.as_bytes()).await?;
            let mut buf = Vec::with_capacity(4096);
            stream.read_to_end(&mut buf).await?;
            Ok::<_, std::io::Error>(buf)
        };
        let raw = tokio::time::timeout(REQUEST_TIMEOUT, exchange)
            .await
            .map_err(|_| DockerError::Protocol(format!("{method} {path}: timed out")))?
            .map_err(unreachable)?;
        let (status, body) = parse_response(&raw)?;
        if (200..300).contains(&status) || status == 304 {
            return Ok((status, body));
        }
        let message = serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|v| v["message"].as_str().map(str::to_owned))
            .unwrap_or_else(|| String::from_utf8_lossy(&body).trim().to_owned());
        Err(DockerError::Api { status, message })
    }
}

/// Splits a complete HTTP/1.1 reply into status and de-framed body.
pub fn parse_response(raw: &[u8]) -> Result<(u16, Vec<u8>), DockerError> {
    let hdr_end = find(raw, b"\r\n\r\n").ok_or_else(|| DockerError::Protocol("no header terminator".into()))?;
    let head = std::str::from_utf8(&raw[..hdr_end]).map_err(|_| DockerError::Protocol("non-utf8 headers".into()))?;
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let status: u16 = status_line
        .strip_prefix("HTTP/1.")
        .and_then(|s| s.get(2..5))
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| DockerError::Protocol(format!("bad status line {status_line:?}")))?;
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for line in lines {
        let Some((k, v)) = line.split_once(':') else { continue };
        let v = v.trim();
        if k.eq_ignore_ascii_case("content-length") {
            content_length = v.parse().ok();
        } else if k.eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        }
    }
    let rest = &raw[hdr_end + 4..];
    let body = if chunked {
        decode_chunked(rest)?
    } else if let Some(n) = content_length {
        rest.get(..n).ok_or_else(|| DockerError::Protocol("body shorter than content-length".into()))?.to_vec()
    } else {
        rest.to_vec()
    };
    Ok((status, body))
}

fn decode_chunked(mut rest: &[u8]) -> Result<Vec<u8>, DockerError> {
    let mut out = Vec::new();
    loop {
        let line_end = find(rest, b"\r\n").ok_or_else(|| DockerError::Protocol("truncated chunk size".into()))?;
        let size_str =
            std::str::from_utf8(&rest[..line_end]).map_err(|_| DockerError::Protocol("chunk size".into()))?;
        let size_str = size_str.split(';').next().unwrap_or_default().trim();
        let size = usize::from_str_radix(size_str, 16).map_err(|_| DockerError::Protocol("chunk size".into()))?;
        rest = &rest[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        let chunk = rest.get(..size).ok_or_else(|| DockerError::Protocol("truncated chunk".into()))?;
        out.extend_from_slice(chunk);
        rest = rest.get(size + 2..).ok_or_else(|| DockerError::Protocol("truncated chunk terminator".into()))?;
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// RFC 3986 percent-encoding of everything but unreserved characters.
pub fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod fake {
    //! A canned Docker daemon on a unix socket for tests: each `(method, path-prefix)` maps to a
    //! `(status, body)`; unknown requests answer 404 like the real daemon.
    use std::{path::PathBuf, sync::Arc};

    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::UnixListener,
        sync::Mutex,
    };

    pub struct FakeDaemon {
        pub socket: PathBuf,
        pub requests: Arc<Mutex<Vec<String>>>,
        _dir: tempfile::TempDir,
    }

    pub type Route = (&'static str, String, u16, String);

    impl FakeDaemon {
        /// `routes`: (method, path prefix, status, body). `chunked_lists` frames every 200 body
        /// as two chunks like the daemon does for `/containers/json`.
        pub fn spawn(routes: Vec<Route>, chunked: bool) -> Self {
            let dir = tempfile::Builder::new().prefix("fa-dock").tempdir_in(std::env::temp_dir()).unwrap();
            let socket = dir.path().join("d.sock");
            let listener = UnixListener::bind(&socket).unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let seen = requests.clone();
            tokio::spawn(async move {
                while let Ok((mut stream, _)) = listener.accept().await {
                    let mut buf = vec![0u8; 8192];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let line = req.lines().next().unwrap_or_default().to_owned();
                    seen.lock().await.push(line.clone());
                    let mut parts = line.split(' ');
                    let (m, p) = (parts.next().unwrap_or_default(), parts.next().unwrap_or_default());
                    let hit = routes.iter().find(|(rm, rp, _, _)| *rm == m && p.starts_with(rp.as_str()));
                    let (status, body) = match hit {
                        Some((_, _, s, b)) => (*s, b.clone()),
                        None => (404, r#"{"message":"no such route"}"#.to_owned()),
                    };
                    let reason = match status {
                        204 => "No Content",
                        304 => "Not Modified",
                        404 => "Not Found",
                        _ => "OK",
                    };
                    let reply = if status == 204 || status == 304 {
                        format!("HTTP/1.1 {status} {reason}\r\nServer: Docker/fake\r\n\r\n")
                    } else if chunked {
                        let (a, b) = body.split_at(body.len() / 2);
                        format!(
                            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{a}\r\n{:x}\r\n{b}\r\n0\r\n\r\n",
                            a.len(),
                            b.len()
                        )
                    } else {
                        format!(
                            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                    };
                    let _ = stream.write_all(reply.as_bytes()).await;
                    let _ = stream.shutdown().await;
                }
            });
            Self { socket, requests, _dir: dir }
        }
    }

    /// A `/containers/json` row of compose service `service` in `project`.
    pub fn summary(project: &str, service: &str, state: &str) -> serde_json::Value {
        serde_json::json!({
            "Id": format!("{service}{}", "0".repeat(60 - service.len().min(60))),
            "Names": [format!("/{project}-{service}-1")],
            "Image": "financial-auto/rust:dev",
            "State": state,
            "Status": if state == "running" { "Up 2 hours" } else { "Exited (0) 5 minutes ago" },
            "Labels": { super::PROJECT_LABEL: project, super::SERVICE_LABEL: service }
        })
    }

    /// A `/containers/{id}/json` document.
    pub fn inspect_doc(project: &str, service: &str, state: &str) -> serde_json::Value {
        serde_json::json!({
            "Id": format!("{service}{}", "0".repeat(60 - service.len().min(60))),
            "Name": format!("/{project}-{service}-1"),
            "RestartCount": 2,
            "Config": { "Image": "financial-auto/rust:dev" },
            "State": {
                "Status": state, "Running": state == "running", "ExitCode": 0,
                "StartedAt": "2026-11-10T08:00:00.000000000Z",
                "FinishedAt": "0001-01-01T00:00:00Z",
                "Health": { "Status": "healthy" }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{fake::*, *};

    #[test]
    fn parses_content_length_and_chunked_bodies() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\nOKtrailing";
        assert_eq!(parse_response(raw).unwrap(), (200, b"OK".to_vec()));
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\n[{\"\r\n2;ext\r\n}]\r\n0\r\n\r\n";
        assert_eq!(parse_response(raw).unwrap(), (200, b"[{\"}]".to_vec()));
        let raw = b"HTTP/1.1 304 Not Modified\r\nServer: Docker/29.8.2 (linux)\r\n\r\n";
        assert_eq!(parse_response(raw).unwrap(), (304, Vec::new()));
        assert!(matches!(parse_response(b"garbage"), Err(DockerError::Protocol(_))));
        assert!(matches!(
            parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nshort"),
            Err(DockerError::Protocol(_))
        ));
        assert!(matches!(
            parse_response(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nab"),
            Err(DockerError::Protocol(_))
        ));
    }

    #[test]
    fn percent_encodes_filters() {
        assert_eq!(percent_encode(r#"{"a":["b=c"]}"#), "%7B%22a%22%3A%5B%22b%3Dc%22%5D%7D");
        assert_eq!(percent_encode("abc-1_2.3~"), "abc-1_2.3~");
    }

    #[test]
    fn socket_resolution_order() {
        let dir = tempfile::tempdir().unwrap();
        let xdg = dir.path().join("docker.sock");
        std::fs::write(&xdg, b"").unwrap();
        let x = || Some(dir.path().to_string_lossy().into_owned());
        assert_eq!(
            Docker::resolve_socket(Some("unix:///tmp/a.sock".into()), Some("tcp://1.2.3.4:2375".into()), x()),
            Some(PathBuf::from("/tmp/a.sock"))
        );
        assert_eq!(Docker::resolve_socket(Some("/tmp/b.sock".into()), None, None), Some(PathBuf::from("/tmp/b.sock")));
        assert_eq!(
            Docker::resolve_socket(None, Some("unix:///run/user/1000/docker.sock".into()), x()),
            Some(PathBuf::from("/run/user/1000/docker.sock"))
        );
        // tcp DOCKER_HOST is ignored; defaults: /var/run/docker.sock when it exists, else $XDG_RUNTIME_DIR/docker.sock
        let got = Docker::resolve_socket(None, Some("tcp://1.2.3.4:2375".into()), x()).unwrap();
        assert!(got.as_os_str() == "/var/run/docker.sock" || got == xdg, "{got:?}");
        assert_eq!(
            Docker::resolve_socket(None, None, Some("/nonexistent-xdg".into())).is_some(),
            PathBuf::from("/var/run/docker.sock").exists()
        );
    }

    #[tokio::test]
    async fn lists_only_the_project_and_parses_chunked_replies() {
        let rows = serde_json::json!([
            summary("financial-auto", "nats", "running"),
            summary("financial-auto", "evo-worker", "exited")
        ]);
        let d = FakeDaemon::spawn(vec![("GET", "/containers/json?".into(), 200, rows.to_string())], true);
        let docker = Docker::at(&d.socket, "financial-auto");
        let list = docker.list_containers().await.unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].name(), "financial-auto-nats-1");
        assert_eq!(list[0].service(), Some("nats"));
        assert_eq!(list[1].state, "exited");
        let req = d.requests.lock().await[0].clone();
        assert_eq!(
            req,
            format!(
                "GET /containers/json?all=1&filters={} HTTP/1.1",
                percent_encode(r#"{"label":["com.docker.compose.project=financial-auto"]}"#)
            )
        );
    }

    #[tokio::test]
    async fn inspect_start_stop_and_errors() {
        let doc = inspect_doc("financial-auto", "broker-gateway", "running").to_string();
        let d = FakeDaemon::spawn(
            vec![
                ("GET", "/containers/abc/json".into(), 200, doc),
                ("POST", "/containers/abc/start".into(), 304, String::new()),
                ("POST", "/containers/abc/stop?t=10".into(), 204, String::new()),
                ("POST", "/containers/abc/restart?t=10".into(), 204, String::new()),
                ("POST", "/containers/boom/start".into(), 500, r#"{"message":"kaboom"}"#.into()),
            ],
            false,
        );
        let docker = Docker::at(&d.socket, "financial-auto");
        let c = docker.inspect("abc").await.unwrap();
        assert_eq!(
            (c.name.as_str(), c.state.as_str(), c.running, c.restart_count),
            ("financial-auto-broker-gateway-1", "running", true, 2)
        );
        assert_eq!(c.health.as_deref(), Some("healthy"));
        assert_eq!(c.started_at.as_deref(), Some("2026-11-10T08:00:00.000000000Z"));
        assert_eq!(c.finished_at, None, "zero time is null");
        assert_eq!(docker.start("abc").await.unwrap(), Applied::AlreadyInState);
        assert_eq!(docker.stop("abc", 10).await.unwrap(), Applied::Done);
        assert_eq!(docker.restart("abc", 10).await.unwrap(), Applied::Done);
        match docker.start("boom").await {
            Err(DockerError::Api { status: 500, message }) => assert_eq!(message, "kaboom"),
            other => panic!("{other:?}"),
        }
        match docker.inspect("missing").await {
            Err(DockerError::Api { status: 404, .. }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_socket_is_unreachable() {
        let docker = Docker::at("/nonexistent/docker.sock", "p");
        assert!(matches!(docker.inspect("x").await, Err(DockerError::Unreachable(_))));
        let none = Docker { socket: None, project: "p".into() };
        assert!(matches!(none.list_containers().await, Err(DockerError::Unreachable(_))));
    }
}
