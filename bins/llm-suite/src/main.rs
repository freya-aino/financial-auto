//! `llm-suite`: the platform's LLM service (D-117).
//!
//! Two jobs, one endpoint:
//!
//! - **Embedder** (`embedder`): every `news.item.>` envelope is embedded once as it arrives and
//!   the vector is stored with the item's knowledge time, so the encoder's text expert reads
//!   stored vectors and never puts an API call on the model path. Without a configured endpoint it
//!   keeps running on the deterministic hashed fallback, which is what lets the text expert exist
//!   before anyone enters a key.
//! - **Agent** (`agent`): an agentic loop over the platform's own control API, with tools the
//!   operator approves, external MCP servers ([`mcp`], D-120), and conversations persisted in
//!   Postgres. It reads freely and can take the audited operator actions control-api already
//!   catalogues; it cannot
//!   flip trading mode, promote a snapshot or move money, and while the platform is in real-money
//!   mode every mutating tool call is refused.
//!
//! The service holds no HTTP surface of its own that the UI talks to: control-api proxies
//! `/api/llm/*` and `/ws/agent` to it, keeping spec 51's invariant that the frontend calls exactly
//! one API. What it does expose is `/metrics` (spec 60 §4) and `/health`.
mod agent;
mod embedder;
mod mcp;
mod tools;

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use fa_bus::Bus;
use fa_core::config::PlatformConfig;
use fa_core::secrets::Secrets;
use futures::StreamExt;

const SERVICE: &str = "llm-suite";

/// Restart window in which the durable consumer resumes where it left off.
const RESUME_WITHIN: Duration = Duration::from_secs(600);

#[derive(Debug, Parser)]
#[command(about = "news embeddings for the text expert + the operator agent (D-117)")]
struct Args {
    /// Platform config (spec 60).
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: PathBuf,
    /// Items held while waiting for the endpoint; over this the oldest is dropped and counted.
    #[arg(long, env = "FA_LLM_QUEUE_CAP", default_value_t = 4096)]
    queue_cap: usize,
    /// Run only the embedder (no agent API).
    #[arg(long)]
    embedder_only: bool,
    /// Bind address of this service's own API, which control-api proxies.
    #[arg(long, env = "FA_LLM_BIND", default_value = "0.0.0.0:7710")]
    bind: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fa_core::logging::init(SERVICE);
    fa_bus::metrics::serve_from_env(SERVICE, env!("CARGO_PKG_VERSION")); // spec 60 §4, D-056
    let args = Args::parse();
    let cfg = PlatformConfig::load(&args.config)?;
    let secrets = Secrets::load().unwrap_or_default();

    let vector_store = fa_hub::configure_vectors(&cfg);
    tracing::info!(graph = fa_hub::configure_graph(&cfg), "graph store");
    tracing::info!(qdrant = vector_store.as_deref().unwrap_or("off"), "vector store");

    let llm = fa_llm::LlmClient::from_env(&cfg.llm, &secrets.llm);
    match &llm {
        Some(c) => tracing::info!(
            base = c.base_url(),
            model = c.model(),
            embedding_model = c.embedding_model(),
            dim = c.embedding_dim(),
            key = c.has_key(),
            "llm endpoint configured"
        ),
        None => tracing::warn!(
            "no [llm] base_url: the agent will answer that it has no endpoint and news embeddings \
             use the deterministic hashed fallback (enter an endpoint in the UI to change this)"
        ),
    }
    let dim = llm.as_ref().map(|c| c.embedding_dim()).unwrap_or(cfg.llm.embedding_dim).max(1);

    let hub = fa_hub::HubClient::from_env();
    let bus = Bus::connect(&cfg.nats.url).await?;

    // The embedder's own Postgres pool: provenance per item and dedupe across restarts.
    let pg = match sqlx::postgres::PgPoolOptions::new()
        .max_connections(2)
        .acquire_timeout(Duration::from_secs(5))
        .connect(&cfg.postgres.url)
        .await
    {
        Ok(p) => Some(p),
        Err(e) => {
            tracing::warn!(error = %e, "postgres unavailable: embeddings will not be recorded and may repeat");
            None
        }
    };
    let store = embedder::Store::new(pg);
    let embedder = tokio::spawn(run_embedder(bus.clone(), hub, llm.clone(), dim, args.queue_cap, store));
    if args.embedder_only {
        tracing::info!("embedder only");
        return embedder.await?;
    }
    let api = tokio::spawn(agent::serve(args.bind.clone(), cfg.clone(), llm, bus));
    tokio::select! {
        r = embedder => r?,
        r = api => r?,
    }
}

/// Consumes `news.item.>` and keeps the vector store in step with it.
///
/// The consumer is durable, so a restart resumes rather than skipping the backlog, and each
/// message is acked when it has been queued — not when it has been embedded. That is deliberate:
/// an endpoint outage must not stall the TEXT stream for every other consumer, and an item whose
/// embedding was dropped is recoverable by re-embedding from the warehouse later, while a stalled
/// stream is not.
async fn run_embedder(
    bus: Bus,
    hub: fa_hub::HubClient,
    llm: Option<fa_llm::LlmClient>,
    dim: usize,
    cap: usize,
    store: embedder::Store,
) -> anyhow::Result<()> {
    use fa_bus::consume::LiveSpec;
    let filter = "news.item.>";
    let stream = LiveSpec::stream_of(filter).ok_or_else(|| anyhow::anyhow!("no stream holds {filter}"))?;
    let spec = LiveSpec::durable(&stream, &format!("{SERVICE}-news"), &[filter]).with_resume_within(RESUME_WITHIN);
    let mut sub = bus.subscribe_durable::<fa_proto::v1::Envelope>(spec).await?;
    let mut queue = embedder::Queue::new(cap);
    let mut ticker = tokio::time::interval(Duration::from_millis(embedder::LINGER_MS));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    tracing::info!(filter, dim, cap, "embedding news items");

    loop {
        let flush = tokio::select! {
            item = sub.next() => match item {
                Some((subject, env)) => {
                    if let Some(p) = embedder::pending_of(&subject, &env) {
                        queue.push(p);
                    }
                    queue.len() >= embedder::BATCH
                }
                None => {
                    tracing::warn!("news subscription ended; draining and stopping");
                    true
                }
            },
            _ = ticker.tick() => !queue.is_empty(),
        };
        if !flush {
            continue;
        }
        let batch = queue.take(embedder::BATCH);
        // Items embedded before a restart are skipped here rather than paid for again.
        let ids: Vec<String> = batch.iter().map(|p| p.doc_id.clone()).collect();
        let known = store.known(&ids).await;
        let batch: Vec<embedder::Pending> = batch.into_iter().filter(|p| !known.contains(&p.doc_id)).collect();
        if batch.is_empty() {
            continue;
        }
        match embedder::embed_and_store(&hub, llm.as_ref(), dim, &batch).await {
            Ok((n, provider)) if n > 0 => {
                let model = llm.as_ref().map(|c| c.embedding_model().to_owned()).unwrap_or_else(|| "hashed".into());
                store.record(&batch, provider, &model, dim).await;
                tracing::info!(n, provider = provider.as_str(), queued = queue.len(), "news vectors written")
            }
            Ok(_) => {}
            // The vector store refusing a write is worth a loud line but not a crash: the next
            // batch may well succeed, and the encoder reads whatever is there.
            Err(e) => tracing::error!(error = %e, n = batch.len(), "storing news vectors failed"),
        }
    }
}
