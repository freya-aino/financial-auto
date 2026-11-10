//! Sends one `ctl.strategy.<verb>` request and prints the reply.
//!
//! ```sh
//! cargo run -p strategy-runtime --example ctl -- pause strategy_id=baseline-momentum-1s
//! cargo run -p strategy-runtime --example ctl -- resume strategy_id=baseline-momentum-1s
//! cargo run -p strategy-runtime --example ctl -- reload path=crates/fa-strategies/baselines.toml
//! cargo run -p strategy-runtime --example ctl -- reload artifact_key=strategies/strategies.3.toml
//! cargo run -p strategy-runtime --example ctl -- reload artifact_key=latest
//! cargo run -p strategy-runtime --example ctl -- status          # JSON: population, stats, primitive sources
//! ```
use std::time::Duration;

use bytes::Bytes;
use fa_bus::{Bus, subjects};
use fa_proto::{
    Message,
    v1::{CtlReply, CtlRequest},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let verb = args.next().ok_or_else(|| anyhow::anyhow!("usage: ctl <reload|pause|resume|status> [key=value...]"))?;
    let req = CtlRequest {
        verb: verb.clone(),
        args: args.filter_map(|kv| kv.split_once('=').map(|(k, v)| (k.to_owned(), v.to_owned()))).collect(),
    };
    let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
    let bus = Bus::connect(&url).await?;
    let fut = bus.client().request(subjects::ctl("strategy", &verb), Bytes::from(req.encode_to_vec()));
    let msg = tokio::time::timeout(Duration::from_secs(10), fut).await??;
    let reply = CtlReply::decode(msg.payload.as_ref())?;
    println!("ok={} message={}", reply.ok, reply.message);
    Ok(())
}
