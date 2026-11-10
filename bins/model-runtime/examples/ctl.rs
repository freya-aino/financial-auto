//! Sends a `ctl.model.<verb>` request and prints the reply (operator / smoke-test helper).
//!
//! ```sh
//! cargo run -p model-runtime --example ctl -- shadow timescale=1m snapshot_id=stat-v1-1m-g0-1794341952
//! cargo run -p model-runtime --example ctl -- shadow timescale=1m snapshot_id=none
//! cargo run -p model-runtime --example ctl -- reload timescale=1m snapshot_id=stat-v1-1m-g0-1794341952
//! cargo run -p model-runtime --example ctl -- adapter timescale=1m
//! ```
use std::time::Duration;

use fa_proto::{
    Message,
    v1::{CtlReply, CtlRequest},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let verb = args.next().ok_or_else(|| anyhow::anyhow!("usage: ctl <reload|shadow|adapter> key=value…"))?;
    let kv = args.filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_owned(), v.to_owned()))).collect();
    let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
    let client = async_nats::connect(url).await?;
    let req = CtlRequest { verb: verb.clone(), args: kv };
    let msg = tokio::time::timeout(
        Duration::from_secs(30),
        client.request(format!("ctl.model.{verb}"), req.encode_to_vec().into()),
    )
    .await??;
    let reply = CtlReply::decode(msg.payload.as_ref())?;
    println!("ok={} {}", reply.ok, reply.message);
    Ok(())
}
