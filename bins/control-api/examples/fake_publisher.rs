//! Dev helper: publishes fake heartbeats and trades so the UI has something to show
//! before the ingestor and model-runtime exist.
//!
//! cargo run -p control-api --example fake_publisher -- nats://127.0.0.1:4222
use std::time::Duration;

use fa_core::InstrumentId;
use fa_proto::{
    Message, now,
    v1::{Envelope, Heartbeat, Trade, envelope::Payload},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = std::env::args().nth(1).unwrap_or_else(|| "nats://127.0.0.1:4222".into());
    let client = fa_bus::async_nats::connect(&url).await?;
    let inst = InstrumentId::new();
    let mut n = 0u64;
    loop {
        n += 1;
        let env = Envelope {
            event_id: fa_core::EventId::new().to_string(),
            source_id: "fake".into(),
            instrument_id: inst.to_string(),
            event_time: Some(now()),
            publish_time: Some(now()),
            ingest_time: Some(now()),
            schema_version: 1,
            payload: Some(Payload::Trade(Trade { price: 100.0 + (n as f64).sin(), size: 1.0, ..Default::default() })),
            ..Default::default()
        };
        client.publish(fa_bus::subjects::md_trade("sim", &inst), env.encode_to_vec().into()).await?;
        let hb = Heartbeat {
            service: "fake-publisher".into(),
            instance_id: "local".into(),
            at: Some(now()),
            version: env!("CARGO_PKG_VERSION").into(),
            counters: [("trades_published".to_owned(), n as f64)].into(),
        };
        client.publish(fa_bus::subjects::heartbeat("fake-publisher"), hb.encode_to_vec().into()).await?;
        client.flush().await?;
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
