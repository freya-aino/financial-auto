//! NATS JetStream access: subject taxonomy, typed protobuf pub/sub and the JetStream
//! stream table (spec 21).

pub mod streams;
pub mod subjects;

use bytes::Bytes;
use fa_proto::Message;
use futures::{Stream, StreamExt};

pub use async_nats;

#[derive(Debug, thiserror::Error)]
pub enum BusError {
    #[error("connect: {0}")]
    Connect(#[from] async_nats::ConnectError),
    #[error("publish: {0}")]
    Publish(#[from] async_nats::PublishError),
    #[error("subscribe: {0}")]
    Subscribe(#[from] async_nats::SubscribeError),
    #[error("decode: {0}")]
    Decode(#[from] fa_proto::prost::DecodeError),
}

/// Thin typed wrapper around a NATS client.
#[derive(Clone)]
pub struct Bus {
    client: async_nats::Client,
}

impl Bus {
    pub async fn connect(url: &str) -> Result<Self, BusError> {
        let client = async_nats::connect(url).await?;
        Ok(Self { client })
    }

    pub fn client(&self) -> &async_nats::Client {
        &self.client
    }

    pub fn jetstream(&self) -> async_nats::jetstream::Context {
        async_nats::jetstream::new(self.client.clone())
    }

    /// Publishes a protobuf message on a core NATS subject.
    pub async fn publish<M: Message>(&self, subject: String, msg: &M) -> Result<(), BusError> {
        self.client.publish(subject, Bytes::from(msg.encode_to_vec())).await?;
        Ok(())
    }

    /// Subscribes to a subject pattern and decodes each message as `M`.
    /// Messages that fail to decode are logged and skipped.
    pub async fn subscribe<M: Message + Default>(
        &self,
        subject: String,
    ) -> Result<impl Stream<Item = (String, M)> + Unpin + use<M>, BusError> {
        let sub = self.client.subscribe(subject).await?;
        // `future::ready` keeps the stream `Unpin`, so callers can use `.next()` directly.
        Ok(sub.filter_map(|m| {
            futures::future::ready(match M::decode(m.payload.as_ref()) {
                Ok(v) => Some((m.subject.to_string(), v)),
                Err(e) => {
                    tracing::warn!(subject = %m.subject, error = %e, "dropping undecodable message");
                    None
                }
            })
        }))
    }

    /// Raw subscription (used by the control-api bus tap).
    pub async fn subscribe_raw(&self, subject: String) -> Result<async_nats::Subscriber, BusError> {
        Ok(self.client.subscribe(subject).await?)
    }
}
