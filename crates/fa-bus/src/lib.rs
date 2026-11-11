//! NATS JetStream access: subject taxonomy, typed protobuf pub/sub and the JetStream
//! stream table (spec 21).

pub mod consume;
pub mod contract;
pub mod dlq;
pub mod ingest;
pub mod jobs;
pub mod replay;
pub mod streams;
pub mod subjects;
pub mod trace;

use bytes::Bytes;
use fa_proto::Message;
use futures::{Stream, StreamExt};

pub use async_nats;
/// Prometheus exporter shared by every service (D-056); re-exported so binaries need no extra dependency.
pub use fa_metrics as metrics;

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

    /// Publishes a protobuf message on a core NATS subject. Heartbeats (`svc.heartbeat.*`) are
    /// additionally mirrored into `fa_heartbeat_counter{service,key}` gauges (D-056).
    pub async fn publish<M: Message>(&self, subject: String, msg: &M) -> Result<(), BusError> {
        let bytes = msg.encode_to_vec();
        if subject.starts_with(subjects::HEARTBEAT_PREFIX) {
            mirror_heartbeat(&bytes);
        }
        if trace::traced(&subject) {
            // `traceparent` continuing the task's trace, or a new one (spec 60 §6.5)
            let h = trace::headers(&trace::outgoing());
            self.client.publish_with_headers(subject, h, Bytes::from(bytes)).await?;
        } else {
            self.client.publish(subject, Bytes::from(bytes)).await?;
        }
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
            log_consume(&m);
            if !contract::accept(&m.subject, m.headers.as_ref()) {
                return futures::future::ready(None);
            }
            futures::future::ready(match M::decode(m.payload.as_ref()) {
                Ok(v) => Some((m.subject.to_string(), v)),
                Err(e) => {
                    tracing::warn!(subject = %m.subject, error = %e, "dropping undecodable message");
                    None
                }
            })
        }))
    }

    /// Like [`subscribe`](Self::subscribe), but also yields the message's trace context, so
    /// the consumer can run its handler under [`trace::scope`] and continue the trace.
    pub async fn subscribe_traced<M: Message + Default>(
        &self,
        subject: String,
    ) -> Result<impl Stream<Item = (String, M, Option<trace::TraceContext>)> + Unpin + use<M>, BusError> {
        let sub = self.client.subscribe(subject).await?;
        Ok(sub.filter_map(|m| {
            let ctx = log_consume(&m);
            if !contract::accept(&m.subject, m.headers.as_ref()) {
                return futures::future::ready(None);
            }
            futures::future::ready(match M::decode(m.payload.as_ref()) {
                Ok(v) => Some((m.subject.to_string(), v, ctx)),
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

/// Logs a consumed message's trace id at `debug` (spec 60 §6.5) and returns its context.
fn log_consume(m: &async_nats::Message) -> Option<trace::TraceContext> {
    let ctx = trace::from_headers(m.headers.as_ref());
    if let Some(c) = &ctx {
        tracing::debug!(subject = %m.subject, trace_id = %c.trace_id_hex(), "consume");
    }
    ctx
}

/// Decodes an encoded `Heartbeat` and records its counters as metrics gauges.
fn mirror_heartbeat(bytes: &[u8]) {
    if let Ok(hb) = fa_proto::v1::Heartbeat::decode(bytes) {
        fa_metrics::heartbeat::record(&hb.service, hb.counters.iter().map(|(k, v)| (k.as_str(), *v)));
    }
}

#[cfg(test)]
mod tests {
    use fa_proto::Message;

    #[test]
    fn heartbeat_publish_mirrors_counters_into_metrics() {
        let hb = fa_proto::v1::Heartbeat {
            service: "unit-svc".into(),
            counters: [("x.count".to_owned(), 7.0)].into_iter().collect(),
            ..Default::default()
        };
        super::mirror_heartbeat(&hb.encode_to_vec());
        let out = fa_metrics::render();
        assert!(out.contains("fa_heartbeat_counter{service=\"unit-svc\",key=\"x.count\"} 7\n"), "{out}");
        assert!(super::subjects::heartbeat("unit-svc").starts_with(super::subjects::HEARTBEAT_PREFIX));
    }
}
