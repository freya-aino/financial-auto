//! Prost-generated types for the `fa.v1` protobuf contracts (spec 21).

#[allow(clippy::all, clippy::pedantic)]
pub mod v1 {
    include!(concat!(env!("OUT_DIR"), "/fa.v1.rs"));
}

pub use prost::{self, Message};
pub use prost_types::Timestamp;

/// Converts a chrono-free `SystemTime` into a protobuf timestamp.
pub fn ts(t: std::time::SystemTime) -> Timestamp {
    Timestamp::from(t)
}

/// Current wall-clock time as a protobuf timestamp.
pub fn now() -> Timestamp {
    ts(std::time::SystemTime::now())
}
