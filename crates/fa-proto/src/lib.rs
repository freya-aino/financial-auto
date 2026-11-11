//! Prost-generated types for the `fa.v1` protobuf contracts (spec 21).

#[allow(clippy::all, clippy::pedantic)]
pub mod v1 {
    include!(concat!(env!("OUT_DIR"), "/fa.v1.rs"));
}

pub use prost::{self, Message};

/// The protobuf package this build is generated from (spec 21 §6: breaking changes ship as a
/// new major package side by side; `fa_bus::contract` rejects majors a consumer cannot decode).
pub const PACKAGE: &str = "fa.v1";
/// The major version of [`PACKAGE`].
pub const MAJOR: u32 = 1;
pub use prost_types::Timestamp;

/// Converts a chrono-free `SystemTime` into a protobuf timestamp.
pub fn ts(t: std::time::SystemTime) -> Timestamp {
    Timestamp::from(t)
}

/// Current wall-clock time as a protobuf timestamp.
pub fn now() -> Timestamp {
    ts(std::time::SystemTime::now())
}
