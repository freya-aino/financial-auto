//! Identifier types: UUID v7 for entities, ULID for events and jobs (D-003).
use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Internal instrument identifier (spec 12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct InstrumentId(pub Uuid);

impl InstrumentId {
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Token used inside NATS subjects: 32 hex chars, no dashes (D-017).
    pub fn subject_token(&self) -> String {
        self.0.simple().to_string()
    }
}

impl Default for InstrumentId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for InstrumentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

macro_rules! ulid_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub ulid::Ulid);

        impl $name {
            pub fn new() -> Self {
                Self(ulid::Ulid::generate())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

ulid_id!(EventId, "Canonical event id (spec 10).");
ulid_id!(JobId, "Work-queue job id, also used as `Nats-Msg-Id` (spec 21).");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subject_token_has_no_dashes() {
        let t = InstrumentId::new().subject_token();
        assert_eq!(t.len(), 32);
        assert!(!t.contains('-'));
    }
}
