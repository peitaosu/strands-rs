//! Snapshot types for versioned agent state capture.

use chrono::{DateTime, Utc};
use serde::de::Error as _;
use serde::{Deserialize, Serialize};

/// Schema version for snapshot envelopes.
pub const SNAPSHOT_SCHEMA_VERSION: &str = "1.0";

/// Preset field sets for snapshot scoping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotPreset {
    /// Conversation messages only.
    Messages,
    /// Messages plus agent state.
    MessagesAndState,
    /// Everything the agent can restore.
    Full,
}

/// A versioned snapshot envelope for agent state.
///
/// Mirrors the Python `strands.types._snapshot.Snapshot`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    /// The agent or orchestrator id the snapshot belongs to.
    pub scope: String,
    #[serde(rename = "schemaVersion")]
    pub schema_version: String,
    /// The captured state payload.
    pub data: serde_json::Value,
    /// Application-specific companion data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_data: Option<serde_json::Value>,
    /// When the snapshot was created.
    pub created_at: DateTime<Utc>,
}

impl Snapshot {
    /// Creates a new snapshot envelope with the current schema version.
    pub fn new(scope: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            scope: scope.into(),
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            data,
            app_data: None,
            created_at: Utc::now(),
        }
    }

    /// Serializes the snapshot envelope to JSON bytes.
    pub fn to_json(&self) -> serde_json::Result<Vec<u8>> {
        serde_json::to_vec(self)
    }

    /// Parses a snapshot envelope from JSON bytes.
    pub fn from_json(bytes: &[u8]) -> serde_json::Result<Self> {
        let snapshot: Self = serde_json::from_slice(bytes)?;
        if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION {
            return Err(serde_json::Error::custom(format!(
                "unsupported snapshot schema version: {}",
                snapshot.schema_version
            )));
        }
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_roundtrip() {
        let snapshot = Snapshot::new("agent-1", serde_json::json!({"messages": []}));
        let bytes = snapshot.to_json().unwrap();

        let restored = Snapshot::from_json(&bytes).unwrap();
        assert_eq!(restored.scope, "agent-1");
        assert_eq!(restored.schema_version, "1.0");
        assert_eq!(restored.data, serde_json::json!({"messages": []}));
    }

    #[test]
    fn test_snapshot_rejects_unknown_version() {
        let bytes = br#"{"scope":"a","schemaVersion":"9.9","data":{},"created_at":"2026-01-01T00:00:00Z"}"#;
        assert!(Snapshot::from_json(bytes).is_err());
    }
}
