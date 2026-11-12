//! Snapshot-based session persistence over the `Storage` protocol.
//!
//! Mirrors the Python `SnapshotSessionManager`: instead of per-message
//! repository records, versioned `Snapshot` envelopes are persisted under
//! `<session_id>/scopes/agent/<agent_id>/snapshots/`.

use std::sync::Arc;

use async_trait::async_trait;

use crate::agent::Agent;
use crate::hooks::{HookEvent, HookProvider};
use crate::storage::NamespacedStorage;
use crate::types::errors::{Result, StrandsError};
use crate::types::snapshot::Snapshot;

/// When a snapshot is persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveLatestStrategy {
    /// Persist after every appended message.
    Message,
    /// Persist after every completed invocation.
    Invocation,
    /// Persist only on explicit `save_snapshot` calls.
    Trigger,
}

/// Session manager persisting agent state as snapshot envelopes.
pub struct SnapshotSessionManager {
    storage: NamespacedStorage,
    session_id: String,
    agent_id: String,
    strategy: SaveLatestStrategy,
}

impl SnapshotSessionManager {
    /// Creates a manager rooted at `backend` for the given session and agent.
    pub fn new(
        backend: Arc<dyn crate::storage::Storage>,
        session_id: impl Into<String>,
        agent_id: impl Into<String>,
        strategy: SaveLatestStrategy,
    ) -> Self {
        let session_id = session_id.into();
        let agent_id = agent_id.into();
        let storage = NamespacedStorage::new(
            backend,
            &format!("{}/scopes/agent/{}", session_id, agent_id),
        );
        Self {
            storage,
            session_id,
            agent_id,
            strategy,
        }
    }

    fn latest_key(&self) -> String {
        "snapshots/snapshot_latest.json".to_string()
    }

    /// Persists the agent's current state as the latest snapshot.
    pub async fn save_snapshot(&self, agent: &Agent) -> Result<()> {
        let snapshot = agent.take_snapshot(&self.agent_id)?;
        let bytes = snapshot.to_json().map_err(|e| {
            StrandsError::SnapshotException {
                message: format!("failed to serialize snapshot: {}", e),
            }
        })?;
        self.storage
            .write(&self.latest_key(), &bytes)
            .await
            .map_err(|e| StrandsError::StorageError {
                message: format!("failed to write snapshot: {}", e),
            })
    }

    /// Restores the agent from the latest snapshot, if one exists.
    pub async fn load_latest(&self, agent: &mut Agent) -> Result<Option<Snapshot>> {
        let bytes = self
            .storage
            .read(&self.latest_key())
            .await
            .map_err(|e| StrandsError::StorageError {
                message: format!("failed to read snapshot: {}", e),
            })?;

        let Some(bytes) = bytes else {
            return Ok(None);
        };

        let snapshot =
            Snapshot::from_json(&bytes).map_err(|e| StrandsError::SnapshotException {
                message: format!("failed to parse snapshot: {}", e),
            })?;
        agent.load_snapshot(snapshot.clone())?;
        Ok(Some(snapshot))
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn strategy(&self) -> SaveLatestStrategy {
        self.strategy
    }

    async fn capture_from_event(&self, message: &crate::types::content::Message) {
        let snapshot = Snapshot::new(
            self.agent_id.clone(),
            serde_json::json!({ "latest_message": message }),
        );
        if let Ok(bytes) = snapshot.to_json() {
            let _ = self.storage.write(&self.latest_key(), &bytes).await;
        }
    }
}

#[async_trait]
impl HookProvider for SnapshotSessionManager {
    async fn on_event(&self, event: &mut HookEvent) {
        match event {
            HookEvent::MessageAdded(added) if self.strategy == SaveLatestStrategy::Message => {
                self.capture_from_event(&added.message).await;
            }
            HookEvent::AfterInvocation(after) if self.strategy == SaveLatestStrategy::Invocation => {
                if let Some(result) = &after.result {
                    self.capture_from_event(&result.message).await;
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
use crate::testing::MockModel;
use crate::storage::{InMemoryStorage, LocalFileStorage, NamespacedStorage};

    fn test_agent() -> Agent {
        Agent::builder()
            .model(MockModel::new(vec![]))
            .agent_id("agent-1")
            .system_prompt("snap")
            .build()
            .unwrap()
    }

    #[test]
    fn test_agent_snapshot_roundtrip() {
        let mut agent = test_agent();
        agent.state_mut().set("counter", 7);

        let snapshot = agent.take_snapshot("agent-1").unwrap();
        let bytes = snapshot.to_json().unwrap();

        let mut restored = test_agent();
        restored
            .load_snapshot(Snapshot::from_json(&bytes).unwrap())
            .unwrap();

        assert_eq!(restored.system_prompt(), Some("snap"));
        assert_eq!(restored.state().get::<i32>("counter"), Some(7));
    }

    #[tokio::test]
    async fn test_snapshot_session_manager_roundtrip() {
        let backend = Arc::new(InMemoryStorage::new());
        let manager = SnapshotSessionManager::new(
            backend.clone(),
            "session-1",
            "agent-1",
            SaveLatestStrategy::Trigger,
        );

        let mut agent = test_agent();
        manager.save_snapshot(&agent).await.unwrap();
        assert!(manager.load_latest(&mut agent).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_snapshot_session_manager_file_backend() {
        let root = std::env::temp_dir().join(format!("strands-snap-{}", uuid::Uuid::new_v4()));
        let backend = Arc::new(LocalFileStorage::new(&root).unwrap());
        let manager = SnapshotSessionManager::new(
            backend,
            "session-1",
            "agent-1",
            SaveLatestStrategy::Trigger,
        );

        let agent = test_agent();
        manager.save_snapshot(&agent).await.unwrap();

        let mut fresh = test_agent();
        manager.load_latest(&mut fresh).await.unwrap();

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn test_hook_driven_message_strategy() {
        let backend = Arc::new(InMemoryStorage::new());
        let nested = NamespacedStorage::new(backend.clone(), "session-1");
        let manager = SnapshotSessionManager::new(
            backend.clone(),
            "session-1",
            "agent-1",
            SaveLatestStrategy::Message,
        );

        let mut event = HookEvent::MessageAdded(crate::hooks::MessageAddedEvent::new(
            crate::types::content::Message::assistant("captured"),
        ));
        manager.on_event(&mut event).await;

        let bytes = nested
            .read("scopes/agent/agent-1/snapshots/snapshot_latest.json")
            .await
            .unwrap()
            .expect("snapshot persisted");
        let snapshot = Snapshot::from_json(&bytes).unwrap();
        assert!(snapshot.data["latest_message"].is_object());
    }

    #[tokio::test]
    async fn test_load_latest_missing_is_none() {
        let backend = Arc::new(InMemoryStorage::new());
        let manager =
            SnapshotSessionManager::new(backend, "s", "a", SaveLatestStrategy::Trigger);
        let mut agent = test_agent();
        assert!(manager.load_latest(&mut agent).await.unwrap().is_none());
    }
}
