//! Session management for persistent conversations.

mod s3_session_manager;
mod snapshot_session_manager;

pub use snapshot_session_manager::{SaveLatestStrategy, SnapshotSessionManager};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::types::content::Message;
use crate::types::errors::{Result, StrandsError};

#[cfg(feature = "s3-session")]
pub use s3_session_manager::S3SessionManager;

/// Session type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SessionType {
    Agent,
}

/// A message within a session agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMessage {
    pub message: Message,
    pub message_id: usize,
    pub redact_message: Option<Message>,
    pub created_at: String,
    pub updated_at: String,
}

impl SessionMessage {
    pub fn from_message(message: Message, index: usize) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            message,
            message_id: index,
            redact_message: None,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn to_message(&self) -> Message {
        self.redact_message.clone().unwrap_or_else(|| self.message.clone())
    }
}

/// Encode bytes values to base64 for JSON serialization.
pub fn encode_bytes_values(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let encoded: serde_json::Map<String, serde_json::Value> = map
                .into_iter()
                .map(|(k, v)| (k, encode_bytes_values(v)))
                .collect();
            serde_json::Value::Object(encoded)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(encode_bytes_values).collect())
        }
        other => other,
    }
}

/// Decode base64-encoded bytes values from JSON.
pub fn decode_bytes_values(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            if map.get("__bytes_encoded__") == Some(&serde_json::Value::Bool(true)) {
                if let Some(serde_json::Value::String(data)) = map.get("data") {
                    if let Ok(decoded) = BASE64.decode(data) {
                        return serde_json::json!({
                            "__decoded_bytes__": decoded
                        });
                    }
                }
            }
            let decoded: serde_json::Map<String, serde_json::Value> = map
                .into_iter()
                .map(|(k, v)| (k, decode_bytes_values(v)))
                .collect();
            serde_json::Value::Object(decoded)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(decode_bytes_values).collect())
        }
        other => other,
    }
}

/// Encode bytes to base64 string with marker.
pub fn encode_bytes(data: &[u8]) -> serde_json::Value {
    serde_json::json!({
        "__bytes_encoded__": true,
        "data": BASE64.encode(data)
    })
}

/// Agent state within a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionAgent {
    pub agent_id: String,
    pub state: HashMap<String, serde_json::Value>,
    pub conversation_manager_state: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub internal_state: HashMap<String, serde_json::Value>,
    pub created_at: String,
    pub updated_at: String,
}

impl SessionAgent {
    pub fn new(agent_id: impl Into<String>) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            agent_id: agent_id.into(),
            state: HashMap::new(),
            conversation_manager_state: HashMap::new(),
            internal_state: HashMap::new(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// Create a SessionAgent from an Agent instance.
    ///
    /// This captures the agent's current state for session persistence.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to create a session state from
    ///
    /// # Returns
    ///
    /// A SessionAgent with the captured state.
    pub fn from_agent(agent: &crate::agent::Agent) -> Result<Self> {
        let now = Utc::now().to_rfc3339();


        let mut internal_state = HashMap::new();
        internal_state.insert(
            "interrupt_state".to_string(),
            serde_json::to_value(agent.interrupt_state().to_dict())
                .unwrap_or_default(),
        );

        Ok(Self {
            agent_id: agent.agent_id.clone(),
            state: agent.state().get_all(),
            conversation_manager_state: agent.conversation_manager().get_state(),
            internal_state,
            created_at: now.clone(),
            updated_at: now,
        })
    }

    /// Initialize an agent's internal state from this session.
    ///
    /// This restores internal state like interrupt state to the agent.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to initialize internal state for
    pub fn initialize_internal_state(&self, agent: &mut crate::agent::Agent) {
        if let Some(interrupt_state_value) = self.internal_state.get("interrupt_state") {
            if let Some(interrupt_data) = interrupt_state_value.as_object() {
                let interrupt_state = crate::types::interrupt::InterruptState::from_dict(
                    interrupt_data
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                );
                agent.set_interrupt_state(interrupt_state);
            }
        }
    }

    /// Restore agent state from this session.
    ///
    /// This restores the user-managed state to the agent.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to restore state to
    pub fn restore_state(&self, agent: &mut crate::agent::Agent) {
        for (key, value) in &self.state {
            agent.state_mut().set(key.clone(), value.clone());
        }
    }

    /// Restore conversation manager state from this session.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to restore conversation manager state to
    ///
    /// # Returns
    ///
    /// Optional prepend messages if the conversation manager provides them.
    pub fn restore_conversation_manager_state(
        &self,
        agent: &mut crate::agent::Agent,
    ) -> Option<Vec<Message>> {
        agent
            .conversation_manager_mut()
            .restore_from_session(self.conversation_manager_state.clone())
    }

    /// Update with current agent state.
    ///
    /// Updates timestamps and captures latest state from agent.
    pub fn update_from_agent(&mut self, agent: &crate::agent::Agent) {
        self.updated_at = Utc::now().to_rfc3339();
        self.state = agent.state().get_all();
        self.conversation_manager_state = agent.conversation_manager().get_state();


        self.internal_state.insert(
            "interrupt_state".to_string(),
            serde_json::to_value(agent.interrupt_state().to_dict())
                .unwrap_or_default(),
        );
    }
}

/// Session data model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Session {
    pub session_id: String,
    pub session_type: SessionType,
    pub created_at: String,
    pub updated_at: String,
}

impl Session {
    pub fn new(session_id: impl Into<String>) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            session_id: session_id.into(),
            session_type: SessionType::Agent,
            created_at: now.clone(),
            updated_at: now,
        }
    }
}

/// Legacy session format for backward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacySession {
    pub session_id: String,
    pub agent_id: String,
    pub messages: Vec<Message>,
    pub state: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl LegacySession {
    pub fn new(session_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            session_id: session_id.into(),
            agent_id: agent_id.into(),
            messages: Vec::new(),
            state: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// Summary of a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub session_type: SessionType,
    pub created_at: String,
    pub updated_at: String,
}

/// Session repository trait for storage backends.
#[async_trait]
pub trait SessionRepository: Send + Sync {
    async fn create_session(&self, session: &Session) -> Result<()>;
    async fn read_session(&self, session_id: &str) -> Result<Option<Session>>;
    async fn delete_session(&self, session_id: &str) -> Result<()>;

    async fn create_agent(&self, session_id: &str, agent: &SessionAgent) -> Result<()>;
    async fn read_agent(&self, session_id: &str, agent_id: &str) -> Result<Option<SessionAgent>>;
    async fn update_agent(&self, session_id: &str, agent: &SessionAgent) -> Result<()>;

    async fn create_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message: &SessionMessage,
    ) -> Result<()>;
    async fn read_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message_id: usize,
    ) -> Result<Option<SessionMessage>>;
    async fn update_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message: &SessionMessage,
    ) -> Result<()>;
    async fn list_messages(
        &self,
        session_id: &str,
        agent_id: &str,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<Vec<SessionMessage>>;

    async fn create_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
        state: &serde_json::Value,
    ) -> Result<()>;
    async fn read_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
    ) -> Result<Option<serde_json::Value>>;
    async fn update_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
        state: &serde_json::Value,
    ) -> Result<()>;
}

/// Trait for implementing session managers.
#[async_trait]
pub trait SessionManager: Send + Sync {
    async fn read_session(&self, session_id: &str) -> Result<Option<Session>>;
    async fn write_session(&self, session: &Session) -> Result<()>;
    async fn delete_session(&self, session_id: &str) -> Result<()>;
    async fn list_sessions(&self) -> Result<Vec<SessionSummary>>;
}

/// In-memory session manager for testing.
#[derive(Default)]
pub struct InMemorySessionManager {
    sessions: std::sync::RwLock<HashMap<String, Session>>,
    agents: std::sync::RwLock<HashMap<String, SessionAgent>>,
    messages: std::sync::RwLock<HashMap<String, Vec<SessionMessage>>>,
    multi_agents: std::sync::RwLock<HashMap<String, serde_json::Value>>,
}

impl InMemorySessionManager {
    pub fn new() -> Self {
        Self::default()
    }

    fn agent_key(session_id: &str, agent_id: &str) -> String {
        format!("{}:{}", session_id, agent_id)
    }

    fn multi_agent_key(session_id: &str, multi_agent_id: &str) -> String {
        format!("{}:ma:{}", session_id, multi_agent_id)
    }
}

#[async_trait]
impl SessionManager for InMemorySessionManager {
    async fn read_session(&self, session_id: &str) -> Result<Option<Session>> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions.get(session_id).cloned())
    }

    async fn write_session(&self, session: &Session) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session.session_id.clone(), session.clone());
        Ok(())
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        sessions.remove(session_id);
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions
            .values()
            .map(|s| SessionSummary {
                session_id: s.session_id.clone(),
                session_type: s.session_type,
                created_at: s.created_at.clone(),
                updated_at: s.updated_at.clone(),
            })
            .collect())
    }
}

#[async_trait]
impl SessionRepository for InMemorySessionManager {
    async fn create_session(&self, session: &Session) -> Result<()> {
        self.write_session(session).await
    }

    async fn read_session(&self, session_id: &str) -> Result<Option<Session>> {
        SessionManager::read_session(self, session_id).await
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        SessionManager::delete_session(self, session_id).await
    }

    async fn create_agent(&self, session_id: &str, agent: &SessionAgent) -> Result<()> {
        let key = Self::agent_key(session_id, &agent.agent_id);
        let mut agents = self.agents.write().unwrap();
        agents.insert(key, agent.clone());
        Ok(())
    }

    async fn read_agent(&self, session_id: &str, agent_id: &str) -> Result<Option<SessionAgent>> {
        let key = Self::agent_key(session_id, agent_id);
        let agents = self.agents.read().unwrap();
        Ok(agents.get(&key).cloned())
    }

    async fn update_agent(&self, session_id: &str, agent: &SessionAgent) -> Result<()> {
        self.create_agent(session_id, agent).await
    }

    async fn create_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message: &SessionMessage,
    ) -> Result<()> {
        let key = Self::agent_key(session_id, agent_id);
        let mut messages = self.messages.write().unwrap();
        messages
            .entry(key)
            .or_default()
            .push(message.clone());
        Ok(())
    }

    async fn read_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message_id: usize,
    ) -> Result<Option<SessionMessage>> {
        let key = Self::agent_key(session_id, agent_id);
        let messages = self.messages.read().unwrap();
        Ok(messages
            .get(&key)
            .and_then(|msgs| msgs.iter().find(|m| m.message_id == message_id).cloned()))
    }

    async fn update_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message: &SessionMessage,
    ) -> Result<()> {
        let key = Self::agent_key(session_id, agent_id);
        let mut messages = self.messages.write().unwrap();
        if let Some(msgs) = messages.get_mut(&key) {
            if let Some(m) = msgs.iter_mut().find(|m| m.message_id == message.message_id) {
                *m = message.clone();
            }
        }
        Ok(())
    }

    async fn list_messages(
        &self,
        session_id: &str,
        agent_id: &str,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<Vec<SessionMessage>> {
        let key = Self::agent_key(session_id, agent_id);
        let messages = self.messages.read().unwrap();
        Ok(messages
            .get(&key)
            .map(|msgs| {
                let end = limit.map(|l| offset + l).unwrap_or(msgs.len());
                msgs.iter()
                    .skip(offset)
                    .take(end - offset)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default())
    }

    async fn create_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
        state: &serde_json::Value,
    ) -> Result<()> {
        let key = Self::multi_agent_key(session_id, multi_agent_id);
        let mut multi_agents = self.multi_agents.write().unwrap();
        multi_agents.insert(key, state.clone());
        Ok(())
    }

    async fn read_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
    ) -> Result<Option<serde_json::Value>> {
        let key = Self::multi_agent_key(session_id, multi_agent_id);
        let multi_agents = self.multi_agents.read().unwrap();
        Ok(multi_agents.get(&key).cloned())
    }

    async fn update_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
        state: &serde_json::Value,
    ) -> Result<()> {
        self.create_multi_agent(session_id, multi_agent_id, state).await
    }
}

/// Session manager that uses a repository for persistence.
///
/// This is the primary session manager for production use, providing
/// agent initialization, message appending, and state synchronization.
pub struct RepositorySessionManager {
    session_repository: Arc<dyn SessionRepository>,
    session_id: String,
    session: Session,
    latest_agent_message: std::sync::RwLock<HashMap<String, Option<SessionMessage>>>,
}

impl RepositorySessionManager {
    /// Create a new RepositorySessionManager.
    ///
    /// If no session with the specified session_id exists, it will be created.
    pub async fn new(
        session_id: impl Into<String>,
        session_repository: Arc<dyn SessionRepository>,
    ) -> Result<Self> {
        let session_id = session_id.into();
        let session = match session_repository.read_session(&session_id).await? {
            Some(s) => s,
            None => {
                debug!("session_id=<{}> | session not found, creating new session", session_id);
                let new_session = Session::new(&session_id);
                session_repository.create_session(&new_session).await?;
                new_session
            }
        };

        Ok(Self {
            session_repository,
            session_id,
            session,
            latest_agent_message: std::sync::RwLock::new(HashMap::new()),
        })
    }

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get the session.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Append a message to the agent's session.
    pub async fn append_message(&self, message: Message, agent_id: &str) -> Result<()> {
        let next_index = {
            let latest = self.latest_agent_message.read().unwrap();
            latest
                .get(agent_id)
                .and_then(|m| m.as_ref())
                .map(|m| m.message_id + 1)
                .unwrap_or(0)
        };

        let session_message = SessionMessage::from_message(message, next_index);
        
        {
            let mut latest = self.latest_agent_message.write().unwrap();
            latest.insert(agent_id.to_string(), Some(session_message.clone()));
        }

        self.session_repository
            .create_message(&self.session_id, agent_id, &session_message)
            .await
    }

    /// Redact the latest message appended to the session.
    pub async fn redact_latest_message(
        &self,
        redact_message: Message,
        agent_id: &str,
    ) -> Result<()> {
        let mut latest = self.latest_agent_message.write().unwrap();
        let latest_message = latest
            .get_mut(agent_id)
            .and_then(|m| m.as_mut())
            .ok_or_else(|| StrandsError::SessionError {
                message: "No message to redact.".to_string(),
            })?;

        latest_message.redact_message = Some(redact_message);
        let message_to_update = latest_message.clone();
        drop(latest);

        self.session_repository
            .update_message(&self.session_id, agent_id, &message_to_update)
            .await
    }

    /// Sync an agent's state to the session repository.
    pub async fn sync_agent(&self, agent: &SessionAgent) -> Result<()> {
        self.session_repository
            .update_agent(&self.session_id, agent)
            .await
    }

    /// Initialize an agent with a session.
    ///
    /// If the agent already exists in the session, its state will be restored.
    /// Otherwise, a new agent entry will be created.
    pub async fn initialize_agent(&self, agent_id: &str) -> Result<Option<SessionAgent>> {
        {
            let latest = self.latest_agent_message.read().unwrap();
            if latest.contains_key(agent_id) {
                return Err(StrandsError::SessionError {
                    message: "The agent_id must be unique in a session.".to_string(),
                });
            }
        }

        {
            let mut latest = self.latest_agent_message.write().unwrap();
            latest.insert(agent_id.to_string(), None);
        }

        let session_agent = self
            .session_repository
            .read_agent(&self.session_id, agent_id)
            .await?;

        if session_agent.is_none() {
            debug!(
                "agent_id=<{}> | session_id=<{}> | creating agent",
                agent_id, self.session_id
            );
            let new_agent = SessionAgent::new(agent_id);
            self.session_repository
                .create_agent(&self.session_id, &new_agent)
                .await?;
            return Ok(None);
        }

        debug!(
            "agent_id=<{}> | session_id=<{}> | restoring agent",
            agent_id, self.session_id
        );

        Ok(session_agent)
    }

    /// Restore messages for an agent.
    pub async fn restore_messages(
        &self,
        agent_id: &str,
        offset: usize,
    ) -> Result<Vec<Message>> {
        let session_messages = self
            .session_repository
            .list_messages(&self.session_id, agent_id, None, offset)
            .await?;

        if !session_messages.is_empty() {
            let mut latest = self.latest_agent_message.write().unwrap();
            latest.insert(
                agent_id.to_string(),
                Some(session_messages.last().unwrap().clone()),
            );
        }

        Ok(session_messages.into_iter().map(|m| m.to_message()).collect())
    }

    /// Sync multi-agent state to the session repository.
    pub async fn sync_multi_agent(
        &self,
        multi_agent_id: &str,
        state: &serde_json::Value,
    ) -> Result<()> {
        self.session_repository
            .update_multi_agent(&self.session_id, multi_agent_id, state)
            .await
    }

    /// Initialize multi-agent state from the session repository.
    pub async fn initialize_multi_agent(
        &self,
        multi_agent_id: &str,
    ) -> Result<Option<serde_json::Value>> {
        let state = self
            .session_repository
            .read_multi_agent(&self.session_id, multi_agent_id)
            .await?;

        if state.is_none() {
            self.session_repository
                .create_multi_agent(&self.session_id, multi_agent_id, &serde_json::json!({}))
                .await?;
        }

        Ok(state)
    }

    /// Fix broken tool use/result pairs in message history.
    pub fn fix_broken_tool_use(&self, messages: Vec<Message>) -> Vec<Message> {
        let mut result = messages;

        if !result.is_empty() {
            let first = &result[0];
            if first.role == crate::types::content::Role::User {
                let has_tool_result = first.content.iter().any(|c| c.tool_result.is_some());
                if has_tool_result {
                    tracing::warn!(
                        "Session message history starts with orphaned toolResult. Removing."
                    );
                    result.remove(0);
                }
            }
        }

        result
    }
}

/// File-based session manager for local filesystem storage.
pub struct FileSessionManager {
    storage_dir: PathBuf,
    session_id: String,
}

impl FileSessionManager {
    const SESSION_PREFIX: &'static str = "session_";
    const AGENT_PREFIX: &'static str = "agent_";
    const MESSAGE_PREFIX: &'static str = "message_";

    pub fn new(session_id: impl Into<String>, storage_dir: Option<PathBuf>) -> Result<Self> {
        let session_id = session_id.into();
        let storage_dir = storage_dir.unwrap_or_else(|| {
            std::env::temp_dir().join("strands").join("sessions")
        });

        std::fs::create_dir_all(&storage_dir).map_err(|e| StrandsError::SessionError {
            message: format!("Failed to create storage directory: {}", e),
        })?;

        Ok(Self {
            storage_dir,
            session_id,
        })
    }

    fn get_session_path(&self, session_id: &str) -> PathBuf {
        self.storage_dir
            .join(format!("{}{}", Self::SESSION_PREFIX, session_id))
    }

    fn get_agent_path(&self, session_id: &str, agent_id: &str) -> PathBuf {
        self.get_session_path(session_id)
            .join("agents")
            .join(format!("{}{}", Self::AGENT_PREFIX, agent_id))
    }

    fn get_message_path(&self, session_id: &str, agent_id: &str, message_id: usize) -> PathBuf {
        self.get_agent_path(session_id, agent_id)
            .join("messages")
            .join(format!("{}{}.json", Self::MESSAGE_PREFIX, message_id))
    }

    fn read_json<T: serde::de::DeserializeOwned>(&self, path: &Path) -> Result<T> {
        let content = std::fs::read_to_string(path).map_err(|e| StrandsError::SessionError {
            message: format!("Failed to read file {}: {}", path.display(), e),
        })?;

        serde_json::from_str(&content).map_err(|e| StrandsError::SessionError {
            message: format!("Invalid JSON in {}: {}", path.display(), e),
        })
    }

    fn write_json<T: serde::Serialize>(&self, path: &Path, data: &T) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| StrandsError::SessionError {
                message: format!("Failed to create directory: {}", e),
            })?;
        }

        let tmp_path = path.with_extension("tmp");
        let content = serde_json::to_string_pretty(data).map_err(|e| StrandsError::SessionError {
            message: format!("Failed to serialize: {}", e),
        })?;

        std::fs::write(&tmp_path, &content).map_err(|e| StrandsError::SessionError {
            message: format!("Failed to write file: {}", e),
        })?;

        std::fs::rename(&tmp_path, path).map_err(|e| StrandsError::SessionError {
            message: format!("Failed to rename file: {}", e),
        })?;

        Ok(())
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

#[async_trait]
impl SessionManager for FileSessionManager {
    async fn read_session(&self, session_id: &str) -> Result<Option<Session>> {
        let path = self.get_session_path(session_id).join("session.json");
        if !path.exists() {
            return Ok(None);
        }
        self.read_json(&path).map(Some)
    }

    async fn write_session(&self, session: &Session) -> Result<()> {
        let session_dir = self.get_session_path(&session.session_id);
        std::fs::create_dir_all(&session_dir).map_err(|e| StrandsError::SessionError {
            message: format!("Failed to create session directory: {}", e),
        })?;
        std::fs::create_dir_all(session_dir.join("agents")).ok();
        std::fs::create_dir_all(session_dir.join("multi_agents")).ok();

        let path = session_dir.join("session.json");
        self.write_json(&path, session)
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        let session_dir = self.get_session_path(session_id);
        if session_dir.exists() {
            std::fs::remove_dir_all(&session_dir).map_err(|e| StrandsError::SessionError {
                message: format!("Failed to delete session: {}", e),
            })?;
        }
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let mut summaries = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.storage_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with(Self::SESSION_PREFIX) {
                    let session_id = name_str.trim_start_matches(Self::SESSION_PREFIX);
                    if let Some(session) = SessionManager::read_session(self, session_id).await? {
                        summaries.push(SessionSummary {
                            session_id: session.session_id,
                            session_type: session.session_type,
                            created_at: session.created_at,
                            updated_at: session.updated_at,
                        });
                    }
                }
            }
        }

        Ok(summaries)
    }
}

#[async_trait]
impl SessionRepository for FileSessionManager {
    async fn create_session(&self, session: &Session) -> Result<()> {
        let session_dir = self.get_session_path(&session.session_id);
        if session_dir.exists() {
            return Err(StrandsError::SessionError {
                message: format!("Session {} already exists", session.session_id),
            });
        }
        self.write_session(session).await
    }

    async fn read_session(&self, session_id: &str) -> Result<Option<Session>> {
        SessionManager::read_session(self, session_id).await
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        SessionManager::delete_session(self, session_id).await
    }

    async fn create_agent(&self, session_id: &str, agent: &SessionAgent) -> Result<()> {
        let agent_dir = self.get_agent_path(session_id, &agent.agent_id);
        std::fs::create_dir_all(&agent_dir).map_err(|e| StrandsError::SessionError {
            message: format!("Failed to create agent directory: {}", e),
        })?;
        std::fs::create_dir_all(agent_dir.join("messages")).ok();

        let path = agent_dir.join("agent.json");
        self.write_json(&path, agent)
    }

    async fn read_agent(&self, session_id: &str, agent_id: &str) -> Result<Option<SessionAgent>> {
        let path = self.get_agent_path(session_id, agent_id).join("agent.json");
        if !path.exists() {
            return Ok(None);
        }
        self.read_json(&path).map(Some)
    }

    async fn update_agent(&self, session_id: &str, agent: &SessionAgent) -> Result<()> {
        let existing = self.read_agent(session_id, &agent.agent_id).await?;
        if existing.is_none() {
            return Err(StrandsError::SessionError {
                message: format!("Agent {} does not exist", agent.agent_id),
            });
        }

        let path = self.get_agent_path(session_id, &agent.agent_id).join("agent.json");
        self.write_json(&path, agent)
    }

    async fn create_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message: &SessionMessage,
    ) -> Result<()> {
        let path = self.get_message_path(session_id, agent_id, message.message_id);
        self.write_json(&path, message)
    }

    async fn read_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message_id: usize,
    ) -> Result<Option<SessionMessage>> {
        let path = self.get_message_path(session_id, agent_id, message_id);
        if !path.exists() {
            return Ok(None);
        }
        self.read_json(&path).map(Some)
    }

    async fn update_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message: &SessionMessage,
    ) -> Result<()> {
        let existing = self.read_message(session_id, agent_id, message.message_id).await?;
        if existing.is_none() {
            return Err(StrandsError::SessionError {
                message: format!("Message {} does not exist", message.message_id),
            });
        }

        let path = self.get_message_path(session_id, agent_id, message.message_id);
        self.write_json(&path, message)
    }

    async fn list_messages(
        &self,
        session_id: &str,
        agent_id: &str,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<Vec<SessionMessage>> {
        let messages_dir = self.get_agent_path(session_id, agent_id).join("messages");
        if !messages_dir.exists() {
            return Ok(Vec::new());
        }

        let mut message_files: Vec<(usize, PathBuf)> = Vec::new();

        for entry in std::fs::read_dir(&messages_dir).map_err(|e| StrandsError::SessionError {
            message: format!("Failed to read messages directory: {}", e),
        })? {
            let entry = entry.map_err(|e| StrandsError::SessionError {
                message: format!("Failed to read directory entry: {}", e),
            })?;

            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if name_str.starts_with(Self::MESSAGE_PREFIX) && name_str.ends_with(".json") {
                let id_str = name_str
                    .trim_start_matches(Self::MESSAGE_PREFIX)
                    .trim_end_matches(".json");
                if let Ok(id) = id_str.parse::<usize>() {
                    message_files.push((id, entry.path()));
                }
            }
        }

        message_files.sort_by_key(|(id, _)| *id);

        let end = limit.map(|l| offset + l).unwrap_or(message_files.len());
        let mut messages = Vec::new();

        for (_, path) in message_files.into_iter().skip(offset).take(end - offset) {
            let message: SessionMessage = self.read_json(&path)?;
            messages.push(message);
        }

        Ok(messages)
    }

    async fn create_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
        state: &serde_json::Value,
    ) -> Result<()> {
        let path = self
            .get_session_path(session_id)
            .join("multi_agents")
            .join(format!("{}.json", multi_agent_id));
        self.write_json(&path, state)
    }

    async fn read_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
    ) -> Result<Option<serde_json::Value>> {
        let path = self
            .get_session_path(session_id)
            .join("multi_agents")
            .join(format!("{}.json", multi_agent_id));
        if !path.exists() {
            return Ok(None);
        }
        self.read_json(&path).map(Some)
    }

    async fn update_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
        state: &serde_json::Value,
    ) -> Result<()> {
        self.create_multi_agent(session_id, multi_agent_id, state).await
    }
}
