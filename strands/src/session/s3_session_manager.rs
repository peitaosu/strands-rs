//! S3-based session manager for cloud storage.

#[cfg(feature = "s3-session")]
use std::sync::Arc;

#[cfg(feature = "s3-session")]
use async_trait::async_trait;

#[cfg(feature = "s3-session")]
use aws_sdk_s3::Client as S3Client;

#[cfg(feature = "s3-session")]
use crate::identifier::{self, Identifier};
#[cfg(feature = "s3-session")]
use crate::types::errors::{Result, StrandsError};

#[cfg(feature = "s3-session")]
use super::{
    Session, SessionAgent, SessionMessage, SessionRepository, SessionSummary, SessionManager,
    RepositorySessionManager,
};

#[cfg(feature = "s3-session")]
const SESSION_PREFIX: &str = "session_";
#[cfg(feature = "s3-session")]
const AGENT_PREFIX: &str = "agent_";
#[cfg(feature = "s3-session")]
const MESSAGE_PREFIX: &str = "message_";
#[cfg(feature = "s3-session")]
const MULTI_AGENT_PREFIX: &str = "multi_agent_";

/// S3-based session manager for cloud storage.
///
/// Creates the following filesystem structure for the session storage:
/// ```text
/// /<sessions_dir>/
/// └── session_<session_id>/
///     ├── session.json                # Session metadata
///     └── agents/
///         └── agent_<agent_id>/
///             ├── agent.json          # Agent metadata
///             └── messages/
///                 ├── message_<id1>.json
///                 └── message_<id2>.json
/// ```
#[cfg(feature = "s3-session")]
pub struct S3SessionManager {
    client: S3Client,
    bucket: String,
    prefix: String,
    session_id: String,
}

#[cfg(feature = "s3-session")]
impl S3SessionManager {
    /// Create a new S3SessionManager with S3 storage.
    ///
    /// # Arguments
    ///
    /// * `session_id` - ID for the session. ID is not allowed to contain path separators.
    /// * `bucket` - S3 bucket name (required)
    /// * `prefix` - S3 key prefix for storage organization
    /// * `client` - AWS S3 client
    pub fn new(
        session_id: impl Into<String>,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        client: S3Client,
    ) -> Result<Self> {
        let session_id = session_id.into();
        identifier::validate(&session_id, Identifier::Session)
            .map_err(|e| StrandsError::SessionError { message: e })?;

        Ok(Self {
            client,
            bucket: bucket.into(),
            prefix: prefix.into(),
            session_id,
        })
    }

    /// Create a new S3SessionManager using default AWS config.
    ///
    /// # Arguments
    ///
    /// * `session_id` - ID for the session
    /// * `bucket` - S3 bucket name
    /// * `prefix` - S3 key prefix for storage organization
    /// * `region` - Optional AWS region
    pub async fn with_default_config(
        session_id: impl Into<String>,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        region: Option<&str>,
    ) -> Result<Self> {
        let mut config_loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
        if let Some(region) = region {
            config_loader = config_loader.region(aws_config::Region::new(region.to_string()));
        }
        let config = config_loader.load().await;
        let client = S3Client::new(&config);
        Self::new(session_id, bucket, prefix, client)
    }

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    fn get_session_path(&self, session_id: &str) -> Result<String> {
        identifier::validate(session_id, Identifier::Session)
            .map_err(|e| StrandsError::SessionError { message: e })?;
        Ok(format!("{}/{}{}/", self.prefix, SESSION_PREFIX, session_id))
    }

    fn get_agent_path(&self, session_id: &str, agent_id: &str) -> Result<String> {
        let session_path = self.get_session_path(session_id)?;
        identifier::validate(agent_id, Identifier::Agent)
            .map_err(|e| StrandsError::SessionError { message: e })?;
        Ok(format!("{}agents/{}{}/", session_path, AGENT_PREFIX, agent_id))
    }

    fn get_message_path(&self, session_id: &str, agent_id: &str, message_id: usize) -> Result<String> {
        let agent_path = self.get_agent_path(session_id, agent_id)?;
        Ok(format!("{}messages/{}{}.json", agent_path, MESSAGE_PREFIX, message_id))
    }

    fn get_multi_agent_path(&self, session_id: &str, multi_agent_id: &str) -> Result<String> {
        let session_path = self.get_session_path(session_id)?;
        identifier::validate(multi_agent_id, Identifier::Agent)
            .map_err(|e| StrandsError::SessionError { message: e })?;
        Ok(format!("{}multi_agents/{}{}/", session_path, MULTI_AGENT_PREFIX, multi_agent_id))
    }

    async fn read_s3_object<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match self.client.get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(response) => {
                let body = response.body.collect().await
                    .map_err(|e| StrandsError::SessionError {
                        message: format!("Failed to read S3 object body: {}", e),
                    })?;
                let content = String::from_utf8(body.into_bytes().to_vec())
                    .map_err(|e| StrandsError::SessionError {
                        message: format!("Invalid UTF-8 in S3 object: {}", e),
                    })?;
                let data = serde_json::from_str(&content)
                    .map_err(|e| StrandsError::SessionError {
                        message: format!("Invalid JSON in S3 object {}: {}", key, e),
                    })?;
                Ok(Some(data))
            }
            Err(e) => {

                let err_str = e.to_string();
                if err_str.contains("NoSuchKey") || err_str.contains("not found") {
                    Ok(None)
                } else {
                    Err(StrandsError::SessionError {
                        message: format!("S3 error reading {}: {}", key, e),
                    })
                }
            }
        }
    }

    async fn write_s3_object<T: serde::Serialize>(&self, key: &str, data: &T) -> Result<()> {
        let content = serde_json::to_string_pretty(data)
            .map_err(|e| StrandsError::SessionError {
                message: format!("Failed to serialize data: {}", e),
            })?;

        self.client.put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(content.into_bytes().into())
            .content_type("application/json")
            .send()
            .await
            .map_err(|e| StrandsError::SessionError {
                message: format!("Failed to write S3 object {}: {}", key, e),
            })?;

        Ok(())
    }

    async fn object_exists(&self, key: &str) -> Result<bool> {
        match self.client.head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("404") || err_str.contains("not found") || err_str.contains("NoSuchKey") {
                    Ok(false)
                } else {
                    Err(StrandsError::SessionError {
                        message: format!("S3 error checking existence of {}: {}", key, e),
                    })
                }
            }
        }
    }

    async fn list_objects_with_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self.client.list_objects_v2()
                .bucket(&self.bucket)
                .prefix(prefix);

            if let Some(token) = continuation_token {
                request = request.continuation_token(token);
            }

            let response = request.send().await
                .map_err(|e| StrandsError::SessionError {
                    message: format!("S3 error listing objects: {}", e),
                })?;

            if let Some(contents) = response.contents {
                for object in contents {
                    if let Some(key) = object.key {
                        keys.push(key);
                    }
                }
            }

            if response.is_truncated.unwrap_or(false) {
                continuation_token = response.next_continuation_token;
            } else {
                break;
            }
        }

        Ok(keys)
    }

    async fn delete_objects(&self, keys: Vec<String>) -> Result<()> {

        for chunk in keys.chunks(1000) {
            let objects: Vec<_> = chunk.iter()
                .map(|key| {
                    aws_sdk_s3::types::ObjectIdentifier::builder()
                        .key(key)
                        .build()
                        .expect("Failed to build ObjectIdentifier")
                })
                .collect();

            let delete = aws_sdk_s3::types::Delete::builder()
                .set_objects(Some(objects))
                .build()
                .map_err(|e| StrandsError::SessionError {
                    message: format!("Failed to build delete request: {}", e),
                })?;

            self.client.delete_objects()
                .bucket(&self.bucket)
                .delete(delete)
                .send()
                .await
                .map_err(|e| StrandsError::SessionError {
                    message: format!("S3 error deleting objects: {}", e),
                })?;
        }

        Ok(())
    }

    /// Create a RepositorySessionManager backed by this S3 storage.
    pub async fn into_repository_manager(self) -> Result<RepositorySessionManager> {
        let session_id = self.session_id.clone();
        RepositorySessionManager::new(session_id, Arc::new(self)).await
    }
}

#[cfg(feature = "s3-session")]
#[async_trait]
impl SessionRepository for S3SessionManager {
    async fn create_session(&self, session: &Session) -> Result<()> {
        let session_key = format!("{}session.json", self.get_session_path(&session.session_id)?);

        if self.object_exists(&session_key).await? {
            return Err(StrandsError::SessionError {
                message: format!("Session {} already exists", session.session_id),
            });
        }

        self.write_s3_object(&session_key, session).await
    }

    async fn read_session(&self, session_id: &str) -> Result<Option<Session>> {
        let session_key = format!("{}session.json", self.get_session_path(session_id)?);
        self.read_s3_object(&session_key).await
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        let session_prefix = self.get_session_path(session_id)?;
        let keys = self.list_objects_with_prefix(&session_prefix).await?;

        if keys.is_empty() {
            return Err(StrandsError::SessionError {
                message: format!("Session {} does not exist", session_id),
            });
        }

        self.delete_objects(keys).await
    }

    async fn create_agent(&self, session_id: &str, agent: &SessionAgent) -> Result<()> {
        let agent_key = format!("{}agent.json", self.get_agent_path(session_id, &agent.agent_id)?);
        self.write_s3_object(&agent_key, agent).await
    }

    async fn read_agent(&self, session_id: &str, agent_id: &str) -> Result<Option<SessionAgent>> {
        let agent_key = format!("{}agent.json", self.get_agent_path(session_id, agent_id)?);
        self.read_s3_object(&agent_key).await
    }

    async fn update_agent(&self, session_id: &str, agent: &SessionAgent) -> Result<()> {
        let previous_agent = self.read_agent(session_id, &agent.agent_id).await?;
        if previous_agent.is_none() {
            return Err(StrandsError::SessionError {
                message: format!("Agent {} in session {} does not exist", agent.agent_id, session_id),
            });
        }

        let mut updated_agent = agent.clone();
        if let Some(prev) = previous_agent {
            updated_agent.created_at = prev.created_at;
        }

        let agent_key = format!("{}agent.json", self.get_agent_path(session_id, &agent.agent_id)?);
        self.write_s3_object(&agent_key, &updated_agent).await
    }

    async fn create_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message: &SessionMessage,
    ) -> Result<()> {
        let message_key = self.get_message_path(session_id, agent_id, message.message_id)?;
        self.write_s3_object(&message_key, message).await
    }

    async fn read_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message_id: usize,
    ) -> Result<Option<SessionMessage>> {
        let message_key = self.get_message_path(session_id, agent_id, message_id)?;
        self.read_s3_object(&message_key).await
    }

    async fn update_message(
        &self,
        session_id: &str,
        agent_id: &str,
        message: &SessionMessage,
    ) -> Result<()> {
        let previous_message = self.read_message(session_id, agent_id, message.message_id).await?;
        if previous_message.is_none() {
            return Err(StrandsError::SessionError {
                message: format!("Message {} does not exist", message.message_id),
            });
        }

        let mut updated_message = message.clone();
        if let Some(prev) = previous_message {
            updated_message.created_at = prev.created_at;
        }

        let message_key = self.get_message_path(session_id, agent_id, message.message_id)?;
        self.write_s3_object(&message_key, &updated_message).await
    }

    async fn list_messages(
        &self,
        session_id: &str,
        agent_id: &str,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<Vec<SessionMessage>> {
        let messages_prefix = format!("{}messages/", self.get_agent_path(session_id, agent_id)?);
        let keys = self.list_objects_with_prefix(&messages_prefix).await?;


        let mut message_index_keys: Vec<(usize, String)> = Vec::new();
        for key in keys {
            if key.ends_with(".json") && key.contains(MESSAGE_PREFIX) {
                if let Some(filename) = key.split('/').last() {
                    let index_str = filename
                        .trim_start_matches(MESSAGE_PREFIX)
                        .trim_end_matches(".json");
                    if let Ok(index) = index_str.parse::<usize>() {
                        message_index_keys.push((index, key));
                    }
                }
            }
        }

        message_index_keys.sort_by_key(|(idx, _)| *idx);


        let message_keys: Vec<String> = if let Some(lim) = limit {
            message_index_keys.into_iter()
                .skip(offset)
                .take(lim)
                .map(|(_, key)| key)
                .collect()
        } else {
            message_index_keys.into_iter()
                .skip(offset)
                .map(|(_, key)| key)
                .collect()
        };


        let mut messages = Vec::new();
        for key in message_keys {
            if let Some(message) = self.read_s3_object::<SessionMessage>(&key).await? {
                messages.push(message);
            }
        }

        Ok(messages)
    }

    async fn create_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
        state: &serde_json::Value,
    ) -> Result<()> {
        let key = format!("{}multi_agent.json", self.get_multi_agent_path(session_id, multi_agent_id)?);
        self.write_s3_object(&key, state).await
    }

    async fn read_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
    ) -> Result<Option<serde_json::Value>> {
        let key = format!("{}multi_agent.json", self.get_multi_agent_path(session_id, multi_agent_id)?);
        self.read_s3_object(&key).await
    }

    async fn update_multi_agent(
        &self,
        session_id: &str,
        multi_agent_id: &str,
        state: &serde_json::Value,
    ) -> Result<()> {
        let previous_state = self.read_multi_agent(session_id, multi_agent_id).await?;
        if previous_state.is_none() {
            return Err(StrandsError::SessionError {
                message: format!("MultiAgent state {} in session {} does not exist", multi_agent_id, session_id),
            });
        }

        let key = format!("{}multi_agent.json", self.get_multi_agent_path(session_id, multi_agent_id)?);
        self.write_s3_object(&key, state).await
    }
}

#[cfg(feature = "s3-session")]
#[async_trait]
impl SessionManager for S3SessionManager {
    async fn read_session(&self, session_id: &str) -> Result<Option<Session>> {
        SessionRepository::read_session(self, session_id).await
    }

    async fn write_session(&self, session: &Session) -> Result<()> {
        let session_key = format!("{}session.json", self.get_session_path(&session.session_id)?);
        self.write_s3_object(&session_key, session).await
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        SessionRepository::delete_session(self, session_id).await
    }

    async fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let sessions_prefix = format!("{}/", self.prefix);
        let keys = self.list_objects_with_prefix(&sessions_prefix).await?;

        let mut summaries = Vec::new();
        let mut seen_sessions = std::collections::HashSet::new();

        for key in keys {

            if key.ends_with("/session.json") {
                if let Some(session) = self.read_s3_object::<Session>(&key).await? {
                    if !seen_sessions.contains(&session.session_id) {
                        seen_sessions.insert(session.session_id.clone());
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

