//! A2A client: calls remote A2A agents over JSON-RPC HTTP.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;

use super::types::{
    A2AMessage, A2APart, A2ARequest, A2AResponse, A2ATask, A2ATaskState, AgentCard,
};
use crate::multiagent::base::{
    InvocationState, MultiAgentBase, MultiAgentEventStream, MultiAgentInput, MultiAgentResult,
    Status,
};
use crate::types::errors::{Result, StrandsError};
use crate::types::streaming::{Metrics, Usage};

/// Default HTTP timeout for A2A operations.
pub const DEFAULT_A2A_TIMEOUT_SECS: u64 = 300;

/// Well-known agent card paths, newest first.
const AGENT_CARD_PATHS: [&str; 2] = ["/.well-known/agent-card.json", "/.well-known/agent.json"];

/// Client wrapper for a remote A2A agent.
///
/// Mirrors the Python `A2AAgent`: discovers the agent card, sends
/// `message/send` JSON-RPC requests, and flattens task artifacts and status
/// messages into a `MultiAgentResult`.
#[derive(Debug, Clone)]
pub struct A2AClient {
    endpoint: String,
    timeout: Duration,
    headers: HashMap<String, String>,
    http: reqwest::Client,
}

impl A2AClient {
    pub fn new(endpoint: impl Into<String>) -> Self {
        let endpoint = endpoint.into().trim_end_matches('/').to_string();
        Self {
            endpoint,
            timeout: Duration::from_secs(DEFAULT_A2A_TIMEOUT_SECS),
            headers: HashMap::new(),
            http: reqwest::Client::new(),
        }
    }

    /// Sets the HTTP timeout for card discovery and message sends.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Adds a header used for both card discovery and message sends
    /// (e.g. `Authorization` for authenticated endpoints).
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Fetches the remote agent card.
    ///
    /// Tries the current well-known path first, then the legacy one.
    pub async fn get_agent_card(&self) -> Result<AgentCard> {
        for path in AGENT_CARD_PATHS {
            let mut request = self
                .http
                .get(format!("{}{}", self.endpoint, path))
                .timeout(self.timeout);
            for (key, value) in &self.headers {
                request = request.header(key, value);
            }

            let response = request.send().await.map_err(|e| StrandsError::NetworkError(e.to_string()))?;
            if response.status().is_success() {
                let card: AgentCard = response
                    .json()
                    .await
                    .map_err(|e| StrandsError::NetworkError(format!("failed to parse agent card: {}", e)))?;
                return Ok(card);
            }
        }

        Err(StrandsError::NetworkError(format!(
            "agent card not found at {}",
            self.endpoint
        )))
    }

    /// Sends a text message via `message/send` and returns the resulting task.
    pub async fn send_message(&self, text: &str) -> Result<A2ATask> {
        let message = A2AMessage {
            role: "user".to_string(),
            parts: vec![A2APart::Text {
                text: text.to_string(),
            }],
            context_id: None,
            task_id: None,
        };

        let request = A2ARequest::new(uuid::Uuid::new_v4().to_string(), "message/send").with_params(
            serde_json::json!({
                "message": message,
            }),
        );

        let response = self.post_json_rpc(request).await?;
        extract_task(response)
    }

    async fn post_json_rpc(&self, request: A2ARequest) -> Result<serde_json::Value> {
        let mut post = self
            .http
            .post(self.endpoint())
            .timeout(self.timeout)
            .json(&request);
        for (key, value) in &self.headers {
            post = post.header(key, value);
        }

        let http_response = post
            .send()
            .await
            .map_err(|e| StrandsError::NetworkError(e.to_string()))?;

        if !http_response.status().is_success() {
            let status = http_response.status();
            let body = http_response.text().await.unwrap_or_default();
            return Err(StrandsError::NetworkError(format!(
                "A2A server returned HTTP {}: {}",
                status, body
            )));
        }

        let envelope: A2AResponse = http_response
            .json()
            .await
            .map_err(|e| StrandsError::NetworkError(format!("failed to parse A2A response: {}", e)))?;

        if let Some(error) = envelope.error {
            return Err(StrandsError::NetworkError(format!(
                "A2A error {}: {}",
                error.code, error.message
            )));
        }

        envelope.result.ok_or_else(|| StrandsError::NetworkError {
            0: format!("A2A response carried no result"),
        })
    }
}

/// Extracts an `A2ATask` from a JSON-RPC result payload.
fn extract_task(result: serde_json::Value) -> Result<A2ATask> {
    serde_json::from_value(result).map_err(|e| StrandsError::NetworkError(format!(
        "failed to parse A2A task: {}",
        e
    )))
}

/// Flattens a task's artifacts and status message into text.
pub fn task_to_text(task: &A2ATask) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(artifacts) = &task.artifacts {
        for artifact in artifacts {
            for part in &artifact.parts {
                if let A2APart::Text { text } = part {
                    parts.push(text.clone());
                }
            }
        }
    }

    if parts.is_empty() {
        if let Some(message) = &task.message {
            for part in &message.parts {
                if let A2APart::Text { text } = part {
                    parts.push(text.clone());
                }
            }
        }
    }

    parts.join("\n")
}

/// Maps an A2A task state onto multi-agent status.
pub fn task_status_to_status(task: &A2ATask) -> Status {
    match task.state {
        A2ATaskState::Completed => Status::Completed,
        A2ATaskState::Failed => Status::Failed,
        A2ATaskState::Cancelled => Status::Interrupted,
        A2ATaskState::Pending | A2ATaskState::Working => Status::Executing,
    }
}

#[async_trait]
impl MultiAgentBase for A2AClient {
    fn id(&self) -> &str {
        "a2a_client"
    }

    async fn invoke_async(
        &mut self,
        task: MultiAgentInput,
        _invocation_state: Option<&InvocationState>,
    ) -> Result<MultiAgentResult> {
        let text = task.to_string_lossy();
        let start = std::time::Instant::now();

        let a2a_task = self.send_message(&text).await?;
        let execution_time_ms = start.elapsed().as_millis() as u64;

        Ok(MultiAgentResult {
            status: task_status_to_status(&a2a_task),
            results: Default::default(),
            accumulated_usage: Usage::default(),
            accumulated_metrics: Metrics {
                latency_ms: execution_time_ms,
                time_to_first_byte_ms: 0,
            },
            execution_count: 1,
            execution_time_ms,
            interrupts: Vec::new(),
        })
    }

    fn stream_async<'a>(
        &'a mut self,
        task: MultiAgentInput,
        invocation_state: Option<&'a InvocationState>,
    ) -> MultiAgentEventStream<'a> {
        Box::pin(async_stream::stream! {
            match self.invoke_async(task, invocation_state).await {
                Ok(result) => {
                    yield crate::multiagent::base::MultiAgentEvent::result(result);
                }
                Err(e) => {
                    tracing::warn!("A2A invocation failed: {}", e);
                }
            }
        })
    }

    fn serialize_state(&self) -> serde_json::Value {
        serde_json::json!({
            "endpoint": self.endpoint,
        })
    }

    fn deserialize_state(&mut self, _payload: &serde_json::Value) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_configuration() {
        let client = A2AClient::new("http://localhost:8080/")
            .with_timeout(Duration::from_secs(5))
            .with_header("Authorization", "Bearer token");

        assert_eq!(client.endpoint(), "http://localhost:8080");
        assert_eq!(client.timeout, Duration::from_secs(5));
        assert_eq!(
            client.headers.get("Authorization").map(|v| v.as_str()),
            Some("Bearer token")
        );
    }

    #[test]
    fn test_task_to_text_from_artifacts() {
        let task: A2ATask = serde_json::from_value(serde_json::json!({
            "id": "t1",
            "contextId": "c1",
            "state": "completed",
            "artifacts": [
                {"name": "answer", "parts": [{"type": "text", "text": "the answer"}]}
            ]
        }))
        .unwrap();

        assert_eq!(task_to_text(&task), "the answer");
        assert_eq!(task_status_to_status(&task), Status::Completed);
    }

    #[test]
    fn test_task_to_text_falls_back_to_status_message() {
        let task: A2ATask = serde_json::from_value(serde_json::json!({
            "id": "t1",
            "contextId": "c1",
            "state": "completed",
            "message": {"role": "agent", "parts": [{"type": "text", "text": "status text"}]}
        }))
        .unwrap();

        assert_eq!(task_to_text(&task), "status text");
    }

    #[test]
    fn test_failed_task_maps_status() {
        let task: A2ATask = serde_json::from_value(serde_json::json!({
            "id": "t1",
            "contextId": "c1",
            "state": "failed"
        }))
        .unwrap();

        assert_eq!(task_status_to_status(&task), Status::Failed);
    }

    #[test]
    fn test_message_send_request_shape() {
        let message = A2AMessage {
            role: "user".to_string(),
            parts: vec![A2APart::Text { text: "hi".to_string() }],
            context_id: None,
            task_id: None,
        };
        let request = A2ARequest::new("1", "message/send")
            .with_params(serde_json::json!({"message": message}));

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["method"], "message/send");
        assert_eq!(json["params"]["message"]["role"], "user");
        assert_eq!(json["params"]["message"]["parts"][0]["text"], "hi");
    }

    #[tokio::test]
    async fn test_get_agent_card_unreachable_errors() {
        let client = A2AClient::new("http://127.0.0.1:1").with_timeout(Duration::from_millis(500));
        assert!(client.get_agent_card().await.is_err());
    }
}
