//! A2A client vended tool: discover and message remote A2A agents.

use async_trait::async_trait;

use crate::multiagent::a2a::{task_to_text, A2AClient};
use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::tools::ToolSpec;

/// Stateless shim over an [`A2AClient`] exposing agent interaction as a tool.
pub struct A2AClientTool {
    client: A2AClient,
    agent_name: String,
}

impl A2AClientTool {
    /// Creates the tool bound to a remote agent endpoint.
    pub fn new(endpoint: impl Into<String>, agent_name: impl Into<String>) -> Self {
        Self {
            client: A2AClient::new(endpoint),
            agent_name: agent_name.into(),
        }
    }

    /// Configures the underlying client.
    pub fn with_client(mut self, client: A2AClient) -> Self {
        self.client = client;
        self
    }

    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }
}

#[async_trait]
impl AgentTool for A2AClientTool {
    fn name(&self) -> &str {
        "a2a_send_message"
    }

    fn description(&self) -> &str {
        "Send a task to the remote A2A agent and return its final text response."
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description()).with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Complete task description for the remote agent"
                }
            },
            "required": ["message"]
        }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let message = match input.get("message").and_then(|m| m.as_str()) {
            Some(message) => message.to_string(),
            None => serde_json::to_string(&input).unwrap_or_default(),
        };

        let task = self
            .client
            .send_message(&message)
            .await
            .map_err(|e| format!("a2a call to '{}' failed: {}", self.agent_name, e))?;

        let text = task_to_text(&task);
        if text.is_empty() {
            return Err(format!(
                "a2a agent '{}' returned no text content (state: {:?})",
                self.agent_name, task.state
            ));
        }

        Ok(ToolResult2::success(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tools::ToolUse;

    fn tool() -> A2AClientTool {
        A2AClientTool::new("http://127.0.0.1:1", "remote-worker")
    }

    #[test]
    fn test_tool_metadata() {
        let tool = tool();
        assert_eq!(tool.name(), "a2a_send_message");
        assert_eq!(tool.agent_name(), "remote-worker");
        let spec = tool.tool_spec();
        assert!(spec.description.contains("A2A"));
    }

    #[tokio::test]
    async fn test_unreachable_endpoint_errors() {
        let context = ToolContext::new(ToolUse::new("a2a_send_message", "t1", serde_json::json!({})));
        let error = tool()
            .invoke(serde_json::json!({"message": "do work"}), &context)
            .await
            .unwrap_err();
        assert!(error.contains("remote-worker"));
    }
}
