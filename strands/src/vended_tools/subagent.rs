//! Subagent vended tool: agent delegation through a tool call.

use std::sync::Arc;

use async_trait::async_trait;

use crate::agent::Agent;
use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::tools::ToolSpec;

/// Runs a sub-agent with the call input as its task.
///
/// Mirrors the Python `subagent` vended tool: the wrapped agent runs with
/// its own message history; the tool result carries the sub-agent's final
/// text output.
pub struct SubagentTool {
    agent: Arc<tokio::sync::Mutex<Agent>>,
    name: String,
    description: String,
}

impl SubagentTool {
    /// Creates a delegation tool around `agent`.
    pub fn new(agent: Agent, name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            agent: Arc::new(tokio::sync::Mutex::new(agent)),
            name: name.into(),
            description: description.into(),
        }
    }

    /// Creates a delegation tool using the agent's own name and description.
    pub fn from_agent(agent: Agent) -> Self {
        let name = agent
            .name()
            .cloned()
            .unwrap_or_else(|| "subagent".to_string());
        let description = agent
            .description
            .clone()
            .unwrap_or_else(|| "Delegate the task to a specialized sub-agent.".to_string());
        Self::new(agent, name, description)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl AgentTool for SubagentTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description()).with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Complete task description for the sub-agent"
                }
            },
            "required": ["task"]
        }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let task = match input.get("task").and_then(|t| t.as_str()) {
            Some(task) => task.to_string(),
            None => serde_json::to_string(&input).unwrap_or_default(),
        };

        let mut agent = self.agent.lock().await;
        let result = agent
            .invoke_async(task)
            .await
            .map_err(|e| format!("subagent failed: {}", e))?;

        Ok(ToolResult2::success(result.text()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockModel;
    use crate::testing::MockResponse;
    use crate::types::tools::ToolUse;

    #[tokio::test]
    async fn test_subagent_runs_wrapped_agent() {
        let agent = Agent::builder()
            .model(MockModel::new(vec![MockResponse::text("subagent done")]))
            .name("worker")
            .build()
            .unwrap();

        let tool = SubagentTool::new(agent, "worker", "Does work");
        let context = ToolContext::new(ToolUse::new("worker", "t1", serde_json::json!({})));

        let result = tool
            .invoke(serde_json::json!({"task": "do the thing"}), &context)
            .await
            .unwrap();

        assert_eq!(result.status, crate::types::tools::ToolResultStatus::Success);
        assert_eq!(result.content[0].text.as_deref(), Some("subagent done"));
    }

    #[tokio::test]
    async fn test_subagent_uses_raw_input_without_task() {
        let agent = Agent::builder()
            .model(MockModel::new(vec![MockResponse::text("ok")]))
            .build()
            .unwrap();

        let tool = SubagentTool::from_agent(agent);
        let context = ToolContext::new(ToolUse::new("subagent", "t1", serde_json::json!({})));

        let result = tool
            .invoke(serde_json::json!({"query": "hi"}), &context)
            .await
            .unwrap();
        assert_eq!(result.content[0].text.as_deref(), Some("ok"));
    }
}
