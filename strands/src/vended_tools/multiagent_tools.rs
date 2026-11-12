//! Multi-agent orchestration as a tool: swarm and graph delegation.
//!
//! Port of the Python `vended_tools.swarm` / `vended_tools.graph` shim: a
//! generic wrapper that runs any `MultiAgentBase` orchestrator from a tool
//! call and returns its flattened node outputs.

use std::sync::Arc;

use async_trait::async_trait;

use crate::multiagent::base::{
    MultiAgentBase, MultiAgentInput, NodeResultValue, Status,
};
use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::tools::ToolSpec;

/// Runs a multi-agent orchestrator as a tool call.
pub struct MultiAgentTool<M: MultiAgentBase + 'static> {
    orchestrator: Arc<tokio::sync::Mutex<M>>,
    name: String,
    description: String,
}

impl<M: MultiAgentBase + 'static> MultiAgentTool<M> {
    /// Wraps an orchestrator under a tool name.
    pub fn new(
        orchestrator: M,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            orchestrator: Arc::new(tokio::sync::Mutex::new(orchestrator)),
            name: name.into(),
            description: description.into(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Flattens an orchestrator result into tool text.
    fn flatten_result(result: &crate::multiagent::base::MultiAgentResult) -> String {
        let mut lines = Vec::new();
        for (node_id, node_result) in &result.results {
            let text = match &node_result.result {
                NodeResultValue::Agent(agent_result) => agent_result.text(),
                NodeResultValue::MultiAgent(nested) => Self::flatten_result(nested),
                NodeResultValue::Error(error) => format!("error: {}", error),
            };
            if !text.is_empty() {
                lines.push(format!("[{}] {}", node_id, text));
            }
        }

        if lines.is_empty() {
            return format!(
                "Multi-agent run ended with status {:?} and no node output.",
                result.status
            );
        }

        lines.join("\n")
    }
}

#[async_trait]
impl<M: MultiAgentBase + 'static> AgentTool for MultiAgentTool<M> {
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
                    "description": "Complete task description for the multi-agent system"
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
            Some(task) => MultiAgentInput::Text(task.to_string()),
            None => MultiAgentInput::Text(serde_json::to_string(&input).unwrap_or_default()),
        };

        let mut orchestrator = self.orchestrator.lock().await;
        let result = orchestrator
            .invoke_async(task, None)
            .await
            .map_err(|e| format!("multi-agent run failed: {}", e))?;

        let succeeded = matches!(
            result.status,
            Status::Completed | Status::Dispatched
        );

        Ok(ToolResult2 {
            status: if succeeded {
                crate::types::tools::ToolResultStatus::Success
            } else {
                crate::types::tools::ToolResultStatus::Error
            },
            content: vec![crate::types::tools::ToolResultContent::text(Self::flatten_result(
                &result,
            ))],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::multiagent::{Swarm, SwarmConfig};
    use crate::testing::MockModel;
    use crate::testing::MockResponse;
    use crate::types::tools::ToolUse;

    fn echo_agent(text: &'static str) -> Agent {
        Agent::builder()
            .model(MockModel::new(vec![MockResponse::text(text)]))
            .name("unnamed")
            .build()
            .unwrap()
    }

    fn swarm() -> Swarm {
        let agent = echo_agent("swarm says hi");
        Swarm::new(vec![agent], Some("unnamed"), SwarmConfig::default()).unwrap()
    }

    #[test]
    fn test_flatten_result_renders_nodes() {
        let mut result = crate::multiagent::base::MultiAgentResult::new();
        result.status = Status::Completed;
        result.results.insert(
            "node_a".to_string(),
            crate::multiagent::base::NodeResult::from_agent(
                crate::agent::AgentResult {
                    stop_reason: crate::types::streaming::StopReason::EndTurn,
                    message: crate::types::content::Message::assistant("node a output"),
                    metrics: Default::default(),
                    state: Default::default(),
                    interrupts: None,
                    structured_output: None,
                    checkpoint: None,
                },
                5,
            ),
        );
        result.results.insert(
            "node_b".to_string(),
            crate::multiagent::base::NodeResult::from_error("boom", 5),
        );

        let flattened = MultiAgentTool::<Swarm>::flatten_result(&result);
        assert!(flattened.contains("[node_a] node a output"));
        assert!(flattened.contains("[node_b] error: boom"));
    }

    #[tokio::test]
    async fn test_swarm_tool_runs() {
        let tool = MultiAgentTool::new(swarm(), "swarm_helper", "Runs the swarm on a task");
        let context = ToolContext::new(ToolUse::new("swarm_helper", "t1", serde_json::json!({})));

        let result = tool
            .invoke(serde_json::json!({"task": "greet"}), &context)
            .await
            .unwrap();

        assert_eq!(result.status, crate::types::tools::ToolResultStatus::Success);
        assert!(result.content[0].text.as_deref().unwrap().contains("swarm says hi"));
    }

    #[test]
    fn test_tool_metadata() {
        let tool = MultiAgentTool::new(swarm(), "swarm_helper", "Runs the swarm on a task");
        assert_eq!(tool.name(), "swarm_helper");
        assert!(tool.tool_spec().description.contains("swarm"));
    }
}
