//! Tool execution with sequential and concurrent modes.

use std::sync::Arc;

use futures::stream::{self, StreamExt};

use super::{AgentTool, InvocationState, ToolEvent, ToolContext};
use crate::types::tools::{ToolResult, ToolResultContent, ToolResultStatus, ToolUse};

/// Execution mode for tool processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Sequential,
    Concurrent { max_parallel: Option<usize> },
}

impl Default for ExecutionMode {
    fn default() -> Self { Self::Concurrent { max_parallel: None } }
}

/// Executor for running tools.
pub struct ToolExecutor {
    mode: ExecutionMode,
}

impl Default for ToolExecutor {
    fn default() -> Self { Self::new() }
}

impl ToolExecutor {
    pub fn new() -> Self { Self { mode: ExecutionMode::default() } }
    pub fn sequential() -> Self { Self { mode: ExecutionMode::Sequential } }
    pub fn concurrent(max_parallel: Option<usize>) -> Self { Self { mode: ExecutionMode::Concurrent { max_parallel } } }

    /// Executes all tools and returns their results.
    pub async fn execute_all(
        &self,
        tools: &[(Arc<dyn AgentTool>, ToolUse)],
        invocation_state: &InvocationState,
    ) -> Vec<(String, Vec<ToolEvent>)> {
        match self.mode {
            ExecutionMode::Sequential => self.execute_sequential(tools, invocation_state).await,
            ExecutionMode::Concurrent { max_parallel } => self.execute_concurrent(tools, invocation_state, max_parallel).await,
        }
    }

    async fn execute_sequential(
        &self,
        tools: &[(Arc<dyn AgentTool>, ToolUse)],
        invocation_state: &InvocationState,
    ) -> Vec<(String, Vec<ToolEvent>)> {
        let mut results = Vec::with_capacity(tools.len());
        for (tool, tool_use) in tools {
            let events = Self::execute_single(tool.clone(), tool_use, invocation_state).await;
            results.push((tool_use.tool_use_id.clone(), events));
        }
        results
    }

    async fn execute_concurrent(
        &self,
        tools: &[(Arc<dyn AgentTool>, ToolUse)],
        invocation_state: &InvocationState,
        max_parallel: Option<usize>,
    ) -> Vec<(String, Vec<ToolEvent>)> {
        let limit = max_parallel.unwrap_or(tools.len());

        let futures = tools.iter().map(|(tool, tool_use)| {
            let tool = tool.clone();
            let tool_use = tool_use.clone();
            let state = invocation_state.clone();
            async move {
                let events = Self::execute_single(tool, &tool_use, &state).await;
                (tool_use.tool_use_id, events)
            }
        });

        stream::iter(futures).buffer_unordered(limit).collect().await
    }

    async fn execute_single(
        tool: Arc<dyn AgentTool>,
        tool_use: &ToolUse,
        invocation_state: &InvocationState,
    ) -> Vec<ToolEvent> {
        let context = ToolContext::with_state(invocation_state.clone());
        let result = match tool.invoke(tool_use.input.clone(), &context).await {
            Ok(r) => ToolResult {
                tool_use_id: tool_use.tool_use_id.clone(),
                status: r.status,
                content: r.content,
            },
            Err(e) => ToolResult {
                tool_use_id: tool_use.tool_use_id.clone(),
                status: ToolResultStatus::Error,
                content: vec![ToolResultContent::text(e)],
            },
        };
        vec![ToolEvent::Result(result)]
    }

    /// Executes a single tool and returns its events.
    pub async fn execute_one(
        &self,
        tool: Arc<dyn AgentTool>,
        tool_use: &ToolUse,
        invocation_state: &InvocationState,
    ) -> Vec<ToolEvent> {
        Self::execute_single(tool, tool_use, invocation_state).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::tools::ToolResult2;
    use crate::types::tools::ToolSpec;

    struct SlowTool { name: String, delay_ms: u64 }

    #[async_trait]
    impl AgentTool for SlowTool {
        fn name(&self) -> &str { &self.name }
        fn description(&self) -> &str { "A slow tool" }
        fn tool_spec(&self) -> ToolSpec { ToolSpec::new(&self.name, "A slow tool") }

        async fn invoke(
            &self,
            _input: serde_json::Value,
            _context: &ToolContext,
        ) -> std::result::Result<ToolResult2, String> {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;
            Ok(ToolResult2::success(format!("done after {}ms", self.delay_ms)))
        }
    }

    #[tokio::test]
    async fn test_sequential_execution() {
        let executor = ToolExecutor::sequential();
        let tool1: Arc<dyn AgentTool> = Arc::new(SlowTool { name: "tool1".to_string(), delay_ms: 10 });
        let tool2: Arc<dyn AgentTool> = Arc::new(SlowTool { name: "tool2".to_string(), delay_ms: 10 });

        let tools = vec![
            (tool1, ToolUse::new("tool1", "1", serde_json::json!({}))),
            (tool2, ToolUse::new("tool2", "2", serde_json::json!({}))),
        ];

        let state = InvocationState::new();
        let results = executor.execute_all(&tools, &state).await;
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_concurrent_execution() {
        let executor = ToolExecutor::concurrent(None);
        let tool1: Arc<dyn AgentTool> = Arc::new(SlowTool { name: "tool1".to_string(), delay_ms: 50 });
        let tool2: Arc<dyn AgentTool> = Arc::new(SlowTool { name: "tool2".to_string(), delay_ms: 50 });

        let tools = vec![
            (tool1, ToolUse::new("tool1", "1", serde_json::json!({}))),
            (tool2, ToolUse::new("tool2", "2", serde_json::json!({}))),
        ];

        let state = InvocationState::new();
        let start = std::time::Instant::now();
        let results = executor.execute_all(&tools, &state).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 2);
        assert!(elapsed.as_millis() < 100);
    }
}
