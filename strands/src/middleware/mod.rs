//! Middleware stages for intercepting agent execution.
//!
//! Mirrors the Python `strands.middleware` subsystem: three fixed stages
//! (`InvokeModel`, `ExecuteTool`, `AgentStream`), each with Input (before)
//! and Output (after) phases plus an optional short-circuit override.

use std::sync::Arc;

use async_trait::async_trait;

use crate::models::Model;
use crate::tools::InvocationState;
use crate::types::content::{Message, Messages};
use crate::types::errors::StrandsError;
use crate::types::streaming::{StopReason, Usage};
use crate::types::tools::{ToolChoice, ToolResult, ToolSpec, ToolUse};

/// Stage identifiers for middleware registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Wraps every model call in the event loop.
    InvokeModel,
    /// Wraps every tool execution, including direct tool calls.
    ExecuteTool,
    /// Wraps the whole agent pass.
    AgentStream,
}

/// Context for the `InvokeModel` stage.
#[derive(Clone)]
pub struct InvokeModelContext {
    pub messages: Messages,
    pub system_prompt: Option<String>,
    pub tool_specs: Vec<ToolSpec>,
    pub tool_choice: Option<ToolChoice>,
    pub invocation_state: InvocationState,
    /// Per-call model replacement; `None` keeps the agent's model.
    pub model: Option<Arc<dyn Model>>,
    pub projected_input_tokens: Option<u64>,
}

impl std::fmt::Debug for InvokeModelContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvokeModelContext")
            .field("messages", &self.messages)
            .field("system_prompt", &self.system_prompt)
            .field("tool_specs", &self.tool_specs)
            .field("tool_choice", &self.tool_choice)
            .field("invocation_state", &self.invocation_state)
            .field("model", &self.model.as_ref().map(|m| m.config().model_id.clone()))
            .field("projected_input_tokens", &self.projected_input_tokens)
            .finish()
    }
}

/// Result of a model call produced by the terminal or an override.
#[derive(Debug, Clone)]
pub struct InvokeModelResult {
    pub stop_reason: StopReason,
    pub message: Message,
    pub usage: Usage,
}

/// Middleware for the `InvokeModel` stage.
#[async_trait]
pub trait InvokeModelMiddleware: Send + Sync {
    /// Input phase: transform the context before the model call.
    async fn before(&self, _context: &mut InvokeModelContext) {}

    /// Output phase: transform the result after the model call.
    async fn after(&self, _context: &InvokeModelContext, _result: &mut InvokeModelResult) {}

    /// Returns Some to short-circuit the model call entirely.
    async fn override_result(
        &self,
        _context: &InvokeModelContext,
    ) -> Option<InvokeModelResult> {
        None
    }
}

/// Context for the `ExecuteTool` stage.
#[derive(Debug, Clone)]
pub struct ExecuteToolContext {
    /// The tool about to run; `None` when the tool is unknown.
    pub tool_name: Option<String>,
    pub tool_use: ToolUse,
    pub invocation_state: InvocationState,
    pub cancel_signal: Option<crate::async_utils::CancelSignal>,
}

/// Result of a tool execution produced by the terminal or an override.
#[derive(Debug, Clone)]
pub struct ExecuteToolResult {
    pub result: ToolResult,
}

/// Middleware for the `ExecuteTool` stage.
#[async_trait]
pub trait ExecuteToolMiddleware: Send + Sync {
    async fn before(&self, _context: &mut ExecuteToolContext) {}
    async fn after(&self, _context: &ExecuteToolContext, _result: &mut ExecuteToolResult) {}

    async fn override_result(
        &self,
        _context: &ExecuteToolContext,
    ) -> Option<ExecuteToolResult> {
        None
    }
}

/// Context for the `AgentStream` stage.
#[derive(Debug, Clone)]
pub struct AgentStreamContext {
    pub messages: Messages,
    pub invocation_state: InvocationState,
}

/// Middleware for the `AgentStream` stage.
#[async_trait]
pub trait AgentStreamMiddleware: Send + Sync {
    async fn before(&self, _context: &mut AgentStreamContext) {}
    async fn after(&self, _context: &AgentStreamContext) {}
}

/// Registry holding handlers for each stage.
#[derive(Default)]
pub struct MiddlewareRegistry {
    invoke_model: Vec<Arc<dyn InvokeModelMiddleware>>,
    execute_tool: Vec<Arc<dyn ExecuteToolMiddleware>>,
    agent_stream: Vec<Arc<dyn AgentStreamMiddleware>>,
}

impl MiddlewareRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_invoke_model(&mut self, handler: Arc<dyn InvokeModelMiddleware>) {
        self.invoke_model.push(handler);
    }

    pub fn add_execute_tool(&mut self, handler: Arc<dyn ExecuteToolMiddleware>) {
        self.execute_tool.push(handler);
    }

    pub fn add_agent_stream(&mut self, handler: Arc<dyn AgentStreamMiddleware>) {
        self.agent_stream.push(handler);
    }

    pub fn is_empty(&self) -> bool {
        self.invoke_model.is_empty()
            && self.execute_tool.is_empty()
            && self.agent_stream.is_empty()
    }

    /// Runs the `InvokeModel` chain around the terminal model call.
    pub async fn run_invoke_model<F, Fut>(
        &self,
        mut context: InvokeModelContext,
        terminal: F,
    ) -> std::result::Result<InvokeModelResult, StrandsError>
    where
        F: FnOnce(InvokeModelContext) -> Fut,
        Fut: std::future::Future<Output = std::result::Result<InvokeModelResult, StrandsError>>,
    {
        for handler in &self.invoke_model {
            handler.before(&mut context).await;
        }

        let mut result = None;
        for handler in &self.invoke_model {
            if let Some(overridden) = handler.override_result(&context).await {
                result = Some(overridden);
                break;
            }
        }

        let mut result = match result {
            Some(result) => result,
            None => terminal(context.clone()).await?,
        };

        for handler in self.invoke_model.iter().rev() {
            handler.after(&context, &mut result).await;
        }

        Ok(result)
    }

    /// Runs the `ExecuteTool` chain around the terminal tool execution.
    pub async fn run_execute_tool<F, Fut>(
        &self,
        mut context: ExecuteToolContext,
        terminal: F,
    ) -> std::result::Result<ExecuteToolResult, StrandsError>
    where
        F: FnOnce(ExecuteToolContext) -> Fut,
        Fut: std::future::Future<Output = std::result::Result<ExecuteToolResult, StrandsError>>,
    {
        for handler in &self.execute_tool {
            handler.before(&mut context).await;
        }

        let mut result = None;
        for handler in &self.execute_tool {
            if let Some(overridden) = handler.override_result(&context).await {
                result = Some(overridden);
                break;
            }
        }

        let mut result = match result {
            Some(result) => result,
            None => terminal(context.clone()).await?,
        };

        for handler in self.execute_tool.iter().rev() {
            handler.after(&context, &mut result).await;
        }

        Ok(result)
    }

    /// Runs the `AgentStream` chain around the terminal agent pass.
    pub async fn run_agent_stream<F, Fut, T>(
        &self,
        mut context: AgentStreamContext,
        terminal: F,
    ) -> std::result::Result<T, StrandsError>
    where
        F: FnOnce(AgentStreamContext) -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, StrandsError>>,
    {
        for handler in &self.agent_stream {
            handler.before(&mut context).await;
        }

        let result = terminal(context.clone()).await?;

        for handler in self.agent_stream.iter().rev() {
            handler.after(&context).await;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_utils::CancelSignal;
    use crate::types::tools::ToolResultStatus;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Recorder {
        hits: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InvokeModelMiddleware for Recorder {
        async fn before(&self, _context: &mut InvokeModelContext) {
            self.hits.fetch_add(1, Ordering::SeqCst);
        }
        async fn after(&self, _context: &InvokeModelContext, _result: &mut InvokeModelResult) {
            self.hits.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl ExecuteToolMiddleware for Recorder {
        async fn before(&self, _context: &mut ExecuteToolContext) {
            self.hits.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn test_invoke_model_chain() {
        let mut registry = MiddlewareRegistry::new();
        let hits = Arc::new(AtomicUsize::new(0));
        registry.add_invoke_model(Arc::new(Recorder { hits: hits.clone() }));

        let result = registry
            .run_invoke_model(
                InvokeModelContext {
                    messages: vec![],
                    system_prompt: None,
                    tool_specs: vec![],
                    tool_choice: None,
                    invocation_state: InvocationState::new(),
                    model: None,
                    projected_input_tokens: None,
                },
                |_context| async move {
                    Ok(InvokeModelResult {
                        stop_reason: StopReason::EndTurn,
                        message: Message::assistant("done"),
                        usage: Usage::default(),
                    })
                },
            )
            .await
            .unwrap();

        assert_eq!(result.message.text_content(), "done");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_invoke_model_override() {
        struct Override;

        #[async_trait]
        impl InvokeModelMiddleware for Override {
            async fn override_result(
                &self,
                _context: &InvokeModelContext,
            ) -> Option<InvokeModelResult> {
                Some(InvokeModelResult {
                    stop_reason: StopReason::EndTurn,
                    message: Message::assistant("short-circuited"),
                    usage: Usage::default(),
                })
            }
        }

        let mut registry = MiddlewareRegistry::new();
        registry.add_invoke_model(Arc::new(Override));

        let result = registry
            .run_invoke_model(
                InvokeModelContext {
                    messages: vec![],
                    system_prompt: None,
                    tool_specs: vec![],
                    tool_choice: None,
                    invocation_state: InvocationState::new(),
                    model: None,
                    projected_input_tokens: None,
                },
                |_context| async move { panic!("terminal must not run") },
            )
            .await
            .unwrap();

        assert_eq!(result.message.text_content(), "short-circuited");
    }

    #[tokio::test]
    async fn test_execute_tool_chain() {
        let mut registry = MiddlewareRegistry::new();
        let hits = Arc::new(AtomicUsize::new(0));
        registry.add_execute_tool(Arc::new(Recorder { hits: hits.clone() }));

        let result = registry
            .run_execute_tool(
                ExecuteToolContext {
                    tool_name: Some("echo".to_string()),
                    tool_use: ToolUse::new("echo", "t1", serde_json::json!({})),
                    invocation_state: InvocationState::new(),
                    cancel_signal: Some(CancelSignal::new()),
                },
                |context| async move {
                    Ok(ExecuteToolResult {
                        result: ToolResult::success(context.tool_use.tool_use_id, "ran"),
                    })
                },
            )
            .await
            .unwrap();

        assert_eq!(result.result.status, ToolResultStatus::Success);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }
}
