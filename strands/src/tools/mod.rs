//! Tool definitions and execution.

pub mod executor;
pub mod helpers;
pub mod loader;
pub mod mcp;
pub mod mcp_auth;
pub mod mcp_instrumentation;
pub mod mcp_tasks;
pub mod registry;
pub mod structured_output;
pub mod validator;
pub mod watcher;

use std::pin::Pin;

use futures::Stream;

use async_trait::async_trait;

use crate::types::tools::{ToolResult, ToolResultContent, ToolResultStatus, ToolSpec, ToolUse};

/// A stream of tool execution events.
pub type ToolEventStream = Pin<Box<dyn Stream<Item = ToolEvent> + Send>>;

/// Generator type for tool execution streams.
pub type ToolGenerator = ToolEventStream;

/// Events emitted during tool execution.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    /// Progress update during execution.
    Progress { message: String },
    /// Streaming data from the tool.
    Stream(serde_json::Value),
    /// Final result of tool execution.
    Result(ToolResult),
    /// Interrupt request from the tool.
    Interrupt { id: String, data: serde_json::Value },
}

impl ToolEvent {
    pub fn progress(message: impl Into<String>) -> Self {
        Self::Progress { message: message.into() }
    }

    pub fn stream(data: serde_json::Value) -> Self { Self::Stream(data) }
    pub fn result(result: ToolResult) -> Self { Self::Result(result) }
    pub fn is_result(&self) -> bool { matches!(self, Self::Result(_)) }

    pub fn as_result(&self) -> Option<&ToolResult> {
        match self {
            Self::Result(r) => Some(r),
            _ => None,
        }
    }
}

/// State passed through tool invocations.
///
/// The data map is shared across clones and mutable through `&self`,
/// mirroring the Python invocation-state dict that tools update live.
#[derive(Debug, Clone, Default)]
pub struct InvocationState {
    data: std::sync::Arc<std::sync::RwLock<std::collections::HashMap<String, serde_json::Value>>>,
    stop_event_loop: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl InvocationState {
    pub fn new() -> Self { Self::default() }

    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.data
            .read()
            .unwrap()
            .get(key)
            .and_then(|v| T::deserialize(v).ok())
    }

    pub fn set(&self, key: impl Into<String>, value: impl serde::Serialize) {
        if let Ok(v) = serde_json::to_value(value) {
            self.data.write().unwrap().insert(key.into(), v);
        }
    }

    /// Returns a snapshot of the state map.
    pub fn get_all(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.data.read().unwrap().clone()
    }

    /// Requests the event loop stop after the current tool cycle.
    pub fn request_stop(&self) {
        self.stop_event_loop
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Returns true when a stop was requested, via the flag or the legacy
    /// `stop_event_loop` data key.
    pub fn should_stop(&self) -> bool {
        self.stop_event_loop
            .load(std::sync::atomic::Ordering::SeqCst)
            || self
                .data
                .read()
                .unwrap()
                .get("stop_event_loop")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
    }
}

impl serde::Serialize for InvocationState {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.data.read().unwrap().serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for InvocationState {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let data = std::collections::HashMap::<String, serde_json::Value>::deserialize(deserializer)?;
        Ok(Self {
            data: std::sync::Arc::new(std::sync::RwLock::new(data)),
            stop_event_loop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }
}

/// Context provided to tool execution.
///
/// Alias for the canonical [`crate::types::tools::ToolContext`].
pub use crate::types::tools::ToolContext;

/// Result returned from async tool invocation.
#[derive(Debug, Clone)]
pub struct ToolResult2 {
    pub status: ToolResultStatus,
    pub content: Vec<ToolResultContent>,
}

impl ToolResult2 {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            status: ToolResultStatus::Success,
            content: vec![ToolResultContent::text(content.into())],
        }
    }

    pub fn success_json(value: serde_json::Value) -> Self {
        Self {
            status: ToolResultStatus::Success,
            content: vec![ToolResultContent::json(value)],
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            status: ToolResultStatus::Error,
            content: vec![ToolResultContent::text(message.into())],
        }
    }
}

/// Trait for implementing agent tools.
#[async_trait]
pub trait AgentTool: Send + Sync {
    /// Returns the unique name of the tool.
    fn name(&self) -> &str;

    /// Returns the tool description.
    fn description(&self) -> &str;

    /// Returns the tool specification.
    fn tool_spec(&self) -> ToolSpec;

    /// Invokes the tool asynchronously.
    async fn invoke(
        &self,
        input: serde_json::Value,
        context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String>;

    /// Legacy name accessor.
    fn tool_name(&self) -> &str { self.name() }

    /// Returns the type of the tool (e.g., "function", "python").
    fn tool_type(&self) -> &str { "function" }

    /// Whether the tool supports hot reloading.
    fn supports_hot_reload(&self) -> bool { false }

    /// Whether this is a dynamically loaded tool.
    fn is_dynamic(&self) -> bool { false }

    /// Returns display properties for the tool.
    fn get_display_properties(&self) -> std::collections::HashMap<String, String> {
        let mut props = std::collections::HashMap::new();
        props.insert("Name".to_string(), self.name().to_string());
        props.insert("Type".to_string(), self.tool_type().to_string());
        props
    }
}

/// Executes an agent tool and returns a stream of events.
pub fn tool_to_stream(
    tool: std::sync::Arc<dyn AgentTool>,
    tool_use: ToolUse,
    invocation_state: InvocationState,
) -> ToolGenerator {
    let input = tool_use.input.clone();
    let tool_use_id = tool_use.tool_use_id.clone();
    let context = ToolContext::with_state(tool_use, invocation_state);

    Box::pin(async_stream::stream! {
        let result = match tool.invoke(input, &context).await {
            Ok(r) => ToolResult {
                tool_use_id,
                status: r.status,
                content: r.content,
            },
            Err(e) => ToolResult {
                tool_use_id,
                status: ToolResultStatus::Error,
                content: vec![ToolResultContent::text(e)],
            },
        };
        yield ToolEvent::Result(result);
    })
}

/// Trait for dynamically loaded tools.
pub trait DynamicAgentTool: AgentTool {
    /// Marks the tool as dynamic.
    fn mark_dynamic(&mut self);
}

/// Executes a tool and returns its event stream.
pub fn execute_tool_stream(
    tool: std::sync::Arc<dyn AgentTool>,
    tool_use: ToolUse,
    invocation_state: InvocationState,
) -> ToolGenerator {
    tool_to_stream(tool, tool_use, invocation_state)
}

pub use loader::{ReloadCallback, ToolLoader, ToolLoaderConfig, ToolWatcher};
pub use mcp_auth::{BearerTokenSource, MCPClientCredentials};
pub use mcp_tasks::{
    parse_task, task_params, tasks_get_params, MCPTask, MCPTaskState, TasksConfig,
    DEFAULT_POLL_INTERVAL, TASKS_EXTENSION, TASKS_PROTOCOL_VERSION, TERMINAL_STATES,
};
pub use mcp::{
    ConnectionState, MCPClient, MCPImageContent, MCPImageSource, MCPResultContent,
    MCPServerConfig, MCPToolResult, MCPToolSpec, MCPTransport, ToolFilters, ToolProvider,
};
pub use registry::{ToolInput, ToolRegistry};
pub use structured_output::{
    flatten_schema, get_required_fields, process_schema_for_optional_fields, schema_to_tool_spec,
    structured_output_spec, validate_against_schema, StructuredOutputContext, StructuredOutputResult,
    StructuredOutputTool,
};
pub use helpers::{
    generate_cancelled_tool_result, generate_missing_tool_result,
    generate_missing_tool_result_content, generate_missing_tool_results_for_message,
    generate_timeout_tool_result, noop_tool, noop_tool_with, NoopTool,
};
pub use validator::{
    is_valid_tool_name, sanitize_tool_name, validate_and_prepare_tools, validate_tool_spec,
    validate_tool_specs, validate_tool_use, validate_tool_uses, ToolUseValidationResult,
    MAX_TOOL_NAME_LENGTH, MIN_TOOL_NAME_LENGTH,
};
pub use mcp_instrumentation::{
    create_mcp_tool_span, extract_trace_context, init_mcp_instrumentation, inject_trace_context,
    is_instrumentation_applied, ExtractableContext, InjectableContext, ItemWithContext,
    MCPInstrumentationConfig, InstrumentationGuard,
};


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct TestTool;

    #[async_trait]
    impl AgentTool for TestTool {
        fn name(&self) -> &str { "test_tool" }
        fn description(&self) -> &str { "A test tool" }
        fn tool_spec(&self) -> ToolSpec { ToolSpec::new("test_tool", "A test tool") }

        async fn invoke(
            &self,
            _input: serde_json::Value,
            _context: &ToolContext,
        ) -> std::result::Result<ToolResult2, String> {
            Ok(ToolResult2::success("Test result"))
        }
    }

    #[tokio::test]
    async fn test_tool_execution() {
        use futures::StreamExt;

        let tool: Arc<dyn AgentTool> = Arc::new(TestTool);
        let tool_use = ToolUse::new("test_tool", "123", serde_json::json!({}));
        let state = InvocationState::new();
        let mut stream = tool_to_stream(tool, tool_use, state);

        if let Some(event) = stream.next().await {
            assert!(event.is_result());
            assert!(event.as_result().unwrap().is_success());
        }
    }
}
