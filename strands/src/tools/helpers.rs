//! Tool helper utilities.
//!
//! This module provides helper functions for tool operations, including:
//! - Creating no-op tools for testing or placeholders
//! - Generating missing tool results for unhandled tool uses

use std::sync::Arc;

use async_trait::async_trait;

use super::{AgentTool, ToolContext, ToolResult2};
use crate::types::content::Message;
use crate::types::tools::{ToolResult, ToolResultContent, ToolResultStatus, ToolSpec, ToolUse};

/// A no-operation tool that always succeeds with a configurable message.
///
/// Useful for testing, placeholders, or as a fallback when actual tool
/// implementation is not available.
#[derive(Debug, Clone)]
pub struct NoopTool {
    name: String,
    description: String,
    response_message: String,
}

impl NoopTool {
    /// Creates a new NoopTool with the given name and a default response.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            name: name.clone(),
            description: format!("No-op tool: {name}"),
            response_message: "Tool executed successfully (no-op)".to_string(),
        }
    }

    /// Creates a NoopTool with a custom description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Creates a NoopTool with a custom response message.
    pub fn with_response(mut self, message: impl Into<String>) -> Self {
        self.response_message = message.into();
        self
    }
}

#[async_trait]
impl AgentTool for NoopTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(&self.name, &self.description)
    }

    async fn invoke(
        &self,
        _input: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult2, String> {
        Ok(ToolResult2::success(&self.response_message))
    }
}

/// Creates a no-op tool with the given name.
///
/// This is a convenience function for creating `NoopTool` instances.
///
/// # Example
///
/// ```ignore
/// let tool = noop_tool("placeholder_tool");
/// ```
pub fn noop_tool(name: impl Into<String>) -> Arc<dyn AgentTool> {
    Arc::new(NoopTool::new(name))
}

/// Creates a no-op tool with custom configuration.
///
/// # Example
///
/// ```ignore
/// let tool = noop_tool_with("test_tool")
///     .with_description("A test tool")
///     .with_response("Test complete");
/// ```
pub fn noop_tool_with(name: impl Into<String>) -> NoopTool {
    NoopTool::new(name)
}

/// Generates tool result content for a missing or unhandled tool.
///
/// This function creates appropriate error content when a tool use references
/// a tool that is not registered or available in the current context.
///
/// # Arguments
///
/// * `tool_name` - The name of the missing tool
///
/// # Returns
///
/// A vector of `ToolResultContent` containing an error message.
pub fn generate_missing_tool_result_content(tool_name: &str) -> Vec<ToolResultContent> {
    vec![ToolResultContent::text(format!(
        "Error: Tool '{}' is not available. The tool may not be registered or has been removed.",
        tool_name
    ))]
}

/// Generates a complete `ToolResult` for a missing tool.
///
/// # Arguments
///
/// * `tool_use` - The tool use that references the missing tool
///
/// # Returns
///
/// A `ToolResult` with error status and appropriate message.
pub fn generate_missing_tool_result(tool_use: &ToolUse) -> ToolResult {
    ToolResult {
        tool_use_id: tool_use.tool_use_id.clone(),
        status: ToolResultStatus::Error,
        content: generate_missing_tool_result_content(&tool_use.name),
    }
}

/// Generates tool results for all tool uses in a message that reference missing tools.
///
/// This function scans the message for tool use content blocks and generates
/// error results for any tools that are not found in the provided list of
/// known tool names.
///
/// # Arguments
///
/// * `message` - The message containing tool use content blocks
/// * `known_tools` - A slice of known tool names
///
/// # Returns
///
/// A vector of `ToolResult` for each missing tool, empty if all tools are found.
pub fn generate_missing_tool_results_for_message(
    message: &Message,
    known_tools: &[&str],
) -> Vec<ToolResult> {
    let mut results = Vec::new();

    for content in &message.content {
        if let Some(tool_use) = content.as_tool_use() {
            if !known_tools.contains(&tool_use.name.as_str()) {
                results.push(generate_missing_tool_result(tool_use));
            }
        }
    }

    results
}

/// Creates a placeholder tool result indicating the tool was cancelled.
///
/// # Arguments
///
/// * `tool_use_id` - The ID of the tool use that was cancelled
/// * `reason` - Optional reason for cancellation
///
/// # Returns
///
/// A `ToolResult` with error status indicating cancellation.
pub fn generate_cancelled_tool_result(tool_use_id: &str, reason: Option<&str>) -> ToolResult {
    let message = reason.unwrap_or("Tool execution was cancelled");
    ToolResult {
        tool_use_id: tool_use_id.to_string(),
        status: ToolResultStatus::Error,
        content: vec![ToolResultContent::text(format!("Cancelled: {}", message))],
    }
}

/// Creates a tool result for when tool execution times out.
///
/// # Arguments
///
/// * `tool_use_id` - The ID of the tool use that timed out
/// * `timeout_seconds` - The timeout duration that was exceeded
///
/// # Returns
///
/// A `ToolResult` with error status indicating timeout.
pub fn generate_timeout_tool_result(tool_use_id: &str, timeout_seconds: u64) -> ToolResult {
    ToolResult {
        tool_use_id: tool_use_id.to_string(),
        status: ToolResultStatus::Error,
        content: vec![ToolResultContent::text(format!(
            "Error: Tool execution timed out after {} seconds",
            timeout_seconds
        ))],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::content::{ContentBlock, Role};

    #[tokio::test]
    async fn test_noop_tool_execution() {
        let tool = noop_tool("test_noop");
        let context = ToolContext::default();
        let result = tool.invoke(serde_json::json!({}), &context).await.unwrap();
        
        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(!result.content.is_empty());
    }

    #[tokio::test]
    async fn test_noop_tool_with_custom_response() {
        let tool = noop_tool_with("custom_noop")
            .with_description("Custom description")
            .with_response("Custom response");
        
        assert_eq!(tool.name(), "custom_noop");
        assert_eq!(tool.description(), "Custom description");
        
        let context = ToolContext::default();
        let result = tool.invoke(serde_json::json!({}), &context).await.unwrap();
        
        assert_eq!(result.status, ToolResultStatus::Success);
    }

    #[test]
    fn test_generate_missing_tool_result_content() {
        let content = generate_missing_tool_result_content("unknown_tool");
        assert_eq!(content.len(), 1);
        
        if let Some(text) = content[0].text.as_ref() {
            assert!(text.contains("unknown_tool"));
            assert!(text.contains("not available"));
        }
    }

    #[test]
    fn test_generate_missing_tool_result() {
        let tool_use = ToolUse::new("missing_tool", "123", serde_json::json!({}));
        let result = generate_missing_tool_result(&tool_use);
        
        assert_eq!(result.tool_use_id, "123");
        assert_eq!(result.status, ToolResultStatus::Error);
    }

    #[test]
    fn test_generate_cancelled_tool_result() {
        let result = generate_cancelled_tool_result("456", Some("User requested stop"));
        
        assert_eq!(result.tool_use_id, "456");
        assert_eq!(result.status, ToolResultStatus::Error);
        if let Some(text) = result.content[0].text.as_ref() {
            assert!(text.contains("Cancelled"));
        }
    }

    #[test]
    fn test_generate_timeout_tool_result() {
        let result = generate_timeout_tool_result("789", 30);
        
        assert_eq!(result.tool_use_id, "789");
        assert_eq!(result.status, ToolResultStatus::Error);
        if let Some(text) = result.content[0].text.as_ref() {
            assert!(text.contains("timed out"));
            assert!(text.contains("30"));
        }
    }

    #[test]
    fn test_generate_missing_tool_results_for_message() {
        let tool_use = ToolUse::new("unknown_tool", "123", serde_json::json!({}));
        let message = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::tool_use(tool_use)],
        };

        let known_tools = vec!["known_tool_1", "known_tool_2"];
        let results = generate_missing_tool_results_for_message(&message, &known_tools);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, ToolResultStatus::Error);
    }
}

