//! Tool-related types for agent tools.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// JSON Schema type alias.
pub type JsonSchema = serde_json::Value;

/// Specification for an agent tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: InputSchema,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<JsonSchema>,
}

impl ToolSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: InputSchema::default(),
            output_schema: None,
        }
    }

    pub fn with_input_schema(mut self, schema: JsonSchema) -> Self {
        self.input_schema = InputSchema { json: schema };
        self
    }

    pub fn with_output_schema(mut self, schema: JsonSchema) -> Self {
        self.output_schema = Some(schema);
        self
    }
}

/// Input schema for a tool.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct InputSchema {
    pub json: JsonSchema,
}

/// A tool configuration containing its specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub tool_spec: ToolSpec,
}

/// A request to use a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolUse {
    pub name: String,
    pub tool_use_id: String,
    pub input: serde_json::Value,
}

impl ToolUse {
    pub fn new(name: impl Into<String>, tool_use_id: impl Into<String>, input: serde_json::Value) -> Self {
        Self { name: name.into(), tool_use_id: tool_use_id.into(), input }
    }

    pub fn get_param<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.input.get(key).and_then(|v| T::deserialize(v).ok())
    }
}

/// Content returned from a tool execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageResultContent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<DocumentResultContent>,
}

impl ToolResultContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self { text: Some(text.into()), ..Default::default() }
    }

    pub fn json(value: serde_json::Value) -> Self {
        Self { json: Some(value), ..Default::default() }
    }
}

/// Image content in a tool result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageResultContent {
    pub format: String,
    pub data: String,
}

/// Document content in a tool result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentResultContent {
    pub format: String,
    pub name: String,
    pub data: String,
}

/// Status of a tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolResultStatus {
    Success,
    Error,
}

impl ToolResultStatus {
    /// Returns the string representation of the status.
    pub fn as_str(&self) -> &'static str {
        match self {
            ToolResultStatus::Success => "success",
            ToolResultStatus::Error => "error",
        }
    }
}

impl std::fmt::Display for ToolResultStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Result from a tool execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResult {
    pub tool_use_id: String,
    pub status: ToolResultStatus,
    pub content: Vec<ToolResultContent>,
}

impl ToolResult {
    pub fn success(tool_use_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            status: ToolResultStatus::Success,
            content: vec![ToolResultContent::text(text)],
        }
    }

    pub fn success_json(tool_use_id: impl Into<String>, json: serde_json::Value) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            status: ToolResultStatus::Success,
            content: vec![ToolResultContent::json(json)],
        }
    }

    pub fn error(tool_use_id: impl Into<String>, error_message: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            status: ToolResultStatus::Error,
            content: vec![ToolResultContent::text(error_message)],
        }
    }

    pub fn is_success(&self) -> bool { self.status == ToolResultStatus::Success }
    pub fn is_error(&self) -> bool { self.status == ToolResultStatus::Error }
}

/// Auto tool choice - model decides whether to use tools.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceAuto {}

/// Any tool choice - model must use at least one tool.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceAny {}

/// Specific tool choice - model must use the named tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceTool {
    pub name: String,
}

/// Configuration for how the model should use tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto(ToolChoiceAuto),
    Any(ToolChoiceAny),
    Tool(ToolChoiceTool),
}

impl Default for ToolChoice {
    fn default() -> Self { Self::Auto(ToolChoiceAuto {}) }
}

impl ToolChoice {
    pub fn auto() -> Self { Self::Auto(ToolChoiceAuto {}) }
    pub fn any() -> Self { Self::Any(ToolChoiceAny {}) }
    pub fn tool(name: impl Into<String>) -> Self { Self::Tool(ToolChoiceTool { name: name.into() }) }
}

/// Tool configuration for a model request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub tools: Vec<Tool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

/// Context provided to a tool during execution.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub tool_use: ToolUse,
    pub invocation_state: HashMap<String, serde_json::Value>,
}

impl ToolContext {
    pub fn new(tool_use: ToolUse) -> Self {
        Self { tool_use, invocation_state: HashMap::new() }
    }

    pub fn with_state(tool_use: ToolUse, state: HashMap<String, serde_json::Value>) -> Self {
        Self { tool_use, invocation_state: state }
    }

    pub fn get_state<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.invocation_state.get(key).and_then(|v| T::deserialize(v).ok())
    }

    pub fn interrupt_id(&self, name: &str) -> String {
        format!(
            "v1:tool_call:{}:{}",
            self.tool_use.tool_use_id,
            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, name.as_bytes())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_spec_creation() {
        let spec = ToolSpec::new("get_weather", "Get weather for a location");
        assert_eq!(spec.name, "get_weather");
        assert_eq!(spec.description, "Get weather for a location");
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("123", "Weather is sunny");
        assert!(result.is_success());
        assert!(!result.is_error());
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("123", "Failed to fetch weather");
        assert!(result.is_error());
        assert!(!result.is_success());
    }

    #[test]
    fn test_tool_choice_variants() {
        let auto = ToolChoice::auto();
        assert!(matches!(auto, ToolChoice::Auto(_)));

        let any = ToolChoice::any();
        assert!(matches!(any, ToolChoice::Any(_)));

        let specific = ToolChoice::tool("my_tool");
        assert!(matches!(specific, ToolChoice::Tool(t) if t.name == "my_tool"));
    }

    #[test]
    fn test_tool_result_content_serialization() {
        let content = ToolResultContent::text("hello");
        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(json, r#"{"text":"hello"}"#);
    }

    #[test]
    fn test_tool_choice_serialization() {
        let auto = ToolChoice::auto();
        let json = serde_json::to_string(&auto).unwrap();
        assert_eq!(json, r#"{"auto":{}}"#);

        let any = ToolChoice::any();
        let json = serde_json::to_string(&any).unwrap();
        assert_eq!(json, r#"{"any":{}}"#);

        let tool = ToolChoice::tool("my_tool");
        let json = serde_json::to_string(&tool).unwrap();
        assert_eq!(json, r#"{"tool":{"name":"my_tool"}}"#);
    }
}
