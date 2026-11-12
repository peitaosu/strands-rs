//! Multi-agent base types and traits.
//!
//! Provides the foundation for multi-agent patterns (Swarm, Graph).

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::AgentResult;
use crate::types::content::ContentBlock;
use crate::types::errors::Result;
use crate::types::streaming::{Metrics, Usage};

/// Execution status for both graphs and nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Pending,
    /// The node was dispatched for execution but has not completed; prevents
    /// fan-in double-execution on resume.
    Dispatched,
    Executing,
    Completed,
    Failed,
    Interrupted,
}

impl Default for Status {
    fn default() -> Self {
        Self::Pending
    }
}

use crate::types::interrupt::InterruptResponseContent;

/// Input type for multi-agent systems.
///
/// This type can represent:
/// - Plain text input
/// - A sequence of content blocks
/// - Interrupt responses for resuming after an interrupt
#[derive(Debug, Clone)]
pub enum MultiAgentInput {
    /// Plain text input.
    Text(String),
    /// A sequence of content blocks (text, images, etc.).
    ContentBlocks(Vec<ContentBlock>),
    /// Interrupt responses for resuming execution after an interrupt.
    InterruptResponses(Vec<InterruptResponseContent>),
}

impl From<&str> for MultiAgentInput {
    fn from(s: &str) -> Self {
        MultiAgentInput::Text(s.to_string())
    }
}

impl From<String> for MultiAgentInput {
    fn from(s: String) -> Self {
        MultiAgentInput::Text(s)
    }
}

impl From<Vec<ContentBlock>> for MultiAgentInput {
    fn from(blocks: Vec<ContentBlock>) -> Self {
        MultiAgentInput::ContentBlocks(blocks)
    }
}

impl From<Vec<InterruptResponseContent>> for MultiAgentInput {
    fn from(responses: Vec<InterruptResponseContent>) -> Self {
        MultiAgentInput::InterruptResponses(responses)
    }
}

impl MultiAgentInput {
    /// Returns the input as text if it is a Text variant.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MultiAgentInput::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the input as content blocks if it is a ContentBlocks variant.
    pub fn as_content_blocks(&self) -> Option<&[ContentBlock]> {
        match self {
            MultiAgentInput::ContentBlocks(blocks) => Some(blocks),
            _ => None,
        }
    }

    /// Returns the input as interrupt responses if it is an InterruptResponses variant.
    pub fn as_interrupt_responses(&self) -> Option<&[InterruptResponseContent]> {
        match self {
            MultiAgentInput::InterruptResponses(responses) => Some(responses),
            _ => None,
        }
    }

    /// Returns true if this is an interrupt response input.
    pub fn is_interrupt_response(&self) -> bool {
        matches!(self, MultiAgentInput::InterruptResponses(_))
    }

    /// Converts the input to a string, losing type information.
    pub fn to_string_lossy(&self) -> String {
        match self {
            MultiAgentInput::Text(s) => s.clone(),
            MultiAgentInput::ContentBlocks(blocks) => blocks
                .iter()
                .filter_map(|b| b.text.as_ref())
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
            MultiAgentInput::InterruptResponses(responses) => responses
                .iter()
                .map(|r| {
                    format!(
                        "{}:{}",
                        r.interrupt_response.interrupt_id,
                        r.interrupt_response.response
                    )
                })
                .collect::<Vec<_>>()
                .join("; "),
        }
    }
}

/// Interrupt for pausing multi-agent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interrupt {
    pub id: String,
    pub tool_name: String,
    pub tool_use_id: String,
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<serde_json::Value>,
}

impl Interrupt {
    pub fn new(id: impl Into<String>, tool_name: impl Into<String>, tool_use_id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tool_name: tool_name.into(),
            tool_use_id: tool_use_id.into(),
            message: None,
            response: None,
        }
    }

    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    pub fn with_response(mut self, response: serde_json::Value) -> Self {
        self.response = Some(response);
        self
    }

    pub fn has_response(&self) -> bool {
        self.response.is_some()
    }
}

/// Result from a single node execution.
#[derive(Debug, Clone)]
pub struct NodeResult {
    pub result: NodeResultValue,
    pub execution_time_ms: u64,
    pub status: Status,
    pub accumulated_usage: Usage,
    pub accumulated_metrics: Metrics,
    pub execution_count: u32,
    pub interrupts: Vec<Interrupt>,
}

/// The value contained in a NodeResult.
#[derive(Debug, Clone)]
pub enum NodeResultValue {
    Agent(AgentResult),
    MultiAgent(Box<MultiAgentResult>),
    Error(String),
}

impl NodeResult {
    pub fn from_agent(result: AgentResult, execution_time_ms: u64) -> Self {
        Self {
            result: NodeResultValue::Agent(result),
            execution_time_ms,
            status: Status::Completed,
            accumulated_usage: Usage::default(),
            accumulated_metrics: Metrics::default(),
            execution_count: 1,
            interrupts: Vec::new(),
        }
    }

    pub fn from_error(error: impl Into<String>, execution_time_ms: u64) -> Self {
        Self {
            result: NodeResultValue::Error(error.into()),
            execution_time_ms,
            status: Status::Failed,
            accumulated_usage: Usage::default(),
            accumulated_metrics: Metrics::default(),
            execution_count: 1,
            interrupts: Vec::new(),
        }
    }

    pub fn get_agent_results(&self) -> Vec<&AgentResult> {
        match &self.result {
            NodeResultValue::Agent(r) => vec![r],
            NodeResultValue::MultiAgent(m) => m
                .results
                .values()
                .flat_map(|nr| nr.get_agent_results())
                .collect(),
            NodeResultValue::Error(_) => vec![],
        }
    }

    pub fn is_error(&self) -> bool {
        matches!(self.result, NodeResultValue::Error(_))
    }

    pub fn is_interrupted(&self) -> bool {
        self.status == Status::Interrupted
    }
}

/// Result from multi-agent execution with accumulated metrics.
#[derive(Debug, Clone, Default)]
pub struct MultiAgentResult {
    pub status: Status,
    pub results: HashMap<String, NodeResult>,
    pub accumulated_usage: Usage,
    pub accumulated_metrics: Metrics,
    pub execution_count: u32,
    pub execution_time_ms: u64,
    pub interrupts: Vec<Interrupt>,
}

impl MultiAgentResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_status(mut self, status: Status) -> Self {
        self.status = status;
        self
    }

    pub fn add_node_result(&mut self, node_id: impl Into<String>, result: NodeResult) {
        self.accumulated_usage.add(&result.accumulated_usage);
        self.accumulated_metrics.latency_ms += result.accumulated_metrics.latency_ms;
        self.execution_count += result.execution_count;
        self.results.insert(node_id.into(), result);
    }
}

/// Events emitted during multi-agent execution.
#[derive(Debug, Clone)]
pub enum MultiAgentEvent {
    /// A node has started execution.
    NodeStart {
        node_id: String,
        node_type: String,
    },
    /// A node has stopped execution.
    NodeStop {
        node_id: String,
        node_result: NodeResult,
    },
    /// A handoff occurred between nodes.
    Handoff {
        from_node_ids: Vec<String>,
        to_node_ids: Vec<String>,
        message: Option<String>,
    },
    /// Streaming event from a node.
    NodeStream {
        node_id: String,
        event: serde_json::Value,
    },
    /// A node was cancelled.
    NodeCancel {
        node_id: String,
        message: String,
    },
    /// A node was interrupted.
    NodeInterrupt {
        node_id: String,
        interrupts: Vec<Interrupt>,
    },
    /// Final result.
    Result(MultiAgentResult),
}

impl MultiAgentEvent {
    pub fn node_start(node_id: impl Into<String>, node_type: impl Into<String>) -> Self {
        Self::NodeStart {
            node_id: node_id.into(),
            node_type: node_type.into(),
        }
    }

    pub fn node_stop(node_id: impl Into<String>, node_result: NodeResult) -> Self {
        Self::NodeStop {
            node_id: node_id.into(),
            node_result,
        }
    }

    pub fn handoff(
        from_node_ids: Vec<String>,
        to_node_ids: Vec<String>,
        message: Option<String>,
    ) -> Self {
        Self::Handoff {
            from_node_ids,
            to_node_ids,
            message,
        }
    }

    pub fn node_stream(node_id: impl Into<String>, event: serde_json::Value) -> Self {
        Self::NodeStream {
            node_id: node_id.into(),
            event,
        }
    }

    pub fn node_cancel(node_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::NodeCancel {
            node_id: node_id.into(),
            message: message.into(),
        }
    }

    pub fn node_interrupt(node_id: impl Into<String>, interrupts: Vec<Interrupt>) -> Self {
        Self::NodeInterrupt {
            node_id: node_id.into(),
            interrupts,
        }
    }

    pub fn result(result: MultiAgentResult) -> Self {
        Self::Result(result)
    }

    pub fn is_result(&self) -> bool {
        matches!(self, Self::Result(_))
    }

    pub fn as_result(&self) -> Option<&MultiAgentResult> {
        match self {
            Self::Result(r) => Some(r),
            _ => None,
        }
    }
}

/// Stream of multi-agent events.
pub type MultiAgentEventStream<'a> =
    std::pin::Pin<Box<dyn futures::Stream<Item = MultiAgentEvent> + Send + 'a>>;

/// Invocation state passed through multi-agent execution.
#[derive(Debug, Clone, Default)]
pub struct InvocationState {
    pub data: HashMap<String, serde_json::Value>,
}

impl InvocationState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.data.get(key).and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl serde::Serialize) {
        if let Ok(v) = serde_json::to_value(value) {
            self.data.insert(key.into(), v);
        }
    }
}

/// Base trait for multi-agent orchestration patterns.
#[async_trait]
pub trait MultiAgentBase: Send + Sync {
    /// Returns the unique ID of this multi-agent system.
    fn id(&self) -> &str;

    /// Invokes the multi-agent system asynchronously.
    async fn invoke_async(
        &mut self,
        task: MultiAgentInput,
        invocation_state: Option<&InvocationState>,
    ) -> Result<MultiAgentResult>;

    /// Streams events during multi-agent execution.
    fn stream_async<'a>(
        &'a mut self,
        task: MultiAgentInput,
        invocation_state: Option<&'a InvocationState>,
    ) -> MultiAgentEventStream<'a>;

    /// Serializes the current state for persistence.
    fn serialize_state(&self) -> serde_json::Value;

    /// Deserializes state from persistence.
    fn deserialize_state(&mut self, payload: &serde_json::Value) -> Result<()>;
}

/// State tracking for interrupt handling.
#[derive(Debug, Clone, Default)]
pub struct InterruptState {
    pub activated: bool,
    pub interrupts: HashMap<String, Interrupt>,
    pub context: HashMap<String, serde_json::Value>,
    pub responses: Option<serde_json::Value>,
}

impl InterruptState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn activate(&mut self) {
        self.activated = true;
    }

    pub fn deactivate(&mut self) {
        self.activated = false;
        self.interrupts.clear();
        self.context.clear();
        self.responses = None;
    }

    pub fn resume(&mut self, responses: serde_json::Value) {
        self.responses = Some(responses);
    }

    /// Add an interrupt to the state.
    pub fn add(&mut self, interrupt: Interrupt) {
        self.interrupts.insert(interrupt.id.clone(), interrupt);
    }

    /// Serialize to a dictionary for session persistence.
    pub fn to_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut dict = HashMap::new();
        dict.insert("activated".to_string(), serde_json::json!(self.activated));
        dict.insert(
            "interrupts".to_string(),
            serde_json::json!(self.interrupts
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::json!({
                    "id": v.id,
                    "tool_name": v.tool_name,
                    "tool_use_id": v.tool_use_id,
                    "message": v.message,
                    "response": v.response,
                })))
                .collect::<HashMap<_, _>>()),
        );
        dict.insert("context".to_string(), serde_json::json!(self.context));
        dict.insert("responses".to_string(), serde_json::json!(self.responses));
        dict
    }

    /// Restore from a dictionary.
    pub fn from_dict(data: HashMap<String, serde_json::Value>) -> Self {
        let activated = data
            .get("activated")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let interrupts = data
            .get("interrupts")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        let id = v.get("id")?.as_str()?.to_string();
                        let tool_name = v.get("tool_name")?.as_str()?.to_string();
                        let tool_use_id = v.get("tool_use_id")?.as_str()?.to_string();
                        let message = v.get("message").and_then(|m| m.as_str().map(|s| s.to_string()));
                        let response = v.get("response").cloned();
                        Some((k.clone(), Interrupt {
                            id,
                            tool_name,
                            tool_use_id,
                            message,
                            response,
                        }))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let context = data
            .get("context")
            .and_then(|v| v.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let responses = data.get("responses").cloned();

        Self {
            activated,
            interrupts,
            context,
            responses,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_default() {
        assert_eq!(Status::default(), Status::Pending);
    }

    #[test]
    fn test_multi_agent_input_from_str() {
        let input = MultiAgentInput::from("test task");
        assert_eq!(input.as_text(), Some("test task"));
    }

    #[test]
    fn test_interrupt_creation() {
        let interrupt = Interrupt::new("int-1", "my_tool", "tu-1")
            .with_message("Please provide more info");
        assert_eq!(interrupt.id, "int-1");
        assert_eq!(interrupt.message, Some("Please provide more info".to_string()));
    }

    #[test]
    fn test_multi_agent_event_variants() {
        let event = MultiAgentEvent::node_start("node1", "agent");
        assert!(!event.is_result());

        let result = MultiAgentResult::new();
        let event = MultiAgentEvent::result(result);
        assert!(event.is_result());
    }
}

