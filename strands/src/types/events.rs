//! Typed event system for the Strands Agents framework.
//!
//! This module defines the event types that are emitted during agent execution,
//! providing a structured way to observe different events of the event loop and
//! agent lifecycle.

use serde::{Deserialize, Serialize};

use crate::types::content::Message;
use crate::types::interrupt::Interrupt;
use crate::types::streaming::{ContentBlockDelta, Metrics, StopReason, StreamEvent, Usage};
use crate::types::tools::{ToolResult, ToolUse};
use crate::types::citations::Citation;

/// Base trait for all typed events in the agent system.
pub trait TypedEvent: Send + Sync {
    /// True if this event should trigger the callback_handler to fire.
    fn is_callback_event(&self) -> bool { true }

    /// Convert this event to a JSON-serializable value.
    fn as_dict(&self) -> serde_json::Value;
}

/// Event emitted at the very beginning of agent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitEventLoopEvent {
    pub init_event_loop: bool,
}

impl InitEventLoopEvent {
    pub fn new() -> Self {
        Self { init_event_loop: true }
    }
}

impl Default for InitEventLoopEvent {
    fn default() -> Self { Self::new() }
}

impl TypedEvent for InitEventLoopEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "init_event_loop": self.init_event_loop })
    }
}

/// Event emitted at the start of each event loop cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartEvent {
    pub start: bool,
}

impl StartEvent {
    pub fn new() -> Self {
        Self { start: true }
    }
}

impl Default for StartEvent {
    fn default() -> Self { Self::new() }
}

impl TypedEvent for StartEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "start": self.start })
    }
}

/// Event emitted when the event loop cycle begins processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartEventLoopEvent {
    pub start_event_loop: bool,
}

impl StartEventLoopEvent {
    pub fn new() -> Self {
        Self { start_event_loop: true }
    }
}

impl Default for StartEventLoopEvent {
    fn default() -> Self { Self::new() }
}

impl TypedEvent for StartEventLoopEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "start_event_loop": self.start_event_loop })
    }
}

/// Event emitted during model response streaming for each raw chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStreamChunkEvent {
    pub event: StreamEvent,
}

impl ModelStreamChunkEvent {
    pub fn new(chunk: StreamEvent) -> Self {
        Self { event: chunk }
    }

    pub fn chunk(&self) -> &StreamEvent {
        &self.event
    }
}

impl TypedEvent for ModelStreamChunkEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "event": self.event })
    }
}

/// Event emitted during model response streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStreamEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<ContentBlockDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_tool_use: Option<serde_json::Value>,
}

impl ModelStreamEvent {
    pub fn text(delta: ContentBlockDelta, text: String) -> Self {
        Self {
            data: Some(text),
            delta: Some(delta),
            reasoning_text: None,
            current_tool_use: None,
        }
    }

    pub fn tool_use(delta: ContentBlockDelta, current_tool_use: serde_json::Value) -> Self {
        Self {
            data: None,
            delta: Some(delta),
            reasoning_text: None,
            current_tool_use: Some(current_tool_use),
        }
    }

    pub fn reasoning(delta: ContentBlockDelta, reasoning_text: String) -> Self {
        Self {
            data: None,
            delta: Some(delta),
            reasoning_text: Some(reasoning_text),
            current_tool_use: None,
        }
    }
}

impl Default for ModelStreamEvent {
    fn default() -> Self {
        Self {
            data: None,
            delta: None,
            reasoning_text: None,
            current_tool_use: None,
        }
    }
}

impl TypedEvent for ModelStreamEvent {
    fn is_callback_event(&self) -> bool {
        self.data.is_some() || self.reasoning_text.is_some() || self.current_tool_use.is_some()
    }

    fn as_dict(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }
}

/// Event emitted during text content streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStreamEvent {
    pub data: String,
    pub delta: ContentBlockDelta,
}

impl TextStreamEvent {
    pub fn new(delta: ContentBlockDelta, text: String) -> Self {
        Self { data: text, delta }
    }
}

impl TypedEvent for TextStreamEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "data": self.data, "delta": self.delta })
    }
}

/// Event emitted during citation streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationStreamEvent {
    pub citation: Citation,
    pub delta: ContentBlockDelta,
}

impl CitationStreamEvent {
    pub fn new(delta: ContentBlockDelta, citation: Citation) -> Self {
        Self { citation, delta }
    }
}

impl TypedEvent for CitationStreamEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "citation": self.citation, "delta": self.delta })
    }
}

/// Event emitted during reasoning text streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTextStreamEvent {
    pub reasoning_text: Option<String>,
    pub delta: ContentBlockDelta,
    pub reasoning: bool,
}

impl ReasoningTextStreamEvent {
    pub fn new(delta: ContentBlockDelta, reasoning_text: Option<String>) -> Self {
        Self { reasoning_text, delta, reasoning: true }
    }
}

impl TypedEvent for ReasoningTextStreamEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "reasoningText": self.reasoning_text, "delta": self.delta, "reasoning": self.reasoning })
    }
}

/// Event emitted when model invocation stops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStopReason {
    pub stop_reason: StopReason,
    pub message: Message,
    pub usage: Usage,
    pub metrics: Metrics,
}

impl ModelStopReason {
    pub fn new(stop_reason: StopReason, message: Message, usage: Usage, metrics: Metrics) -> Self {
        Self { stop_reason, message, usage, metrics }
    }
}

impl TypedEvent for ModelStopReason {
    fn is_callback_event(&self) -> bool { false }

    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "stop": [self.stop_reason, self.message, self.usage, self.metrics] })
    }
}

/// Event emitted when the agent execution completes normally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLoopStopEvent {
    pub stop_reason: StopReason,
    pub message: Message,
    pub request_state: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interrupts: Option<Vec<Interrupt>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<serde_json::Value>,
}

impl EventLoopStopEvent {
    pub fn new(
        stop_reason: StopReason,
        message: Message,
        request_state: serde_json::Value,
    ) -> Self {
        Self {
            stop_reason,
            message,
            request_state,
            interrupts: None,
            structured_output: None,
        }
    }

    pub fn with_interrupts(mut self, interrupts: Vec<Interrupt>) -> Self {
        self.interrupts = Some(interrupts);
        self
    }

    pub fn with_structured_output(mut self, output: serde_json::Value) -> Self {
        self.structured_output = Some(output);
        self
    }
}

impl TypedEvent for EventLoopStopEvent {
    fn is_callback_event(&self) -> bool { false }

    fn as_dict(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }
}

/// Event emitted when the event loop is throttled due to rate limiting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLoopThrottleEvent {
    pub event_loop_throttled_delay: u64,
}

impl EventLoopThrottleEvent {
    pub fn new(delay: u64) -> Self {
        Self { event_loop_throttled_delay: delay }
    }
}

impl TypedEvent for EventLoopThrottleEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "event_loop_throttled_delay": self.event_loop_throttled_delay })
    }
}

/// Event emitted when a tool execution completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultEvent {
    pub tool_result: ToolResult,
}

impl ToolResultEvent {
    pub fn new(tool_result: ToolResult) -> Self {
        Self { tool_result }
    }

    pub fn tool_use_id(&self) -> &str {
        &self.tool_result.tool_use_id
    }
}

impl TypedEvent for ToolResultEvent {
    fn is_callback_event(&self) -> bool { false }

    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "type": "tool_result", "tool_result": self.tool_result })
    }
}

/// Event emitted when a tool yields sub-events as part of tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolStreamEvent {
    pub tool_use: ToolUse,
    pub data: serde_json::Value,
}

impl ToolStreamEvent {
    pub fn new(tool_use: ToolUse, data: serde_json::Value) -> Self {
        Self { tool_use, data }
    }

    pub fn tool_use_id(&self) -> &str {
        &self.tool_use.tool_use_id
    }
}

impl TypedEvent for ToolStreamEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "type": "tool_stream", "tool_stream_event": { "tool_use": self.tool_use, "data": self.data } })
    }
}

/// Event emitted when a user cancels a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCancelEvent {
    pub tool_use: ToolUse,
    pub message: String,
}

impl ToolCancelEvent {
    pub fn new(tool_use: ToolUse, message: String) -> Self {
        Self { tool_use, message }
    }

    pub fn tool_use_id(&self) -> &str {
        &self.tool_use.tool_use_id
    }
}

impl TypedEvent for ToolCancelEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "tool_cancel_event": { "tool_use": self.tool_use, "message": self.message } })
    }
}

/// Event emitted when a tool is interrupted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInterruptEvent {
    pub tool_use: ToolUse,
    pub interrupts: Vec<Interrupt>,
}

impl ToolInterruptEvent {
    pub fn new(tool_use: ToolUse, interrupts: Vec<Interrupt>) -> Self {
        Self { tool_use, interrupts }
    }

    pub fn tool_use_id(&self) -> &str {
        &self.tool_use.tool_use_id
    }
}

impl TypedEvent for ToolInterruptEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "tool_interrupt_event": { "tool_use": self.tool_use, "interrupts": self.interrupts } })
    }
}

/// Event emitted when the model invocation has completed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMessageEvent {
    pub message: Message,
}

impl ModelMessageEvent {
    pub fn new(message: Message) -> Self {
        Self { message }
    }
}

impl TypedEvent for ModelMessageEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "message": self.message })
    }
}

/// Event emitted when tool results are formatted as a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultMessageEvent {
    pub message: Message,
}

impl ToolResultMessageEvent {
    pub fn new(message: Message) -> Self {
        Self { message }
    }
}

impl TypedEvent for ToolResultMessageEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "message": self.message })
    }
}

/// Event emitted when the agent execution is forcibly stopped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceStopEvent {
    pub force_stop: bool,
    pub force_stop_reason: String,
}

impl ForceStopEvent {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            force_stop: true,
            force_stop_reason: reason.into(),
        }
    }
}

impl TypedEvent for ForceStopEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "force_stop": self.force_stop, "force_stop_reason": self.force_stop_reason })
    }
}

/// Multi-agent event emitted when node execution starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentNodeStartEvent {
    pub node_id: String,
    pub node_type: String,
}

impl MultiAgentNodeStartEvent {
    pub fn new(node_id: impl Into<String>, node_type: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            node_type: node_type.into(),
        }
    }
}

impl TypedEvent for MultiAgentNodeStartEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "type": "multiagent_node_start", "node_id": self.node_id, "node_type": self.node_type })
    }
}

/// Multi-agent event emitted when node execution stops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentNodeStopEvent {
    pub node_id: String,
    pub node_result: serde_json::Value,
}

impl MultiAgentNodeStopEvent {
    pub fn new(node_id: impl Into<String>, node_result: serde_json::Value) -> Self {
        Self {
            node_id: node_id.into(),
            node_result,
        }
    }
}

impl TypedEvent for MultiAgentNodeStopEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "type": "multiagent_node_stop", "node_id": self.node_id, "node_result": self.node_result })
    }
}

/// Multi-agent handoff event for node transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentHandoffEvent {
    pub from_node_ids: Vec<String>,
    pub to_node_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl MultiAgentHandoffEvent {
    pub fn new(from_node_ids: Vec<String>, to_node_ids: Vec<String>) -> Self {
        Self {
            from_node_ids,
            to_node_ids,
            message: None,
        }
    }

    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }
}

impl TypedEvent for MultiAgentHandoffEvent {
    fn as_dict(&self) -> serde_json::Value {
        let mut value = serde_json::json!({
            "type": "multiagent_handoff",
            "from_node_ids": self.from_node_ids,
            "to_node_ids": self.to_node_ids,
        });
        if let Some(ref msg) = self.message {
            value["message"] = serde_json::json!(msg);
        }
        value
    }
}

/// Multi-agent node stream event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentNodeStreamEvent {
    pub node_id: String,
    pub event: serde_json::Value,
}

impl MultiAgentNodeStreamEvent {
    pub fn new(node_id: impl Into<String>, event: serde_json::Value) -> Self {
        Self {
            node_id: node_id.into(),
            event,
        }
    }
}

impl TypedEvent for MultiAgentNodeStreamEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "type": "multiagent_node_stream", "node_id": self.node_id, "event": self.event })
    }
}

/// Multi-agent node cancel event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentNodeCancelEvent {
    pub node_id: String,
    pub message: String,
}

impl MultiAgentNodeCancelEvent {
    pub fn new(node_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            message: message.into(),
        }
    }
}

impl TypedEvent for MultiAgentNodeCancelEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "type": "multiagent_node_cancel", "node_id": self.node_id, "message": self.message })
    }
}

/// Multi-agent node interrupt event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentNodeInterruptEvent {
    pub node_id: String,
    pub interrupts: Vec<Interrupt>,
}

impl MultiAgentNodeInterruptEvent {
    pub fn new(node_id: impl Into<String>, interrupts: Vec<Interrupt>) -> Self {
        Self {
            node_id: node_id.into(),
            interrupts,
        }
    }
}

impl TypedEvent for MultiAgentNodeInterruptEvent {
    fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({ "type": "multiagent_node_interrupt", "node_id": self.node_id, "interrupts": self.interrupts })
    }
}

/// Enum wrapper for all event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentEvent {
    InitEventLoop(InitEventLoopEvent),
    Start(StartEvent),
    StartEventLoop(StartEventLoopEvent),
    ModelStreamChunk(ModelStreamChunkEvent),
    ModelStream(ModelStreamEvent),
    TextStream(TextStreamEvent),
    CitationStream(CitationStreamEvent),
    ReasoningTextStream(ReasoningTextStreamEvent),
    ModelStopReason(ModelStopReason),
    EventLoopStop(EventLoopStopEvent),
    EventLoopThrottle(EventLoopThrottleEvent),
    ToolResult(ToolResultEvent),
    ToolStream(ToolStreamEvent),
    ToolCancel(ToolCancelEvent),
    ToolInterrupt(ToolInterruptEvent),
    ModelMessage(ModelMessageEvent),
    ToolResultMessage(ToolResultMessageEvent),
    ForceStop(ForceStopEvent),
    MultiAgentNodeStart(MultiAgentNodeStartEvent),
    MultiAgentNodeStop(MultiAgentNodeStopEvent),
    MultiAgentHandoff(MultiAgentHandoffEvent),
    MultiAgentNodeStream(MultiAgentNodeStreamEvent),
    MultiAgentNodeCancel(MultiAgentNodeCancelEvent),
    MultiAgentNodeInterrupt(MultiAgentNodeInterruptEvent),
}

impl AgentEvent {
    pub fn is_callback_event(&self) -> bool {
        match self {
            AgentEvent::ModelStopReason(_) => false,
            AgentEvent::EventLoopStop(_) => false,
            AgentEvent::ToolResult(_) => false,
            AgentEvent::ModelStream(e) => e.is_callback_event(),
            _ => true,
        }
    }

    pub fn as_dict(&self) -> serde_json::Value {
        match self {
            AgentEvent::InitEventLoop(e) => e.as_dict(),
            AgentEvent::Start(e) => e.as_dict(),
            AgentEvent::StartEventLoop(e) => e.as_dict(),
            AgentEvent::ModelStreamChunk(e) => e.as_dict(),
            AgentEvent::ModelStream(e) => e.as_dict(),
            AgentEvent::TextStream(e) => e.as_dict(),
            AgentEvent::CitationStream(e) => e.as_dict(),
            AgentEvent::ReasoningTextStream(e) => e.as_dict(),
            AgentEvent::ModelStopReason(e) => e.as_dict(),
            AgentEvent::EventLoopStop(e) => e.as_dict(),
            AgentEvent::EventLoopThrottle(e) => e.as_dict(),
            AgentEvent::ToolResult(e) => e.as_dict(),
            AgentEvent::ToolStream(e) => e.as_dict(),
            AgentEvent::ToolCancel(e) => e.as_dict(),
            AgentEvent::ToolInterrupt(e) => e.as_dict(),
            AgentEvent::ModelMessage(e) => e.as_dict(),
            AgentEvent::ToolResultMessage(e) => e.as_dict(),
            AgentEvent::ForceStop(e) => e.as_dict(),
            AgentEvent::MultiAgentNodeStart(e) => e.as_dict(),
            AgentEvent::MultiAgentNodeStop(e) => e.as_dict(),
            AgentEvent::MultiAgentHandoff(e) => e.as_dict(),
            AgentEvent::MultiAgentNodeStream(e) => e.as_dict(),
            AgentEvent::MultiAgentNodeCancel(e) => e.as_dict(),
            AgentEvent::MultiAgentNodeInterrupt(e) => e.as_dict(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_event_loop_event() {
        let event = InitEventLoopEvent::new();
        assert!(event.is_callback_event());
        let dict = event.as_dict();
        assert_eq!(dict["init_event_loop"], true);
    }

    #[test]
    fn test_model_stream_event() {
        let delta = ContentBlockDelta::default();
        let event = ModelStreamEvent::text(delta, "Hello".to_string());
        assert!(event.is_callback_event());
    }

    #[test]
    fn test_empty_model_stream_event() {
        let event = ModelStreamEvent::default();
        assert!(!event.is_callback_event());
    }

    #[test]
    fn test_force_stop_event() {
        let event = ForceStopEvent::new("Test reason");
        assert!(event.force_stop);
        assert_eq!(event.force_stop_reason, "Test reason");
    }
}

