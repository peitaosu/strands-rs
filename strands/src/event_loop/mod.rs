//! Event loop types for agent execution.
//!
//! This module defines the event types that are emitted during agent execution,
//! providing a structured way to observe different events in the event loop and
//! agent lifecycle.

mod recover_max_tokens;
mod retry;

pub use recover_max_tokens::recover_message_on_max_tokens_reached;
pub use retry::ModelRetryStrategy;

use std::any::Any;
use std::collections::HashMap;

use crate::agent::AgentResult;
use crate::hooks::Interrupt;
use crate::telemetry::EventLoopMetrics;
use crate::types::citations::Citation;
use crate::types::content::Message;
use crate::types::streaming::{ContentBlockDelta, Metrics, StopReason, StreamEvent, Usage};
use crate::types::tools::{ToolResult as ToolResultType, ToolUse};

/// Base trait for typed events.
pub trait TypedEventBase: Send + Sync + std::fmt::Debug {
    /// Whether this event should trigger the callback_handler.
    fn is_callback_event(&self) -> bool {
        true
    }

    /// Convert to a raw dictionary for emitting purposes.
    fn as_dict(&self) -> HashMap<String, serde_json::Value>;

    /// Prepare the event for emission by adding invocation state.
    fn prepare(&mut self, _invocation_state: &HashMap<String, serde_json::Value>) {}

    /// Returns the event as Any for downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// Event emitted at the very beginning of agent execution.
#[derive(Debug, Clone, Default)]
pub struct InitEventLoopEvent {
    pub state: HashMap<String, serde_json::Value>,
}

impl InitEventLoopEvent {
    pub fn new() -> Self {
        Self {
            state: HashMap::new(),
        }
    }
}

impl TypedEventBase for InitEventLoopEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("init_event_loop".to_string(), serde_json::Value::Bool(true));
        map.extend(self.state.clone());
        map
    }

    fn prepare(&mut self, invocation_state: &HashMap<String, serde_json::Value>) {
        self.state.extend(invocation_state.clone());
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted at the start of each event loop cycle (deprecated, use StartEventLoopEvent).
#[derive(Debug, Clone, Default)]
pub struct StartEvent;

impl TypedEventBase for StartEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("start".to_string(), serde_json::Value::Bool(true));
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when the event loop cycle begins processing.
#[derive(Debug, Clone, Default)]
pub struct StartEventLoopEvent;

impl TypedEventBase for StartEventLoopEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("start_event_loop".to_string(), serde_json::Value::Bool(true));
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted during model response streaming for each raw chunk.
#[derive(Debug, Clone)]
pub struct ModelStreamChunkEvent {
    pub chunk: StreamEvent,
}

impl ModelStreamChunkEvent {
    pub fn new(chunk: StreamEvent) -> Self {
        Self { chunk }
    }
}

impl TypedEventBase for ModelStreamChunkEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert(
            "event".to_string(),
            serde_json::to_value(&self.chunk).unwrap_or_default(),
        );
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted during model response streaming.
#[derive(Debug, Clone, Default)]
pub struct ModelStreamEvent {
    pub delta_data: HashMap<String, serde_json::Value>,
}

impl ModelStreamEvent {
    pub fn new(delta_data: HashMap<String, serde_json::Value>) -> Self {
        Self { delta_data }
    }
}

impl TypedEventBase for ModelStreamEvent {
    fn is_callback_event(&self) -> bool {
        !self.delta_data.is_empty()
    }

    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        self.delta_data.clone()
    }

    fn prepare(&mut self, invocation_state: &HashMap<String, serde_json::Value>) {
        if self.delta_data.contains_key("delta") {
            self.delta_data.extend(invocation_state.clone());
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted during tool use input streaming.
#[derive(Debug, Clone)]
pub struct ToolUseStreamEvent {
    pub delta: ContentBlockDelta,
    pub current_tool_use: HashMap<String, serde_json::Value>,
}

impl ToolUseStreamEvent {
    pub fn new(delta: ContentBlockDelta, current_tool_use: HashMap<String, serde_json::Value>) -> Self {
        Self { delta, current_tool_use }
    }
}

impl TypedEventBase for ToolUseStreamEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("type".to_string(), serde_json::Value::String("tool_use_stream".to_string()));
        map.insert("delta".to_string(), serde_json::to_value(&self.delta).unwrap_or_default());
        map.insert("current_tool_use".to_string(), serde_json::to_value(&self.current_tool_use).unwrap_or_default());
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted during text content streaming.
#[derive(Debug, Clone)]
pub struct TextStreamEvent {
    pub delta: ContentBlockDelta,
    pub text: String,
}

impl TextStreamEvent {
    pub fn new(delta: ContentBlockDelta, text: impl Into<String>) -> Self {
        Self { delta, text: text.into() }
    }
}

impl TypedEventBase for TextStreamEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("data".to_string(), serde_json::Value::String(self.text.clone()));
        map.insert("delta".to_string(), serde_json::to_value(&self.delta).unwrap_or_default());
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted during citation streaming.
#[derive(Debug, Clone)]
pub struct CitationStreamEvent {
    pub delta: ContentBlockDelta,
    pub citation: Citation,
}

impl CitationStreamEvent {
    pub fn new(delta: ContentBlockDelta, citation: Citation) -> Self {
        Self { delta, citation }
    }
}

impl TypedEventBase for CitationStreamEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("citation".to_string(), serde_json::to_value(&self.citation).unwrap_or_default());
        map.insert("delta".to_string(), serde_json::to_value(&self.delta).unwrap_or_default());
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted during reasoning text streaming.
#[derive(Debug, Clone)]
pub struct ReasoningTextStreamEvent {
    pub delta: ContentBlockDelta,
    pub reasoning_text: Option<String>,
}

impl ReasoningTextStreamEvent {
    pub fn new(delta: ContentBlockDelta, reasoning_text: Option<String>) -> Self {
        Self { delta, reasoning_text }
    }
}

impl TypedEventBase for ReasoningTextStreamEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("reasoningText".to_string(), serde_json::to_value(&self.reasoning_text).unwrap_or_default());
        map.insert("delta".to_string(), serde_json::to_value(&self.delta).unwrap_or_default());
        map.insert("reasoning".to_string(), serde_json::Value::Bool(true));
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted during redacted content streaming.
#[derive(Debug, Clone)]
pub struct ReasoningRedactedContentStreamEvent {
    pub delta: ContentBlockDelta,
    pub redacted_content: Option<Vec<u8>>,
}

impl ReasoningRedactedContentStreamEvent {
    pub fn new(delta: ContentBlockDelta, redacted_content: Option<Vec<u8>>) -> Self {
        Self { delta, redacted_content }
    }
}

impl TypedEventBase for ReasoningRedactedContentStreamEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("reasoningRedactedContent".to_string(), serde_json::to_value(&self.redacted_content).unwrap_or_default());
        map.insert("delta".to_string(), serde_json::to_value(&self.delta).unwrap_or_default());
        map.insert("reasoning".to_string(), serde_json::Value::Bool(true));
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted during reasoning signature streaming.
#[derive(Debug, Clone)]
pub struct ReasoningSignatureStreamEvent {
    pub delta: ContentBlockDelta,
    pub reasoning_signature: Option<String>,
}

impl ReasoningSignatureStreamEvent {
    pub fn new(delta: ContentBlockDelta, reasoning_signature: Option<String>) -> Self {
        Self { delta, reasoning_signature }
    }
}

impl TypedEventBase for ReasoningSignatureStreamEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("reasoning_signature".to_string(), serde_json::to_value(&self.reasoning_signature).unwrap_or_default());
        map.insert("delta".to_string(), serde_json::to_value(&self.delta).unwrap_or_default());
        map.insert("reasoning".to_string(), serde_json::Value::Bool(true));
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when model stops generating.
#[derive(Debug, Clone)]
pub struct ModelStopReasonEvent {
    pub stop_reason: StopReason,
    pub message: Message,
    pub usage: Usage,
    pub metrics: Metrics,
}

impl ModelStopReasonEvent {
    pub fn new(stop_reason: StopReason, message: Message, usage: Usage, metrics: Metrics) -> Self {
        Self { stop_reason, message, usage, metrics }
    }
}

impl TypedEventBase for ModelStopReasonEvent {
    fn is_callback_event(&self) -> bool {
        false
    }

    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert(
            "stop".to_string(),
            serde_json::json!([
                self.stop_reason,
                self.message,
                self.usage,
                self.metrics
            ]),
        );
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when the agent execution completes normally.
#[derive(Debug, Clone)]
pub struct EventLoopStopEvent {
    pub stop_reason: StopReason,
    pub message: Message,
    pub metrics: EventLoopMetrics,
    pub request_state: serde_json::Value,
    pub interrupts: Option<Vec<Interrupt>>,
    pub structured_output: Option<serde_json::Value>,
}

impl EventLoopStopEvent {
    pub fn new(
        stop_reason: StopReason,
        message: Message,
        metrics: EventLoopMetrics,
        request_state: serde_json::Value,
    ) -> Self {
        Self {
            stop_reason,
            message,
            metrics,
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

    fn interrupts_to_json(&self) -> serde_json::Value {
        match &self.interrupts {
            Some(interrupts) => serde_json::Value::Array(
                interrupts
                    .iter()
                    .map(|i| {
                        serde_json::json!({
                            "id": i.id,
                            "name": i.name,
                            "reason": i.reason
                        })
                    })
                    .collect(),
            ),
            None => serde_json::Value::Null,
        }
    }
}

impl TypedEventBase for EventLoopStopEvent {
    fn is_callback_event(&self) -> bool {
        false
    }

    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert(
            "stop".to_string(),
            serde_json::json!([
                self.stop_reason,
                self.message,
                self.metrics.get_summary(),
                self.request_state,
                self.interrupts_to_json(),
                self.structured_output
            ]),
        );
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when structured output is detected and processed.
#[derive(Debug, Clone)]
pub struct StructuredOutputEvent {
    pub structured_output: serde_json::Value,
}

impl StructuredOutputEvent {
    pub fn new(structured_output: serde_json::Value) -> Self {
        Self { structured_output }
    }
}

impl TypedEventBase for StructuredOutputEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("structured_output".to_string(), self.structured_output.clone());
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when the event loop is throttled due to rate limiting.
#[derive(Debug, Clone)]
pub struct EventLoopThrottleEvent {
    pub delay: u32,
    pub state: HashMap<String, serde_json::Value>,
}

impl EventLoopThrottleEvent {
    pub fn new(delay: u32) -> Self {
        Self {
            delay,
            state: HashMap::new(),
        }
    }
}

impl TypedEventBase for EventLoopThrottleEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("event_loop_throttled_delay".to_string(), serde_json::Value::Number(self.delay.into()));
        map.extend(self.state.clone());
        map
    }

    fn prepare(&mut self, invocation_state: &HashMap<String, serde_json::Value>) {
        self.state.extend(invocation_state.clone());
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when a tool execution completes.
#[derive(Debug, Clone)]
pub struct ToolResultEvent {
    pub tool_result: ToolResultType,
}

impl ToolResultEvent {
    pub fn new(tool_result: ToolResultType) -> Self {
        Self { tool_result }
    }

    pub fn tool_use_id(&self) -> &str {
        &self.tool_result.tool_use_id
    }
}

impl TypedEventBase for ToolResultEvent {
    fn is_callback_event(&self) -> bool {
        false
    }

    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("type".to_string(), serde_json::Value::String("tool_result".to_string()));
        map.insert("tool_result".to_string(), serde_json::to_value(&self.tool_result).unwrap_or_default());
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when a tool yields sub-events during execution.
#[derive(Debug, Clone)]
pub struct ToolStreamEvent {
    pub tool_use: ToolUse,
    pub tool_stream_data: serde_json::Value,
}

impl ToolStreamEvent {
    pub fn new(tool_use: ToolUse, tool_stream_data: serde_json::Value) -> Self {
        Self { tool_use, tool_stream_data }
    }

    pub fn tool_use_id(&self) -> &str {
        &self.tool_use.tool_use_id
    }
}

impl TypedEventBase for ToolStreamEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("type".to_string(), serde_json::Value::String("tool_stream".to_string()));
        map.insert(
            "tool_stream_event".to_string(),
            serde_json::json!({
                "tool_use": self.tool_use,
                "data": self.tool_stream_data
            }),
        );
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when a tool call is cancelled.
#[derive(Debug, Clone)]
pub struct ToolCancelEvent {
    pub tool_use: ToolUse,
    pub message: String,
}

impl ToolCancelEvent {
    pub fn new(tool_use: ToolUse, message: impl Into<String>) -> Self {
        Self { tool_use, message: message.into() }
    }

    pub fn tool_use_id(&self) -> &str {
        &self.tool_use.tool_use_id
    }
}

impl TypedEventBase for ToolCancelEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert(
            "tool_cancel_event".to_string(),
            serde_json::json!({
                "tool_use": self.tool_use,
                "message": self.message
            }),
        );
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when a tool is interrupted.
#[derive(Debug, Clone)]
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

impl TypedEventBase for ToolInterruptEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert(
            "tool_interrupt_event".to_string(),
            serde_json::json!({
                "tool_use": self.tool_use,
                "interrupts": self.interrupts.iter().map(|i| {
                    serde_json::json!({
                        "id": i.id,
                        "name": i.name,
                        "reason": i.reason
                    })
                }).collect::<Vec<_>>()
            }),
        );
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when the model invocation has completed.
#[derive(Debug, Clone)]
pub struct ModelMessageEvent {
    pub message: Message,
}

impl ModelMessageEvent {
    pub fn new(message: Message) -> Self {
        Self { message }
    }
}

impl TypedEventBase for ModelMessageEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("message".to_string(), serde_json::to_value(&self.message).unwrap_or_default());
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when tool results are formatted as a message.
#[derive(Debug, Clone)]
pub struct ToolResultMessageEvent {
    pub message: Message,
}

impl ToolResultMessageEvent {
    pub fn new(message: Message) -> Self {
        Self { message }
    }
}

impl TypedEventBase for ToolResultMessageEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("message".to_string(), serde_json::to_value(&self.message).unwrap_or_default());
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when the agent execution is forcibly stopped.
#[derive(Debug, Clone)]
pub struct ForceStopEvent {
    pub reason: String,
}

impl ForceStopEvent {
    pub fn new(reason: impl Into<String>) -> Self {
        Self { reason: reason.into() }
    }

    pub fn from_error(err: &dyn std::error::Error) -> Self {
        Self { reason: err.to_string() }
    }
}

impl TypedEventBase for ForceStopEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("force_stop".to_string(), serde_json::Value::Bool(true));
        map.insert("force_stop_reason".to_string(), serde_json::Value::String(self.reason.clone()));
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Event emitted when agent execution completes with a result.
#[derive(Debug, Clone)]
pub struct AgentResultEvent {
    pub result: AgentResult,
}

impl AgentResultEvent {
    pub fn new(result: AgentResult) -> Self {
        Self { result }
    }
}

impl TypedEventBase for AgentResultEvent {
    fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("result".to_string(), serde_json::to_value(&self.result).unwrap_or_default());
        map
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Unified typed event enum for backward compatibility.
#[derive(Debug, Clone)]
pub enum TypedEvent {
    InitEventLoop(InitEventLoopEvent),
    Start(StartEvent),
    StartEventLoop(StartEventLoopEvent),
    ModelStreamChunk(ModelStreamChunkEvent),
    ModelStream(ModelStreamEvent),
    ToolUseStream(ToolUseStreamEvent),
    TextStream(TextStreamEvent),
    CitationStream(CitationStreamEvent),
    ReasoningTextStream(ReasoningTextStreamEvent),
    ReasoningRedactedContentStream(ReasoningRedactedContentStreamEvent),
    ReasoningSignatureStream(ReasoningSignatureStreamEvent),
    ModelStopReason(ModelStopReasonEvent),
    EventLoopStop(EventLoopStopEvent),
    StructuredOutput(StructuredOutputEvent),
    EventLoopThrottle(EventLoopThrottleEvent),
    ToolResult(ToolResultEvent),
    ToolStream(ToolStreamEvent),
    ToolCancel(ToolCancelEvent),
    ToolInterrupt(ToolInterruptEvent),
    ModelMessage(ModelMessageEvent),
    ToolResultMessage(ToolResultMessageEvent),
    ForceStop(ForceStopEvent),
    AgentResult(AgentResultEvent),
}

impl TypedEvent {
    pub fn is_result(&self) -> bool {
        matches!(self, Self::AgentResult(_) | Self::EventLoopStop(_))
    }

    pub fn as_result(&self) -> Option<&AgentResult> {
        match self {
            Self::AgentResult(e) => Some(&e.result),
            _ => None,
        }
    }

    pub fn as_model_message(&self) -> Option<&Message> {
        match self {
            Self::ModelMessage(e) => Some(&e.message),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<String> {
        match self {
            Self::TextStream(e) => Some(e.text.clone()),
            Self::ModelMessage(e) => Some(e.message.text_content()),
            Self::AgentResult(e) => Some(e.result.text()),
            _ => None,
        }
    }

    pub fn as_text_delta(&self) -> Option<&str> {
        match self {
            Self::TextStream(e) => Some(&e.text),
            Self::ModelStreamChunk(e) => e.chunk.as_text_delta(),
            _ => None,
        }
    }

    pub fn is_callback_event(&self) -> bool {
        match self {
            Self::InitEventLoop(e) => e.is_callback_event(),
            Self::Start(e) => e.is_callback_event(),
            Self::StartEventLoop(e) => e.is_callback_event(),
            Self::ModelStreamChunk(e) => e.is_callback_event(),
            Self::ModelStream(e) => e.is_callback_event(),
            Self::ToolUseStream(e) => e.is_callback_event(),
            Self::TextStream(e) => e.is_callback_event(),
            Self::CitationStream(e) => e.is_callback_event(),
            Self::ReasoningTextStream(e) => e.is_callback_event(),
            Self::ReasoningRedactedContentStream(e) => e.is_callback_event(),
            Self::ReasoningSignatureStream(e) => e.is_callback_event(),
            Self::ModelStopReason(e) => e.is_callback_event(),
            Self::EventLoopStop(e) => e.is_callback_event(),
            Self::StructuredOutput(e) => e.is_callback_event(),
            Self::EventLoopThrottle(e) => e.is_callback_event(),
            Self::ToolResult(e) => e.is_callback_event(),
            Self::ToolStream(e) => e.is_callback_event(),
            Self::ToolCancel(e) => e.is_callback_event(),
            Self::ToolInterrupt(e) => e.is_callback_event(),
            Self::ModelMessage(e) => e.is_callback_event(),
            Self::ToolResultMessage(e) => e.is_callback_event(),
            Self::ForceStop(e) => e.is_callback_event(),
            Self::AgentResult(e) => e.is_callback_event(),
        }
    }

    pub fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        match self {
            Self::InitEventLoop(e) => e.as_dict(),
            Self::Start(e) => e.as_dict(),
            Self::StartEventLoop(e) => e.as_dict(),
            Self::ModelStreamChunk(e) => e.as_dict(),
            Self::ModelStream(e) => e.as_dict(),
            Self::ToolUseStream(e) => e.as_dict(),
            Self::TextStream(e) => e.as_dict(),
            Self::CitationStream(e) => e.as_dict(),
            Self::ReasoningTextStream(e) => e.as_dict(),
            Self::ReasoningRedactedContentStream(e) => e.as_dict(),
            Self::ReasoningSignatureStream(e) => e.as_dict(),
            Self::ModelStopReason(e) => e.as_dict(),
            Self::EventLoopStop(e) => e.as_dict(),
            Self::StructuredOutput(e) => e.as_dict(),
            Self::EventLoopThrottle(e) => e.as_dict(),
            Self::ToolResult(e) => e.as_dict(),
            Self::ToolStream(e) => e.as_dict(),
            Self::ToolCancel(e) => e.as_dict(),
            Self::ToolInterrupt(e) => e.as_dict(),
            Self::ModelMessage(e) => e.as_dict(),
            Self::ToolResultMessage(e) => e.as_dict(),
            Self::ForceStop(e) => e.as_dict(),
            Self::AgentResult(e) => e.as_dict(),
        }
    }
}

impl TypedEvent {
    pub fn init_event_loop() -> Self {
        Self::InitEventLoop(InitEventLoopEvent::new())
    }

    pub fn start() -> Self {
        Self::Start(StartEvent)
    }

    pub fn start_event_loop() -> Self {
        Self::StartEventLoop(StartEventLoopEvent)
    }

    pub fn text_stream(delta: ContentBlockDelta, text: impl Into<String>) -> Self {
        Self::TextStream(TextStreamEvent::new(delta, text))
    }

    pub fn model_message(message: Message) -> Self {
        Self::ModelMessage(ModelMessageEvent::new(message))
    }

    pub fn tool_result(tool_result: ToolResultType) -> Self {
        Self::ToolResult(ToolResultEvent::new(tool_result))
    }

    pub fn agent_result(result: AgentResult) -> Self {
        Self::AgentResult(AgentResultEvent::new(result))
    }

    pub fn force_stop(reason: impl Into<String>) -> Self {
        Self::ForceStop(ForceStopEvent::new(reason))
    }

    pub fn throttle(delay: u32) -> Self {
        Self::EventLoopThrottle(EventLoopThrottleEvent::new(delay))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_event_loop_event() {
        let event = InitEventLoopEvent::new();
        let dict = event.as_dict();
        assert_eq!(dict.get("init_event_loop"), Some(&serde_json::Value::Bool(true)));
    }

    #[test]
    fn test_start_event() {
        let event = StartEvent;
        let dict = event.as_dict();
        assert_eq!(dict.get("start"), Some(&serde_json::Value::Bool(true)));
    }

    #[test]
    fn test_force_stop_event() {
        let event = ForceStopEvent::new("Test reason");
        let dict = event.as_dict();
        assert_eq!(dict.get("force_stop"), Some(&serde_json::Value::Bool(true)));
        assert_eq!(dict.get("force_stop_reason"), Some(&serde_json::Value::String("Test reason".to_string())));
    }

    #[test]
    fn test_typed_event_enum() {
        let event = TypedEvent::start();
        assert!(!event.is_result());

        let event = TypedEvent::force_stop("error");
        let dict = event.as_dict();
        assert!(dict.contains_key("force_stop"));
    }
}
