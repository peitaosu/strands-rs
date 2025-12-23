//! Streaming types for model responses.

use serde::{Deserialize, Serialize};

use super::content::Role;

/// Reason why model generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    ContentFiltered,
    GuardrailIntervention,
    Interrupt,
}

impl Default for StopReason {
    fn default() -> Self { Self::EndTurn }
}

impl StopReason {
    /// Returns the string representation of the stop reason.
    pub fn as_str(&self) -> &'static str {
        match self {
            StopReason::EndTurn => "end_turn",
            StopReason::ToolUse => "tool_use",
            StopReason::MaxTokens => "max_tokens",
            StopReason::StopSequence => "stop_sequence",
            StopReason::ContentFiltered => "content_filtered",
            StopReason::GuardrailIntervention => "guardrail_intervention",
            StopReason::Interrupt => "interrupt",
        }
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,

    #[serde(default)]
    pub cache_read_input_tokens: u32,

    #[serde(default)]
    pub cache_write_input_tokens: u32,
}

impl Usage {
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cache_read_input_tokens: 0,
            cache_write_input_tokens: 0,
        }
    }

    pub fn add(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
        self.cache_write_input_tokens += other.cache_write_input_tokens;
    }
}

/// Performance metrics for a model call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    pub latency_ms: u64,

    #[serde(default)]
    pub time_to_first_byte_ms: u64,
}

/// Event indicating message generation has started.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageStartEvent {
    pub role: Role,
}

/// Tool use information at content block start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockStartToolUse {
    pub name: String,
    pub tool_use_id: String,
}

/// Content block start data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockStart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<ContentBlockStartToolUse>,
}

/// Event indicating a content block has started.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockStartEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block_index: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<ContentBlockStart>,
}

/// Tool use delta within a content block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockDeltaToolUse {
    pub input: String,
}

/// Reasoning content delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningContentBlockDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted_content: Option<Vec<u8>>,
}

/// Citation delta in streaming response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CitationsDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_content: Option<Vec<CitationSourceContentDelta>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// Source content delta for citations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CitationSourceContentDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Incremental content block update.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<ContentBlockDeltaToolUse>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<ReasoningContentBlockDelta>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation: Option<CitationsDelta>,
}

/// Event containing a content block delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockDeltaEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block_index: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<ContentBlockDelta>,
}

/// Event indicating a content block has stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockStopEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block_index: Option<u32>,
}

/// Event indicating message generation has stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MessageStopEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_model_response_fields: Option<serde_json::Value>,
}

/// Event containing usage and metrics metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MetadataEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<serde_json::Value>,
}

/// An exception event from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ExceptionEvent {
    pub message: String,
}

/// A stream error event from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStreamErrorEvent {
    pub message: String,
    pub original_message: String,
    pub original_status_code: i32,
}

/// Event for content redaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RedactContentEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redact_user_content_message: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub redact_assistant_content_message: Option<String>,
}

/// A streaming event from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct StreamEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_start: Option<MessageStartEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block_start: Option<ContentBlockStartEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block_delta: Option<ContentBlockDeltaEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block_stop: Option<ContentBlockStopEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_stop: Option<MessageStopEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MetadataEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub redact_content: Option<RedactContentEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal_server_exception: Option<ExceptionEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_stream_error_exception: Option<ModelStreamErrorEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub throttling_exception: Option<ExceptionEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_exception: Option<ExceptionEvent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_unavailable_exception: Option<ExceptionEvent>,
}

impl StreamEvent {
    pub fn message_start(role: Role) -> Self {
        Self { message_start: Some(MessageStartEvent { role }), ..Default::default() }
    }

    pub fn content_block_start(index: u32, start: Option<ContentBlockStart>) -> Self {
        Self {
            content_block_start: Some(ContentBlockStartEvent {
                content_block_index: Some(index),
                start,
            }),
            ..Default::default()
        }
    }

    pub fn content_block_delta(index: u32, delta: ContentBlockDelta) -> Self {
        Self {
            content_block_delta: Some(ContentBlockDeltaEvent {
                content_block_index: Some(index),
                delta: Some(delta),
            }),
            ..Default::default()
        }
    }

    pub fn text_delta(index: u32, text: impl Into<String>) -> Self {
        Self::content_block_delta(index, ContentBlockDelta { text: Some(text.into()), ..Default::default() })
    }

    pub fn tool_use_delta(index: u32, input: impl Into<String>) -> Self {
        Self::content_block_delta(index, ContentBlockDelta {
            tool_use: Some(ContentBlockDeltaToolUse { input: input.into() }),
            ..Default::default()
        })
    }

    pub fn tool_use_start(index: u32, name: impl Into<String>, tool_use_id: impl Into<String>) -> Self {
        Self {
            content_block_start: Some(ContentBlockStartEvent {
                content_block_index: Some(index),
                start: Some(ContentBlockStart {
                    tool_use: Some(ContentBlockStartToolUse {
                        name: name.into(),
                        tool_use_id: tool_use_id.into(),
                    }),
                }),
            }),
            ..Default::default()
        }
    }

    pub fn reasoning_delta(index: u32, text: impl Into<String>) -> Self {
        Self::content_block_delta(index, ContentBlockDelta {
            reasoning_content: Some(ReasoningContentBlockDelta {
                text: Some(text.into()),
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    pub fn content_block_stop(index: u32) -> Self {
        Self {
            content_block_stop: Some(ContentBlockStopEvent { content_block_index: Some(index) }),
            ..Default::default()
        }
    }

    pub fn message_stop(stop_reason: StopReason) -> Self {
        Self {
            message_stop: Some(MessageStopEvent { stop_reason: Some(stop_reason), additional_model_response_fields: None }),
            ..Default::default()
        }
    }

    pub fn metadata(usage: Usage, metrics: Metrics) -> Self {
        Self {
            metadata: Some(MetadataEvent { usage: Some(usage), metrics: Some(metrics), trace: None }),
            ..Default::default()
        }
    }

    pub fn is_text_delta(&self) -> bool {
        self.content_block_delta.as_ref().and_then(|e| e.delta.as_ref()).map(|d| d.text.is_some()).unwrap_or(false)
    }

    pub fn as_text_delta(&self) -> Option<&str> {
        self.content_block_delta.as_ref().and_then(|e| e.delta.as_ref()).and_then(|d| d.text.as_deref())
    }

    pub fn is_message_stop(&self) -> bool { self.message_stop.is_some() }
    pub fn stop_reason(&self) -> Option<StopReason> { self.message_stop.as_ref().and_then(|e| e.stop_reason) }

    pub fn is_error(&self) -> bool {
        self.internal_server_exception.is_some()
            || self.model_stream_error_exception.is_some()
            || self.throttling_exception.is_some()
            || self.validation_exception.is_some()
            || self.service_unavailable_exception.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_add() {
        let mut usage1 = Usage::new(100, 50);
        let usage2 = Usage::new(200, 100);
        usage1.add(&usage2);
        assert_eq!(usage1.input_tokens, 300);
        assert_eq!(usage1.output_tokens, 150);
        assert_eq!(usage1.total_tokens, 450);
    }

    #[test]
    fn test_stop_reason_serialization() {
        assert_eq!(serde_json::to_string(&StopReason::EndTurn).unwrap(), "\"end_turn\"");
        assert_eq!(serde_json::to_string(&StopReason::ToolUse).unwrap(), "\"tool_use\"");
    }

    #[test]
    fn test_stream_event_text_delta() {
        let event = StreamEvent::text_delta(0, "Hello");
        assert!(event.is_text_delta());
        assert_eq!(event.as_text_delta(), Some("Hello"));
    }

    #[test]
    fn test_stream_event_serialization() {
        let event = StreamEvent::text_delta(0, "hi");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("contentBlockDelta"));
    }
}
