//! Anthropic Claude model provider.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{Model, ModelConfig, StreamEventStream};
use crate::types::{
    content::{ContentBlock, Message, Role, SystemContentBlock},
    errors::StrandsError,
    streaming::{
        ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockDeltaToolUse, ContentBlockStart,
        ContentBlockStartEvent, ContentBlockStartToolUse, ContentBlockStopEvent, MessageStartEvent,
        MessageStopEvent, MetadataEvent, Metrics, StopReason, StreamEvent, Usage,
    },
    tools::{ToolChoice, ToolSpec},
};

const DEFAULT_MODEL_ID: &str = "claude-sonnet-4-20250514";
const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Claude model provider.
#[derive(Clone)]
pub struct AnthropicModel {
    config: ModelConfig,
    api_key: String,
    max_tokens: u32,
    client: Client,
}

impl std::fmt::Debug for AnthropicModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicModel")
            .field("config", &self.config)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text { #[serde(rename = "type")] content_type: String, text: String },
    ToolUse { #[serde(rename = "type")] content_type: String, id: String, name: String, input: serde_json::Value },
    ToolResult { #[serde(rename = "type")] content_type: String, tool_use_id: String, content: Vec<AnthropicToolResultContent>, is_error: bool },
}

#[derive(Debug, Serialize)]
struct AnthropicToolResultContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    content_block: Option<AnthropicContentBlock>,
    #[serde(default)]
    delta: Option<AnthropicDelta>,
    #[serde(default)]
    message: Option<AnthropicMessageInfo>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageInfo {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl AnthropicModel {
    pub fn new(api_key: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            config: ModelConfig::new(DEFAULT_MODEL_ID),
            api_key: api_key.into(),
            max_tokens,
            client: Client::new(),
        }
    }

    pub fn with_model(mut self, model_id: impl Into<String>) -> Self {
        self.config.model_id = model_id.into();
        self
    }

    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = config;
        self
    }

    fn format_messages(&self, messages: &[Message]) -> Vec<AnthropicMessage> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };

                let content: Vec<AnthropicContent> = msg
                    .content
                    .iter()
                    .filter_map(|block| self.format_content_block(block))
                    .collect();

                AnthropicMessage {
                    role: role.to_string(),
                    content,
                }
            })
            .collect()
    }

    fn format_content_block(&self, block: &ContentBlock) -> Option<AnthropicContent> {
        if let Some(ref text) = block.text {
            return Some(AnthropicContent::Text {
                content_type: "text".to_string(),
                text: text.clone(),
            });
        }

        if let Some(ref tu) = block.tool_use {
            return Some(AnthropicContent::ToolUse {
                content_type: "tool_use".to_string(),
                id: tu.tool_use_id.clone(),
                name: tu.name.clone(),
                input: tu.input.clone(),
            });
        }

        if let Some(ref tr) = block.tool_result {
            let content: Vec<AnthropicToolResultContent> = tr
                .content
                .iter()
                .filter_map(|c| {
                    c.text.as_ref().map(|t| AnthropicToolResultContent {
                        content_type: "text".to_string(),
                        text: t.clone(),
                    })
                })
                .collect();

            let is_error = tr.status == crate::types::tools::ToolResultStatus::Error;

            return Some(AnthropicContent::ToolResult {
                content_type: "tool_result".to_string(),
                tool_use_id: tr.tool_use_id.clone(),
                content,
                is_error,
            });
        }

        None
    }

    fn format_tools(&self, tool_specs: &[ToolSpec]) -> Vec<AnthropicTool> {
        tool_specs
            .iter()
            .map(|spec| AnthropicTool {
                name: spec.name.clone(),
                description: spec.description.clone(),
                input_schema: spec.input_schema.json.clone(),
            })
            .collect()
    }

    fn format_tool_choice(&self, tool_choice: Option<ToolChoice>) -> Option<serde_json::Value> {
        tool_choice.map(|tc| match tc {
            ToolChoice::Auto(_) => serde_json::json!({ "type": "auto" }),
            ToolChoice::Any(_) => serde_json::json!({ "type": "any" }),
            ToolChoice::Tool(t) => serde_json::json!({ "type": "tool", "name": t.name }),
        })
    }

    fn map_stop_reason(reason: &str) -> StopReason {
        match reason {
            "tool_use" => StopReason::ToolUse,
            "max_tokens" => StopReason::MaxTokens,
            "end_turn" | "stop_sequence" => StopReason::EndTurn,
            _ => StopReason::EndTurn,
        }
    }
}

#[async_trait]
impl Model for AnthropicModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn update_config(&mut self, config: ModelConfig) {
        self.config = config;
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        tool_specs: Option<&'a [ToolSpec]>,
        system_prompt: Option<&'a str>,
        tool_choice: Option<ToolChoice>,
        _system_prompt_content: Option<&'a [SystemContentBlock]>,
    ) -> StreamEventStream<'a> {
        let api_key = self.api_key.clone();
        let client = self.client.clone();

        let request = AnthropicRequest {
            model: self.config.model_id.clone(),
            messages: self.format_messages(messages),
            max_tokens: self.max_tokens,
            stream: true,
            system: system_prompt.map(String::from),
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            tools: tool_specs.map(|s| self.format_tools(s)).unwrap_or_default(),
            tool_choice: self.format_tool_choice(tool_choice),
        };

        Box::pin(async_stream::stream! {
            let response = match client
                .post(ANTHROPIC_API_URL)
                .header("x-api-key", &api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    yield Err(StrandsError::NetworkError(e.to_string()));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                if status.as_u16() == 429 {
                    yield Err(StrandsError::ModelThrottled { message: body });
                } else if body.contains("prompt is too long") || body.contains("context") {
                    yield Err(StrandsError::ContextWindowOverflow { message: body });
                } else {
                    yield Err(StrandsError::model_error(format!("HTTP {status}: {body}")));
                }
                return;
            }

            use futures::StreamExt;
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut final_usage: Option<AnthropicUsage> = None;
            let mut stop_reason_str: Option<String> = None;

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                    Err(e) => {
                        yield Err(StrandsError::NetworkError(e.to_string()));
                        return;
                    }
                };

                buffer.push_str(&chunk);

                let lines: Vec<String> = buffer.lines().map(String::from).collect();
                buffer.clear();

                for line in &lines {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    if let Some(json_str) = line.strip_prefix("data: ") {
                        if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(json_str) {
                            match event.event_type.as_str() {
                                "message_start" => {
                                    yield Ok(StreamEvent {
                                        message_start: Some(MessageStartEvent { role: Role::Assistant }),
                                        ..Default::default()
                                    });
                                }

                                "content_block_start" => {
                                    let index = event.index.unwrap_or(0) as u32;
                                    let start = event.content_block.as_ref().and_then(|cb| {
                                        if cb.block_type == "tool_use" {
                                            Some(ContentBlockStart {
                                                tool_use: Some(ContentBlockStartToolUse {
                                                    name: cb.name.clone().unwrap_or_default(),
                                                    tool_use_id: cb.id.clone().unwrap_or_default(),
                                                }),
                                            })
                                        } else {
                                            None
                                        }
                                    });

                                    yield Ok(StreamEvent {
                                        content_block_start: Some(ContentBlockStartEvent {
                                            content_block_index: Some(index),
                                            start,
                                        }),
                                        ..Default::default()
                                    });
                                }

                                "content_block_delta" => {
                                    let index = event.index.unwrap_or(0) as u32;
                                    if let Some(ref delta) = event.delta {
                                        let block_delta = match delta.delta_type.as_str() {
                                            "text_delta" => ContentBlockDelta {
                                                text: delta.text.clone(),
                                                ..Default::default()
                                            },
                                            "input_json_delta" => ContentBlockDelta {
                                                tool_use: Some(ContentBlockDeltaToolUse {
                                                    input: delta.partial_json.clone().unwrap_or_default(),
                                                }),
                                                ..Default::default()
                                            },
                                            _ => ContentBlockDelta::default(),
                                        };

                                        yield Ok(StreamEvent {
                                            content_block_delta: Some(ContentBlockDeltaEvent {
                                                content_block_index: Some(index),
                                                delta: Some(block_delta),
                                            }),
                                            ..Default::default()
                                        });
                                    }
                                }

                                "content_block_stop" => {
                                    let index = event.index.unwrap_or(0) as u32;
                                    yield Ok(StreamEvent {
                                        content_block_stop: Some(ContentBlockStopEvent {
                                            content_block_index: Some(index),
                                        }),
                                        ..Default::default()
                                    });
                                }

                                "message_delta" => {
                                    if let Some(ref usage) = event.usage {
                                        final_usage = Some(AnthropicUsage {
                                            input_tokens: usage.input_tokens,
                                            output_tokens: usage.output_tokens,
                                        });
                                    }
                                    if let Some(ref delta) = event.delta {
                                        if let Some(ref text) = delta.text {
                                            stop_reason_str = Some(text.clone());
                                        }
                                    }
                                }

                                "message_stop" => {
                                    let reason = event.message
                                        .as_ref()
                                        .and_then(|m| m.stop_reason.as_ref())
                                        .map(|s| Self::map_stop_reason(s))
                                        .or_else(|| stop_reason_str.as_ref().map(|s| Self::map_stop_reason(s)))
                                        .unwrap_or(StopReason::EndTurn);

                                    yield Ok(StreamEvent {
                                        message_stop: Some(MessageStopEvent {
                                            stop_reason: Some(reason),
                                            additional_model_response_fields: None,
                                        }),
                                        ..Default::default()
                                    });
                                }

                                _ => {}
                            }
                        }
                    }
                }
            }

            if let Some(usage) = final_usage {
                yield Ok(StreamEvent {
                    metadata: Some(MetadataEvent {
                        usage: Some(Usage {
                            input_tokens: usage.input_tokens,
                            output_tokens: usage.output_tokens,
                            total_tokens: usage.input_tokens + usage.output_tokens,
                            cache_read_input_tokens: 0,
                            cache_write_input_tokens: 0,
                        }),
                        metrics: Some(Metrics {
                            latency_ms: 0,
                            time_to_first_byte_ms: 0,
                        }),
                        trace: None,
                    }),
                    ..Default::default()
                });
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_model_creation() {
        let model = AnthropicModel::new("test-key", 4096).with_model("claude-3-opus-20240229");
        assert_eq!(model.config().model_id, "claude-3-opus-20240229");
        assert_eq!(model.max_tokens, 4096);
    }
}

