//! Ollama local model provider.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{Model, ModelConfig, StreamEventStream};
use crate::types::{
    content::{Message, Role, SystemContentBlock},
    errors::StrandsError,
    streaming::{
        ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockDeltaToolUse, ContentBlockStart,
        ContentBlockStartEvent, ContentBlockStartToolUse, ContentBlockStopEvent, MessageStartEvent,
        MessageStopEvent, MetadataEvent, Metrics, StopReason, StreamEvent, Usage,
    },
    tools::{ToolChoice, ToolSpec},
};

const DEFAULT_MODEL_ID: &str = "llama3";
const DEFAULT_HOST: &str = "http://localhost:11434";

/// Response substrings that indicate a context-window overflow.
const OLLAMA_OVERFLOW_MESSAGES: [&str; 2] = [
    "input length exceeds context length",
    "exceeds context length",
];

/// Ollama local model provider.
#[derive(Clone)]
pub struct OllamaModel {
    config: ModelConfig,
    host: String,
    keep_alive: Option<String>,
    client: Client,
}

impl std::fmt::Debug for OllamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaModel")
            .field("config", &self.config)
            .field("host", &self.host)
            .finish()
    }
}

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OllamaTool>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OllamaFunctionCall {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OllamaFunctionDef,
}

#[derive(Debug, Serialize)]
struct OllamaFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamResponse {
    #[serde(default)]
    message: OllamaResponseMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    total_duration: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

impl OllamaModel {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            config: ModelConfig::new(model_id),
            host: DEFAULT_HOST.to_string(),
            keep_alive: Some("5m".to_string()),
            client: Client::new(),
        }
    }

    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Sets how long the model stays loaded after a request (e.g. `"5m"`, `0`).
    pub fn with_keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = config;
        self
    }

    fn format_messages(&self, messages: &[Message], system_prompt: Option<&str>) -> Vec<OllamaMessage> {
        let mut formatted = Vec::new();

        if let Some(prompt) = system_prompt {
            formatted.push(OllamaMessage {
                role: "system".to_string(),
                content: prompt.to_string(),
                images: None,
                tool_calls: None,
            });
        }

        for msg in messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            let mut text_content = String::new();
            let mut tool_calls = Vec::new();

            for block in &msg.content {
                if let Some(ref text) = block.text {
                    text_content.push_str(text);
                }

                if let Some(ref tu) = block.tool_use {
                    tool_calls.push(OllamaToolCall {
                        function: OllamaFunctionCall {
                            name: tu.name.clone(),
                            arguments: tu.input.clone(),
                        },
                    });
                }

                if let Some(ref tr) = block.tool_result {
                    let content = tr
                        .content
                        .iter()
                        .filter_map(|c| c.text.clone())
                        .collect::<Vec<_>>()
                        .join("\n");

                    formatted.push(OllamaMessage {
                        role: "tool".to_string(),
                        content,
                        images: None,
                        tool_calls: None,
                    });
                }
            }

            if !text_content.is_empty() || !tool_calls.is_empty() {
                formatted.push(OllamaMessage {
                    role: role.to_string(),
                    content: text_content,
                    images: None,
                    tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                });
            }
        }

        formatted
    }

    fn format_tools(&self, tool_specs: &[ToolSpec]) -> Vec<OllamaTool> {
        tool_specs
            .iter()
            .map(|spec| OllamaTool {
                tool_type: "function".to_string(),
                function: OllamaFunctionDef {
                    name: spec.name.clone(),
                    description: spec.description.clone(),
                    parameters: spec.input_schema.json.clone(),
                },
            })
            .collect()
    }
}

impl Default for OllamaModel {
    fn default() -> Self {
        Self::new(DEFAULT_MODEL_ID)
    }
}

#[async_trait]
impl Model for OllamaModel {
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
        _tool_choice: Option<ToolChoice>,
        _system_prompt_content: Option<&'a [SystemContentBlock]>,
    ) -> StreamEventStream<'a> {
        let url = format!("{}/api/chat", self.host);
        let client = self.client.clone();

        let options = OllamaOptions {
            num_predict: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
        };

        let request = OllamaRequest {
            model: self.config.model_id.clone(),
            messages: self.format_messages(messages, system_prompt),
            stream: true,
            options: Some(options),
            keep_alive: self.keep_alive.clone(),
            tools: tool_specs.map(|s| self.format_tools(s)).unwrap_or_default(),
        };

        Box::pin(async_stream::stream! {
            let response = match client
                .post(&url)
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
                if OLLAMA_OVERFLOW_MESSAGES.iter().any(|m| body.contains(m)) {
                    yield Err(StrandsError::ContextWindowOverflow { message: body });
                } else {
                    yield Err(StrandsError::model_error(format!("HTTP {status}: {body}")));
                }
                return;
            }

            yield Ok(StreamEvent {
                message_start: Some(MessageStartEvent { role: Role::Assistant }),
                ..Default::default()
            });

            yield Ok(StreamEvent {
                content_block_start: Some(ContentBlockStartEvent {
                    content_block_index: Some(0),
                    start: None,
                }),
                ..Default::default()
            });

            use futures::StreamExt;
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut tool_calls_found: Vec<OllamaToolCall> = Vec::new();
            let mut final_response: Option<OllamaStreamResponse> = None;

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                    Err(e) => {
                        yield Err(StrandsError::NetworkError(e.to_string()));
                        return;
                    }
                };

                buffer.push_str(&chunk);

                for line in super::drain_complete_lines(&mut buffer) {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    if let Ok(resp) = serde_json::from_str::<OllamaStreamResponse>(line) {
                        if !resp.message.content.is_empty() {
                            yield Ok(StreamEvent {
                                content_block_delta: Some(ContentBlockDeltaEvent {
                                    content_block_index: Some(0),
                                    delta: Some(ContentBlockDelta {
                                        text: Some(resp.message.content.clone()),
                                        ..Default::default()
                                    }),
                                }),
                                ..Default::default()
                            });
                        }

                        if let Some(ref tcs) = resp.message.tool_calls {
                            tool_calls_found.extend(tcs.clone());
                        }

                        if resp.done {
                            final_response = Some(resp);
                            break;
                        }
                    }
                }
            }

            yield Ok(StreamEvent {
                content_block_stop: Some(ContentBlockStopEvent {
                    content_block_index: Some(0),
                }),
                ..Default::default()
            });

            let mut tool_index = 1u32;
            for tc in &tool_calls_found {
                yield Ok(StreamEvent {
                    content_block_start: Some(ContentBlockStartEvent {
                        content_block_index: Some(tool_index),
                        start: Some(ContentBlockStart {
                            tool_use: Some(ContentBlockStartToolUse {
                                name: tc.function.name.clone(),
                                tool_use_id: tc.function.name.clone(),
                            }),
                        }),
                    }),
                    ..Default::default()
                });

                yield Ok(StreamEvent {
                    content_block_delta: Some(ContentBlockDeltaEvent {
                        content_block_index: Some(tool_index),
                        delta: Some(ContentBlockDelta {
                            tool_use: Some(ContentBlockDeltaToolUse {
                                input: serde_json::to_string(&tc.function.arguments).unwrap_or_default(),
                            }),
                            ..Default::default()
                        }),
                    }),
                    ..Default::default()
                });

                yield Ok(StreamEvent {
                    content_block_stop: Some(ContentBlockStopEvent {
                        content_block_index: Some(tool_index),
                    }),
                    ..Default::default()
                });

                tool_index += 1;
            }

            let stop_reason = if !tool_calls_found.is_empty() {
                StopReason::ToolUse
            } else if final_response.as_ref().and_then(|r| r.done_reason.as_ref()).map(|s| s == "length").unwrap_or(false) {
                StopReason::MaxTokens
            } else {
                StopReason::EndTurn
            };

            yield Ok(StreamEvent {
                message_stop: Some(MessageStopEvent {
                    stop_reason: Some(stop_reason),
                    additional_model_response_fields: None,
                }),
                ..Default::default()
            });

            if let Some(ref resp) = final_response {
                let input_tokens = resp.prompt_eval_count.unwrap_or(0);
                let output_tokens = resp.eval_count.unwrap_or(0);
                let latency_ms = resp.total_duration.map(|d| d / 1_000_000).unwrap_or(0);

                yield Ok(StreamEvent {
                    metadata: Some(MetadataEvent {
                        usage: Some(Usage {
                            input_tokens,
                            output_tokens,
                            total_tokens: input_tokens + output_tokens,
                            cache_read_input_tokens: 0,
                            cache_write_input_tokens: 0,
                        }),
                        metrics: Some(Metrics {
                            latency_ms,
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
    fn test_ollama_model_creation() {
        let model = OllamaModel::new("llama3.2");
        assert_eq!(model.config().model_id, "llama3.2");
    }

    #[test]
    fn test_ollama_with_host() {
        let model = OllamaModel::new("llama3").with_host("http://192.168.1.100:11434");
        assert_eq!(model.host, "http://192.168.1.100:11434");
    }

    #[test]
    fn test_ollama_default() {
        let model = OllamaModel::default();
        assert_eq!(model.config().model_id, "llama3");
        assert_eq!(model.host, "http://localhost:11434");
    }
}

