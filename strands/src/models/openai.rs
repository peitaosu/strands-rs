//! OpenAI model provider.

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

const DEFAULT_MODEL_ID: &str = "gpt-4o";
const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI model provider.
#[derive(Clone)]
pub struct OpenAIModel {
    config: ModelConfig,
    api_key: String,
    base_url: Option<String>,
    client: Client,
}

impl std::fmt::Debug for OpenAIModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIModel")
            .field("config", &self.config)
            .field("base_url", &self.base_url)
            .finish()
    }
}

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    stream_options: StreamOptions,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAIFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunctionDef,
}

#[derive(Debug, Serialize)]
struct OpenAIFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChunk {
    choices: Vec<OpenAIChoice>,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    delta: OpenAIDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[derive(Debug, Deserialize, Clone)]
struct OpenAIToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<OpenAIFunctionDelta>,
}

#[derive(Debug, Deserialize, Clone)]
struct OpenAIFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl OpenAIModel {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            config: ModelConfig::new(DEFAULT_MODEL_ID),
            api_key: api_key.into(),
            base_url: None,
            client: Client::new(),
        }
    }

    pub fn with_model(mut self, model_id: impl Into<String>) -> Self {
        self.config.model_id = model_id.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = config;
        self
    }

    fn format_messages(&self, messages: &[Message], system_prompt: Option<&str>) -> Vec<OpenAIMessage> {
        let mut formatted = Vec::new();

        if let Some(prompt) = system_prompt {
            formatted.push(OpenAIMessage {
                role: "system".to_string(),
                content: serde_json::Value::String(prompt.to_string()),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        for msg in messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            let mut text_content = Vec::new();
            let mut tool_calls = Vec::new();
            let mut tool_results = Vec::new();

            for block in &msg.content {
                if let Some(ref text) = block.text {
                    text_content.push(serde_json::json!({ "type": "text", "text": text }));
                }

                if let Some(ref tu) = block.tool_use {
                    tool_calls.push(OpenAIToolCall {
                        id: tu.tool_use_id.clone(),
                        call_type: "function".to_string(),
                        function: OpenAIFunction {
                            name: tu.name.clone(),
                            arguments: serde_json::to_string(&tu.input).unwrap_or_default(),
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
                    tool_results.push((tr.tool_use_id.clone(), content));
                }
            }

            if !tool_calls.is_empty() {
                formatted.push(OpenAIMessage {
                    role: role.to_string(),
                    content: if text_content.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::Array(text_content.clone())
                    },
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                });
            } else if !text_content.is_empty() {
                formatted.push(OpenAIMessage {
                    role: role.to_string(),
                    content: serde_json::Value::Array(text_content),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }

            for (tool_id, content) in tool_results {
                formatted.push(OpenAIMessage {
                    role: "tool".to_string(),
                    content: serde_json::Value::String(content),
                    tool_calls: None,
                    tool_call_id: Some(tool_id),
                });
            }
        }

        formatted
    }

    fn format_tools(&self, tool_specs: &[ToolSpec]) -> Vec<OpenAITool> {
        tool_specs
            .iter()
            .map(|spec| OpenAITool {
                tool_type: "function".to_string(),
                function: OpenAIFunctionDef {
                    name: spec.name.clone(),
                    description: spec.description.clone(),
                    parameters: spec.input_schema.json.clone(),
                },
            })
            .collect()
    }

    fn format_tool_choice(&self, tool_choice: Option<ToolChoice>) -> Option<serde_json::Value> {
        tool_choice.map(|tc| match tc {
            ToolChoice::Auto(_) => serde_json::json!("auto"),
            ToolChoice::Any(_) => serde_json::json!("required"),
            ToolChoice::Tool(t) => serde_json::json!({
                "type": "function",
                "function": { "name": t.name }
            }),
        })
    }

    fn map_stop_reason(reason: &str) -> StopReason {
        match reason {
            "tool_calls" => StopReason::ToolUse,
            "length" => StopReason::MaxTokens,
            "content_filter" => StopReason::ContentFiltered,
            _ => StopReason::EndTurn,
        }
    }
}

#[async_trait]
impl Model for OpenAIModel {
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
        let url = self.base_url.clone().unwrap_or_else(|| OPENAI_API_URL.to_string());
        let api_key = self.api_key.clone();
        let client = self.client.clone();

        let request = OpenAIRequest {
            model: self.config.model_id.clone(),
            messages: self.format_messages(messages, system_prompt),
            stream: true,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            tools: tool_specs.map(|s| self.format_tools(s)).unwrap_or_default(),
            tool_choice: self.format_tool_choice(tool_choice),
            stream_options: StreamOptions { include_usage: true },
        };

        Box::pin(async_stream::stream! {
            let response = match client
                .post(&url)
                .header("Authorization", format!("Bearer {api_key}"))
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
                } else if body.contains("context_length_exceeded") {
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

            let mut content_started = false;
            let mut tool_calls: std::collections::HashMap<usize, (String, String, String)> = std::collections::HashMap::new();
            let mut finish_reason = None;
            let mut final_usage = None;

            use futures::StreamExt;
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            loop {
                for line in buffer.lines() {
                    let line = line.trim();
                    if line.is_empty() || line == "data: [DONE]" {
                        continue;
                    }

                    if let Some(json_str) = line.strip_prefix("data: ") {
                        if let Ok(chunk) = serde_json::from_str::<OpenAIStreamChunk>(json_str) {
                            if let Some(usage) = chunk.usage {
                                final_usage = Some(usage);
                            }

                            for choice in chunk.choices {
                                if let Some(ref content) = choice.delta.content {
                                    if !content_started {
                                        yield Ok(StreamEvent {
                                            content_block_start: Some(ContentBlockStartEvent {
                                                content_block_index: Some(0),
                                                start: None,
                                            }),
                                            ..Default::default()
                                        });
                                        content_started = true;
                                    }

                                    yield Ok(StreamEvent {
                                        content_block_delta: Some(ContentBlockDeltaEvent {
                                            content_block_index: Some(0),
                                            delta: Some(ContentBlockDelta {
                                                text: Some(content.clone()),
                                                ..Default::default()
                                            }),
                                        }),
                                        ..Default::default()
                                    });
                                }

                                if let Some(ref tcs) = choice.delta.tool_calls {
                                    for tc in tcs {
                                        let entry = tool_calls.entry(tc.index).or_insert_with(|| {
                                            (String::new(), String::new(), String::new())
                                        });
                                        if let Some(ref id) = tc.id {
                                            entry.0 = id.clone();
                                        }
                                        if let Some(ref f) = tc.function {
                                            if let Some(ref name) = f.name {
                                                entry.1 = name.clone();
                                            }
                                            if let Some(ref args) = f.arguments {
                                                entry.2.push_str(args);
                                            }
                                        }
                                    }
                                }

                                if let Some(ref reason) = choice.finish_reason {
                                    finish_reason = Some(reason.clone());
                                }
                            }
                        }
                    }
                }

                match byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        buffer = String::from_utf8_lossy(&bytes).to_string();
                    }
                    _ => break,
                }
            }

            if content_started {
                yield Ok(StreamEvent {
                    content_block_stop: Some(ContentBlockStopEvent {
                        content_block_index: Some(0),
                    }),
                    ..Default::default()
                });
            }

            let mut tool_index = 1u32;
            for (_idx, (id, name, args)) in tool_calls {
                yield Ok(StreamEvent {
                    content_block_start: Some(ContentBlockStartEvent {
                        content_block_index: Some(tool_index),
                        start: Some(ContentBlockStart {
                            tool_use: Some(ContentBlockStartToolUse {
                                name: name.clone(),
                                tool_use_id: id.clone(),
                            }),
                        }),
                    }),
                    ..Default::default()
                });

                yield Ok(StreamEvent {
                    content_block_delta: Some(ContentBlockDeltaEvent {
                        content_block_index: Some(tool_index),
                        delta: Some(ContentBlockDelta {
                            tool_use: Some(ContentBlockDeltaToolUse { input: args }),
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

            let stop = finish_reason.as_deref().map(Self::map_stop_reason).unwrap_or(StopReason::EndTurn);

            yield Ok(StreamEvent {
                message_stop: Some(MessageStopEvent {
                    stop_reason: Some(stop),
                    additional_model_response_fields: None,
                }),
                ..Default::default()
            });

            if let Some(usage) = final_usage {
                yield Ok(StreamEvent {
                    metadata: Some(MetadataEvent {
                        usage: Some(Usage {
                            input_tokens: usage.prompt_tokens,
                            output_tokens: usage.completion_tokens,
                            total_tokens: usage.total_tokens,
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
    fn test_openai_model_creation() {
        let model = OpenAIModel::new("test-key").with_model("gpt-4o-mini");
        assert_eq!(model.config().model_id, "gpt-4o-mini");
    }

    #[test]
    fn test_openai_with_base_url() {
        let model = OpenAIModel::new("test-key").with_base_url("https://custom.api.com");
        assert_eq!(model.base_url, Some("https://custom.api.com".to_string()));
    }
}

