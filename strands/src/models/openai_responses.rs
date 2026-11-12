//! OpenAI Responses API model provider.
//!
//! Port of the Python `strands.models.openai_responses.OpenAIResponsesModel`:
//! streams the Responses API's typed events into the SDK `StreamEvent`
//! protocol, supports server-side stateful conversations via
//! `previous_response_id`, and maps tool calls to `toolUse` blocks.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use base64::Engine;
use reqwest::Client;
use serde_json::{json, Value};

use super::{Model, ModelConfig, StreamEventStream};
use crate::types::content::{ContentBlock, Message, Role, SystemContentBlock};
use crate::types::errors::StrandsError;
use crate::types::streaming::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockDeltaToolUse, ContentBlockStart,
    ContentBlockStartEvent, ContentBlockStartToolUse, ContentBlockStopEvent, MessageStartEvent,
    MessageStopEvent, MetadataEvent, Metrics, ReasoningContentBlockDelta, StopReason, StreamEvent,
    Usage,
};
use crate::types::tools::{ToolChoice, ToolSpec};

const DEFAULT_MODEL_ID: &str = "gpt-5.6";
const RESPONSES_API_URL: &str = "https://api.openai.com/v1/responses";
const MAX_MEDIA_BYTES: usize = 20 * 1024 * 1024;

/// OpenAI Responses API model provider.
#[derive(Clone)]
pub struct OpenAIResponsesModel {
    config: ModelConfig,
    api_key: String,
    base_url: Option<String>,
    stateful: bool,
    use_native_token_count: bool,
    response_id: Arc<Mutex<Option<String>>>,
    client: Client,
}

impl std::fmt::Debug for OpenAIResponsesModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIResponsesModel")
            .field("config", &self.config)
            .field("base_url", &self.base_url)
            .field("stateful", &self.stateful)
            .finish()
    }
}

impl OpenAIResponsesModel {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            config: ModelConfig::new(DEFAULT_MODEL_ID),
            api_key: api_key.into(),
            base_url: None,
            stateful: false,
            use_native_token_count: false,
            response_id: Arc::new(Mutex::new(None)),
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

    /// Enables server-side conversation state management.
    pub fn with_stateful(mut self, stateful: bool) -> Self {
        self.stateful = stateful;
        self
    }

    /// Enables the native input token count endpoint, falling back to the
    /// local estimator when the call fails.
    pub fn with_use_native_token_count(mut self, enabled: bool) -> Self {
        self.use_native_token_count = enabled;
        self
    }

    fn format_tool_choice(tool_choice: Option<&ToolChoice>) -> Value {
        match tool_choice {
            None => Value::Null,
            Some(ToolChoice::Auto(_)) => json!("auto"),
            Some(ToolChoice::Any(_)) => json!("required"),
            Some(ToolChoice::Tool(t)) => json!({"type": "function", "name": t.name}),
        }
    }

    fn format_request_messages(messages: &[Message]) -> Vec<Value> {
        let mut formatted = Vec::new();

        for message in messages {
            let role = match message.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            let mut formatted_contents: Vec<Value> = Vec::new();
            for content in &message.content {
                if content.tool_use.is_some() || content.tool_result.is_some() {
                    continue;
                }
                if let Some(part) = Self::format_message_content(content, role) {
                    formatted_contents.push(part);
                }
            }

            if role == "assistant" {
                let text = formatted_contents
                    .iter()
                    .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    formatted.push(json!({"role": "assistant", "content": text}));
                }
            } else if !formatted_contents.is_empty() {
                formatted.push(json!({"role": "user", "content": formatted_contents}));
            }

            for content in &message.content {
                if let Some(tool_use) = &content.tool_use {
                    formatted.push(json!({
                        "type": "function_call",
                        "call_id": tool_use.tool_use_id,
                        "name": tool_use.name,
                        "arguments": serde_json::to_string(&tool_use.input)
                            .unwrap_or_else(|_| "{}".to_string()),
                    }));
                }
                if let Some(tool_result) = &content.tool_result {
                    formatted.push(Self::format_tool_message(tool_result));
                }
            }
        }

        formatted
            .into_iter()
            .filter(|item| {
                let has_content = item
                    .get("content")
                    .map(|c| !c.is_null() && c.as_str().map(|s| !s.is_empty()).unwrap_or(true))
                    .unwrap_or(false);
                let item_type = item.get("type").and_then(|t| t.as_str());
                has_content || matches!(item_type, Some("function_call") | Some("function_call_output"))
            })
            .collect()
    }

    fn format_message_content(content: &ContentBlock, role: &str) -> Option<Value> {
        let text_type = if role == "assistant" { "output_text" } else { "input_text" };

        if let Some(text) = &content.text {
            return Some(json!({"type": text_type, "text": text}));
        }
        if let Some(image) = &content.image {
            let data_url = encode_media_data_url(
                image.source.bytes.as_deref().unwrap_or_default(),
                &image.format,
            )
            .ok()?;
            return Some(json!({"type": "input_image", "image_url": data_url}));
        }
        if let Some(document) = &content.document {
            let data_url = encode_media_data_url(
                document.source.bytes.as_deref().unwrap_or_default(),
                &document.format,
            )
            .ok()?;
            let filename = format!("{}.{}", document.name, document.format);
            return Some(json!({
                "type": "input_file",
                "filename": filename,
                "file_data": data_url,
            }));
        }
        if let Some(citations) = &content.citations_content {
            let text = citations
                .content
                .as_ref()
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|part| part.text.as_deref())
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();
            return Some(json!({"type": text_type, "text": text}));
        }
        None
    }

    fn format_tool_message(tool_result: &crate::types::tools::ToolResult) -> Value {
        let mut output_parts: Vec<Value> = Vec::new();
        let mut has_media = false;

        for content in &tool_result.content {
            if let Some(json_value) = &content.json {
                output_parts.push(json!({
                    "type": "input_text",
                    "text": serde_json::to_string(json_value).unwrap_or_default(),
                }));
            } else if let Some(text) = &content.text {
                output_parts.push(json!({"type": "input_text", "text": text}));
            } else if let Some(image) = &content.image {
                has_media = true;
                if let Ok(data_url) = encode_media_data_url(&image.data, &image.format) {
                    output_parts.push(json!({"type": "input_image", "image_url": data_url}));
                }
            } else if let Some(document) = &content.document {
                has_media = true;
                if let Ok(data_url) = encode_media_data_url(&document.data, &document.format) {
                    output_parts.push(json!({
                        "type": "input_file",
                        "filename": format!("{}.{}", document.name, document.format),
                        "file_data": data_url,
                    }));
                }
            }
        }

        let output = if has_media {
            Value::Array(output_parts)
        } else {
            let text = output_parts
                .iter()
                .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            json!(text)
        };

        json!({
            "type": "function_call_output",
            "call_id": tool_result.tool_use_id,
            "output": output,
        })
    }

    pub(crate) fn format_request(
        &self,
        messages: &[Message],
        tool_specs: Option<&[ToolSpec]>,
        system_prompt: Option<&str>,
        tool_choice: Option<&ToolChoice>,
    ) -> Value {
        let input_items = Self::format_request_messages(messages);
        let mut request = json!({
            "model": self.config.model_id,
            "input": input_items,
            "stream": true,
            "store": self.stateful,
        });

        if self.stateful {
            if let Some(response_id) = self.response_id.lock().unwrap().as_ref() {
                request["previous_response_id"] = json!(response_id);
            }
        }

        if let Some(prompt) = system_prompt {
            request["instructions"] = json!(prompt);
        }

        if let Some(max_tokens) = self.config.max_tokens {
            request["max_output_tokens"] = json!(max_tokens);
        }
        if let Some(temperature) = self.config.temperature {
            request["temperature"] = json!(temperature);
        }
        if let Some(top_p) = self.config.top_p {
            request["top_p"] = json!(top_p);
        }
        for (key, value) in &self.config.additional {
            request[key.as_str()] = value.clone();
        }

        if let Some(specs) = tool_specs {
            let mut tools: Vec<Value> = request
                .get("tools")
                .and_then(|t| t.as_array())
                .cloned()
                .unwrap_or_default();
            tools.extend(specs.iter().map(|spec| {
                json!({
                    "type": "function",
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema.json,
                })
            }));
            request["tools"] = json!(tools);

            let choice = Self::format_tool_choice(tool_choice);
            if !choice.is_null() {
                request["tool_choice"] = choice;
            }
        }

        request
    }

    fn error_class(body: &str, status: u16) -> StrandsError {
        if status == 429 {
            StrandsError::ModelThrottled { message: body.to_string() }
        } else if body.contains("context_length_exceeded") || body.contains("context window") {
            StrandsError::ContextWindowOverflow { message: body.to_string() }
        } else {
            StrandsError::model_error(format!("HTTP {status}: {body}"))
        }
    }
}

/// Encodes raw bytes into a `data:` URL for the Responses API media inputs.
fn encode_media_data_url(encoded: &str, format: &str) -> Result<String, StrandsError> {
    use base64::engine::general_purpose::STANDARD;

    let bytes = STANDARD
        .decode(encoded)
        .map_err(|e| StrandsError::model_error(format!("invalid media encoding: {}", e)))?;
    if bytes.len() > MAX_MEDIA_BYTES {
        return Err(StrandsError::model_error(format!(
            "media exceeds maximum size of {} bytes",
            MAX_MEDIA_BYTES
        )));
    }
    let mime = match format {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        "csv" => "text/csv",
        "doc" => "application/msword",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xls" => "application/vnd.ms-excel",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "html" => "text/html",
        "txt" | "md" => "text/plain",
        _ => "application/octet-stream",
    };
    Ok(format!("data:{};base64,{}", mime, encoded))
}

#[derive(Debug, Default)]
struct ToolCallInfo {
    name: String,
    call_id: String,
    item_id: String,
    arguments: String,
}

#[async_trait]
impl Model for OpenAIResponsesModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn update_config(&mut self, config: ModelConfig) {
        self.config = config;
    }

    async fn count_tokens(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
    ) -> Result<u64, StrandsError> {
        if !self.use_native_token_count {
            return Ok(super::estimate_tokens(messages, system_prompt));
        }

        let mut request = self.format_request(messages, None, system_prompt, None);
        let count_fields = ["model", "input", "instructions", "tools"];
        if let Some(object) = request.as_object_mut() {
            object.retain(|key, _| count_fields.contains(&key.as_str()));
        }

        let url = self
            .base_url
            .clone()
            .unwrap_or_else(|| RESPONSES_API_URL.to_string());
        let count_url = format!("{}/input_tokens", url.trim_end_matches('/'));

        let response = self
            .client
            .post(&count_url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await;

        if let Ok(response) = response {
            if response.status().is_success() {
                if let Ok(body) = response.json::<Value>().await {
                    if let Some(tokens) = body.get("input_tokens").and_then(|t| t.as_u64()) {
                        return Ok(tokens);
                    }
                }
            }
        }

        Ok(super::estimate_tokens(messages, system_prompt))
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        tool_specs: Option<&'a [ToolSpec]>,
        system_prompt: Option<&'a str>,
        tool_choice: Option<ToolChoice>,
        _system_prompt_content: Option<&'a [SystemContentBlock]>,
    ) -> StreamEventStream<'a> {
        let request = self.format_request(messages, tool_specs, system_prompt, tool_choice.as_ref());
        let url = self
            .base_url
            .clone()
            .unwrap_or_else(|| RESPONSES_API_URL.to_string());
        let api_key = self.api_key.clone();
        let response_id_slot = self.response_id.clone();
        let timeout = std::time::Duration::from_secs(300);

        Box::pin(async_stream::stream! {
            let client = Client::builder().timeout(timeout).build();
            let client = match client {
                Ok(client) => client,
                Err(e) => {
                    yield Err(StrandsError::NetworkError(e.to_string()));
                    return;
                }
            };

            let response = client
                .post(&url)
                .bearer_auth(&api_key)
                .json(&request)
                .send()
                .await;

            let response = match response {
                Ok(response) if response.status().is_success() => response,
                Ok(response) => {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    yield Err(Self::error_class(&body, status.as_u16()));
                    return;
                }
                Err(e) => {
                    yield Err(StrandsError::NetworkError(e.to_string()));
                    return;
                }
            };

            yield Ok(StreamEvent {
                message_start: Some(MessageStartEvent { role: Role::Assistant }),
                ..Default::default()
            });

            let mut tool_calls: Vec<ToolCallInfo> = Vec::new();
            let mut final_usage: Option<Usage> = None;
            let mut data_type: Option<&'static str> = None;
            let mut stop_reason: Option<StopReason> = None;
            let mut block_index: u32 = 0;

            use futures::StreamExt;
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            loop {
                match byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                    }
                    _ => break,
                }

                for line in super::drain_complete_lines(&mut buffer) {
                    let line = line.trim();
                    let Some(json_str) = line.strip_prefix("data: ") else {
                        continue;
                    };
                    if json_str == "[DONE]" {
                        continue;
                    }
                    let Ok(event) = serde_json::from_str::<Value>(json_str) else {
                        continue;
                    };

                    let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");

                    match event_type {
                        "response.created" => {
                            if let Some(id) = event
                                .get("response")
                                .and_then(|r| r.get("id"))
                                .and_then(|i| i.as_str())
                            {
                                *response_id_slot.lock().unwrap() = Some(id.to_string());
                            }
                        }
                        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                            if data_type != Some("reasoning") {
                                block_index += 1;
                                data_type = Some("reasoning");
                            }
                            if let Some(delta) = event.get("delta").and_then(|d| d.as_str()) {
                                yield Ok(StreamEvent {
                                    content_block_delta: Some(ContentBlockDeltaEvent {
                                        content_block_index: Some(block_index),
                                        delta: Some(ContentBlockDelta {
                                            reasoning_content: Some(ReasoningContentBlockDelta {
                                                text: Some(delta.to_string()),
                                                ..Default::default()
                                            }),
                                            ..Default::default()
                                        }),
                                    }),
                                    ..Default::default()
                                });
                            }
                        }
                        "response.output_text.delta" => {
                            if data_type != Some("text") {
                                block_index += 1;
                                data_type = Some("text");
                            }
                            if let Some(delta) = event.get("delta").and_then(|d| d.as_str()) {
                                yield Ok(StreamEvent {
                                    content_block_delta: Some(ContentBlockDeltaEvent {
                                        content_block_index: Some(block_index),
                                        delta: Some(ContentBlockDelta {
                                            text: Some(delta.to_string()),
                                            ..Default::default()
                                        }),
                                    }),
                                    ..Default::default()
                                });
                            }
                        }
                        "response.output_item.added" => {
                            let item = event.get("item");
                            if item
                                .and_then(|i| i.get("type"))
                                .and_then(|t| t.as_str())
                                == Some("function_call")
                            {
                                tool_calls.push(ToolCallInfo {
                                    name: item
                                        .and_then(|i| i.get("name"))
                                        .and_then(|n| n.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    call_id: item
                                        .and_then(|i| i.get("call_id"))
                                        .and_then(|c| c.as_str())
                                        .unwrap_or("unknown")
                                        .to_string(),
                                    item_id: item
                                        .and_then(|i| i.get("id"))
                                        .and_then(|i| i.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    arguments: String::new(),
                                });
                            }
                        }
                        "response.function_call_arguments.delta" => {
                            let item_id = event.get("item_id").and_then(|i| i.as_str());
                            let delta = event.get("delta").and_then(|d| d.as_str()).unwrap_or("");
                            if let Some(item_id) = item_id {
                                if let Some(info) = tool_calls
                                    .iter_mut()
                                    .find(|info| info.item_id == item_id)
                                {
                                    info.arguments.push_str(delta);
                                }
                            }
                        }
                        "response.function_call_arguments.done" => {
                            let item_id = event.get("item_id").and_then(|i| i.as_str());
                            let arguments = event.get("arguments").and_then(|a| a.as_str());
                            if let (Some(item_id), Some(arguments)) = (item_id, arguments) {
                                if let Some(info) = tool_calls
                                    .iter_mut()
                                    .find(|info| info.item_id == item_id)
                                {
                                    info.arguments = arguments.to_string();
                                }
                            }
                        }
                        "response.failed" | "error" => {
                            let message = if event_type == "response.failed" {
                                event
                                    .get("response")
                                    .and_then(|r| r.get("error"))
                                    .and_then(|e| e.get("message"))
                                    .and_then(|m| m.as_str())
                            } else {
                                event.get("message").and_then(|m| m.as_str())
                            };
                            yield Err(StrandsError::model_error(
                                message.unwrap_or("OpenAI Responses stream error").to_string(),
                            ));
                            return;
                        }
                        "response.incomplete" => {
                            let response_data = event.get("response");
                            if let Some(usage) = response_data.and_then(|r| r.get("usage")) {
                                final_usage = Some(parse_usage(usage));
                            }
                            let reason = response_data
                                .and_then(|r| r.get("incomplete_details"))
                                .and_then(|d| d.get("reason"))
                                .and_then(|r| r.as_str());
                            if reason == Some("max_output_tokens") {
                                stop_reason = Some(StopReason::MaxTokens);
                            } else if reason == Some("content_filter") {
                                stop_reason = Some(StopReason::ContentFiltered);
                            }
                            break;
                        }
                        "response.completed" => {
                            if let Some(usage) =
                                event.get("response").and_then(|r| r.get("usage"))
                            {
                                final_usage = Some(parse_usage(usage));
                            }
                            break;
                        }
                        _ => {}
                    }
                }
            }

            if data_type.is_some() {
                yield Ok(StreamEvent {
                    content_block_stop: Some(ContentBlockStopEvent {
                        content_block_index: Some(block_index),
                    }),
                    ..Default::default()
                });
            }

            let has_tool_calls = !tool_calls.is_empty();
            for info in &tool_calls {
                block_index += 1;
                yield Ok(StreamEvent {
                    content_block_start: Some(ContentBlockStartEvent {
                        content_block_index: Some(block_index),
                        start: Some(ContentBlockStart {
                            tool_use: Some(ContentBlockStartToolUse {
                                name: info.name.clone(),
                                tool_use_id: info.call_id.clone(),
                            }),
                        }),
                    }),
                    ..Default::default()
                });
                yield Ok(StreamEvent {
                    content_block_delta: Some(ContentBlockDeltaEvent {
                        content_block_index: Some(block_index),
                        delta: Some(ContentBlockDelta {
                            tool_use: Some(ContentBlockDeltaToolUse {
                                input: info.arguments.clone(),
                            }),
                            ..Default::default()
                        }),
                    }),
                    ..Default::default()
                });
                yield Ok(StreamEvent {
                    content_block_stop: Some(ContentBlockStopEvent {
                        content_block_index: Some(block_index),
                    }),
                    ..Default::default()
                });
            }

            let finish_reason = if has_tool_calls {
                StopReason::ToolUse
            } else {
                stop_reason.unwrap_or(StopReason::EndTurn)
            };

            yield Ok(StreamEvent {
                message_stop: Some(MessageStopEvent {
                    stop_reason: Some(finish_reason),
                    additional_model_response_fields: None,
                }),
                ..Default::default()
            });

            if let Some(usage) = final_usage {
                yield Ok(StreamEvent {
                    metadata: Some(MetadataEvent {
                        usage: Some(usage),
                        metrics: Some(Metrics::default()),
                        trace: None,
                    }),
                    ..Default::default()
                });
            }
        })
    }
}

/// Maps a Responses API usage object onto the SDK usage shape.
fn parse_usage(raw: &Value) -> Usage {
    let input = raw.get("input_tokens").and_then(|t| t.as_u64()).unwrap_or(0);
    let output = raw.get("output_tokens").and_then(|t| t.as_u64()).unwrap_or(0);
    let mut usage = Usage::new(input as u32, output as u32);
    if let Some(details) = raw.get("input_tokens_details").and_then(|d| d.get("cached_tokens")) {
        if let Some(cached) = details.as_u64() {
            usage.cache_read_input_tokens = cached as u32;
        }
    }
    usage
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::content::ContentBlock;
    use crate::types::tools::ToolResult;

    fn responses_model() -> OpenAIResponsesModel {
        OpenAIResponsesModel::new("test-key").with_model("gpt-5.6")
    }

    #[test]
    fn test_format_request_basic() {
        let model = responses_model();
        let messages = vec![Message::user("hello")];
        let request = model.format_request(&messages, None, Some("be terse"), None);

        assert_eq!(request["model"], "gpt-5.6");
        assert_eq!(request["instructions"], "be terse");
        assert_eq!(request["stream"], true);
        assert_eq!(request["store"], false);
        assert_eq!(request["input"][0]["role"], "user");
        assert_eq!(request["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(request["input"][0]["content"][0]["text"], "hello");
    }

    #[test]
    fn test_format_request_stateful_chaining() {
        let model = responses_model().with_stateful(true);
        *model.response_id.lock().unwrap() = Some("resp_123".to_string());

        let messages = vec![Message::user("next")];
        let request = model.format_request(&messages, None, None, None);

        assert_eq!(request["store"], true);
        assert_eq!(request["previous_response_id"], "resp_123");
    }

    #[test]
    fn test_format_request_tools_and_choice() {
        let model = responses_model();
        let spec = ToolSpec::new("get_weather", "Get weather")
            .with_input_schema(json!({"type": "object", "properties": {}}));
        let messages = vec![Message::user("weather?")];

        let request = model.format_request(&messages, Some(&[spec]), None, Some(&ToolChoice::any()));
        assert_eq!(request["tools"][0]["type"], "function");
        assert_eq!(request["tools"][0]["name"], "get_weather");
        assert_eq!(request["tool_choice"], "required");
    }

    #[test]
    fn test_format_tool_choice_variants() {
        assert_eq!(OpenAIResponsesModel::format_tool_choice(None), Value::Null);
        assert_eq!(
            OpenAIResponsesModel::format_tool_choice(Some(&ToolChoice::auto())),
            json!("auto")
        );
        assert_eq!(
            OpenAIResponsesModel::format_tool_choice(Some(&ToolChoice::tool("my_tool"))),
            json!({"type": "function", "name": "my_tool"})
        );
    }

    #[test]
    fn test_tool_use_and_result_replay() {
        let model = responses_model();
        let tool_use = crate::types::tools::ToolUse::new("calc", "call-1", json!({"a": 1}));
        let tool_result = ToolResult::success("call-1", "42");

        let messages = vec![
            Message::assistant(""),
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::tool_use(tool_use),
                    ContentBlock::tool_result(tool_result),
                ],
                tracking_id: None,
                metadata: None,
            },
        ];

        let request = model.format_request(&messages, None, None, None);
        let items = request["input"].as_array().expect("input items");
        assert_eq!(items[0]["type"], "function_call");
        assert_eq!(items[0]["call_id"], "call-1");
        assert_eq!(items[1]["type"], "function_call_output");
        assert_eq!(items[1]["output"], "42");
    }

    #[test]
    fn test_error_classification() {
        assert!(matches!(
            OpenAIResponsesModel::error_class("rate limited", 429),
            StrandsError::ModelThrottled { .. }
        ));
        assert!(matches!(
            OpenAIResponsesModel::error_class("context_length_exceeded here", 400),
            StrandsError::ContextWindowOverflow { .. }
        ));
        assert!(matches!(
            OpenAIResponsesModel::error_class("boom", 500),
            StrandsError::ModelError { .. }
        ));
    }

    #[test]
    fn test_media_data_url() {
        let encoded = base64::engine::general_purpose::STANDARD.encode(b"fake-image");
        let url = encode_media_data_url(&encoded, "png").unwrap();
        assert!(url.starts_with("data:image/png;base64,"));

        let oversized = base64::engine::general_purpose::STANDARD.encode(vec![0u8; MAX_MEDIA_BYTES + 1]);
        assert!(encode_media_data_url(&oversized, "png").is_err());
    }

    #[tokio::test]
    async fn test_count_tokens_heuristic_fallback() {
        let model = responses_model();
        let messages = vec![Message::user("four chars")];
        let tokens = model.count_tokens(&messages, None).await.unwrap();
        assert!(tokens > 0);
    }
}
