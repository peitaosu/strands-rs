//! Mistral AI model provider.
//!
//! Docs: https://docs.mistral.ai/

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::models::{Model, ModelConfig, StreamEventStream};
use crate::types::content::{Message, Role, SystemContentBlock};
use crate::types::errors::StrandsError;
use crate::types::streaming::{StopReason, StreamEvent};
use crate::types::tools::{ToolChoice, ToolSpec};

/// Configuration for Mistral models.
#[derive(Debug, Clone, Default)]
pub struct MistralConfig {
    /// Mistral model ID (e.g., "mistral-large-latest").
    pub model_id: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u32>,
    /// Controls randomness (0.0 to 1.0).
    pub temperature: Option<f32>,
    /// Controls diversity via nucleus sampling.
    pub top_p: Option<f32>,
    /// API key for authentication.
    pub api_key: Option<String>,
}

impl MistralConfig {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

/// Mistral API request format.
#[derive(Debug, Serialize)]
struct MistralRequest {
    model: String,
    messages: Vec<MistralMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MistralTool>>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct MistralMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MistralToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MistralToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: MistralFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct MistralFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct MistralTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: MistralFunctionDef,
}

#[derive(Debug, Serialize)]
struct MistralFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Mistral AI model provider.
pub struct MistralModel {
    config: ModelConfig,
    mistral_config: MistralConfig,
    client: reqwest::Client,
}

impl MistralModel {
    const BASE_URL: &'static str = "https://api.mistral.ai/v1";

    pub fn new(config: MistralConfig) -> Self {
        let model_config = ModelConfig {
            model_id: config.model_id.clone(),
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            ..Default::default()
        };

        Self {
            config: model_config,
            mistral_config: config,
            client: reqwest::Client::new(),
        }
    }

    fn api_key(&self) -> Result<String, StrandsError> {
        self.mistral_config
            .api_key
            .clone()
            .or_else(|| std::env::var("MISTRAL_API_KEY").ok())
            .ok_or_else(|| StrandsError::ConfigurationError {
                message: "Mistral API key not configured. Set MISTRAL_API_KEY or provide api_key".into(),
            })
    }

    fn convert_messages(&self, messages: &[Message], system_prompt: Option<&str>) -> Vec<MistralMessage> {
        let mut result = Vec::new();

        if let Some(prompt) = system_prompt {
            result.push(MistralMessage {
                role: "system".to_string(),
                content: prompt.to_string(),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        for msg in messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            let content = msg.text_content();

            let tool_calls: Option<Vec<MistralToolCall>> = {
                let calls: Vec<_> = msg
                    .content
                    .iter()
                    .filter_map(|b| b.tool_use.as_ref())
                    .map(|tu| MistralToolCall {
                        id: tu.tool_use_id.clone(),
                        call_type: "function".to_string(),
                        function: MistralFunction {
                            name: tu.name.clone(),
                            arguments: serde_json::to_string(&tu.input).unwrap_or_default(),
                        },
                    })
                    .collect();

                if calls.is_empty() {
                    None
                } else {
                    Some(calls)
                }
            };

            if tool_calls.is_some() {
                result.push(MistralMessage {
                    role: role.to_string(),
                    content,
                    tool_calls,
                    tool_call_id: None,
                });
            } else if msg.has_tool_result() {
                for block in &msg.content {
                    if let Some(tr) = &block.tool_result {
                        let content_text = tr
                            .content
                            .iter()
                            .filter_map(|c| c.text.as_ref())
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("");

                        result.push(MistralMessage {
                            role: "tool".to_string(),
                            content: content_text,
                            tool_calls: None,
                            tool_call_id: Some(tr.tool_use_id.clone()),
                        });
                    }
                }
            } else {
                result.push(MistralMessage {
                    role: role.to_string(),
                    content,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }

        result
    }

    fn convert_tools(&self, tool_specs: &[ToolSpec]) -> Vec<MistralTool> {
        tool_specs
            .iter()
            .map(|spec| MistralTool {
                tool_type: "function".to_string(),
                function: MistralFunctionDef {
                    name: spec.name.clone(),
                    description: spec.description.clone(),
                    parameters: serde_json::to_value(&spec.input_schema).unwrap_or_default(),
                },
            })
            .collect()
    }
}

#[async_trait]
impl Model for MistralModel {
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
        let messages = messages.to_vec();
        let tool_specs = tool_specs.map(|t| t.to_vec());
        let system_prompt = system_prompt.map(|s| s.to_string());

        Box::pin(async_stream::stream! {
            let api_key = match self.api_key() {
                Ok(key) => key,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let mistral_messages = self.convert_messages(&messages, system_prompt.as_deref());
            let tools = tool_specs.as_ref().map(|specs| self.convert_tools(specs));

            let request = MistralRequest {
                model: self.config.model_id.clone(),
                messages: mistral_messages,
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
                top_p: self.config.top_p,
                tools,
                stream: true,
            };

            let url = format!("{}/chat/completions", Self::BASE_URL);

            let response = match self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
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
                    yield Err(StrandsError::ModelThrottled {
                        message: "Mistral rate limit exceeded".into(),
                    });
                } else {
                    yield Err(StrandsError::ModelError {
                        message: format!("Mistral API error {}: {}", status, body),
                        source: None,
                    });
                }
                return;
            }

            yield Ok(StreamEvent::message_start(Role::Assistant));

            let body = match response.text().await {
                Ok(b) => b,
                Err(e) => {
                    yield Err(StrandsError::NetworkError(e.to_string()));
                    return;
                }
            };

            for line in body.lines() {
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    if data == "[DONE]" {
                        break;
                    }

                    if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                            for choice in choices {
                                if let Some(delta) = choice.get("delta") {
                                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                        yield Ok(StreamEvent::text_delta(0, content));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            yield Ok(StreamEvent::message_stop(StopReason::EndTurn));
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_config() {
        let config = MistralConfig::new("mistral-large-latest")
            .with_api_key("test-key")
            .with_temperature(0.7);

        assert_eq!(config.model_id, "mistral-large-latest");
        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.temperature, Some(0.7));
    }
}

