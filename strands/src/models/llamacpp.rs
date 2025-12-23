//! llama.cpp model provider.
//!
//! Provides integration with llama.cpp servers running in OpenAI-compatible mode.
//! Docs: https://github.com/ggml-org/llama.cpp

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::models::{Model, ModelConfig, StreamEventStream};
use crate::types::content::{Message, Role, SystemContentBlock};
use crate::types::errors::StrandsError;
use crate::types::streaming::{StopReason, StreamEvent};
use crate::types::tools::{ToolChoice, ToolSpec};

/// Configuration for llama.cpp models.
#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// Model identifier (default: "default").
    pub model_id: String,
    /// Base URL for the llama.cpp server.
    pub base_url: String,
    /// Additional model parameters.
    pub params: HashMap<String, serde_json::Value>,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_id: "default".to_string(),
            base_url: "http://localhost:8080".to_string(),
            params: HashMap::new(),
        }
    }
}

impl LlamaCppConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            ..Default::default()
        }
    }

    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.params.insert(key.into(), value);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.params.insert("temperature".to_string(), serde_json::json!(temperature));
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.params.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
        self
    }
}

/// OpenAI-compatible request format for llama.cpp.
#[derive(Debug, Serialize)]
struct LlamaCppRequest {
    model: String,
    messages: Vec<LlamaCppMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<LlamaCppTool>>,
    stream: bool,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LlamaCppMessage {
    role: String,
    content: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct LlamaCppTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: LlamaCppFunction,
}

#[derive(Debug, Serialize)]
struct LlamaCppFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// llama.cpp model provider.
pub struct LlamaCppModel {
    config: ModelConfig,
    llamacpp_config: LlamaCppConfig,
    client: reqwest::Client,
}

impl LlamaCppModel {
    pub fn new(config: LlamaCppConfig) -> Self {
        let model_config = ModelConfig::new(&config.model_id);

        Self {
            config: model_config,
            llamacpp_config: config,
            client: reqwest::Client::new(),
        }
    }

    fn convert_messages(&self, messages: &[Message], system_prompt: Option<&str>) -> Vec<LlamaCppMessage> {
        let mut result = Vec::new();

        if let Some(prompt) = system_prompt {
            result.push(LlamaCppMessage {
                role: "system".to_string(),
                content: serde_json::json!(prompt),
            });
        }

        for msg in messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            let content = msg.text_content();

            result.push(LlamaCppMessage {
                role: role.to_string(),
                content: serde_json::json!(content),
            });
        }

        result
    }

    fn convert_tools(&self, tool_specs: &[ToolSpec]) -> Vec<LlamaCppTool> {
        tool_specs
            .iter()
            .map(|spec| LlamaCppTool {
                tool_type: "function".to_string(),
                function: LlamaCppFunction {
                    name: spec.name.clone(),
                    description: spec.description.clone(),
                    parameters: serde_json::to_value(&spec.input_schema).unwrap_or_default(),
                },
            })
            .collect()
    }
}

#[async_trait]
impl Model for LlamaCppModel {
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
            let llamacpp_messages = self.convert_messages(&messages, system_prompt.as_deref());
            let tools = tool_specs.as_ref().map(|specs| self.convert_tools(specs));

            let max_tokens = self.llamacpp_config.params
                .get("max_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);

            let temperature = self.llamacpp_config.params
                .get("temperature")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32);

            let request = LlamaCppRequest {
                model: self.config.model_id.clone(),
                messages: llamacpp_messages,
                max_tokens,
                temperature,
                tools,
                stream: true,
                extra: self.llamacpp_config.params.clone(),
            };

            let url = format!("{}/v1/chat/completions", self.llamacpp_config.base_url);

            let response = match self.client
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

                if status.as_u16() == 429 {
                    yield Err(StrandsError::ModelThrottled {
                        message: "llama.cpp rate limit exceeded".into(),
                    });
                } else {
                    yield Err(StrandsError::ModelError {
                        message: format!("llama.cpp API error {}: {}", status, body),
                        source: None,
                    });
                }
                return;
            }

            yield Ok(StreamEvent::message_start(crate::types::content::Role::Assistant));

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
    fn test_llamacpp_config() {
        let config = LlamaCppConfig::new("http://localhost:8080")
            .with_model_id("my-model")
            .with_temperature(0.7);

        assert_eq!(config.base_url, "http://localhost:8080");
        assert_eq!(config.model_id, "my-model");
        assert!(config.params.contains_key("temperature"));
    }
}

