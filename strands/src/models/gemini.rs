//! Google Gemini model provider.
//!
//! Docs: https://ai.google.dev/api

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::models::{Model, ModelConfig, StreamEventStream};
use crate::types::content::{Message, SystemContentBlock};
use crate::types::errors::StrandsError;
use crate::types::streaming::StreamEvent;
use crate::types::tools::{ToolChoice, ToolSpec};

/// Configuration for Gemini models.
#[derive(Debug, Clone, Default)]
pub struct GeminiConfig {
    /// Gemini model ID (e.g., "gemini-2.5-flash").
    pub model_id: String,
    /// Additional model parameters (e.g., temperature).
    pub params: HashMap<String, serde_json::Value>,
    /// API key for authentication.
    pub api_key: Option<String>,
    /// Base URL for the API.
    pub base_url: Option<String>,
}

impl GeminiConfig {
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

    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.params.insert(key.into(), value);
        self
    }
}

/// Gemini API request format.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    FunctionCall { function_call: GeminiFunctionCall },
    FunctionResponse { function_response: GeminiFunctionResponse },
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Google Gemini model provider.
pub struct GeminiModel {
    config: ModelConfig,
    gemini_config: GeminiConfig,
    client: reqwest::Client,
}

impl GeminiModel {
    const DEFAULT_BASE_URL: &'static str = "https://generativelanguage.googleapis.com/v1beta";

    pub fn new(config: GeminiConfig) -> Self {
        let model_config = ModelConfig::new(&config.model_id);

        Self {
            config: model_config,
            gemini_config: config,
            client: reqwest::Client::new(),
        }
    }

    fn base_url(&self) -> &str {
        self.gemini_config
            .base_url
            .as_deref()
            .unwrap_or(Self::DEFAULT_BASE_URL)
    }

    fn api_key(&self) -> Result<&str, StrandsError> {
        self.gemini_config
            .api_key
            .as_deref()
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok().as_deref().map(|_| ""))
            .ok_or_else(|| StrandsError::ConfigurationError {
                message: "Gemini API key not configured. Set GOOGLE_API_KEY or provide api_key".into(),
            })
    }

    fn convert_messages(&self, messages: &[Message]) -> Vec<GeminiContent> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    crate::types::content::Role::User => "user",
                    crate::types::content::Role::Assistant => "model",
                };

                let parts: Vec<GeminiPart> = msg
                    .content
                    .iter()
                    .filter_map(|block| {
                        if let Some(text) = &block.text {
                            Some(GeminiPart::Text { text: text.clone() })
                        } else if let Some(tool_use) = &block.tool_use {
                            Some(GeminiPart::FunctionCall {
                                function_call: GeminiFunctionCall {
                                    name: tool_use.name.clone(),
                                    args: tool_use.input.clone(),
                                },
                            })
                        } else if let Some(tool_result) = &block.tool_result {
                            Some(GeminiPart::FunctionResponse {
                                function_response: GeminiFunctionResponse {
                                    name: tool_result.tool_use_id.clone(),
                                    response: serde_json::json!({
                                        "content": tool_result.content
                                    }),
                                },
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                GeminiContent {
                    role: role.to_string(),
                    parts,
                }
            })
            .collect()
    }

    fn convert_tools(&self, tool_specs: &[ToolSpec]) -> Vec<GeminiTool> {
        let declarations: Vec<GeminiFunctionDeclaration> = tool_specs
            .iter()
            .map(|spec| GeminiFunctionDeclaration {
                name: spec.name.clone(),
                description: spec.description.clone(),
                parameters: serde_json::to_value(&spec.input_schema).unwrap_or_default(),
            })
            .collect();

        vec![GeminiTool {
            function_declarations: declarations,
        }]
    }
}

#[async_trait]
impl Model for GeminiModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn update_config(&mut self, config: ModelConfig) {
        self.config = config;
    }

    /// Counts input tokens through the native Gemini `countTokens` API.
    async fn count_tokens(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
    ) -> Result<u64, StrandsError> {
        let api_key = self.api_key()?;
        let api_key = if api_key.is_empty() {
            std::env::var("GOOGLE_API_KEY").map_err(|_| StrandsError::ConfigurationError {
                message: "Gemini API key not configured. Set GOOGLE_API_KEY or provide api_key".into(),
            })?
        } else {
            api_key.to_string()
        };

        let mut contents = self.convert_messages(messages);
        if let Some(prompt) = system_prompt {
            contents.insert(
                0,
                GeminiContent {
                    role: "user".to_string(),
                    parts: vec![GeminiPart::Text { text: prompt.to_string() }],
                },
            );
        }

        let url = format!("{}/models/{}:countTokens", self.base_url(), self.config.model_id);
        let response = self
            .client
            .post(&url)
            .query(&[("key", api_key.as_str())])
            .json(&serde_json::json!({ "contents": contents }))
            .send()
            .await
            .map_err(|e| StrandsError::ProviderTokenCountError {
                message: format!("gemini countTokens request failed: {}", e),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(StrandsError::ProviderTokenCountError {
                message: format!("gemini countTokens failed: HTTP {status}: {body}"),
            });
        }

        let payload: serde_json::Value = response
            .json()
            .await
            .map_err(|e| StrandsError::ProviderTokenCountError {
                message: format!("failed to parse countTokens response: {}", e),
            })?;

        payload
            .get("totalCount")
            .and_then(|t| t.as_u64())
            .map(|tokens| tokens)
            .ok_or_else(|| StrandsError::ProviderTokenCountError {
                message: "countTokens response missing totalCount".to_string(),
            })
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
                Ok(key) => key.to_string(),
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let api_key = if api_key.is_empty() {
                match std::env::var("GOOGLE_API_KEY") {
                    Ok(key) => key,
                    Err(_) => {
                        yield Err(StrandsError::ConfigurationError {
                            message: "GOOGLE_API_KEY not set".into(),
                        });
                        return;
                    }
                }
            } else {
                api_key
            };

            let contents = self.convert_messages(&messages);

            let system_instruction = system_prompt.map(|prompt| GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart::Text { text: prompt }],
            });

            let tools = tool_specs.as_ref().map(|specs| self.convert_tools(specs));

            let request = GeminiRequest {
                contents,
                system_instruction,
                tools,
                generation_config: if self.gemini_config.params.is_empty() {
                    None
                } else {
                    Some(serde_json::to_value(&self.gemini_config.params).unwrap_or_default())
                },
            };

            let url = format!(
                "{}/models/{}:streamGenerateContent?key={}&alt=sse",
                self.base_url(),
                self.config.model_id,
                api_key
            );

            let response = match self.client
                .post(&url)
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
                        message: "Gemini rate limit exceeded".into(),
                    });
                } else {
                    yield Err(StrandsError::ModelError {
                        message: format!("Gemini API error {}: {}", status, body),
                        source: None,
                    });
                }
                return;
            }

            yield Ok(StreamEvent::message_start(crate::types::content::Role::Assistant));
            yield Ok(StreamEvent::content_block_start(0, None));

            let body = match response.text().await {
                Ok(b) => b,
                Err(e) => {
                    yield Err(StrandsError::NetworkError(e.to_string()));
                    return;
                }
            };

            let mut tool_used = false;
            let mut finish_reason = "STOP";
            let mut input_tokens = 0u64;
            let mut output_tokens = 0u64;

            for line in body.lines() {
                let line = line.trim();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if data.trim() == "[DONE]" {
                        continue;
                    }

                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(usage) = parsed.get("usageMetadata") {
                            if let Some(prompt_tokens) = usage.get("promptTokenCount").and_then(|v| v.as_u64()) {
                                input_tokens = prompt_tokens;
                            }
                            if let Some(candidates_tokens) = usage.get("candidatesTokenCount").and_then(|v| v.as_u64()) {
                                output_tokens = candidates_tokens;
                            }
                        }

                        if let Some(candidates) = parsed.get("candidates").and_then(|c| c.as_array()) {
                            for candidate in candidates {
                                if let Some(reason) = candidate.get("finishReason").and_then(|r| r.as_str()) {
                                    finish_reason = match reason {
                                        "MAX_TOKENS" => "MAX_TOKENS",
                                        "SAFETY" => "SAFETY",
                                        "STOP" | _ => "STOP",
                                    };
                                }

                                if let Some(content) = candidate.get("content") {
                                    if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
                                        for part in parts {
                                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                                let is_thought = part.get("thought").and_then(|t| t.as_bool()).unwrap_or(false);
                                                if is_thought {
                                                    yield Ok(StreamEvent::reasoning_delta(0, text));
                                                } else {
                                                    yield Ok(StreamEvent::text_delta(0, text));
                                                }
                                            }

                                            if let Some(function_call) = part.get("functionCall") {
                                                if let (Some(name), Some(args)) = (
                                                    function_call.get("name").and_then(|n| n.as_str()),
                                                    function_call.get("args"),
                                                ) {
                                                    tool_used = true;
                                                    yield Ok(StreamEvent::tool_use_start(
                                                        1,
                                                        name,
                                                        name,
                                                    ));
                                                    yield Ok(StreamEvent::tool_use_delta(
                                                        1,
                                                        &serde_json::to_string(args).unwrap_or_default(),
                                                    ));
                                                    yield Ok(StreamEvent::content_block_stop(1));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            yield Ok(StreamEvent::content_block_stop(0));

            let stop_reason = if tool_used {
                crate::types::streaming::StopReason::ToolUse
            } else {
                match finish_reason {
                    "MAX_TOKENS" => crate::types::streaming::StopReason::MaxTokens,
                    _ => crate::types::streaming::StopReason::EndTurn,
                }
            };

            yield Ok(StreamEvent::message_stop(stop_reason));

            yield Ok(StreamEvent::metadata(
                crate::types::streaming::Usage::new(input_tokens as u32, output_tokens as u32),
                crate::types::streaming::Metrics::default(),
            ));
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_config() {
        let config = GeminiConfig::new("gemini-2.5-flash")
            .with_api_key("test-key")
            .with_param("temperature", serde_json::json!(0.7));

        assert_eq!(config.model_id, "gemini-2.5-flash");
        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert!(config.params.contains_key("temperature"));
    }
}

