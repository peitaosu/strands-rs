//! LiteLLM model provider.
//!
//! LiteLLM provides a unified interface to multiple LLM providers.
//! See: https://docs.litellm.ai/

use crate::types::content::{Message, SystemContentBlock};
use crate::types::errors::StrandsError;
use crate::types::tools::{ToolChoice, ToolSpec};

use super::{Model, ModelConfig, StreamEventStream};

/// Configuration for LiteLLM models.
#[derive(Debug, Clone, Default)]
pub struct LiteLLMConfig {
    /// Model ID (e.g., "openai/gpt-4o", "anthropic/claude-3-sonnet").
    pub model_id: String,
    /// Model parameters (e.g., max_tokens, temperature).
    pub params: Option<serde_json::Value>,
    /// Client arguments for the LiteLLM client.
    pub client_args: Option<serde_json::Value>,
}

impl LiteLLMConfig {
    /// Create a new LiteLLM config.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            params: None,
            client_args: None,
        }
    }

    /// Set model parameters.
    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params);
        self
    }

    /// Set client arguments.
    pub fn with_client_args(mut self, client_args: serde_json::Value) -> Self {
        self.client_args = Some(client_args);
        self
    }
}

/// LiteLLM model provider implementation.
///
/// This provides a unified interface to multiple LLM providers through LiteLLM.
pub struct LiteLLMModel {
    config: ModelConfig,
    litellm_config: LiteLLMConfig,
}

impl LiteLLMModel {
    /// Create a new LiteLLM model.
    pub fn new(config: LiteLLMConfig) -> Self {
        Self {
            config: ModelConfig::new(&config.model_id),
            litellm_config: config,
        }
    }

    /// Get the LiteLLM configuration.
    pub fn litellm_config(&self) -> &LiteLLMConfig {
        &self.litellm_config
    }

    /// Update the LiteLLM configuration.
    pub fn update_litellm_config(&mut self, config: LiteLLMConfig) {
        self.config = ModelConfig::new(&config.model_id);
        self.litellm_config = config;
    }
}

impl Model for LiteLLMModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn update_config(&mut self, config: ModelConfig) {
        self.config = config;
    }

    fn stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _tool_specs: Option<&'a [ToolSpec]>,
        _system_prompt: Option<&'a str>,
        _tool_choice: Option<ToolChoice>,
        _system_prompt_content: Option<&'a [SystemContentBlock]>,
    ) -> StreamEventStream<'a> {
        Box::pin(futures::stream::once(async {
            Err(StrandsError::ModelError {
                message: "LiteLLM integration requires litellm-rs or HTTP client implementation".into(),
                source: None,
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_litellm_config() {
        let config = LiteLLMConfig::new("openai/gpt-4o")
            .with_params(serde_json::json!({"max_tokens": 1000}));
        
        assert_eq!(config.model_id, "openai/gpt-4o");
        assert!(config.params.is_some());
    }

    #[test]
    fn test_litellm_model_creation() {
        let config = LiteLLMConfig::new("anthropic/claude-3-sonnet");
        let model = LiteLLMModel::new(config);
        
        assert_eq!(model.config().model_id, "anthropic/claude-3-sonnet");
    }
}
