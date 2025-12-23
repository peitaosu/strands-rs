//! Llama API model provider.
//!
//! This provider integrates with Meta's Llama API.
//! See: https://llama.developer.meta.com/

use crate::types::content::{Message, SystemContentBlock};
use crate::types::errors::StrandsError;
use crate::types::tools::{ToolChoice, ToolSpec};

use super::{Model, ModelConfig, StreamEventStream};

/// Configuration for Llama API models.
#[derive(Debug, Clone, Default)]
pub struct LlamaAPIConfig {
    /// Model ID (e.g., "Llama-4-Maverick-17B-128E-Instruct-FP8").
    pub model_id: String,
    /// Repetition penalty.
    pub repetition_penalty: Option<f64>,
    /// Temperature for sampling.
    pub temperature: Option<f64>,
    /// Top-p for nucleus sampling.
    pub top_p: Option<f64>,
    /// Maximum completion tokens.
    pub max_completion_tokens: Option<u32>,
    /// Top-k for sampling.
    pub top_k: Option<u32>,
    /// API key for authentication.
    pub api_key: Option<String>,
}

impl LlamaAPIConfig {
    /// Create a new Llama API config.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top-p.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set max completion tokens.
    pub fn with_max_completion_tokens(mut self, max_tokens: u32) -> Self {
        self.max_completion_tokens = Some(max_tokens);
        self
    }

    /// Set repetition penalty.
    pub fn with_repetition_penalty(mut self, penalty: f64) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }

    /// Set top-k.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }
}

/// Llama API model provider implementation.
pub struct LlamaAPIModel {
    config: ModelConfig,
    llamaapi_config: LlamaAPIConfig,
}

impl LlamaAPIModel {
    /// Create a new Llama API model.
    pub fn new(config: LlamaAPIConfig) -> Self {
        Self {
            config: ModelConfig::new(&config.model_id),
            llamaapi_config: config,
        }
    }

    /// Get the Llama API configuration.
    pub fn llamaapi_config(&self) -> &LlamaAPIConfig {
        &self.llamaapi_config
    }

    /// Update the Llama API configuration.
    pub fn update_llamaapi_config(&mut self, config: LlamaAPIConfig) {
        self.config = ModelConfig::new(&config.model_id);
        self.llamaapi_config = config;
    }
}

impl Model for LlamaAPIModel {
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
                message: "Llama API integration requires HTTP client implementation".into(),
                source: None,
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llamaapi_config() {
        let config = LlamaAPIConfig::new("Llama-4-Maverick-17B-128E-Instruct-FP8")
            .with_temperature(0.7)
            .with_max_completion_tokens(1000);
        
        assert_eq!(config.model_id, "Llama-4-Maverick-17B-128E-Instruct-FP8");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_completion_tokens, Some(1000));
    }

    #[test]
    fn test_llamaapi_model_creation() {
        let config = LlamaAPIConfig::new("test-model");
        let model = LlamaAPIModel::new(config);
        
        assert_eq!(model.config().model_id, "test-model");
    }
}
