//! Writer model provider.
//!
//! This provider integrates with Writer's Palmyra models.
//! See: https://dev.writer.com/home/introduction

use std::collections::HashMap;

use crate::types::content::{Message, SystemContentBlock};
use crate::types::errors::StrandsError;
use crate::types::tools::{ToolChoice, ToolSpec};

use super::{Model, ModelConfig, StreamEventStream};

/// Configuration for Writer models.
#[derive(Debug, Clone, Default)]
pub struct WriterConfig {
    /// Model name to use (e.g., palmyra-x5, palmyra-x4).
    pub model_id: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u32>,
    /// Stop sequences.
    pub stop: Option<Vec<String>>,
    /// Additional options for streaming.
    pub stream_options: Option<HashMap<String, serde_json::Value>>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Top-p (nucleus sampling).
    pub top_p: Option<f64>,
    /// API key for authentication.
    pub api_key: Option<String>,
    /// Base URL for the API.
    pub base_url: Option<String>,
}

impl WriterConfig {
    /// Create a new Writer config.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
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

    /// Set stop sequences.
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Set API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }
}

/// Writer model provider implementation.
pub struct WriterModel {
    config: ModelConfig,
    writer_config: WriterConfig,
}

impl WriterModel {
    /// Create a new Writer model.
    pub fn new(config: WriterConfig) -> Self {
        Self {
            config: ModelConfig::new(&config.model_id),
            writer_config: config,
        }
    }

    /// Get the Writer configuration.
    pub fn writer_config(&self) -> &WriterConfig {
        &self.writer_config
    }

    /// Update the Writer configuration.
    pub fn update_writer_config(&mut self, config: WriterConfig) {
        self.config = ModelConfig::new(&config.model_id);
        self.writer_config = config;
    }

    /// Check if this is a Palmyra X5 model (supports vision).
    pub fn is_palmyra_x5(&self) -> bool {
        self.writer_config.model_id == "palmyra-x5"
    }
}

impl Model for WriterModel {
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
                message: "Writer integration requires HTTP client implementation".into(),
                source: None,
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_config() {
        let config = WriterConfig::new("palmyra-x5")
            .with_temperature(0.7)
            .with_max_tokens(1000);
        
        assert_eq!(config.model_id, "palmyra-x5");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(1000));
    }

    #[test]
    fn test_writer_model_creation() {
        let config = WriterConfig::new("palmyra-x4");
        let model = WriterModel::new(config);
        
        assert_eq!(model.config().model_id, "palmyra-x4");
        assert!(!model.is_palmyra_x5());
    }

    #[test]
    fn test_palmyra_x5_detection() {
        let config = WriterConfig::new("palmyra-x5");
        let model = WriterModel::new(config);
        
        assert!(model.is_palmyra_x5());
    }
}
