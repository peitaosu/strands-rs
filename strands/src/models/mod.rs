//! Model traits and implementations.
//!
//! All model providers are always compiled (no feature flags required).

pub mod anthropic;
pub mod bedrock;
pub mod validation;
pub mod gemini;
pub mod litellm;
pub mod llamaapi;
pub mod llamacpp;
pub mod mistral;
pub mod ollama;
pub mod openai;
pub mod sagemaker;
pub mod writer;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::types::{
    content::{Message, SystemContentBlock},
    errors::StrandsError,
    streaming::StreamEvent,
    tools::{ToolChoice, ToolSpec},
};

/// Configuration for a model.
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    pub model_id: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub additional: std::collections::HashMap<String, serde_json::Value>,
}

impl ModelConfig {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self { model_id: model_id.into(), ..Default::default() }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
}

/// A stream of model response events.
pub type StreamEventStream<'a> = Pin<Box<dyn Stream<Item = Result<StreamEvent, StrandsError>> + Send + 'a>>;

/// Trait for model implementations.
#[async_trait]
pub trait Model: Send + Sync {
    /// Returns the model configuration.
    fn config(&self) -> &ModelConfig;

    /// Updates the model configuration.
    fn update_config(&mut self, config: ModelConfig);

    /// Streams a response from the model.
    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        tool_specs: Option<&'a [ToolSpec]>,
        system_prompt: Option<&'a str>,
        tool_choice: Option<ToolChoice>,
        system_prompt_content: Option<&'a [SystemContentBlock]>,
    ) -> StreamEventStream<'a>;
}

/// Extension trait for models with additional functionality.
#[async_trait]
pub trait ModelExt: Model {
    /// Generates a structured output from the model.
    async fn structured_output<T>(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
    ) -> Result<T, StrandsError>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema + Send,
    {
        use futures::StreamExt;

        let mut content = String::new();
        let mut stream = self.stream(messages, None, system_prompt, None, None);

        while let Some(event) = stream.next().await {
            let event = event?;
            if let Some(text) = event.as_text_delta() {
                content.push_str(text);
            }
        }

        serde_json::from_str(&content).map_err(|e| StrandsError::StructuredOutputError {
            message: format!("Failed to parse structured output: {e}"),
        })
    }
}

impl<T: Model> ModelExt for T {}

pub use anthropic::AnthropicModel;
pub use bedrock::BedrockModel;
pub use gemini::{GeminiConfig, GeminiModel};
pub use litellm::{LiteLLMConfig, LiteLLMModel};
pub use llamaapi::{LlamaAPIConfig, LlamaAPIModel};
pub use llamacpp::{LlamaCppConfig, LlamaCppModel};
pub use mistral::{MistralConfig, MistralModel};
pub use ollama::OllamaModel;
pub use openai::OpenAIModel;
pub use sagemaker::{SageMakerEndpointConfig, SageMakerModel, SageMakerPayloadConfig};
pub use writer::{WriterConfig, WriterModel};
pub use validation::{
    config_keys, validate_config_keys, warn_on_tool_choice_not_supported,
};
