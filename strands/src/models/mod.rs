//! Model traits and implementations.
//!
//! All model providers are always compiled (no feature flags required).

pub mod anthropic;
pub mod bedrock;
pub mod cache_config;
pub mod model_info;
pub mod routing;
pub mod validation;
pub mod gemini;
pub mod litellm;
pub mod llamaapi;
pub mod llamacpp;
pub mod mistral;
pub mod ollama;
pub mod openai;
pub mod openai_responses;
pub mod sagemaker;

#[cfg(test)]
mod sse_tests;
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

/// Estimates input tokens as characters/4 over message and system text.
pub fn estimate_tokens(messages: &[Message], system_prompt: Option<&str>) -> u64 {
    let chars: usize = messages
        .iter()
        .map(|message| message.text_content().chars().count())
        .sum();
    let chars = chars + system_prompt.map(|p| p.chars().count()).unwrap_or(0);
    (chars as u64).div_ceil(4)
}

/// Splits complete newline-terminated lines off a stream buffer, retaining
/// any trailing partial line for the next chunk.
///
/// SSE and NDJSON parsers must use this instead of replacing or clearing the
/// buffer wholesale, or lines split across chunk boundaries are lost.
pub(crate) fn drain_complete_lines(buffer: &mut String) -> Vec<String> {
    let complete_end = buffer.rfind('\n').map(|index| index + 1).unwrap_or(0);
    if complete_end == 0 {
        return Vec::new();
    }
    let drained: String = buffer.drain(..complete_end).collect();
    drained.lines().map(str::to_string).collect()
}

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

    /// Estimates the input token count for the messages.
    ///
    /// Default: a characters/4 heuristic. Providers with a native token
    /// count API override this.
    async fn count_tokens(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
    ) -> Result<u64, StrandsError> {
        Ok(estimate_tokens(messages, system_prompt))
    }

    /// Returns the shared model info facts for the configured model id.
    fn model_info(&self) -> model_info::ModelInfo {
        model_info::lookup_model_info(&self.config().model_id)
    }

    /// Returns true when the model keeps conversation state server-side.
    fn stateful(&self) -> bool {
        self.model_info().capabilities.stateful.unwrap_or(false)
    }

    /// Returns the context window limit for the configured model id.
    fn context_window_limit(&self) -> Option<u64> {
        self.model_info().limits.context_window
    }

    /// Estimates context utilization as a fraction of the context window.
    fn estimate_utilization(&self, input_tokens: u64) -> f64 {
        let limit = self
            .context_window_limit()
            .unwrap_or(model_info::DEFAULT_CONTEXT_WINDOW_LIMIT);
        input_tokens as f64 / limit as f64
    }
}

/// Extension trait for models with additional functionality.
#[async_trait]
pub trait ModelExt: Model + Sized {
    /// Generates a structured output from the model.
    async fn structured_output<T>(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
    ) -> Result<T, StrandsError>
    where
        T: serde::de::DeserializeOwned + Send,
    {
        structured_output_from(self, messages, system_prompt).await
    }
}

impl<T: Model> ModelExt for T {}

/// Accumulates a model's text stream and parses it as structured output.
///
/// Free-function form so trait objects (`&dyn Model`) can use it too.
pub async fn structured_output_from<T>(
    model: &dyn Model,
    messages: &[Message],
    system_prompt: Option<&str>,
) -> Result<T, StrandsError>
where
    T: serde::de::DeserializeOwned,
{
    use futures::StreamExt;

    let mut content = String::new();
    let mut stream = model.stream(messages, None, system_prompt, None, None);

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

pub use anthropic::AnthropicModel;
pub use cache_config::{CacheConfig, CacheStrategy};
pub use model_info::{
    get_context_window_limit, lookup_model, lookup_model_info, ModelCapabilities, ModelInfo,
    ModelLimits, PromptCaching, DEFAULT_CONTEXT_WINDOW_LIMIT,
};
pub use bedrock::BedrockModel;
pub use gemini::{GeminiConfig, GeminiModel};
pub use litellm::{LiteLLMConfig, LiteLLMModel};
pub use llamaapi::{LlamaAPIConfig, LlamaAPIModel};
pub use llamacpp::{LlamaCppConfig, LlamaCppModel};
pub use mistral::{MistralConfig, MistralModel};
pub use ollama::OllamaModel;
pub use openai::OpenAIModel;
pub use openai_responses::OpenAIResponsesModel;
pub use routing::{ClassifierStrategy, FallbackStrategy, ModelRouter, RoutingCandidate};
pub use sagemaker::{SageMakerEndpointConfig, SageMakerModel, SageMakerPayloadConfig};

/// Alias matching the upstream Python rename to `SageMakerAIModel`.
pub type SageMakerAIModel = SageMakerModel;
pub use writer::{WriterConfig, WriterModel};
pub use validation::{
    config_keys, validate_config_keys, warn_on_tool_choice_not_supported,
};
