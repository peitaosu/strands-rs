//! Error types for the Strands SDK.

use thiserror::Error;

/// The main error type for Strands SDK operations.
#[derive(Error, Debug)]
pub enum StrandsError {
    /// An error occurred during model invocation.
    #[error("Model error: {message}")]
    ModelError {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// The model is being throttled due to rate limits.
    #[error("Model throttled: {message}")]
    ModelThrottled { message: String },

    /// An error occurred during tool execution.
    #[error("Tool execution error in '{tool_name}': {message}")]
    ToolError { tool_name: String, message: String },

    /// The requested tool was not found in the registry.
    #[error("Tool not found: {tool_name}")]
    ToolNotFound { tool_name: String },

    /// Tool validation failed.
    #[error("Tool validation error: {message}")]
    ToolValidationError { message: String },

    /// Invalid tool name.
    #[error("Invalid tool name '{name}': {reason}")]
    InvalidToolName { name: String, reason: String },

    /// Duplicate tool name in registry.
    #[error("Duplicate tool name: {name}")]
    DuplicateToolName { name: String },

    /// Invalid tool use - tool not found in registry.
    #[error("Invalid tool use: '{name}' not found. Available tools: {available_tools:?}")]
    InvalidToolUseName { name: String, available_tools: Vec<String> },

    /// The context window has been exceeded.
    #[error("Context window overflow: {message}")]
    ContextWindowOverflow { message: String },

    /// The model reached the maximum token limit.
    #[error("Max tokens reached")]
    MaxTokensReached,

    /// Content was filtered by safety systems.
    #[error("Content filtered: {message}")]
    ContentFiltered { message: String },

    /// A guardrail intervention occurred.
    #[error("Guardrail intervention: {message}")]
    GuardrailIntervention { message: String },

    /// An error occurred in the event loop.
    #[error("Event loop error: {message}")]
    EventLoopError { message: String },

    /// A session-related error occurred.
    #[error("Session error: {message}")]
    SessionError { message: String },

    /// JSON serialization/deserialization error.
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Configuration error.
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    /// Invalid input was provided.
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    /// An internal error occurred.
    #[error("Internal error: {message}")]
    InternalError { message: String },

    /// Structured output parsing failed.
    #[error("Structured output error: {message}")]
    StructuredOutputError { message: String },

    /// The agent was interrupted.
    #[error("Agent interrupted: {message}")]
    Interrupted { message: String },

    /// An error occurred in multi-agent orchestration.
    #[error("Multi-agent error: {message}")]
    MultiAgentError { message: String },

    /// A network error occurred.
    #[error("Network error: {0}")]
    NetworkError(String),

    /// An AWS-specific error occurred.
    #[error("AWS error: {0}")]
    AwsError(String),

    /// MCP client initialization error.
    #[error("MCP client initialization error: {message}")]
    MCPClientInitializationError { message: String },

    /// Tool provider error.
    #[error("Tool provider error: {message}")]
    ToolProviderError { message: String },

    /// Feature not yet implemented.
    #[error("Not implemented: {feature}")]
    NotImplemented { feature: String },
}

impl StrandsError {
    pub fn model_error(message: impl Into<String>) -> Self {
        Self::ModelError { message: message.into(), source: None }
    }

    pub fn model_error_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::ModelError { message: message.into(), source: Some(Box::new(source)) }
    }

    pub fn tool_error(tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ToolError { tool_name: tool_name.into(), message: message.into() }
    }

    pub fn config_error(message: impl Into<String>) -> Self {
        Self::ConfigurationError { message: message.into() }
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput { message: message.into() }
    }

    /// Returns true if the error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::ModelThrottled { .. } | Self::NetworkError(_))
    }
}

/// A type alias for Results using StrandsError.
pub type Result<T> = std::result::Result<T, StrandsError>;
