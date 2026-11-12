//! Configuration validation utilities for model providers.

use std::collections::HashSet;

use tracing::warn;

use crate::types::tools::ToolChoice;

/// Validates that config keys are valid for the expected configuration type.
///
/// This function checks if all provided keys are valid configuration parameters
/// and emits warnings for any invalid keys found.
///
/// # Arguments
///
/// * `provided_keys` - The keys provided in the configuration
/// * `valid_keys` - The set of valid configuration keys
/// * `config_name` - The name of the configuration type (for error messages)
///
/// # Example
///
/// ```ignore
/// let provided = HashSet::from(["model_id", "invalid_key"]);
/// let valid = HashSet::from(["model_id", "max_tokens", "temperature"]);
/// validate_config_keys(&provided, &valid, "BedrockConfig");
/// ```
pub fn validate_config_keys(
    provided_keys: &HashSet<&str>,
    valid_keys: &HashSet<&str>,
    config_name: &str,
) {
    let invalid_keys: Vec<&&str> = provided_keys.difference(valid_keys).collect();

    if !invalid_keys.is_empty() {
        let mut invalid_sorted: Vec<_> = invalid_keys.iter().map(|k| **k).collect();
        invalid_sorted.sort();

        let mut valid_sorted: Vec<_> = valid_keys.iter().copied().collect();
        valid_sorted.sort();

        warn!(
            config = config_name,
            invalid_keys = ?invalid_sorted,
            valid_keys = ?valid_sorted,
            "Invalid configuration parameters provided"
        );
    }
}

/// Emits a warning if a tool choice is provided but not supported by the provider.
///
/// Some model providers don't support tool choice configuration. This function
/// should be called by those providers to warn users that their tool choice
/// will be ignored.
///
/// # Arguments
///
/// * `tool_choice` - The tool choice option that was provided
/// * `provider_name` - The name of the model provider (for the warning message)
pub fn warn_on_tool_choice_not_supported(tool_choice: Option<&ToolChoice>, provider_name: &str) {
    if tool_choice.is_some() {
        warn!(
            provider = provider_name,
            "A ToolChoice was provided to this provider but is not supported and will be ignored"
        );
    }
}

/// Common valid configuration keys for model providers.
    pub mod config_keys {
    use std::collections::HashSet;

    /// Base configuration keys common to most providers.
    pub fn base_config_keys() -> HashSet<&'static str> {
        HashSet::from([
            "model_id",
            "max_tokens",
            "temperature",
            "top_p",
            "stop_sequences",
        ])
    }

    /// Bedrock-specific configuration keys.
    pub fn bedrock_config_keys() -> HashSet<&'static str> {
        let mut keys = base_config_keys();
        keys.extend([
            "guardrail_id",
            "guardrail_version",
            "guardrail_trace",
            "cache_prompt",
            "cache_tools",
            "cache_config",
            "service_tier",
            "include_tool_result_status",
            "streaming",
            "additional_request_fields",
            "additional_response_field_paths",
        ]);
        keys
    }

    /// OpenAI-specific configuration keys.
    pub fn openai_config_keys() -> HashSet<&'static str> {
        let mut keys = base_config_keys();
        keys.extend([
            "api_key",
            "base_url",
            "organization",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "response_format",
            "cache_config",
            "bedrock_mantle_config",
        ]);
        keys
    }

    /// Anthropic-specific configuration keys.
    pub fn anthropic_config_keys() -> HashSet<&'static str> {
        let mut keys = base_config_keys();
        keys.extend([
            "api_key",
            "base_url",
        ]);
        keys
    }

    /// Ollama-specific configuration keys.
    pub fn ollama_config_keys() -> HashSet<&'static str> {
        let mut keys = base_config_keys();
        keys.extend([
            "host",
            "timeout",
            "keep_alive",
            "additional_args",
            "cache_config",
        ]);
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_config_keys_no_invalid() {
        let provided = HashSet::from(["model_id", "max_tokens"]);
        let valid = HashSet::from(["model_id", "max_tokens", "temperature"]);
        
        validate_config_keys(&provided, &valid, "TestConfig");
    }

    #[test]
    fn test_validate_config_keys_with_invalid() {
        let provided = HashSet::from(["model_id", "invalid_key", "another_invalid"]);
        let valid = HashSet::from(["model_id", "max_tokens"]);
        
        validate_config_keys(&provided, &valid, "TestConfig");
    }

    #[test]
    fn test_warn_on_tool_choice_not_supported_none() {
        warn_on_tool_choice_not_supported(None, "TestProvider");
    }

    #[test]
    fn test_warn_on_tool_choice_not_supported_some() {
        let tool_choice = ToolChoice::auto();
        warn_on_tool_choice_not_supported(Some(&tool_choice), "TestProvider");
    }

    #[test]
    fn test_bedrock_config_keys() {
        let keys = config_keys::bedrock_config_keys();
        assert!(keys.contains("model_id"));
        assert!(keys.contains("guardrail_id"));
        assert!(keys.contains("cache_prompt"));
        assert!(keys.contains("cache_config"));
        assert!(keys.contains("service_tier"));
    }
}

