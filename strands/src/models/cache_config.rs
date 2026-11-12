//! Prompt-caching configuration shared across providers.
//!
//! Port of the Python `CacheConfig` TypedDict (`models/model.py`): providers
//! map these fields onto their native caching mechanisms. Fields a provider
//! does not implement are ignored.

use serde::{Deserialize, Serialize};

/// Which caching mechanism to drive for a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheStrategy {
    /// Provider default: cache points where the provider supports them.
    Auto,
    /// Force the Anthropic-style cache-point mechanism.
    Anthropic,
}

/// Cache lifetime values providers accept (`"5m"`, `"1h"`, or `false` to disable).
pub type CacheTtl = serde_json::Value;

/// Prompt-caching configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheConfig {
    /// Caching mechanism; `auto` uses whatever the provider supports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy: Option<CacheStrategy>,

    /// Default cache entry lifetime, or `false` to disable caching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CacheTtl>,

    /// Lifetime for system-prompt cache points, or `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt_ttl: Option<CacheTtl>,

    /// Lifetime for message cache points, or `false` to disable them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages_ttl: Option<CacheTtl>,

    /// Stable cache key for providers that route cache reads on a key
    /// (`false` disables key-based routing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<CacheTtl>,

    /// Lifetime for tool-list cache points, or `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_ttl: Option<CacheTtl>,
}

impl CacheConfig {
    /// Returns true when message-level caching is enabled.
    pub fn messages_enabled(&self) -> bool {
        !matches!(self.messages_ttl.as_ref(), Some(serde_json::Value::Bool(false)))
    }

    /// Returns true when system-prompt caching is enabled.
    pub fn system_prompt_enabled(&self) -> bool {
        !matches!(self.system_prompt_ttl.as_ref(), Some(serde_json::Value::Bool(false)))
    }

    /// Returns the configured cache-point TTL when it names a lifetime.
    pub fn ttl_value(&self) -> Option<&str> {
        match self.ttl.as_ref() {
            Some(serde_json::Value::String(ttl)) => Some(ttl.as_str()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_default_enables_everything() {
        let config = CacheConfig::default();
        assert!(config.messages_enabled());
        assert!(config.system_prompt_enabled());
        assert!(config.ttl_value().is_none());
    }

    #[test]
    fn test_disabled_via_false() {
        let config: CacheConfig = serde_json::from_value(json!({"messagesTtl": false})).unwrap();
        assert!(!config.messages_enabled());
        assert!(config.system_prompt_enabled());
    }

    #[test]
    fn test_ttl_value() {
        let config: CacheConfig = serde_json::from_value(json!({"ttl": "5m"})).unwrap();
        assert_eq!(config.ttl_value(), Some("5m"));
    }

    #[test]
    fn test_strategy_serialization() {
        let config: CacheConfig = serde_json::from_value(json!({"strategy": "anthropic"})).unwrap();
        assert_eq!(config.strategy, Some(CacheStrategy::Anthropic));
    }
}
