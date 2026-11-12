//! Shared model info registry.
//!
//! Mirrors the Python `strands.models._model_info` module: capability and
//! limit facts per model id, loaded from the shared `model-info.json` data
//! that every SDK copies from `shared/model-info.json`.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

/// Context window limit assumed for a model whose window is unknown.
pub const DEFAULT_CONTEXT_WINDOW_LIMIT: u64 = 200_000;

/// How the model caches prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCaching {
    Unsupported,
    Auto,
    CachePoints,
}

/// What the model can do.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// Whether the model keeps conversation state server-side.
    pub stateful: Option<bool>,
    /// Whether and how the model caches prompts.
    pub prompt_caching: Option<PromptCaching>,
    /// Whether the model accepts a tool choice that forces a tool call.
    pub forced_tool_choice: Option<bool>,
    /// Whether the provider can make text output conform to a JSON schema.
    pub native_structured_output: Option<bool>,
    /// Whether the model summarizes history server-side.
    pub server_side_summary: Option<bool>,
}

/// Size limits of the model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelLimits {
    /// Context window size in tokens.
    pub context_window: Option<u64>,
    /// Largest output token count the model accepts.
    pub max_output_tokens: Option<u64>,
}

/// Everything the SDK knows about a model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelInfo {
    pub capabilities: ModelCapabilities,
    pub limits: ModelLimits,
}

impl ModelInfo {
    /// Merges partial records, later entries winning. Unknown fields inherit.
    pub fn merge(parts: &[Option<&ModelInfo>]) -> ModelInfo {
        let mut merged = ModelInfo::default();
        for part in parts.iter().flatten() {
            merged.capabilities.stateful = part.capabilities.stateful.or(merged.capabilities.stateful);
            merged.capabilities.prompt_caching = part
                .capabilities
                .prompt_caching
                .or(merged.capabilities.prompt_caching);
            merged.capabilities.forced_tool_choice = part
                .capabilities
                .forced_tool_choice
                .or(merged.capabilities.forced_tool_choice);
            merged.capabilities.native_structured_output = part
                .capabilities
                .native_structured_output
                .or(merged.capabilities.native_structured_output);
            merged.capabilities.server_side_summary = part
                .capabilities
                .server_side_summary
                .or(merged.capabilities.server_side_summary);
            merged.limits.context_window = part.limits.context_window.or(merged.limits.context_window);
            merged.limits.max_output_tokens = part
                .limits
                .max_output_tokens
                .or(merged.limits.max_output_tokens);
        }
        merged
    }
}

#[derive(Debug, Deserialize)]
struct RawSection {
    #[serde(default)]
    #[serde(rename = "stateful")]
    stateful: Option<bool>,
    #[serde(default)]
    #[serde(rename = "promptCaching")]
    prompt_caching: Option<String>,
    #[serde(default)]
    #[serde(rename = "forcedToolChoice")]
    forced_tool_choice: Option<bool>,
    #[serde(default)]
    #[serde(rename = "nativeStructuredOutput")]
    native_structured_output: Option<bool>,
    #[serde(default)]
    #[serde(rename = "serverSideSummary")]
    server_side_summary: Option<bool>,
    #[serde(default)]
    #[serde(rename = "contextWindow")]
    context_window: Option<u64>,
    #[serde(default)]
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RawModelEntry {
    #[serde(default)]
    capabilities: Option<RawSection>,
    #[serde(default)]
    limits: Option<RawSection>,
}

#[derive(Debug, Deserialize)]
struct RawData {
    models: HashMap<String, RawModelEntry>,
}

fn parse_section(raw: Option<&RawSection>) -> (ModelCapabilities, ModelLimits) {
    let Some(section) = raw else {
        return (ModelCapabilities::default(), ModelLimits::default());
    };
    let capabilities = ModelCapabilities {
        stateful: section.stateful,
        prompt_caching: section.prompt_caching.as_deref().and_then(|value| {
            match value {
                "unsupported" => Some(PromptCaching::Unsupported),
                "auto" => Some(PromptCaching::Auto),
                "cachePoints" => Some(PromptCaching::CachePoints),
                _ => None,
            }
        }),
        forced_tool_choice: section.forced_tool_choice,
        native_structured_output: section.native_structured_output,
        server_side_summary: section.server_side_summary,
    };
    let limits = ModelLimits {
        context_window: section.context_window,
        max_output_tokens: section.max_output_tokens,
    };
    (capabilities, limits)
}

fn registry() -> &'static HashMap<String, ModelInfo> {
    static REGISTRY: OnceLock<HashMap<String, ModelInfo>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let raw = include_str!("model_info.json");
        let data: RawData = serde_json::from_str(raw).unwrap_or(RawData {
            models: HashMap::new(),
        });
        data.models
            .into_iter()
            .map(|(id, entry)| {
                let (capabilities, limits) =
                    (parse_section(entry.capabilities.as_ref()), parse_section(entry.limits.as_ref()));
                (
                    id,
                    ModelInfo {
                        capabilities: capabilities.0,
                        limits: limits.1,
                    },
                )
            })
            .collect()
    })
}

/// Find the entry for a model id.
///
/// Tries the id as given, then without a trailing Bedrock version suffix,
/// then without a trailing date, then with each leading dot-separated
/// segment removed in turn.
pub fn lookup_model(model_id: &str) -> Option<&'static ModelInfo> {
    let registry = registry();
    let mut candidate = model_id.to_string();
    loop {
        let unversioned = strip_version_suffix(&candidate);
        for key in [
            candidate.as_str(),
            unversioned.as_str(),
            strip_date_suffix(&unversioned),
        ] {
            if let Some(info) = registry.get(key) {
                return Some(info);
            }
        }
        let dot_index = match candidate.find('.') {
            Some(index) => index,
            None => return None,
        };
        candidate = candidate[dot_index + 1..].to_string();
    }
}

/// Strips a trailing Bedrock version suffix (`-v1:0`, `-1:0`).
fn strip_version_suffix(candidate: &str) -> String {
    if !candidate.is_ascii() {
        return candidate.to_string();
    }
    let bytes = candidate.as_bytes();
    let mut end = bytes.len();
    // Longest match first: `-v\d+:\d+`, then `-v\d+`, then `-\d+:\d+`.
    for pattern in [strip_v_colon, strip_v_digits, strip_digits_colon] {
        if let Some(stripped_end) = pattern(candidate, bytes) {
            end = stripped_end;
            break;
        }
    }
    candidate[..end].to_string()
}

fn strip_v_colon(_candidate: &str, bytes: &[u8]) -> Option<usize> {
    // `-v<digits>:<digits>`
    let mut i = bytes.len();
    let mut digits_after_colon = 0;
    while i > 0 && bytes[i - 1].is_ascii_digit() {
        i -= 1;
        digits_after_colon += 1;
    }
    if digits_after_colon == 0 || i == 0 || bytes[i - 1] != b':' {
        return None;
    }
    i -= 1;
    let mut digits_before_colon = 0;
    while i > 0 && bytes[i - 1].is_ascii_digit() {
        i -= 1;
        digits_before_colon += 1;
    }
    if digits_before_colon == 0 || i < 2 || bytes[i - 1] != b'v' || bytes[i - 2] != b'-' {
        return None;
    }
    Some(i - 2)
}

fn strip_v_digits(_candidate: &str, bytes: &[u8]) -> Option<usize> {
    // `-v<digits>`
    let mut i = bytes.len();
    let mut digits = 0;
    while i > 0 && bytes[i - 1].is_ascii_digit() {
        i -= 1;
        digits += 1;
    }
    if digits == 0 || i < 2 || bytes[i - 1] != b'v' || bytes[i - 2] != b'-' {
        return None;
    }
    Some(i - 2)
}

fn strip_digits_colon(_candidate: &str, bytes: &[u8]) -> Option<usize> {
    // `-<digits>:<digits>`
    let mut i = bytes.len();
    let mut digits_after_colon = 0;
    while i > 0 && bytes[i - 1].is_ascii_digit() {
        i -= 1;
        digits_after_colon += 1;
    }
    if digits_after_colon == 0 || i == 0 || bytes[i - 1] != b':' {
        return None;
    }
    i -= 1;
    let mut digits_before_colon = 0;
    while i > 0 && bytes[i - 1].is_ascii_digit() {
        i -= 1;
        digits_before_colon += 1;
    }
    if digits_before_colon == 0 || i == 0 || bytes[i - 1] != b'-' {
        return None;
    }
    Some(i - 1)
}

/// Strips a trailing date snapshot suffix (`-20260401`, `@20251001`).
fn strip_date_suffix(candidate: &str) -> &str {
    if !candidate.is_ascii() {
        return candidate;
    }
    let bytes = candidate.as_bytes();
    if bytes.len() < 9 {
        return candidate;
    }
    let tail = &bytes[bytes.len() - 8..];
    let separator = bytes[bytes.len() - 9];
    let is_digits = tail.iter().all(|b| b.is_ascii_digit());
    if is_digits && (separator == b'-' || separator == b'@') {
        &candidate[..candidate.len() - 9]
    } else {
        candidate
    }
}

/// What the shared data knows about a model id (empty when nothing).
pub fn lookup_model_info(model_id: &str) -> ModelInfo {
    ModelInfo::merge(&[lookup_model(model_id)])
}

/// The context window limit for a model id, if known.
pub fn get_context_window_limit(model_id: &str) -> Option<u64> {
    lookup_model(model_id).and_then(|info| info.limits.context_window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_context_window() {
        assert_eq!(DEFAULT_CONTEXT_WINDOW_LIMIT, 200_000);
    }

    #[test]
    fn test_lookup_direct() {
        let info = lookup_model("claude-sonnet-4-6").expect("claude-sonnet-4-6 should be known");
        assert_eq!(info.limits.context_window, Some(1_000_000));
    }

    #[test]
    fn test_lookup_bedrock_style_id() {
        let info = lookup_model("us.anthropic.claude-sonnet-4-6-20260101-v1:0")
            .expect("bedrock id should resolve to the base model");
        assert_eq!(info.limits.context_window, Some(1_000_000));
    }

    #[test]
    fn test_lookup_unknown_model() {
        assert!(lookup_model("no-such-model-xyz").is_none());
        assert!(get_context_window_limit("no-such-model-xyz").is_none());
    }

    #[test]
    fn test_merge_precedence() {
        let override_info = ModelInfo {
            capabilities: ModelCapabilities::default(),
            limits: ModelLimits {
                context_window: Some(42),
                max_output_tokens: None,
            },
        };
        let merged = ModelInfo::merge(&[lookup_model("claude-sonnet-4-6"), Some(&override_info)]);
        assert_eq!(merged.limits.context_window, Some(42));
    }

    #[test]
    fn test_non_ascii_model_ids_do_not_panic() {
        let weird = "модель-м Claude3-5-sonnet-v1:0";
        assert!(lookup_model(weird).is_none());
        assert!(get_context_window_limit(weird).is_none());

        let unicode_date = "claudé-1234567é";
        assert!(lookup_model(unicode_date).is_none());
    }
}
