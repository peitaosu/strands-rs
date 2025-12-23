//! Guardrail type definitions for content safety.

use serde::{Deserialize, Serialize};

/// Guardrail configuration for model invocations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardrailConfig {
    pub guardrail_identifier: String,
    pub guardrail_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<GuardrailTrace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_processing_mode: Option<String>,
}

/// Trace level for guardrail processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GuardrailTrace {
    Enabled,
    Disabled,
}

/// Guardrail assessment result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardrailAssessment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic_policy: Option<GuardrailTopicPolicyAssessment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_policy: Option<GuardrailContentPolicyAssessment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word_policy: Option<GuardrailWordPolicyAssessment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensitive_information_policy: Option<GuardrailSensitiveInformationPolicyAssessment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contextual_grounding_policy: Option<GuardrailContextualGroundingPolicyAssessment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invocation_metrics: Option<GuardrailInvocationMetrics>,
}

/// Topic policy assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailTopicPolicyAssessment {
    pub topics: Vec<GuardrailTopic>,
}

/// A topic identified by guardrails.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailTopic {
    pub name: String,
    #[serde(rename = "type")]
    pub topic_type: String,
    pub action: String,
}

/// Content policy assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailContentPolicyAssessment {
    pub filters: Vec<GuardrailContentFilter>,
}

/// A content filter result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailContentFilter {
    #[serde(rename = "type")]
    pub filter_type: String,
    pub confidence: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter_strength: Option<String>,
    pub action: String,
}

/// Word policy assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardrailWordPolicyAssessment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_words: Option<Vec<GuardrailCustomWord>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub managed_word_lists: Option<Vec<GuardrailManagedWord>>,
}

/// A custom word matched by guardrails.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailCustomWord {
    #[serde(rename = "match")]
    pub word_match: String,
    pub action: String,
}

/// A managed word matched by guardrails.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailManagedWord {
    #[serde(rename = "match")]
    pub word_match: String,
    #[serde(rename = "type")]
    pub word_type: String,
    pub action: String,
}

/// Sensitive information policy assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardrailSensitiveInformationPolicyAssessment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pii_entities: Option<Vec<GuardrailPiiEntity>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regexes: Option<Vec<GuardrailRegex>>,
}

/// A PII entity detected by guardrails.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailPiiEntity {
    #[serde(rename = "match")]
    pub entity_match: String,
    #[serde(rename = "type")]
    pub entity_type: String,
    pub action: String,
}

/// A regex match by guardrails.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailRegex {
    pub name: String,
    #[serde(rename = "match")]
    pub regex_match: String,
    pub regex: String,
    pub action: String,
}

/// Contextual grounding policy assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailContextualGroundingPolicyAssessment {
    pub filters: Vec<GuardrailGroundingFilter>,
}

/// A grounding filter result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailGroundingFilter {
    #[serde(rename = "type")]
    pub filter_type: String,
    pub threshold: f64,
    pub score: f64,
    pub action: String,
}

/// Guardrail invocation metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardrailInvocationMetrics {
    pub guardrail_processing_latency: u64,
    pub usage: GuardrailUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guardrail_coverage: Option<GuardrailCoverage>,
}

/// Guardrail token usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardrailUsage {
    pub topic_policy_units: u32,
    pub content_policy_units: u32,
    pub word_policy_units: u32,
    pub sensitive_information_policy_units: u32,
    pub sensitive_information_policy_free_units: u32,
    pub contextual_grounding_policy_units: u32,
}

/// Guardrail coverage information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardrailCoverage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_characters: Option<GuardrailTextCharactersCoverage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<GuardrailImageCoverage>,
}

/// Text characters coverage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailTextCharactersCoverage {
    pub guarded: u32,
    pub total: u32,
}

/// Image coverage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailImageCoverage {
    pub guarded: u32,
    pub total: u32,
}

