//! Content types for messages and content blocks.

use serde::{Deserialize, Serialize};

use super::tools::{ToolResult, ToolUse};

/// Represents the role of a message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Role {
    /// Returns the string representation of the role.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// A block of content that can contain text, images, tools, or other media.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageContent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<DocumentContent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoContent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<ToolUse>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<ToolResult>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<ReasoningContentBlock>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_point: Option<CachePoint>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<SummaryBlock>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub guard_content: Option<GuardContent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations_content: Option<CitationsContentBlock>,
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self { text: Some(text.into()), ..Default::default() }
    }

    pub fn tool_use(tool_use: ToolUse) -> Self {
        Self { tool_use: Some(tool_use), ..Default::default() }
    }

    pub fn tool_result(result: ToolResult) -> Self {
        Self { tool_result: Some(result), ..Default::default() }
    }

    pub fn is_text(&self) -> bool { self.text.is_some() }
    pub fn is_tool_use(&self) -> bool { self.tool_use.is_some() }
    pub fn is_tool_result(&self) -> bool { self.tool_result.is_some() }
    pub fn as_text(&self) -> Option<&str> { self.text.as_deref() }
    pub fn as_tool_use(&self) -> Option<&ToolUse> { self.tool_use.as_ref() }
    pub fn as_tool_result(&self) -> Option<&ToolResult> { self.tool_result.as_ref() }
}

impl From<String> for ContentBlock {
    fn from(text: String) -> Self { Self::text(text) }
}

impl From<&str> for ContentBlock {
    fn from(text: &str) -> Self { Self::text(text) }
}

/// Image content with format and source data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    pub format: String,
    pub source: ImageSource,
}

/// Source data for an image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<String>,
}

/// Document content with format, name, and source data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentContent {
    pub format: String,
    pub name: String,
    pub source: DocumentSource,
}

/// Source data for a document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<String>,
}

/// Video content with format and source data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoContent {
    pub format: String,
    pub source: VideoSource,
}

/// Source data for a video.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<String>,
}

/// Content block containing model reasoning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningContentBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_text: Option<ReasoningTextBlock>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted_content: Option<Vec<u8>>,
}

/// Text block within reasoning content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningTextBlock {
    pub text: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// A cache point marker for prompt caching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CachePoint {
    #[serde(rename = "type")]
    pub cache_type: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

/// A summarized content block produced by conversation management.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SummaryBlock {
    pub summary: String,
}

/// Content that has been processed by guardrails.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardContent {
    pub text: GuardContentText,
}

/// Text content from guardrail processing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardContentText {
    pub qualifiers: Vec<String>,
    pub text: String,
}

/// Metadata attached to a message by the event loop.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<crate::types::streaming::Usage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<crate::types::streaming::Metrics>,

    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub custom: std::collections::HashMap<String, serde_json::Value>,
}

/// A message in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracking_id: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MessageMetadata>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
            tracking_id: None,
            metadata: None,
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::text(text)],
            tracking_id: None,
            metadata: None,
        }
    }

    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content, tracking_id: None, metadata: None }
    }

    pub fn with_tracking_id(mut self, tracking_id: impl Into<String>) -> Self {
        self.tracking_id = Some(tracking_id.into());
        self
    }

    pub fn with_metadata(mut self, metadata: MessageMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Assigns a monotonic UUIDv7 tracking id when none is present and returns it.
    pub fn ensure_tracking_id(&mut self) -> String {
        if let Some(id) = &self.tracking_id {
            return id.clone();
        }
        let id = crate::identifier::new_uuid7();
        self.tracking_id = Some(id.clone());
        id
    }

    /// Strips event-loop-only fields, matching the projection models receive.
    pub fn to_model_message(&self) -> Message {
        Message {
            role: self.role,
            content: self.content.clone(),
            tracking_id: None,
            metadata: None,
        }
    }
}

impl Default for Message {
    fn default() -> Self { Self::new(Role::User, Vec::new()) }
}

impl Message {
    pub fn text_content(&self) -> String {
        self.content.iter().filter_map(|b| b.as_text()).collect::<Vec<_>>().join("")
    }

    pub fn has_tool_use(&self) -> bool {
        self.content.iter().any(|b| b.is_tool_use())
    }

    pub fn tool_uses(&self) -> Vec<&ToolUse> {
        self.content.iter().filter_map(|b| b.as_tool_use()).collect()
    }

    pub fn has_tool_result(&self) -> bool {
        self.content.iter().any(|b| b.is_tool_result())
    }
}

/// A collection of messages representing a conversation.
pub type Messages = Vec<Message>;

/// A content block in a system prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SystemContentBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_point: Option<CachePoint>,
}

impl SystemContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self { text: Some(text.into()), cache_point: None }
    }
}

/// Content block containing citations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CitationsContentBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<Citation>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<CitationGeneratedContent>>,
}

/// A citation reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Citation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_content: Option<Vec<CitationSourceContent>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// Source content for a citation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CitationSourceContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Generated content with citation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CitationGeneratedContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_serialization() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(serde_json::to_string(&Role::Assistant).unwrap(), "\"assistant\"");
    }

    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello, world!");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text_content(), "Hello, world!");
    }

    #[test]
    fn test_content_block_from_string() {
        let block: ContentBlock = "test".into();
        assert!(block.is_text());
        assert_eq!(block.as_text(), Some("test"));
    }

    #[test]
    fn test_content_block_serialization() {
        let block = ContentBlock::text("hello");
        let json = serde_json::to_string(&block).unwrap();
        assert_eq!(json, r#"{"text":"hello"}"#);
    }
}
