//! Citation type definitions modeled after the Bedrock API.

use serde::{Deserialize, Serialize};

/// Configuration for enabling citations on documents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CitationsConfig {
    pub enabled: bool,
}

/// Specifies a character-level location within a document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentCharLocation {
    pub document_index: usize,
    pub start: usize,
    pub end: usize,
}

/// Specifies a chunk-level location within a document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentChunkLocation {
    pub document_index: usize,
    pub start: usize,
    pub end: usize,
}

/// Specifies a page-level location within a document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentPageLocation {
    pub document_index: usize,
    pub start: usize,
    pub end: usize,
}

/// Specifies a search result location within the content array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResultLocation {
    pub search_result_index: usize,
    pub start: usize,
    pub end: usize,
}

/// Provides URL and domain information for a cited website.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebLocation {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
}

/// Union type for citation locations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CitationLocation {
    DocumentChar(DocumentCharLocation),
    DocumentPage(DocumentPageLocation),
    DocumentChunk(DocumentChunkLocation),
    SearchResultLocation(SearchResultLocation),
    Web(WebLocation),
}

/// Contains text content from a source document being cited.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CitationSourceContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Contains generated text content corresponding to a citation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CitationGeneratedContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// A citation that references a source document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Citation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<CitationLocation>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_content: Option<Vec<CitationSourceContent>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// A content block containing citations and generated text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CitationsContentBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<Citation>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<CitationGeneratedContent>>,
}

/// Delta for streaming citation updates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CitationsDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<Citation>>,
}

