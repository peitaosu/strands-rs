//! Media-related type definitions for the SDK.

use serde::{Deserialize, Serialize};

use super::citations::CitationsConfig;

/// Supported document formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentFormat {
    Pdf,
    Csv,
    Doc,
    Docx,
    Xls,
    Xlsx,
    Html,
    Txt,
    Md,
}

impl DocumentFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Csv => "csv",
            Self::Doc => "doc",
            Self::Docx => "docx",
            Self::Xls => "xls",
            Self::Xlsx => "xlsx",
            Self::Html => "html",
            Self::Txt => "txt",
            Self::Md => "md",
        }
    }
}

/// Contains the content of a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    #[serde(with = "base64_bytes")]
    pub bytes: Vec<u8>,
}

impl DocumentSource {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

/// A document to include in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentContent {
    pub format: DocumentFormat,
    pub name: String,
    pub source: DocumentSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<CitationsConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

impl DocumentContent {
    pub fn new(format: DocumentFormat, name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            format,
            name: name.into(),
            source: DocumentSource::new(bytes),
            citations: None,
            context: None,
        }
    }

    pub fn with_citations(mut self, config: CitationsConfig) -> Self {
        self.citations = Some(config);
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Supported image formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Gif,
    Webp,
}

impl ImageFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Gif => "gif",
            Self::Webp => "webp",
        }
    }
}

/// Contains the content of an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(with = "base64_bytes")]
    pub bytes: Vec<u8>,
}

impl ImageSource {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

/// An image to include in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContent {
    pub format: ImageFormat,
    pub source: ImageSource,
}

impl ImageContent {
    pub fn new(format: ImageFormat, bytes: Vec<u8>) -> Self {
        Self {
            format,
            source: ImageSource::new(bytes),
        }
    }
}

/// Supported video formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoFormat {
    Flv,
    Mkv,
    Mov,
    Mpeg,
    Mpg,
    Mp4,
    #[serde(rename = "three_gp")]
    ThreeGp,
    Webm,
    Wmv,
}

impl VideoFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Flv => "flv",
            Self::Mkv => "mkv",
            Self::Mov => "mov",
            Self::Mpeg => "mpeg",
            Self::Mpg => "mpg",
            Self::Mp4 => "mp4",
            Self::ThreeGp => "three_gp",
            Self::Webm => "webm",
            Self::Wmv => "wmv",
        }
    }
}

/// Contains the content of a video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoSource {
    #[serde(with = "base64_bytes")]
    pub bytes: Vec<u8>,
}

impl VideoSource {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

/// A video to include in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoContent {
    pub format: VideoFormat,
    pub source: VideoSource,
}

impl VideoContent {
    pub fn new(format: VideoFormat, bytes: Vec<u8>) -> Self {
        Self {
            format,
            source: VideoSource::new(bytes),
        }
    }
}

/// Module for base64 encoding/decoding bytes in serde.
mod base64_bytes {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        STANDARD.decode(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_content() {
        let doc = DocumentContent::new(DocumentFormat::Pdf, "test.pdf", vec![1, 2, 3]);
        assert_eq!(doc.format, DocumentFormat::Pdf);
        assert_eq!(doc.name, "test.pdf");
    }

    #[test]
    fn test_image_content() {
        let img = ImageContent::new(ImageFormat::Png, vec![1, 2, 3]);
        assert_eq!(img.format, ImageFormat::Png);
    }

    #[test]
    fn test_video_content() {
        let vid = VideoContent::new(VideoFormat::Mp4, vec![1, 2, 3]);
        assert_eq!(vid.format, VideoFormat::Mp4);
    }

    #[test]
    fn test_document_format_as_str() {
        assert_eq!(DocumentFormat::Pdf.as_str(), "pdf");
        assert_eq!(DocumentFormat::Docx.as_str(), "docx");
    }
}

