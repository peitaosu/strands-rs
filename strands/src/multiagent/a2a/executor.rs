//! Strands Agent executor for the A2A protocol.
//!
//! This module provides the StrandsA2AExecutor, which adapts a Strands Agent
//! to be used as an executor in the A2A protocol.

use std::collections::HashMap;
use std::pin::pin;
use std::sync::{Arc, LazyLock};

use base64::Engine;
use tokio::sync::Mutex;

use super::types::{A2AArtifact, A2AError, A2AMessage, A2APart, A2ATask, A2ATaskState};
use crate::agent::Agent;
use crate::types::content::{
    ContentBlock, DocumentContent, DocumentSource, ImageContent, ImageSource, VideoContent,
    VideoSource,
};

/// Default formats for each file type when MIME type is unavailable or unrecognized.
static DEFAULT_FORMATS: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("document", "txt");
    m.insert("image", "png");
    m.insert("video", "mp4");
    m.insert("unknown", "txt");
    m
});

/// Special case format mappings where format differs from extension.
static FORMAT_MAPPINGS: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("jpg", "jpeg");
    m.insert("htm", "html");
    m.insert("3gp", "three_gp");
    m.insert("3gpp", "three_gp");
    m.insert("3g2", "three_gp");
    m
});

/// Executor that adapts a Strands Agent to the A2A protocol.
pub struct StrandsA2AExecutor {
    agent: Arc<Mutex<Agent>>,
}

impl StrandsA2AExecutor {
    /// Create a new A2A executor wrapping a Strands Agent.
    pub fn new(agent: Agent) -> Self {
        Self {
            agent: Arc::new(Mutex::new(agent)),
        }
    }

    /// Execute a request using the Strands Agent.
    pub async fn execute(&self, message: A2AMessage) -> Result<A2ATask, A2AError> {
        let content_blocks = self.convert_a2a_parts_to_content_blocks(&message.parts)?;

        let mut agent = self.agent.lock().await;

        let task_id = uuid::Uuid::new_v4().to_string();
        let context_id = message.context_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let mut task = A2ATask::new(&task_id, &context_id);
        task.state = A2ATaskState::Working;

        match agent.invoke_async(content_blocks).await {
            Ok(result) => {
                task.state = A2ATaskState::Completed;

                let response_text = result.text();
                let artifact = A2AArtifact {
                    name: "response".to_string(),
                    parts: vec![A2APart::text(response_text)],
                    index: Some(0),
                };

                task.artifacts = Some(vec![artifact]);
                task.message = Some(A2AMessage::agent(
                    vec![A2APart::text(result.text())],
                    Some(context_id),
                    Some(task_id),
                ));

                Ok(task)
            }
            Err(e) => {
                task.state = A2ATaskState::Failed;
                Err(A2AError::internal(e.to_string()))
            }
        }
    }

    /// Execute a request with streaming.
    pub async fn execute_streaming<F>(
        &self,
        message: A2AMessage,
        mut on_update: F,
    ) -> Result<A2ATask, A2AError>
    where
        F: FnMut(A2ATask) + Send,
    {
        use futures::StreamExt;

        let content_blocks = self.convert_a2a_parts_to_content_blocks(&message.parts)?;

        let mut agent = self.agent.lock().await;

        let task_id = uuid::Uuid::new_v4().to_string();
        let context_id = message.context_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let mut task = A2ATask::new(&task_id, &context_id);
        task.state = A2ATaskState::Working;

        on_update(task.clone());

        let stream = agent.stream_async(content_blocks).await;
        let mut pinned_stream = pin!(stream);
        let mut accumulated_text = String::new();

        while let Some(event) = pinned_stream.next().await {
            match event {
                Ok(stream_event) => {
                    if let Some(text) = stream_event.as_text() {
                        accumulated_text.push_str(&text);

                        let mut update_task = task.clone();
                        update_task.message = Some(A2AMessage::agent(
                            vec![A2APart::text(&accumulated_text)],
                            Some(context_id.clone()),
                            Some(task_id.clone()),
                        ));

                        on_update(update_task);
                    }
                }
                Err(e) => {
                    task.state = A2ATaskState::Failed;
                    return Err(A2AError::internal(e.to_string()));
                }
            }
        }

        task.state = A2ATaskState::Completed;

        let artifact = A2AArtifact {
            name: "response".to_string(),
            parts: vec![A2APart::text(&accumulated_text)],
            index: Some(0),
        };

        task.artifacts = Some(vec![artifact]);
        task.message = Some(A2AMessage::agent(
            vec![A2APart::text(accumulated_text)],
            Some(context_id),
            Some(task_id),
        ));

        Ok(task)
    }

    /// Convert A2A message parts to Strands content blocks.
    fn convert_a2a_parts_to_content_blocks(
        &self,
        parts: &[A2APart],
    ) -> Result<Vec<ContentBlock>, A2AError> {
        let mut content_blocks = Vec::new();

        for part in parts {
            match part {
                A2APart::Text { text } => {
                    content_blocks.push(ContentBlock::text(text));
                }
                A2APart::Data { data } => {
                    let text = serde_json::to_string_pretty(data)
                        .map(|json| format!("[Structured Data]\n{}", json))
                        .unwrap_or_else(|_| data.to_string());
                    content_blocks.push(ContentBlock::text(text));
                }
                A2APart::File { file } => {
                    let file_type = Self::classify_file_type(file.mime_type.as_deref());
                    let file_format = Self::get_file_format_from_mime_type(
                        file.mime_type.as_deref(),
                        file_type,
                    );
                    let file_name = Self::strip_file_extension(&file.name);

                    if let Some(ref bytes_str) = file.bytes {
                        match base64::engine::general_purpose::STANDARD.decode(bytes_str) {
                            Ok(decoded_bytes) => {
                                let bytes_base64 = base64::engine::general_purpose::STANDARD
                                    .encode(&decoded_bytes);

                                match file_type {
                                    "image" => {
                                        content_blocks.push(ContentBlock {
                                            image: Some(ImageContent {
                                                format: file_format,
                                                source: ImageSource {
                                                    bytes: Some(bytes_base64),
                                                },
                                            }),
                                            ..Default::default()
                                        });
                                    }
                                    "video" => {
                                        content_blocks.push(ContentBlock {
                                            video: Some(VideoContent {
                                                format: file_format,
                                                source: VideoSource {
                                                    bytes: Some(bytes_base64),
                                                },
                                            }),
                                            ..Default::default()
                                        });
                                    }
                                    _ => {
                                        content_blocks.push(ContentBlock {
                                            document: Some(DocumentContent {
                                                format: file_format,
                                                name: file_name.to_string(),
                                                source: DocumentSource {
                                                    bytes: Some(bytes_base64),
                                                },
                                            }),
                                            ..Default::default()
                                        });
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to decode base64 data for file '{}': {}",
                                    file.name,
                                    e
                                );
                                let text = format!(
                                    "[File: {} ({:?})] - Failed to decode base64 data",
                                    file.name, file.mime_type
                                );
                                content_blocks.push(ContentBlock::text(text));
                            }
                        }
                    } else if let Some(ref uri) = file.uri {
                        let text = format!(
                            "[File: {} ({:?})] - Referenced file at: {}",
                            file_name, file.mime_type, uri
                        );
                        content_blocks.push(ContentBlock::text(text));
                    } else {
                        let text = format!("[File: {}]", file.name);
                        content_blocks.push(ContentBlock::text(text));
                    }
                }
            }
        }

        if content_blocks.is_empty() {
            return Err(A2AError::invalid_request("No content blocks available"));
        }

        Ok(content_blocks)
    }

    /// Convert Strands content blocks to A2A message parts.
    pub fn convert_content_blocks_to_a2a_parts(blocks: &[ContentBlock]) -> Vec<A2APart> {
        blocks
            .iter()
            .filter_map(|block| {
                if let Some(text) = &block.text {
                    Some(A2APart::text(text))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get file format from MIME type.
    ///
    /// Uses the MIME type to determine the appropriate file format.
    /// Falls back to default formats if MIME type is unavailable or unrecognized.
    pub fn get_file_format_from_mime_type(mime_type: Option<&str>, file_type: &str) -> String {
        let Some(mime_type) = mime_type else {
            return DEFAULT_FORMATS.get(file_type).copied().unwrap_or("txt").to_string();
        };

        let mime_lower = mime_type.to_lowercase();

        if let Some(subtype) = mime_lower.split('/').last() {
            if let Some(mapped) = FORMAT_MAPPINGS.get(subtype) {
                return mapped.to_string();
            }
        }

        if let Some(subtype) = mime_lower.split('/').last() {
            if let Some(mapped) = FORMAT_MAPPINGS.get(subtype) {
                return mapped.to_string();
            }
            return subtype.to_string();
        }

        DEFAULT_FORMATS.get(file_type).copied().unwrap_or("txt").to_string()
    }

    /// Strip the file extension from a file name.
    pub fn strip_file_extension(file_name: &str) -> &str {
        if let Some(pos) = file_name.rfind('.') {
            &file_name[..pos]
        } else {
            file_name
        }
    }

    /// Classify file type based on MIME type.
    pub fn classify_file_type(mime_type: Option<&str>) -> &'static str {
        let Some(mime) = mime_type else {
            return "unknown";
        };

        let mime_lower = mime.to_lowercase();

        if mime_lower.starts_with("image/") {
            "image"
        } else if mime_lower.starts_with("video/") {
            "video"
        } else if mime_lower.starts_with("text/")
            || mime_lower.starts_with("application/pdf")
            || mime_lower.starts_with("application/msword")
            || mime_lower.contains("document")
        {
            "document"
        } else {
            "unknown"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a2a_part_text() {
        let part = A2APart::text("Hello");
        match part {
            A2APart::Text { text } => assert_eq!(text, "Hello"),
            _ => panic!("Expected text part"),
        }
    }

    #[test]
    fn test_a2a_part_data() {
        let part = A2APart::data(serde_json::json!({"key": "value"}));
        match part {
            A2APart::Data { data } => assert_eq!(data["key"], "value"),
            _ => panic!("Expected data part"),
        }
    }

    #[test]
    fn test_content_blocks_to_parts() {
        let blocks = vec![
            ContentBlock::text("Hello"),
            ContentBlock::text("World"),
        ];

        let parts = StrandsA2AExecutor::convert_content_blocks_to_a2a_parts(&blocks);
        assert_eq!(parts.len(), 2);
    }
}

