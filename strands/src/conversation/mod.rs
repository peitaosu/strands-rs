//! Conversation management for context window optimization.

use std::collections::HashMap;
use std::sync::Arc;

use crate::types::content::{ContentBlock, Message, Role};
use crate::types::errors::StrandsError;

/// Default summarization prompt.
pub const DEFAULT_SUMMARIZATION_PROMPT: &str = r#"You are a conversation summarizer. Provide a concise summary of the conversation history.

Format Requirements:
- You MUST create a structured and concise summary in bullet-point format.
- You MUST NOT respond conversationally.
- You MUST NOT address the user directly.
- You MUST NOT comment on tool availability.

Assumptions:
- You MUST NOT assume tool executions failed unless otherwise stated.

Task:
Your task is to create a structured summary document:
- It MUST contain bullet points with key topics and questions covered
- It MUST contain bullet points for all significant tools executed and their results
- It MUST contain bullet points for any code or technical information shared
- It MUST contain a section of key insights gained
- It MUST format the summary in the third person

Example format:

## Conversation Summary
* Topic 1: Key information
* Topic 2: Key information
*
## Tools Executed
* Tool X: Result Y"#;

/// Trait for implementing conversation managers.
pub trait ConversationManager: Send + Sync {
    /// Applies management strategy to the agent's messages.
    fn apply_management(&self, messages: &mut Vec<Message>);

    /// Reduces context when an error occurs (e.g., context overflow).
    fn reduce_context(&self, messages: &mut Vec<Message>, error: &StrandsError);

    /// Returns the current state for session persistence.
    fn get_state(&self) -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }

    /// Restores state from a session. Returns optional prepend messages.
    fn restore_from_session(&mut self, _state: HashMap<String, serde_json::Value>) -> Option<Vec<Message>> {
        None
    }

    /// Returns the count of messages removed by this manager.
    fn removed_message_count(&self) -> usize {
        0
    }
}

/// A no-op conversation manager.
#[derive(Debug, Clone, Default)]
pub struct NullConversationManager;

impl ConversationManager for NullConversationManager {
    fn apply_management(&self, _messages: &mut Vec<Message>) {}
    fn reduce_context(&self, _messages: &mut Vec<Message>, _error: &StrandsError) {}
}

/// Sliding window conversation manager that keeps recent messages.
#[derive(Debug, Clone)]
pub struct SlidingWindowConversationManager {
    pub window_size: usize,
    removed_message_count: usize,
}

impl Default for SlidingWindowConversationManager {
    fn default() -> Self {
        Self {
            window_size: 40,
            removed_message_count: 0,
        }
    }
}

impl SlidingWindowConversationManager {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            removed_message_count: 0,
        }
    }

    fn adjust_split_point_for_tool_pairs(
        &self,
        messages: &[Message],
        split_point: usize,
    ) -> Result<usize, StrandsError> {
        if split_point > messages.len() {
            return Err(StrandsError::ContextWindowOverflow {
                message: "Split point exceeds message array length".to_string(),
            });
        }

        if split_point == messages.len() {
            return Ok(split_point);
        }

        let mut adjusted = split_point;

        while adjusted < messages.len() {
            let msg = &messages[adjusted];
            let has_tool_result = msg.content.iter().any(|c| c.tool_result.is_some());
            let has_tool_use = msg.content.iter().any(|c| c.tool_use.is_some());

            let next_has_tool_result = if adjusted + 1 < messages.len() {
                messages[adjusted + 1]
                    .content
                    .iter()
                    .any(|c| c.tool_result.is_some())
            } else {
                false
            };

            if has_tool_result || (has_tool_use && adjusted + 1 < messages.len() && !next_has_tool_result)
            {
                adjusted += 1;
            } else {
                break;
            }
        }

        if adjusted >= messages.len() {
            return Err(StrandsError::ContextWindowOverflow {
                message: "Unable to trim conversation context!".to_string(),
            });
        }

        Ok(adjusted)
    }
}

impl ConversationManager for SlidingWindowConversationManager {
    fn apply_management(&self, messages: &mut Vec<Message>) {
        if messages.len() > self.window_size {
            let to_remove = messages.len() - self.window_size;
            if let Ok(adjusted) = self.adjust_split_point_for_tool_pairs(messages, to_remove) {
                messages.drain(..adjusted);
            }
        }
    }

    fn reduce_context(&self, messages: &mut Vec<Message>, _error: &StrandsError) {
        let keep = messages.len() / 2;
        if keep > 0 {
            let to_remove = messages.len() - keep;
            if let Ok(adjusted) = self.adjust_split_point_for_tool_pairs(messages, to_remove) {
                messages.drain(..adjusted);
            }
        }
    }

    fn get_state(&self) -> HashMap<String, serde_json::Value> {
        let mut state = HashMap::new();
        state.insert(
            "removed_message_count".to_string(),
            serde_json::json!(self.removed_message_count),
        );
        state.insert(
            "window_size".to_string(),
            serde_json::json!(self.window_size),
        );
        state
    }

    fn removed_message_count(&self) -> usize {
        self.removed_message_count
    }
}

/// Summarization function type for SummarizingConversationManager.
pub type SummarizeFn = Arc<dyn Fn(&[Message]) -> Message + Send + Sync>;

/// Summarizing conversation manager that summarizes older context.
pub struct SummarizingConversationManager {
    pub summary_ratio: f64,
    pub preserve_recent_messages: usize,
    pub summarization_prompt: String,
    summarize_fn: Option<SummarizeFn>,
    summary_message: Option<Message>,
    removed_message_count: usize,
}

impl Default for SummarizingConversationManager {
    fn default() -> Self {
        Self {
            summary_ratio: 0.3,
            preserve_recent_messages: 10,
            summarization_prompt: DEFAULT_SUMMARIZATION_PROMPT.to_string(),
            summarize_fn: None,
            summary_message: None,
            removed_message_count: 0,
        }
    }
}

impl SummarizingConversationManager {
    pub fn new(summary_ratio: f64, preserve_recent_messages: usize) -> Self {
        Self {
            summary_ratio: summary_ratio.clamp(0.1, 0.8),
            preserve_recent_messages,
            ..Default::default()
        }
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.summarization_prompt = prompt.into();
        self
    }

    pub fn with_summarize_fn(mut self, f: SummarizeFn) -> Self {
        self.summarize_fn = Some(f);
        self
    }

    fn adjust_split_point_for_tool_pairs(
        &self,
        messages: &[Message],
        split_point: usize,
    ) -> Result<usize, StrandsError> {
        if split_point > messages.len() {
            return Err(StrandsError::ContextWindowOverflow {
                message: "Split point exceeds message array length".to_string(),
            });
        }

        if split_point == messages.len() {
            return Ok(split_point);
        }

        let mut adjusted = split_point;

        while adjusted < messages.len() {
            let msg = &messages[adjusted];
            let has_tool_result = msg.content.iter().any(|c| c.tool_result.is_some());
            let has_tool_use = msg.content.iter().any(|c| c.tool_use.is_some());

            let next_has_tool_result = if adjusted + 1 < messages.len() {
                messages[adjusted + 1]
                    .content
                    .iter()
                    .any(|c| c.tool_result.is_some())
            } else {
                false
            };

            if has_tool_result || (has_tool_use && adjusted + 1 < messages.len() && !next_has_tool_result)
            {
                adjusted += 1;
            } else {
                break;
            }
        }

        if adjusted >= messages.len() {
            return Err(StrandsError::ContextWindowOverflow {
                message: "Unable to trim conversation context!".to_string(),
            });
        }

        Ok(adjusted)
    }

    fn generate_summary(&self, messages: &[Message]) -> Message {
        if let Some(ref f) = self.summarize_fn {
            f(messages)
        } else {

            let summary_text = messages
                .iter()
                .filter_map(|m| {
                    m.content.iter().find_map(|c| c.text.clone())
                })
                .collect::<Vec<_>>()
                .join("\n");

            Message::new(
                Role::User,
                vec![ContentBlock::text(format!(
                    "## Conversation Summary\n{}",
                    summary_text
                ))],
            )
        }
    }
}

impl ConversationManager for SummarizingConversationManager {
    fn apply_management(&self, _messages: &mut Vec<Message>) {

    }

    fn reduce_context(&self, messages: &mut Vec<Message>, _error: &StrandsError) {
        let messages_to_summarize_count =
            (messages.len() as f64 * self.summary_ratio).max(1.0) as usize;

        let messages_to_summarize_count = messages_to_summarize_count
            .min(messages.len().saturating_sub(self.preserve_recent_messages));

        if messages_to_summarize_count == 0 {
            return;
        }

        let adjusted = match self.adjust_split_point_for_tool_pairs(messages, messages_to_summarize_count) {
            Ok(a) => a,
            Err(_) => return,
        };

        if adjusted == 0 {
            return;
        }

        let messages_to_summarize: Vec<_> = messages.drain(..adjusted).collect();
        let summary = self.generate_summary(&messages_to_summarize);

        messages.insert(0, summary);
    }

    fn get_state(&self) -> HashMap<String, serde_json::Value> {
        let mut state = HashMap::new();
        state.insert(
            "removed_message_count".to_string(),
            serde_json::json!(self.removed_message_count),
        );
        if let Some(ref summary) = self.summary_message {
            if let Ok(v) = serde_json::to_value(summary) {
                state.insert("summary_message".to_string(), v);
            }
        }
        state
    }

    fn restore_from_session(&mut self, state: HashMap<String, serde_json::Value>) -> Option<Vec<Message>> {
        if let Some(v) = state.get("removed_message_count") {
            if let Some(count) = v.as_u64() {
                self.removed_message_count = count as usize;
            }
        }

        if let Some(v) = state.get("summary_message") {
            if let Ok(msg) = serde_json::from_value(v.clone()) {
                self.summary_message = Some(msg);
                return self.summary_message.clone().map(|m| vec![m]);
            }
        }

        None
    }

    fn removed_message_count(&self) -> usize {
        self.removed_message_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::content::Role;

    #[test]
    fn test_sliding_window_applies_management() {
        let manager = SlidingWindowConversationManager::new(3);
        let mut messages = vec![
            Message::new(Role::User, vec![ContentBlock::text("1")]),
            Message::new(Role::Assistant, vec![ContentBlock::text("2")]),
            Message::new(Role::User, vec![ContentBlock::text("3")]),
            Message::new(Role::Assistant, vec![ContentBlock::text("4")]),
            Message::new(Role::User, vec![ContentBlock::text("5")]),
        ];

        manager.apply_management(&mut messages);
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn test_null_conversation_manager() {
        let manager = NullConversationManager;
        let mut messages = vec![
            Message::new(Role::User, vec![ContentBlock::text("test")]),
        ];

        manager.apply_management(&mut messages);
        assert_eq!(messages.len(), 1);
    }
}
