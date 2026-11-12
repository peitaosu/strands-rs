//! Context management utilities.

use super::ConversationManager;
use crate::types::content::{ContentBlock, Message};
use crate::types::errors::StrandsError;
use std::collections::HashMap;

/// Marker prepended to truncated text.
const TRUNCATION_MARKER: &str = "\n...[content truncated]...\n";

/// Truncates text to `max_chars`, keeping the tail and marking the cut.
///
/// Port of the Python `_context_manager/methods/truncate.py` behavior.
pub fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let marker_len = TRUNCATION_MARKER.chars().count();
    let keep = max_chars.saturating_sub(marker_len);
    let skipped: String = text.chars().skip(text.chars().count() - keep).collect();
    format!("{}{}", TRUNCATION_MARKER, skipped)
}

/// Keeps the tail of the conversation within a character budget by
/// truncating older message text.
#[derive(Debug, Clone)]
pub struct TruncatingConversationManager {
    max_chars: usize,
}

impl TruncatingConversationManager {
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }

    fn message_chars(message: &Message) -> usize {
        message.text_content().chars().count()
    }
}

impl ConversationManager for TruncatingConversationManager {
    /// Truncates the oldest messages' text until the conversation fits the
    /// character budget, dropping messages that cannot shrink further.
    fn apply_management(&self, messages: &mut Vec<Message>) {
        loop {
            let total: usize = messages.iter().map(Self::message_chars).sum();
            if total <= self.max_chars {
                return;
            }

            let Some(first) = messages.first_mut() else {
                return;
            };

            let original_text = first.text_content();
            if original_text.is_empty() {
                if messages.len() <= 1 {
                    return;
                }
                messages.remove(0);
                continue;
            }

            let overflow = total - self.max_chars;
            let allowed = original_text.chars().count().saturating_sub(overflow);
            let truncated = truncate_text(&original_text, allowed.max(1));

            if truncated.chars().count() >= original_text.chars().count() {
                if messages.len() <= 1 {
                    return;
                }
                messages.remove(0);
                continue;
            }

            first.content = vec![ContentBlock::text(truncated)];
        }
    }

    fn reduce_context(&self, messages: &mut Vec<Message>, _error: &StrandsError) {
        loop {
            let total: usize = messages.iter().map(Self::message_chars).sum();
            if total <= self.max_chars || messages.len() <= 1 {
                return;
            }
            messages.remove(0);
        }
    }

    fn get_state(&self) -> HashMap<String, serde_json::Value> {
        HashMap::from([("max_chars".to_string(), serde_json::json!(self.max_chars))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_text_under_limit() {
        assert_eq!(truncate_text("short", 100), "short");
    }

    #[test]
    fn test_truncate_text_keeps_tail() {
        let text = "a".repeat(100) + "tail";
        let truncated = truncate_text(&text, 40);
        assert!(truncated.starts_with("\n...[content truncated]...\n"));
        assert!(truncated.ends_with("tail"));
        assert!(truncated.chars().count() <= 40);
    }

    #[test]
    fn test_reduce_context_drops_oldest() {
        let manager = TruncatingConversationManager::new(20);
        let mut messages = vec![
            Message::user("0123456789012345678901234567890"),
            Message::assistant("short"),
        ];

        manager.reduce_context(&mut messages, &StrandsError::ContextWindowOverflow {
            message: "overflow".to_string(),
        });

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text_content(), "short");
    }

    #[test]
    fn test_apply_management_truncates_oldest() {
        let manager = TruncatingConversationManager::new(40);
        let mut messages = vec![
            Message::user("0123456789012345678901234567890123456789"),
            Message::assistant("tail"),
        ];

        manager.apply_management(&mut messages);

        let total: usize = messages.iter().map(|m| m.text_content().chars().count()).sum();
        assert!(total <= 40);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].text_content(), "tail");
        assert!(messages[0]
            .text_content()
            .starts_with("\n...[content truncated]...\n"));
    }

    #[test]
    fn test_apply_management_noop_under_budget() {
        let manager = TruncatingConversationManager::new(1000);
        let mut messages = vec![Message::user("small"), Message::assistant("tiny")];
        let before = messages.clone();

        manager.apply_management(&mut messages);
        assert_eq!(messages, before);
    }

    #[test]
    fn test_apply_management_drops_oversized_non_shrinkable() {
        let manager = TruncatingConversationManager::new(10);
        let mut messages = vec![
            Message::user("0123456789"),
            Message::assistant("tail"),
        ];

        manager.apply_management(&mut messages);

        let total: usize = messages.iter().map(|m| m.text_content().chars().count()).sum();
        assert!(total <= 10);
        assert_eq!(messages.last().unwrap().text_content(), "tail");
    }
}
