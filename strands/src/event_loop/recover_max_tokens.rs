//! Message recovery utilities for handling max token limit scenarios.
//!
//! This module provides functionality to recover and clean up incomplete messages that occur
//! when model responses are truncated due to maximum token limits being reached. It specifically
//! handles cases where tool use blocks are incomplete or malformed due to truncation.

use crate::types::content::{ContentBlock, Message};

/// Recover and clean up messages when max token limits are reached.
///
/// When a model response is truncated due to maximum token limits, all tool use blocks
/// should be replaced with informative error messages since they may be incomplete or
/// unreliable. This function inspects the message content and:
///
/// 1. Identifies all tool use blocks (regardless of validity)
/// 2. Replaces all tool uses with informative error messages
/// 3. Preserves all non-tool content blocks (text, images, etc.)
/// 4. Returns a cleaned message suitable for conversation history
///
/// This recovery mechanism ensures that the conversation can continue gracefully even when
/// model responses are truncated, providing clear feedback about what happened and preventing
/// potentially incomplete or corrupted tool executions.
///
/// # Arguments
///
/// * `message` - The potentially incomplete message from the model that was truncated
///               due to max token limits.
///
/// # Returns
///
/// A cleaned Message with all tool uses replaced by explanatory text content.
/// The returned message maintains the same role as the input message.
///
/// # Example
///
/// If a message contains any tool use (complete or incomplete):
/// ```text
/// {"toolUse": {"name": "calculator", "input": {"expression": "2+2"}, "toolUseId": "123"}}
/// ```
///
/// It will be replaced with:
/// ```text
/// {"text": "The selected tool calculator's tool use was incomplete due to maximum token limits being reached."}
/// ```
pub fn recover_message_on_max_tokens_reached(message: Message) -> Message {
    tracing::info!("handling max_tokens stop reason - replacing all tool uses with error messages");

    let mut valid_content: Vec<ContentBlock> = Vec::new();

    for content in message.content {
        if let Some(tool_use) = &content.tool_use {

            let display_name = if tool_use.name.is_empty() {
                "<unknown>".to_string()
            } else {
                tool_use.name.clone()
            };

            tracing::warn!(
                "tool_name=<{}> | replacing with error message due to max_tokens truncation",
                display_name
            );

            valid_content.push(ContentBlock::text(format!(
                "The selected tool {}'s tool use was incomplete due to maximum token limits being reached.",
                display_name
            )));
        } else {

            valid_content.push(content);
        }
    }

    Message {
        role: message.role,
        content: valid_content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::content::Role;
    use crate::types::tools::ToolUse;

    fn make_tool_use(name: &str) -> ContentBlock {
        ContentBlock::tool_use(ToolUse {
            name: name.to_string(),
            tool_use_id: "test-id".to_string(),
            input: serde_json::json!({}),
        })
    }

    #[test]
    fn test_recover_message_with_incomplete_tool_use() {
        let message = Message {
            role: Role::Assistant,
            content: vec![make_tool_use("calculator")],
        };

        let result = recover_message_on_max_tokens_reached(message);

        assert_eq!(result.role, Role::Assistant);
        assert_eq!(result.content.len(), 1);

        let text = result.content[0].text.as_ref().expect("Expected text content");
        assert!(text.contains("calculator"));
        assert!(text.contains("maximum token limits"));
    }

    #[test]
    fn test_recover_message_with_missing_tool_name() {
        let message = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::tool_use(ToolUse {
                name: String::new(),
                tool_use_id: "test-id".to_string(),
                input: serde_json::json!({}),
            })],
        };

        let result = recover_message_on_max_tokens_reached(message);

        let text = result.content[0].text.as_ref().expect("Expected text content");
        assert!(text.contains("<unknown>"));
    }

    #[test]
    fn test_recover_message_preserves_non_tool_content() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::text("Hello"),
                make_tool_use("calculator"),
                ContentBlock::text("World"),
            ],
        };

        let result = recover_message_on_max_tokens_reached(message);

        assert_eq!(result.content.len(), 3);

        let text0 = result.content[0].text.as_ref().expect("Expected text");
        assert_eq!(text0, "Hello");

        let text1 = result.content[1].text.as_ref().expect("Expected text");
        assert!(text1.contains("calculator"));

        let text2 = result.content[2].text.as_ref().expect("Expected text");
        assert_eq!(text2, "World");
    }

    #[test]
    fn test_recover_message_with_empty_content() {
        let message = Message {
            role: Role::Assistant,
            content: vec![],
        };

        let result = recover_message_on_max_tokens_reached(message);

        assert_eq!(result.content.len(), 0);
    }

    #[test]
    fn test_recover_message_multiple_incomplete_tools() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                make_tool_use("tool1"),
                make_tool_use("tool2"),
            ],
        };

        let result = recover_message_on_max_tokens_reached(message);

        assert_eq!(result.content.len(), 2);

        for content in &result.content {
            let text = content.text.as_ref().expect("Expected text content");
            assert!(text.contains("maximum token limits"));
        }
    }

    #[test]
    fn test_recover_message_preserves_user_role() {
        let message = Message {
            role: Role::User,
            content: vec![make_tool_use("test")],
        };

        let result = recover_message_on_max_tokens_reached(message);

        assert_eq!(result.role, Role::User);
    }
}

