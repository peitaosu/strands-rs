//! Agent-related type definitions for the SDK.

use super::content::{ContentBlock, Messages};
use super::interrupt::InterruptResponseContent;

/// Input type for agent invocation.
#[derive(Debug, Clone)]
pub enum AgentInput {
    /// Simple text input.
    Text(String),
    /// Content blocks input.
    ContentBlocks(Vec<ContentBlock>),
    /// Interrupt response content.
    InterruptResponses(Vec<InterruptResponseContent>),
    /// Full messages input.
    Messages(Messages),
    /// No input (resume).
    None,
}

impl From<String> for AgentInput {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for AgentInput {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<Vec<ContentBlock>> for AgentInput {
    fn from(blocks: Vec<ContentBlock>) -> Self {
        Self::ContentBlocks(blocks)
    }
}

impl From<Vec<InterruptResponseContent>> for AgentInput {
    fn from(responses: Vec<InterruptResponseContent>) -> Self {
        Self::InterruptResponses(responses)
    }
}

impl From<Messages> for AgentInput {
    fn from(messages: Messages) -> Self {
        Self::Messages(messages)
    }
}

impl From<Option<String>> for AgentInput {
    fn from(opt: Option<String>) -> Self {
        match opt {
            Some(s) => Self::Text(s),
            None => Self::None,
        }
    }
}

impl Default for AgentInput {
    fn default() -> Self {
        Self::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_input_from_string() {
        let input: AgentInput = "Hello".into();
        assert!(matches!(input, AgentInput::Text(_)));
    }

    #[test]
    fn test_agent_input_from_none() {
        let input: AgentInput = None::<String>.into();
        assert!(matches!(input, AgentInput::None));
    }

    #[test]
    fn test_agent_input_default() {
        let input = AgentInput::default();
        assert!(matches!(input, AgentInput::None));
    }
}

