//! Invocation limits for agent requests.

use serde::{Deserialize, Serialize};

/// Limits for a top-level agent request.
///
/// Each limit applies across every model call and tool cycle in the request.
/// Omitted limits are unlimited, matching the Python `Limits` TypedDict.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    /// Maximum number of turns, where one turn is one model call plus any
    /// tool execution that follows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turns: Option<u32>,

    /// Maximum cumulative model-generated output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,

    /// Maximum cumulative total tokens across every model call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,

    /// Maximum failed structured output attempts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_output_attempts: Option<u32>,
}

impl Limits {
    /// Returns the first limit reached, highest priority first.
    ///
    /// Priority: turns, total_tokens, output_tokens, structured_output_attempts.
    pub fn check(
        &self,
        turns: u32,
        total_tokens: u64,
        output_tokens: u64,
        structured_output_attempts: u32,
    ) -> Option<crate::types::streaming::StopReason> {
        use crate::types::streaming::StopReason;

        if self.turns.map(|limit| turns > limit).unwrap_or(false) {
            return Some(StopReason::LimitTurns);
        }
        if self.total_tokens.map(|limit| total_tokens >= limit).unwrap_or(false) {
            return Some(StopReason::LimitTotalTokens);
        }
        if self.output_tokens.map(|limit| output_tokens >= limit).unwrap_or(false) {
            return Some(StopReason::LimitOutputTokens);
        }
        if self
            .structured_output_attempts
            .map(|limit| structured_output_attempts >= limit)
            .unwrap_or(false)
        {
            return Some(StopReason::LimitStructuredOutputAttempts);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::streaming::StopReason;

    #[test]
    fn test_no_limits() {
        assert!(Limits::default().check(100, 100_000, 100_000, 100).is_none());
    }

    #[test]
    fn test_turn_limit() {
        let limits = Limits { turns: Some(2), ..Default::default() };
        assert!(limits.check(1, 0, 0, 0).is_none());
        assert!(limits.check(2, 0, 0, 0).is_none());
        assert_eq!(limits.check(3, 0, 0, 0), Some(StopReason::LimitTurns));
    }

    #[test]
    fn test_turn_limit_of_one_allows_one_turn() {
        let limits = Limits { turns: Some(1), ..Default::default() };
        assert!(limits.check(1, 0, 0, 0).is_none());
        assert_eq!(limits.check(2, 0, 0, 0), Some(StopReason::LimitTurns));
    }

    #[test]
    fn test_token_limits() {
        let limits = Limits {
            total_tokens: Some(1000),
            output_tokens: Some(500),
            ..Default::default()
        };
        assert_eq!(limits.check(0, 1000, 100, 0), Some(StopReason::LimitTotalTokens));
        assert_eq!(limits.check(0, 100, 500, 0), Some(StopReason::LimitOutputTokens));
    }

    #[test]
    fn test_priority_order() {
        let limits = Limits {
            turns: Some(1),
            total_tokens: Some(1),
            ..Default::default()
        };
        assert_eq!(limits.check(5, 5, 5, 0), Some(StopReason::LimitTurns));
    }
}
