//! Loop-control vended tools: sleep, stop, and handoff to user.

use std::time::Duration;

use async_trait::async_trait;

use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::tools::ToolSpec;

/// Maximum duration the `sleep` tool may pause for.
pub const MAX_SLEEP_SECONDS: u64 = 300;

/// Bounded cooperative pause of the agent loop.
pub struct SleepTool;

/// Clamps a requested duration to the allowed bounds.
fn clamp_seconds(requested: f64) -> f64 {
    requested.clamp(0.0, MAX_SLEEP_SECONDS as f64)
}

impl SleepTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SleepTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentTool for SleepTool {
    fn name(&self) -> &str {
        "sleep"
    }

    fn description(&self) -> &str {
        "Pause execution for a bounded number of seconds. Use this to wait for \
         external processes or rate limits instead of retrying immediately."
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description()).with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": format!("Seconds to sleep, between 0 and {}", MAX_SLEEP_SECONDS)
                }
            },
            "required": ["seconds"]
        }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let seconds = clamp_seconds(
            input.get("seconds").and_then(|v| v.as_f64()).unwrap_or(0.0),
        );

        if context.is_cancelled() {
            return Ok(ToolResult2::error("Cancelled"));
        }

        tokio::time::sleep(Duration::from_secs_f64(seconds)).await;
        Ok(ToolResult2::success(format!("Slept for {} seconds", seconds)))
    }
}

/// Gracefully ends the agent loop after the current tool cycle.
pub struct StopTool;

impl StopTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StopTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentTool for StopTool {
    fn name(&self) -> &str {
        "stop"
    }

    fn description(&self) -> &str {
        "End the agent loop gracefully. Call this when the task is complete \
         and no further tool calls are needed."
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description()).with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Final message to report before stopping"
                }
            }
        }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        context.invocation_state.request_stop();
        let message = input
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Agent loop stopped by request.");
        Ok(ToolResult2::success(message))
    }
}

/// Surfaces a message to the user and ends the current loop cycle.
pub struct HandoffToUserTool;

impl HandoffToUserTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HandoffToUserTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentTool for HandoffToUserTool {
    fn name(&self) -> &str {
        "handoff_to_user"
    }

    fn description(&self) -> &str {
        "Pause the agent loop and surface a message to the user. Call this \
         when you need information, a decision, or an action only the user \
         can provide."
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description()).with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "What the user needs to know or decide"
                }
            },
            "required": ["message"]
        }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let message = input
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or_default()
            .to_string();

        context.invocation_state.request_stop();
        context
            .invocation_state
            .set("handoff_to_user_message", message.clone());

        Ok(ToolResult2::success(format!(
            "Handed off to the user: {}",
            message
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tools::ToolUse;

    fn context() -> ToolContext {
        ToolContext::new(ToolUse::new("t", "t1", serde_json::json!({})))
    }

    #[test]
    fn test_sleep_clamps_duration() {
        assert_eq!(clamp_seconds(999_999.0), MAX_SLEEP_SECONDS as f64);
        assert_eq!(clamp_seconds(-5.0), 0.0);
        assert_eq!(clamp_seconds(1.5), 1.5);
    }

    #[tokio::test]
    async fn test_sleep_reports_duration() {
        let tool = SleepTool::new();
        let context = context();

        let result = tool
            .invoke(serde_json::json!({"seconds": 0.01}), &context)
            .await
            .unwrap();
        assert!(result
            .content[0]
            .text
            .as_deref()
            .unwrap_or_default()
            .contains("0.01 seconds"));
    }

    #[tokio::test]
    async fn test_stop_sets_loop_flag() {
        let tool = StopTool::new();
        let context = context();

        let result = tool
            .invoke(serde_json::json!({"message": "done"}), &context)
            .await
            .unwrap();
        assert!(result.content[0].text.as_deref().unwrap_or_default().contains("done"));
        assert!(context.invocation_state.should_stop());
    }

    #[tokio::test]
    async fn test_handoff_sets_loop_flag_and_message() {
        let tool = HandoffToUserTool::new();
        let context = context();

        tool.invoke(serde_json::json!({"message": "need approval"}), &context)
            .await
            .unwrap();

        assert!(context.invocation_state.should_stop());
        assert_eq!(
            context.invocation_state.get::<String>("handoff_to_user_message"),
            Some("need approval".to_string())
        );
    }
}
