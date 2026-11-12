//! SEP-2663 task-augmented MCP execution.
//!
//! Port of the Python `tools/mcp/mcp_tasks.py`: task-augmented tool calls
//! carry a `task` object in the JSON-RPC params; the caller polls or waits
//! on `tasks/get` until the task reaches a terminal state.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// The MCP tasks extension URI.
pub const TASKS_EXTENSION: &str = "io.modelcontextprotocol/tasks";

/// Protocol version the tasks support.
pub const TASKS_PROTOCOL_VERSION: &str = "2026-07-28";

/// Default poll interval for task waits.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Terminal task states that stop polling.
pub const TERMINAL_STATES: [&str; 4] = ["completed", "failed", "cancelled", "input_declined"];

/// How a task-augmented call should be executed.
#[derive(Debug, Clone, PartialEq)]
pub enum TasksConfig {
    /// Execute inline without task augmentation.
    Disabled,
    /// Always create a task and poll it.
    Always { poll_interval: Duration },
    /// Create a task only when the server reports `task_support` for the tool.
    Auto { poll_interval: Duration },
}

impl Default for TasksConfig {
    fn default() -> Self {
        TasksConfig::Auto {
            poll_interval: DEFAULT_POLL_INTERVAL,
        }
    }
}

impl TasksConfig {
    pub fn poll_interval(&self) -> Duration {
        match self {
            TasksConfig::Disabled => DEFAULT_POLL_INTERVAL,
            TasksConfig::Always { poll_interval } | TasksConfig::Auto { poll_interval } => {
                *poll_interval
            }
        }
    }

    pub fn should_use_task(&self, tool_task_support: Option<bool>) -> bool {
        match self {
            TasksConfig::Disabled => false,
            TasksConfig::Always { .. } => true,
            TasksConfig::Auto { .. } => tool_task_support.unwrap_or(false),
        }
    }
}

/// State of a task-augmented tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MCPTaskState {
    Working,
    InputRequired,
    Completed,
    Failed,
    Cancelled,
    InputDeclined,
}

impl MCPTaskState {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            MCPTaskState::Completed
                | MCPTaskState::Failed
                | MCPTaskState::Cancelled
                | MCPTaskState::InputDeclined
        )
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            MCPTaskState::Working => "working",
            MCPTaskState::InputRequired => "input_required",
            MCPTaskState::Completed => "completed",
            MCPTaskState::Failed => "failed",
            MCPTaskState::Cancelled => "cancelled",
            MCPTaskState::InputDeclined => "input_declined",
        }
    }
}

/// Result payload for `tasks/get`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MCPTask {
    pub task_id: String,
    pub state: MCPTaskState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
}

/// Builds the `task` params object for a task-augmented `tools/call`.
pub fn task_params(task_id: Option<&str>) -> serde_json::Value {
    match task_id {
        Some(task_id) => serde_json::json!({ "taskId": task_id }),
        None => serde_json::json!({}),
    }
}

/// Builds a `tasks/get` request payload.
pub fn tasks_get_params(task_id: &str) -> serde_json::Value {
    serde_json::json!({ "taskId": task_id })
}

/// Parses a `tasks/get` or task-augmented `tools/call` result into an [`MCPTask`].
pub fn parse_task(result: &serde_json::Value) -> Option<MCPTask> {
    let task = result.get("task")?;
    let task_id = task.get("taskId").or_else(|| task.get("task_id"))?.as_str()?.to_string();
    let state = task
        .get("state")
        .or_else(|| task.get("status"))?
        .as_str()?;
    let state = serde_json::from_value::<MCPTaskState>(serde_json::Value::String(state.to_string()))
        .ok()?;

    Some(MCPTask {
        task_id,
        state,
        status_message: task
            .get("statusMessage")
            .or_else(|| task.get("status_message"))
            .and_then(|m| m.as_str())
            .map(|m| m.to_string()),
        result: task.get("result").cloned().filter(|r| !r.is_null()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_metadata() {
        assert_eq!(TASKS_EXTENSION, "io.modelcontextprotocol/tasks");
        assert_eq!(TASKS_PROTOCOL_VERSION, "2026-07-28");
    }

    #[test]
    fn test_should_use_task() {
        let disabled = TasksConfig::Disabled;
        assert!(!disabled.should_use_task(Some(true)));

        let always = TasksConfig::Always {
            poll_interval: Duration::from_secs(1),
        };
        assert!(always.should_use_task(None));
        assert!(always.should_use_task(Some(false)));

        let auto = TasksConfig::Auto {
            poll_interval: Duration::from_secs(1),
        };
        assert!(auto.should_use_task(Some(true)));
        assert!(!auto.should_use_task(Some(false)));
        assert!(!auto.should_use_task(None));
    }

    #[test]
    fn test_terminal_states() {
        assert!(MCPTaskState::Completed.is_terminal());
        assert!(MCPTaskState::Failed.is_terminal());
        assert!(MCPTaskState::Cancelled.is_terminal());
        assert!(MCPTaskState::InputDeclined.is_terminal());
        assert!(!MCPTaskState::Working.is_terminal());
        assert!(!MCPTaskState::InputRequired.is_terminal());
    }

    #[test]
    fn test_parse_task_result() {
        let parsed = parse_task(&serde_json::json!({
            "task": {
                "taskId": "t-1",
                "state": "completed",
                "result": {"content": [{"type": "text", "text": "done"}]}
            }
        }))
        .unwrap();

        assert_eq!(parsed.task_id, "t-1");
        assert_eq!(parsed.state, MCPTaskState::Completed);
        assert!(parsed.result.is_some());
    }

    #[test]
    fn test_parse_task_snake_case_fallback() {
        let parsed = parse_task(&serde_json::json!({
            "task": {"task_id": "t-2", "state": "working", "status_message": "crunching"}
        }))
        .unwrap();

        assert_eq!(parsed.task_id, "t-2");
        assert_eq!(parsed.state, MCPTaskState::Working);
        assert_eq!(parsed.status_message.as_deref(), Some("crunching"));
    }

    #[test]
    fn test_parse_missing_task_returns_none() {
        assert!(parse_task(&serde_json::json!({"other": 1})).is_none());
    }
}
