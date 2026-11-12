//! Shell and Python REPL tools routed through a [`Sandbox`].

use std::sync::Arc;

use async_trait::async_trait;

use crate::sandbox::{ExecOptions, Sandbox};
use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::errors::StrandsError;
use crate::types::tools::ToolSpec;

/// Maximum output returned to the model, in bytes.
pub const MAX_OUTPUT_BYTES: usize = 256 * 1024;

fn truncate_output(output: &str) -> String {
    if output.len() <= MAX_OUTPUT_BYTES {
        return output.to_string();
    }
    let cut: String = output.chars().take(MAX_OUTPUT_BYTES).collect();
    format!("{}\n...[output truncated]...", cut)
}

fn render_result(result: &crate::sandbox::ExecutionResult) -> String {
    let status = if result.is_success() { "succeeded" } else { "failed" };
    let mut text = format!(
        "Command {} with exit code {}.\n--- stdout ---\n{}\n--- stderr ---\n{}",
        status,
        result.exit_code,
        truncate_output(&result.stdout),
        truncate_output(&result.stderr),
    );
    for file in &result.output_files {
        text.push_str(&format!(
            "\n--- output file: {} ({}, {} bytes) ---",
            file.name,
            file.mime_type,
            file.content.len()
        ));
    }
    text
}

/// Runs a shell command inside the sandbox.
pub struct ShellTool {
    sandbox: Arc<dyn Sandbox>,
    default_timeout_secs: u64,
}

impl ShellTool {
    pub fn new(sandbox: Arc<dyn Sandbox>) -> Self {
        Self {
            sandbox,
            default_timeout_secs: 120,
        }
    }

    /// Overrides the default command timeout.
    pub fn with_default_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.default_timeout_secs = timeout_secs;
        self
    }
}

#[async_trait]
impl AgentTool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command in the sandbox environment and return its output."
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description()).with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "cwd": {"type": "string", "description": "Working directory"},
                "timeout_secs": {"type": "integer", "description": "Command timeout in seconds"}
            },
            "required": ["command"]
        }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let command = input
            .get("command")
            .and_then(|c| c.as_str())
            .ok_or("missing command")?;

        if context.is_cancelled() {
            return Ok(ToolResult2::error("Cancelled"));
        }

        let timeout_secs = input
            .get("timeout_secs")
            .and_then(|t| t.as_u64())
            .unwrap_or(self.default_timeout_secs);

        let options = ExecOptions {
            timeout: Some(std::time::Duration::from_secs(timeout_secs)),
            cwd: input
                .get("cwd")
                .and_then(|c| c.as_str())
                .map(|c| c.to_string()),
            env: Default::default(),
        };

        match self.sandbox.execute(command, options).await {
            Ok(result) => Ok(ToolResult2::success(render_result(&result))),
            Err(StrandsError::SandboxError { message }) => Ok(ToolResult2::error(message)),
            Err(e) => Ok(ToolResult2::error(e.to_string())),
        }
    }
}

/// Runs Python code through the sandbox; each call gets a fresh interpreter.
pub struct PythonReplTool {
    sandbox: Arc<dyn Sandbox>,
    interpreter: String,
}

impl PythonReplTool {
    pub fn new(sandbox: Arc<dyn Sandbox>) -> Self {
        Self {
            sandbox,
            interpreter: "python3".to_string(),
        }
    }

    /// Overrides the interpreter used to run the code.
    pub fn with_interpreter(mut self, interpreter: impl Into<String>) -> Self {
        self.interpreter = interpreter.into();
        self
    }
}

#[async_trait]
impl AgentTool for PythonReplTool {
    fn name(&self) -> &str {
        "python_repl"
    }

    fn description(&self) -> &str {
        "Execute Python code in the sandbox. Each call runs in a fresh interpreter; \
         print() to return output."
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description()).with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source code to execute"}
            },
            "required": ["code"]
        }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let code = input
            .get("code")
            .and_then(|c| c.as_str())
            .ok_or("missing code")?;

        if context.is_cancelled() {
            return Ok(ToolResult2::error("Cancelled"));
        }

        let options = ExecOptions {
            timeout: Some(std::time::Duration::from_secs(self.default_timeout())),
            ..Default::default()
        };

        match self.sandbox.execute_code(code, &self.interpreter, options).await {
            Ok(result) => Ok(ToolResult2::success(render_result(&result))),
            Err(StrandsError::SandboxError { message }) => Ok(ToolResult2::error(message)),
            Err(e) => Ok(ToolResult2::error(e.to_string())),
        }
    }
}

impl PythonReplTool {
    fn default_timeout(&self) -> u64 {
        120
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::{ExecutionResult, NotASandboxLocalEnvironment};
    use crate::types::tools::ToolUse;

    fn context() -> ToolContext {
        ToolContext::new(ToolUse::new("t", "t1", serde_json::json!({})))
    }

    fn sandbox() -> Arc<dyn Sandbox> {
        Arc::new(NotASandboxLocalEnvironment::new())
    }

    #[tokio::test]
    async fn test_shell_tool_runs_command() {
        let tool = ShellTool::new(sandbox());
        let result = tool
            .invoke(serde_json::json!({"command": "echo sandboxed"}), &context())
            .await
            .unwrap();
        assert!(result.content[0].text.as_deref().unwrap().contains("sandboxed"));
    }

    #[tokio::test]
    async fn test_shell_tool_reports_errors() {
        let tool = ShellTool::new(sandbox());
        let result = tool
            .invoke(
                serde_json::json!({"command": "echo boom >&2; exit 7", "timeout_secs": 5}),
                &context(),
            )
            .await
            .unwrap();
        let text = result.content[0].text.as_deref().unwrap();
        assert!(text.contains("exit code 7"));
        assert!(text.contains("boom"));
    }

    #[tokio::test]
    async fn test_python_repl_runs_code() {
        let tool = PythonReplTool::new(sandbox());
        let result = tool
            .invoke(
                serde_json::json!({"code": "print('repl works')"}),
                &context(),
            )
            .await
            .unwrap();
        let text = result.content[0].text.as_deref().unwrap().to_string();
        // The interpreter may be absent in the test environment; either the
        // code ran or the failure is reported cleanly.
        assert!(text.contains("repl works") || text.contains("failed"));
    }

    #[test]
    fn test_truncate_output() {
        let long = "x".repeat(MAX_OUTPUT_BYTES + 10);
        let truncated = truncate_output(&long);
        assert!(truncated.contains("...[output truncated]..."));
        assert_eq!(truncate_output("short"), "short");
    }

    #[test]
    fn test_render_result() {
        let result = ExecutionResult {
            exit_code: 0,
            stdout: "out".to_string(),
            stderr: String::new(),
            output_files: Vec::new(),
        };
        let rendered = render_result(&result);
        assert!(rendered.contains("succeeded"));
        assert!(rendered.contains("--- stdout ---"));
    }
}
