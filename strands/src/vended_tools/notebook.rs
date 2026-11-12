//! Notebook vended tool: text notebooks stored in invocation state.

use async_trait::async_trait;

use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::tools::ToolSpec;

/// Notebook state key inside invocation state.
const NOTEBOOKS_KEY: &str = "notebooks";

/// Create, read, write, append, and clear text notebooks.
///
/// Notebooks live in the agent's invocation state under `notebooks`, keyed
/// by notebook name, mirroring the Python `notebook` vended tool.
pub struct NotebookTool;

impl NotebookTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NotebookTool {
    fn default() -> Self {
        Self::new()
    }
}

fn read_notebooks(context: &ToolContext) -> serde_json::Map<String, serde_json::Value> {
    context
        .invocation_state
        .get::<serde_json::Map<String, serde_json::Value>>(NOTEBOOKS_KEY)
        .unwrap_or_default()
}

fn write_notebooks(context: &ToolContext, notebooks: serde_json::Map<String, serde_json::Value>) {
    context
        .invocation_state
        .set(NOTEBOOKS_KEY, serde_json::Value::Object(notebooks));
}

#[async_trait]
impl AgentTool for NotebookTool {
    fn name(&self) -> &str {
        "notebook"
    }

    fn description(&self) -> &str {
        "Manage text notebooks stored in agent state. Commands: create, read, \
         write, append, clear, list."
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description()).with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Notebook command",
                    "enum": ["create", "read", "write", "append", "clear", "list"]
                },
                "notebook_name": {"type": "string", "description": "Notebook name"},
                "content": {"type": "string", "description": "Content for write/append"}
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
        let notebook_name = input.get("notebook_name").and_then(|n| n.as_str());
        let content = input.get("content").and_then(|c| c.as_str()).unwrap_or("");

        let mut notebooks = read_notebooks(context);

        match command {
            "list" => {
                write_notebooks(context, notebooks);
                let notebooks = read_notebooks(context);
                let names: Vec<String> = notebooks.keys().cloned().collect();
                Ok(ToolResult2::success_json(serde_json::json!({
                    "notebooks": names,
                })))
            }
            "create" => {
                let name = notebook_name.ok_or("missing notebook_name")?;
                if notebooks.contains_key(name) {
                    return Err(format!("notebook '{}' already exists", name));
                }
                notebooks.insert(name.to_string(), serde_json::json!(""));
                write_notebooks(context, notebooks);
                Ok(ToolResult2::success(format!("Created notebook '{}'", name)))
            }
            "read" => {
                let name = notebook_name.ok_or("missing notebook_name")?;
                let text = notebooks
                    .get(name)
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| format!("notebook '{}' not found", name))?;
                Ok(ToolResult2::success_json(serde_json::json!({
                    "name": name,
                    "content": text,
                })))
            }
            "write" => {
                let name = notebook_name.ok_or("missing notebook_name")?;
                if !notebooks.contains_key(name) {
                    return Err(format!("notebook '{}' not found", name));
                }
                notebooks.insert(name.to_string(), serde_json::json!(content));
                write_notebooks(context, notebooks);
                Ok(ToolResult2::success(format!("Wrote notebook '{}'", name)))
            }
            "append" => {
                let name = notebook_name.ok_or("missing notebook_name")?;
                let existing = notebooks
                    .get(name)
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| format!("notebook '{}' not found", name))?
                    .to_string();
                let updated = if existing.is_empty() {
                    content.to_string()
                } else {
                    format!("{}\n{}", existing, content)
                };
                notebooks.insert(name.to_string(), serde_json::json!(updated));
                write_notebooks(context, notebooks);
                Ok(ToolResult2::success(format!("Appended to notebook '{}'", name)))
            }
            "clear" => {
                let name = notebook_name.ok_or("missing notebook_name")?;
                if !notebooks.contains_key(name) {
                    return Err(format!("notebook '{}' not found", name));
                }
                notebooks.insert(name.to_string(), serde_json::json!(""));
                write_notebooks(context, notebooks);
                Ok(ToolResult2::success(format!("Cleared notebook '{}'", name)))
            }
            other => Err(format!("unknown notebook command: {}", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tools::ToolUse;

    fn context() -> ToolContext {
        ToolContext::new(ToolUse::new("t", "t1", serde_json::json!({})))
    }

    #[tokio::test]
    async fn test_notebook_lifecycle() {
        let tool = NotebookTool::new();
        let context = context();

        tool.invoke(
            serde_json::json!({"command": "create", "notebook_name": "notes"}),
            &context,
        )
        .await
        .unwrap();

        tool.invoke(
            serde_json::json!({"command": "append", "notebook_name": "notes", "content": "line one"}),
            &context,
        )
        .await
        .unwrap();
        tool.invoke(
            serde_json::json!({"command": "append", "notebook_name": "notes", "content": "line two"}),
            &context,
        )
        .await
        .unwrap();

        let read = tool
            .invoke(serde_json::json!({"command": "read", "notebook_name": "notes"}), &context)
            .await
            .unwrap();
        let payload = read.content[0].json.as_ref().unwrap();
        assert_eq!(payload["content"], "line one\nline two");

        let listed = tool.invoke(serde_json::json!({"command": "list"}), &context).await.unwrap();
        assert_eq!(listed.content[0].json.as_ref().unwrap()["notebooks"][0], "notes");

        tool.invoke(serde_json::json!({"command": "clear", "notebook_name": "notes"}), &context)
            .await
            .unwrap();
        let cleared = tool
            .invoke(serde_json::json!({"command": "read", "notebook_name": "notes"}), &context)
            .await
            .unwrap();
        assert_eq!(cleared.content[0].json.as_ref().unwrap()["content"], "");
    }

    #[tokio::test]
    async fn test_create_rejects_duplicate() {
        let tool = NotebookTool::new();
        let context = context();

        tool.invoke(serde_json::json!({"command": "create", "notebook_name": "n"}), &context)
            .await
            .unwrap();
        let error = tool
            .invoke(serde_json::json!({"command": "create", "notebook_name": "n"}), &context)
            .await
            .unwrap_err();
        assert!(error.contains("already exists"));
    }

    #[tokio::test]
    async fn test_read_missing_notebook_errors() {
        let tool = NotebookTool::new();
        let error = tool
            .invoke(serde_json::json!({"command": "read", "notebook_name": "ghost"}), &context())
            .await
            .unwrap_err();
        assert!(error.contains("not found"));
    }
}
