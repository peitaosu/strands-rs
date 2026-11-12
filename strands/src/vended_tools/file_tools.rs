//! File-manipulation vended tools: read, write, and edit.
//!
//! All paths are resolved against a fixed workspace root; keys escaping the
//! root are rejected.

use std::path::PathBuf;

use async_trait::async_trait;

use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::tools::ToolSpec;

/// Resolves a relative path under `root`, rejecting escapes.
fn resolve_under_root(root: &PathBuf, path: &str) -> Result<PathBuf, String> {
    let relative = path.trim_start_matches('/');
    if relative.is_empty() {
        return Err("path must not be empty".to_string());
    }
    let resolved = root.join(relative);
    let parent = resolved
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| root.clone());
    let canonical_parent = parent.canonicalize().unwrap_or(parent);
    if !canonical_parent.starts_with(root) {
        return Err(format!("path escapes the workspace root: {}", path));
    }
    Ok(resolved)
}

/// Canonicalizes the workspace root so containment checks work on macOS
/// where `/tmp` is a symlink.
fn canonical_root(root: impl Into<PathBuf>) -> PathBuf {
    let root = root.into();
    root.canonicalize().unwrap_or(root)
}

fn path_spec(name: &str, description: &str, properties: serde_json::Value) -> ToolSpec {
    ToolSpec::new(name, description).with_input_schema(serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": ["path"]
    }))
}

/// Reads a file with optional line range, `cat -n` style numbering.
pub struct ReadFileTool {
    root: PathBuf,
}

impl ReadFileTool {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: canonical_root(root) }
    }
}

#[async_trait]
impl AgentTool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a file from the workspace. Returns the content with line numbers."
    }

    fn tool_spec(&self) -> ToolSpec {
        path_spec(
            self.name(),
            self.description(),
            serde_json::json!({
                "path": {"type": "string", "description": "File path relative to the workspace root"},
                "offset": {"type": "integer", "description": "First line to return, 1-based"},
                "limit": {"type": "integer", "description": "Maximum number of lines to return"}
            }),
        )
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let path = input
            .get("path")
            .and_then(|p| p.as_str())
            .ok_or("missing path")?;
        let resolved = resolve_under_root(&self.root, path)?;
        let bytes = tokio::fs::read(&resolved)
            .await
            .map_err(|e| format!("failed to read {}: {}", path, e))?;
        let text = String::from_utf8_lossy(&bytes);

        let offset = input.get("offset").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
        let limit = input.get("limit").and_then(|v| v.as_u64()).map(|v| v as usize);

        let numbered: Vec<String> = text
            .lines()
            .skip(offset - 1)
            .take(limit.unwrap_or(usize::MAX))
            .enumerate()
            .map(|(index, line)| format!("{:>6}\t{}", offset + index, line))
            .collect();

        Ok(ToolResult2::success(if numbered.is_empty() {
            String::new()
        } else {
            numbered.join("\n")
        }))
    }
}

/// Creates or overwrites a file in the workspace.
pub struct WriteFileTool {
    root: PathBuf,
}

impl WriteFileTool {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: canonical_root(root) }
    }
}

#[async_trait]
impl AgentTool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Create or overwrite a file in the workspace with the given content."
    }

    fn tool_spec(&self) -> ToolSpec {
        path_spec(
            self.name(),
            self.description(),
            serde_json::json!({
                "path": {"type": "string", "description": "File path relative to the workspace root"},
                "content": {"type": "string", "description": "Full file content"}
            }),
        )
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let path = input
            .get("path")
            .and_then(|p| p.as_str())
            .ok_or("missing path")?;
        let content = input
            .get("content")
            .and_then(|c| c.as_str())
            .ok_or("missing content")?;

        let resolved = resolve_under_root(&self.root, path)?;
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("failed to create directories: {}", e))?;
        }
        tokio::fs::write(&resolved, content)
            .await
            .map_err(|e| format!("failed to write {}: {}", path, e))?;

        Ok(ToolResult2::success(format!("Wrote {} bytes to {}", content.len(), path)))
    }
}

/// Replaces text in a file; `old_str` must match exactly once.
pub struct EditFileTool {
    root: PathBuf,
}

impl EditFileTool {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: canonical_root(root) }
    }
}

#[async_trait]
impl AgentTool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace text in a workspace file. old_str must appear exactly once."
    }

    fn tool_spec(&self) -> ToolSpec {
        path_spec(
            self.name(),
            self.description(),
            serde_json::json!({
                "path": {"type": "string", "description": "File path relative to the workspace root"},
                "old_str": {"type": "string", "description": "Text to replace"},
                "new_str": {"type": "string", "description": "Replacement text"}
            }),
        )
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let path = input
            .get("path")
            .and_then(|p| p.as_str())
            .ok_or("missing path")?;
        let old_str = input
            .get("old_str")
            .and_then(|s| s.as_str())
            .ok_or("missing old_str")?;
        let new_str = input
            .get("new_str")
            .and_then(|s| s.as_str())
            .ok_or("missing new_str")?;

        let resolved = resolve_under_root(&self.root, path)?;
        let content = tokio::fs::read_to_string(&resolved)
            .await
            .map_err(|e| format!("failed to read {}: {}", path, e))?;

        let matches = content.matches(old_str).count();
        if matches == 0 {
            return Err(format!("old_str not found in {}", path));
        }
        if matches > 1 {
            return Err(format!("old_str appears {} times in {}; it must be unique", matches, path));
        }

        let updated = content.replacen(old_str, new_str, 1);
        tokio::fs::write(&resolved, &updated)
            .await
            .map_err(|e| format!("failed to write {}: {}", path, e))?;

        Ok(ToolResult2::success(format!("Edited {}", path)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tools::ToolUse;

    fn context() -> ToolContext {
        ToolContext::new(ToolUse::new("t", "t1", serde_json::json!({})))
    }

    fn temp_root() -> PathBuf {
        let root = std::env::temp_dir().join(format!("strands-vt-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[tokio::test]
    async fn test_write_then_read() {
        let root = temp_root();
        let write = WriteFileTool::new(&root);
        let read = ReadFileTool::new(&root);
        let context = context();

        write
            .invoke(serde_json::json!({"path": "src/main.rs", "content": "fn main() {}\n"}), &context)
            .await
            .unwrap();

        let read_result = read
            .invoke(serde_json::json!({"path": "src/main.rs"}), &context)
            .await
            .unwrap();
        assert!(read_result.content[0].text.as_ref().unwrap().contains("fn main()"));

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn test_read_line_numbers_and_range() {
        let root = temp_root();
        let write = WriteFileTool::new(&root);
        let read = ReadFileTool::new(&root);
        let context = context();

        write
            .invoke(serde_json::json!({"path": "lines.txt", "content": "a\nb\nc"}), &context)
            .await
            .unwrap();

        let result = read
            .invoke(serde_json::json!({"path": "lines.txt", "offset": 2, "limit": 1}), &context)
            .await
            .unwrap();
        assert_eq!(result.content[0].text.as_ref().unwrap().trim_end(), "     2\tb");

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn test_edit_requires_unique_match() {
        let root = temp_root();
        let write = WriteFileTool::new(&root);
        let edit = EditFileTool::new(&root);
        let context = context();

        write
            .invoke(serde_json::json!({"path": "f.txt", "content": "alpha beta"}), &context)
            .await
            .unwrap();

        let error = edit
            .invoke(serde_json::json!({"path": "f.txt", "old_str": "a", "new_str": "z"}), &context)
            .await
            .unwrap_err();
        assert!(error.contains("must be unique"));

        edit.invoke(
            serde_json::json!({"path": "f.txt", "old_str": "alpha", "new_str": "gamma"}),
            &context,
        )
        .await
        .unwrap();

        let content = std::fs::read_to_string(root.join("f.txt")).unwrap();
        assert_eq!(content, "gamma beta");

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn test_rejects_path_escape() {
        let root = temp_root();
        let read = ReadFileTool::new(&root);
        let context = context();

        let error = read
            .invoke(serde_json::json!({"path": "../../etc/passwd"}), &context)
            .await
            .unwrap_err();
        assert!(error.contains("escapes") || error.contains("failed to read"));

        std::fs::remove_dir_all(&root).unwrap();
    }
}
