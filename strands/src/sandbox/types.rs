//! Sandbox execution types.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Which output stream a chunk came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamType {
    Stdout,
    Stderr,
}

/// A typed chunk of streaming output from command or code execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamChunk {
    pub data: String,
    pub stream_type: StreamType,
}

impl StreamChunk {
    pub fn stdout(data: impl Into<String>) -> Self {
        Self { data: data.into(), stream_type: StreamType::Stdout }
    }

    pub fn stderr(data: impl Into<String>) -> Self {
        Self { data: data.into(), stream_type: StreamType::Stderr }
    }
}

/// Metadata about a file or directory in a sandbox.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileInfo {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_dir: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

/// A file produced as output by code execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputFile {
    pub name: String,
    pub content: Vec<u8>,
    #[serde(default = "default_mime_type")]
    pub mime_type: String,
}

fn default_mime_type() -> String {
    "application/octet-stream".to_string()
}

/// Result of command or code execution in a sandbox.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_files: Vec<OutputFile>,
}

impl ExecutionResult {
    pub fn is_success(&self) -> bool {
        self.exit_code == 0
    }
}

/// Options applied to one sandbox execution.
#[derive(Debug, Clone, Default)]
pub struct ExecOptions {
    /// Maximum execution time; `None` means no timeout.
    pub timeout: Option<Duration>,
    /// Working directory; `None` uses the sandbox default.
    pub cwd: Option<String>,
    /// Environment variables set for this command.
    pub env: HashMap<String, String>,
}

/// Events streamed by sandbox primitives.
#[derive(Debug, Clone)]
pub enum SandboxEvent {
    Chunk(StreamChunk),
    Result(ExecutionResult),
}

/// Validates an environment variable name against POSIX naming rules.
pub fn validate_env_key(key: &str) -> Result<(), String> {
    let mut chars = key.chars();
    let valid = match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {
            chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(format!("Invalid environment variable name: {}", key))
    }
}

/// Single-quotes a value for safe inclusion in a POSIX shell string.
pub fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', r#"'"'"'"#))
}

/// Builds a shell `export KEY=VALUE && ...` prefix, or an empty string.
///
/// Mirrors the Python `build_shell_env_prefix`: `export` is used so the
/// variables are inherited by every pipeline stage.
pub fn build_shell_env_prefix(env: &HashMap<String, String>) -> Result<String, String> {
    if env.is_empty() {
        return Ok(String::new());
    }
    let mut prefix = String::new();
    for (key, value) in env {
        validate_env_key(key)?;
        prefix.push_str(&format!(
            "export {}={} && ",
            key,
            shell_quote(value)
        ));
    }
    Ok(prefix)
}

/// Validates a language identifier before interpolating it into a shell command.
pub fn validate_language(language: &str) -> Result<(), String> {
    let valid = !language.is_empty()
        && language
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '-' | '.'));
    if valid {
        Ok(())
    } else {
        Err(format!("Invalid language identifier: {}", language))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_env_key() {
        assert!(validate_env_key("PATH").is_ok());
        assert!(validate_env_key("_A1").is_ok());
        assert!(validate_env_key("1BAD").is_err());
        assert!(validate_env_key("BAD KEY").is_err());
        assert!(validate_env_key("").is_err());
    }

    #[test]
    fn test_shell_quote() {
        assert_eq!(shell_quote("simple"), "simple");
        assert_eq!(shell_quote(""), "''");
        assert_eq!(shell_quote("has space"), "'has space'");
        assert_eq!(shell_quote("it's"), r#"'it'"'"'s'"#);
    }

    #[test]
    fn test_build_shell_env_prefix() {
        let mut env = HashMap::new();
        env.insert("FOO".to_string(), "bar baz".to_string());
        let prefix = build_shell_env_prefix(&env).unwrap();
        assert_eq!(prefix, "export FOO='bar baz' && ");

        assert_eq!(build_shell_env_prefix(&HashMap::new()).unwrap(), "");

        let mut bad = HashMap::new();
        bad.insert("BAD KEY".to_string(), "v".to_string());
        assert!(build_shell_env_prefix(&bad).is_err());
    }

    #[test]
    fn test_validate_language() {
        assert!(validate_language("python3").is_ok());
        assert!(validate_language("node").is_ok());
        assert!(validate_language("python; rm -rf /").is_err());
        assert!(validate_language("").is_err());
    }
}
