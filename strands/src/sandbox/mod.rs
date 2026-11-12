//! Sandbox abstraction: isolated execution environments for tools.
//!
//! Skeletal port of the Python `strands.sandbox` package: a `Sandbox` trait
//! with command execution as the primitive, shell-based defaults for code
//! execution and file operations, and three backends — the local host
//! (`NotASandboxLocalEnvironment`, explicitly *not* a security boundary),
//! Docker containers, and SSH targets.

pub mod docker;
pub mod local;
pub mod ssh;
pub mod types;

pub use docker::DockerSandbox;
pub use local::NotASandboxLocalEnvironment;
pub use ssh::SshSandbox;
pub use types::{
    build_shell_env_prefix, shell_quote, validate_env_key, validate_language, ExecOptions,
    ExecutionResult, FileInfo, OutputFile, SandboxEvent, StreamChunk, StreamType,
};

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use futures::stream::{BoxStream, StreamExt};

use crate::types::errors::StrandsError;

/// Marker prepended to code piped over a quoted heredoc.
pub const CODE_HEREDOC_MARKER: &str = "SANDBOX_CODE_EOF";

/// Abstract execution environment.
///
/// `execute_streaming` is the single primitive; the other operations default
/// to shell-based implementations piped through it. Backends override them
/// natively where the transport allows.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// Executes a shell command, streaming stdout/stderr chunks and ending
    /// with the final [`SandboxEvent::Result`].
    fn execute_streaming<'a>(
        &'a self,
        command: &'a str,
        options: ExecOptions,
    ) -> BoxStream<'a, Result<SandboxEvent, StrandsError>>;

    /// Consumes the stream and returns the final result.
    async fn execute(
        &self,
        command: &str,
        options: ExecOptions,
    ) -> Result<ExecutionResult, StrandsError> {
        let stream = self.execute_streaming(command, options);
        consume_to_result(stream).await
    }

    /// Executes code by base64-piping it to a language interpreter.
    ///
    /// The code is base64-encoded and decoded inside a quoted heredoc on the
    /// target, so shell metacharacters, quotes, and newlines travel without
    /// injection risk. The language is validated first.
    async fn execute_code(
        &self,
        code: &str,
        language: &str,
        options: ExecOptions,
    ) -> Result<ExecutionResult, StrandsError> {
        validate_language(language).map_err(|message| StrandsError::SandboxError { message })?;

        let encoded = base64::engine::general_purpose::STANDARD.encode(code.as_bytes());
        let command = format!(
            "base64 -d << '{}'\n{}\n{} | {}",
            CODE_HEREDOC_MARKER, encoded, CODE_HEREDOC_MARKER, language
        );

        self.execute(&command, options).await
    }

    /// Reads a file as raw bytes via base64 over the shell.
    ///
    /// Locale-independent sentinel exit codes: 77 missing, 78 directory,
    /// 79 permission denied.
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, StrandsError> {
        let quoted = shell_quote(path);
        let command = format!(
            "if [ -d {q} ]; then exit 78; elif [ ! -e {q} ]; then exit 77; elif [ ! -r {q} ]; then exit 79; fi; \
             base64 {q}",
            q = quoted
        );

        let result = self.execute(&command, ExecOptions::default()).await?;
        if result.exit_code == 77 {
            return Err(StrandsError::SandboxError {
                message: format!("path not found: {}", path),
            });
        }
        if result.exit_code == 78 {
            return Err(StrandsError::SandboxError {
                message: format!("path is a directory: {}", path),
            });
        }
        if result.exit_code == 79 {
            return Err(StrandsError::SandboxError {
                message: format!("permission denied: {}", path),
            });
        }
        if result.exit_code != 0 {
            return Err(StrandsError::SandboxError {
                message: format!("failed to read {}: {}", path, result.stderr),
            });
        }

        base64::engine::general_purpose::STANDARD
            .decode(result.stdout.trim().as_bytes())
            .map_err(|e| StrandsError::SandboxError {
                message: format!("invalid base64 output reading {}: {}", path, e),
            })
    }

    /// Writes raw bytes to a file via base64 over the shell.
    async fn write_file(&self, path: &str, content: &[u8]) -> Result<(), StrandsError> {
        let encoded = base64::engine::general_purpose::STANDARD.encode(content);
        let command = format!(
            "mkdir -p \"$(dirname {})\" && base64 -d << '{}' > {}\n{}\n{}",
            shell_quote(path),
            CODE_HEREDOC_MARKER,
            shell_quote(path),
            encoded,
            CODE_HEREDOC_MARKER,
        );

        let result = self.execute(&command, ExecOptions::default()).await?;
        if result.exit_code != 0 {
            return Err(StrandsError::SandboxError {
                message: format!("failed to write {}: {}", path, result.stderr),
            });
        }
        Ok(())
    }

    /// Removes a file, ignoring a missing target.
    async fn remove_file(&self, path: &str) -> Result<(), StrandsError> {
        let command = format!("rm -f {}", shell_quote(path));
        let result = self.execute(&command, ExecOptions::default()).await?;
        if result.exit_code != 0 {
            return Err(StrandsError::SandboxError {
                message: format!("failed to remove {}: {}", path, result.stderr),
            });
        }
        Ok(())
    }

    /// Lists a directory's entries as `FileInfo` records.
    async fn list_files(&self, path: &str) -> Result<Vec<FileInfo>, StrandsError> {
        let command = format!(
            "for entry in {}/* {}/.*; do \
               [ -e \"$entry\" ] || continue; \
               name=$(basename \"$entry\"); \
               if [ -d \"$entry\" ]; then echo \"$name|dir|-\"; \
               else echo \"$name|file|$(stat -c %s \"$entry\" 2>/dev/null || echo -)\"; fi; \
             done | sort -u",
            shell_quote(path),
            shell_quote(path)
        );

        let result = self.execute(&command, ExecOptions::default()).await?;
        if result.exit_code != 0 {
            return Err(StrandsError::SandboxError {
                message: format!("failed to list {}: {}", path, result.stderr),
            });
        }

        Ok(result
            .stdout
            .lines()
            .filter_map(|line| {
                let mut parts = line.trim().splitn(3, '|');
                let name = parts.next()?.to_string();
                let is_dir = parts.next().map(|kind| kind == "dir");
                let size = parts.next().and_then(|size| size.parse::<u64>().ok());
                Some(FileInfo {
                    name,
                    is_dir,
                    size,
                })
            })
            .collect())
    }
}

/// Drains a sandbox stream into its final [`ExecutionResult`], accumulating
/// chunk output along the way.
pub async fn consume_to_result(
    mut stream: BoxStream<'_, Result<SandboxEvent, StrandsError>>,
) -> Result<ExecutionResult, StrandsError> {
    let mut stdout = String::new();
    let mut stderr = String::new();
    let mut final_result: Option<ExecutionResult> = None;

    while let Some(event) = stream.next().await {
        match event? {
            SandboxEvent::Chunk(chunk) => match chunk.stream_type {
                StreamType::Stdout => stdout.push_str(&chunk.data),
                StreamType::Stderr => stderr.push_str(&chunk.data),
            },
            SandboxEvent::Result(result) => {
                final_result = Some(result);
            }
        }
    }

    final_result
        .map(|mut result| {
            if result.stdout.is_empty() {
                result.stdout = stdout;
            }
            if result.stderr.is_empty() {
                result.stderr = stderr;
            }
            result
        })
        .ok_or_else(|| StrandsError::SandboxError {
            message: "sandbox stream ended without a result".to_string(),
        })
}

/// Type alias for shared sandbox handles.
pub type SharedSandbox = Arc<dyn Sandbox>;

/// Reads one output pipe into chunk events until EOF.
pub(crate) async fn read_pipe(
    mut pipe: impl tokio::io::AsyncRead + Unpin,
    stream_type: StreamType,
) -> Vec<SandboxEvent> {
    use tokio::io::AsyncReadExt;

    let mut events = Vec::new();
    let mut buffer = vec![0u8; 8192];
    loop {
        match pipe.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(read) => events.push(SandboxEvent::Chunk(StreamChunk {
                data: String::from_utf8_lossy(&buffer[..read]).to_string(),
                stream_type,
            })),
        }
    }
    events
}
