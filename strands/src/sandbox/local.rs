//! Host execution environment used as the default when no sandbox is set.
//!
//! `NotASandboxLocalEnvironment` runs commands directly on the host with
//! **no isolation**. The deliberately blunt name mirrors the Python and
//! TypeScript SDKs: this is the fallback, not a security boundary.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use tokio::process::Command;

use super::types::{
    build_shell_env_prefix, shell_quote, ExecOptions, ExecutionResult, SandboxEvent, StreamType,
};
use super::Sandbox;
use crate::types::errors::StrandsError;

/// Local host execution environment without isolation.
#[derive(Debug, Clone, Default)]
pub struct NotASandboxLocalEnvironment;

impl NotASandboxLocalEnvironment {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Sandbox for NotASandboxLocalEnvironment {
    fn execute_streaming<'a>(
        &'a self,
        command: &'a str,
        options: ExecOptions,
    ) -> BoxStream<'a, Result<SandboxEvent, StrandsError>> {
        Box::pin(async_stream::stream! {
            let env_prefix = match build_shell_env_prefix(&options.env) {
                Ok(prefix) => prefix,
                Err(message) => {
                    yield Err(StrandsError::SandboxError { message });
                    return;
                }
            };

            let full_command = match &options.cwd {
                Some(cwd) => format!("cd {} && {}{}", shell_quote(cwd), env_prefix, command),
                None => format!("{}{}", env_prefix, command),
            };

            let mut child = match Command::new("sh")
                .arg("-c")
                .arg(&full_command)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .spawn()
            {
                Ok(child) => child,
                Err(e) => {
                    yield Err(StrandsError::SandboxError {
                        message: format!("failed to spawn shell: {}", e),
                    });
                    return;
                }
            };

            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            let timeout = options.timeout.unwrap_or(Duration::from_secs(86400));

            let ran = tokio::time::timeout(
                timeout,
                async {
                    let (mut stdout_events, stderr_events) = tokio::join!(
                        async {
                            match stdout {
                                Some(pipe) => super::read_pipe(pipe, StreamType::Stdout).await,
                                None => Vec::new(),
                            }
                        },
                        async {
                            match stderr {
                                Some(pipe) => super::read_pipe(pipe, StreamType::Stderr).await,
                                None => Vec::new(),
                            }
                        }
                    );
                    stdout_events.extend(stderr_events);
                    let exit_code: Result<i32, std::io::Error> = match child.wait().await {
                        Ok(status) => Ok(status.code().unwrap_or(-1)),
                        Err(e) => Err(e),
                    };
                    (stdout_events, exit_code)
                },
            )
            .await;

            let (stdout_events, exit_code) = match ran {
                Ok((events, Ok(code))) => (events, code),
                Ok((_, Err(e))) => {
                    yield Err(StrandsError::SandboxError {
                        message: format!("failed to wait for shell: {}", e),
                    });
                    return;
                }
                Err(_) => {
                    let _ = child.kill().await;
                    yield Err(StrandsError::SandboxError {
                        message: format!("command timed out after {}s", timeout.as_secs()),
                    });
                    return;
                }
            };

            for event in stdout_events {
                yield Ok(event);
            }

            yield Ok(SandboxEvent::Result(ExecutionResult {
                exit_code,
                stdout: String::new(),
                stderr: String::new(),
                output_files: Vec::new(),
            }));
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_execute_echo() {
        let result = NotASandboxLocalEnvironment::new()
            .execute("echo hello", ExecOptions::default())
            .await
            .unwrap();
        assert!(result.is_success());
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_execute_captures_stderr_and_exit_code() {
        let result = NotASandboxLocalEnvironment::new()
            .execute("echo out; echo err >&2; exit 3", ExecOptions::default())
            .await
            .unwrap();
        assert_eq!(result.exit_code, 3);
        assert!(result.stdout.contains("out"));
        assert!(result.stderr.contains("err"));
    }

    #[tokio::test]
    async fn test_execute_env_and_cwd() {
        let mut env = std::collections::HashMap::new();
        env.insert("SANDBOX_TEST_VAR".to_string(), "set".to_string());

        let options = ExecOptions {
            cwd: Some("/tmp".to_string()),
            env,
            ..Default::default()
        };

        let result = NotASandboxLocalEnvironment::new()
            .execute("echo $SANDBOX_TEST_VAR; pwd", options)
            .await
            .unwrap();
        assert!(result.stdout.contains("set"));
        assert!(result.stdout.contains("/tmp"));
    }

    #[tokio::test]
    async fn test_execute_timeout() {
        let options = ExecOptions {
            timeout: Some(Duration::from_millis(100)),
            ..Default::default()
        };
        let error = NotASandboxLocalEnvironment::new()
            .execute("sleep 5", options)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn test_streaming_yields_chunks_then_result() {
        let sandbox = NotASandboxLocalEnvironment::new();
        let mut stream = sandbox.execute_streaming("echo streamed", ExecOptions::default());
        let mut saw_chunk = false;
        let mut saw_result = false;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                SandboxEvent::Chunk(_) => saw_chunk = true,
                SandboxEvent::Result(_) => saw_result = true,
            }
        }
        assert!(saw_chunk && saw_result);
    }
}
