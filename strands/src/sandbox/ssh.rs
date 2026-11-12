//! SSH sandbox: runs commands on a remote host over `ssh`.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use tokio::process::Command;

use super::types::{build_shell_env_prefix, shell_quote, ExecOptions, ExecutionResult, SandboxEvent, StreamType};
use super::Sandbox;
use crate::types::errors::StrandsError;

/// Runs commands on a remote host through an `ssh` connection.
///
/// Environment variables travel as a shell `export` prefix because SSH
/// concatenates the command into a single remote shell string.
#[derive(Debug, Clone)]
pub struct SshSandbox {
    target: String,
    working_dir: Option<String>,
    ssh_args: Vec<String>,
}

impl SshSandbox {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
            working_dir: None,
            ssh_args: Vec::new(),
        }
    }

    /// Sets the default working directory on the remote host.
    pub fn with_working_dir(mut self, working_dir: impl Into<String>) -> Self {
        self.working_dir = Some(working_dir.into());
        self
    }

    /// Adds extra arguments to the ssh invocation (e.g. `-p 2222`, `-i key`).
    pub fn with_ssh_args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.ssh_args = args.into_iter().map(Into::into).collect();
        self
    }

    fn ssh_command(&self, command: &str, options: &ExecOptions) -> Result<String, StrandsError> {
        let env_prefix = build_shell_env_prefix(&options.env)
            .map_err(|message| StrandsError::SandboxError { message })?;

        let effective_cwd = options.cwd.as_ref().or(self.working_dir.as_ref());
        let remote = match effective_cwd {
            Some(cwd) => format!(
                "cd {} && {}{}",
                shell_quote(cwd),
                env_prefix,
                command
            ),
            None => format!("{}{}", env_prefix, command),
        };
        Ok(remote)
    }
}

#[async_trait]
impl Sandbox for SshSandbox {
    fn execute_streaming<'a>(
        &'a self,
        command: &'a str,
        options: ExecOptions,
    ) -> BoxStream<'a, Result<SandboxEvent, StrandsError>> {
        Box::pin(async_stream::stream! {
            let remote = match self.ssh_command(command, &options) {
                Ok(remote) => remote,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let mut ssh = Command::new("ssh");
            ssh.args(&self.ssh_args)
                .arg("-o")
                .arg("BatchMode=yes")
                .arg(&self.target)
                .arg("sh")
                .arg("-c")
                .arg(&remote)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);

            let mut child = match ssh.spawn() {
                Ok(child) => child,
                Err(e) => {
                    yield Err(StrandsError::SandboxError {
                        message: format!("failed to start ssh: {}", e),
                    });
                    return;
                }
            };

            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

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

            let timeout = options.timeout.unwrap_or(Duration::from_secs(86400));
            let exit_code = match tokio::time::timeout(timeout, child.wait()).await {
                Ok(Ok(status)) => status.code().unwrap_or(-1),
                Ok(Err(e)) => {
                    yield Err(StrandsError::SandboxError {
                        message: format!("ssh execution failed: {}", e),
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
    use std::collections::HashMap;

    #[test]
    fn test_ssh_command_shape() {
        let sandbox = SshSandbox::new("host.example")
            .with_working_dir("/srv")
            .with_ssh_args(["-p", "2222"]);

        let mut env = HashMap::new();
        env.insert("A".to_string(), "b c".to_string());
        let options = ExecOptions {
            env,
            ..Default::default()
        };

        let remote = sandbox.ssh_command("echo hi", &options).unwrap();
        assert!(remote.starts_with("cd /srv && export A='b c' && echo hi"));
    }

    #[test]
    fn test_ssh_command_rejects_bad_env() {
        let sandbox = SshSandbox::new("host");
        let options = ExecOptions {
            env: HashMap::from([("BAD KEY".to_string(), "v".to_string())]),
            ..Default::default()
        };
        assert!(sandbox.ssh_command("ls", &options).is_err());
    }
}
