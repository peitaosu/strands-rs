//! Docker container sandbox.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use tokio::process::Command;

use super::types::{validate_env_key, ExecOptions, ExecutionResult, SandboxEvent, StreamType};
use super::Sandbox;
use crate::types::errors::StrandsError;

/// Runs commands inside an existing Docker container via `docker exec`.
#[derive(Debug, Clone)]
pub struct DockerSandbox {
    container: String,
    working_dir: Option<String>,
    user: Option<String>,
}

impl DockerSandbox {
    pub fn new(container: impl Into<String>) -> Self {
        Self {
            container: container.into(),
            working_dir: None,
            user: None,
        }
    }

    /// Sets the working directory inside the container.
    pub fn with_working_dir(mut self, working_dir: impl Into<String>) -> Self {
        self.working_dir = Some(working_dir.into());
        self
    }

    /// Sets the user the commands run as.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    fn docker_exec(&self, command: &str, options: &ExecOptions) -> Result<Command, StrandsError> {
        for key in options.env.keys() {
            validate_env_key(key).map_err(|message| StrandsError::SandboxError { message })?;
        }

        let mut exec = Command::new("docker");
        exec.arg("exec").arg("-i");
        if let Some(user) = &self.user {
            exec.arg("-u").arg(user);
        }
        let working_dir = options.cwd.as_ref().or(self.working_dir.as_ref());
        if let Some(working_dir) = working_dir {
            exec.arg("-w").arg(working_dir);
        }
        for (key, value) in &options.env {
            exec.arg("-e").arg(format!("{}={}", key, value));
        }
        exec.arg(&self.container).arg("sh").arg("-c").arg(command);
        Ok(exec)
    }
}

#[async_trait]
impl Sandbox for DockerSandbox {
    fn execute_streaming<'a>(
        &'a self,
        command: &'a str,
        options: ExecOptions,
    ) -> BoxStream<'a, Result<SandboxEvent, StrandsError>> {
        Box::pin(async_stream::stream! {
            let mut exec = match self.docker_exec(command, &options) {
                Ok(exec) => exec,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            exec.stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);

            let mut child = match exec.spawn() {
                Ok(child) => child,
                Err(e) => {
                    yield Err(StrandsError::SandboxError {
                        message: format!("failed to start docker exec: {}", e),
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
                        message: format!("docker exec failed: {}", e),
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

    #[test]
    fn test_docker_exec_command_shape() {
        let sandbox = DockerSandbox::new("web")
            .with_working_dir("/app")
            .with_user("node");
        let options = ExecOptions {
            env: std::collections::HashMap::from([("FOO".to_string(), "bar".to_string())]),
            ..Default::default()
        };

        let command = sandbox.docker_exec("echo hi", &options).unwrap();
        let program = command.as_std().get_program().to_string_lossy().to_string();
        let args: Vec<String> = command
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();

        assert_eq!(program, "docker");
        assert!(args.contains(&"exec".to_string()));
        assert!(args.contains(&"-u".to_string()) && args.contains(&"node".to_string()));
        assert!(args.contains(&"-w".to_string()) && args.contains(&"/app".to_string()));
        assert!(args.contains(&"-e".to_string()) && args.contains(&"FOO=bar".to_string()));
        assert!(args.contains(&"web".to_string()));
        assert!(args.contains(&"echo hi".to_string()));
    }

    #[test]
    fn test_docker_exec_rejects_bad_env_key() {
        let sandbox = DockerSandbox::new("web");
        let options = ExecOptions {
            env: std::collections::HashMap::from([("BAD KEY".to_string(), "v".to_string())]),
            ..Default::default()
        };
        assert!(sandbox.docker_exec("ls", &options).is_err());
    }
}
