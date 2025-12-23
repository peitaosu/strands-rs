//! A2A-compatible server for Strands Agent.
//!
//! This module provides the A2AServer, which wraps a Strands Agent
//! and exposes it via the A2A protocol.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use super::executor::StrandsA2AExecutor;
use super::types::{
    A2AError, A2ARequest, A2AResponse, A2ATask, A2ATaskState, AgentCard, AgentSkill,
};
use crate::agent::Agent;

/// Configuration for the A2A server.
#[derive(Debug, Clone)]
pub struct A2AServerConfig {
    /// Host to bind the server to.
    pub host: String,
    /// Port to bind the server to.
    pub port: u16,
    /// Public HTTP URL where the agent is accessible.
    pub http_url: Option<String>,
    /// Whether to serve at root path.
    pub serve_at_root: bool,
    /// Version of the agent.
    pub version: String,
    /// Skills exposed by the agent.
    pub skills: Vec<AgentSkill>,
}

impl Default for A2AServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 9000,
            http_url: None,
            serve_at_root: false,
            version: "0.0.1".to_string(),
            skills: Vec::new(),
        }
    }
}

impl A2AServerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    pub fn with_skill(mut self, skill: AgentSkill) -> Self {
        self.skills.push(skill);
        self
    }
}

/// A2A-compatible server wrapping a Strands Agent.
pub struct A2AServer {
    config: A2AServerConfig,
    executor: Arc<StrandsA2AExecutor>,
    agent_name: String,
    agent_description: Option<String>,
    tasks: Arc<RwLock<HashMap<String, A2ATask>>>,
}

impl A2AServer {
    /// Create a new A2A server from a Strands Agent.
    pub fn new(agent: Agent, config: A2AServerConfig) -> Self {
        let agent_name = agent.name().map(|s| s.to_string()).unwrap_or_else(|| "Strands Agent".to_string());
        let agent_description = None;

        Self {
            config,
            executor: Arc::new(StrandsA2AExecutor::new(agent)),
            agent_name,
            agent_description,
            tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the agent card describing this agent.
    pub fn agent_card(&self) -> AgentCard {
        let url = self
            .config
            .http_url
            .clone()
            .unwrap_or_else(|| format!("http://{}:{}/", self.config.host, self.config.port));

        let mut card = AgentCard::new(&self.agent_name, url, &self.config.version)
            .with_streaming(true);

        if let Some(desc) = &self.agent_description {
            card = card.with_description(desc);
        }

        if !self.config.skills.is_empty() {
            card = card.with_skills(self.config.skills.clone());
        }

        card
    }

    /// Handle an A2A JSON-RPC request.
    pub async fn handle_request(&self, request: A2ARequest) -> A2AResponse {
        match request.method.as_str() {
            "agent/card" => self.handle_agent_card(request.id).await,
            "tasks/send" => self.handle_tasks_send(request).await,
            "tasks/get" => self.handle_tasks_get(request).await,
            "tasks/cancel" => self.handle_tasks_cancel(request).await,
            _ => A2AResponse::error(
                request.id,
                A2AError::method_not_found(format!("Unknown method: {}", request.method)),
            ),
        }
    }

    async fn handle_agent_card(&self, id: serde_json::Value) -> A2AResponse {
        let card = self.agent_card();
        A2AResponse::success(id, serde_json::to_value(card).unwrap_or_default())
    }

    async fn handle_tasks_send(&self, request: A2ARequest) -> A2AResponse {
        let params = match request.params {
            Some(p) => p,
            None => {
                return A2AResponse::error(
                    request.id,
                    A2AError::invalid_request("Missing params"),
                );
            }
        };

        let message = match serde_json::from_value(params.get("message").cloned().unwrap_or_default()) {
            Ok(m) => m,
            Err(e) => {
                return A2AResponse::error(
                    request.id,
                    A2AError::invalid_request(format!("Invalid message: {}", e)),
                );
            }
        };

        match self.executor.execute(message).await {
            Ok(task) => {
                let mut tasks = self.tasks.write().await;
                tasks.insert(task.id.clone(), task.clone());
                A2AResponse::success(request.id, serde_json::to_value(task).unwrap_or_default())
            }
            Err(e) => A2AResponse::error(request.id, e),
        }
    }

    async fn handle_tasks_get(&self, request: A2ARequest) -> A2AResponse {
        let params = match request.params {
            Some(p) => p,
            None => {
                return A2AResponse::error(
                    request.id,
                    A2AError::invalid_request("Missing params"),
                );
            }
        };

        let task_id = match params.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => {
                return A2AResponse::error(
                    request.id,
                    A2AError::invalid_request("Missing task id"),
                );
            }
        };

        let tasks = self.tasks.read().await;
        match tasks.get(task_id) {
            Some(task) => {
                A2AResponse::success(request.id, serde_json::to_value(task).unwrap_or_default())
            }
            None => A2AResponse::error(
                request.id,
                A2AError::invalid_request(format!("Task not found: {}", task_id)),
            ),
        }
    }

    async fn handle_tasks_cancel(&self, request: A2ARequest) -> A2AResponse {
        let params = match request.params {
            Some(p) => p,
            None => {
                return A2AResponse::error(
                    request.id,
                    A2AError::invalid_request("Missing params"),
                );
            }
        };

        let task_id = match params.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => {
                return A2AResponse::error(
                    request.id,
                    A2AError::invalid_request("Missing task id"),
                );
            }
        };

        let mut tasks = self.tasks.write().await;
        match tasks.get_mut(task_id) {
            Some(task) => {
                task.state = A2ATaskState::Cancelled;
                A2AResponse::success(request.id, serde_json::to_value(task.clone()).unwrap_or_default())
            }
            None => A2AResponse::error(
                request.id,
                A2AError::invalid_request(format!("Task not found: {}", task_id)),
            ),
        }
    }

    /// Get the host address.
    pub fn host(&self) -> &str {
        &self.config.host
    }

    /// Get the port.
    pub fn port(&self) -> u16 {
        self.config.port
    }

    /// Get the executor.
    pub fn executor(&self) -> &Arc<StrandsA2AExecutor> {
        &self.executor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config() {
        let config = A2AServerConfig::new()
            .with_host("0.0.0.0")
            .with_port(8080)
            .with_version("1.0.0");

        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.version, "1.0.0");
    }

    #[test]
    fn test_agent_card_creation() {
        let card = AgentCard::new("Test Agent", "http://localhost:9000/", "1.0.0")
            .with_streaming(true)
            .with_description("A test agent");

        assert_eq!(card.name, "Test Agent");
        assert_eq!(card.version, "1.0.0");
        assert!(card.capabilities.streaming);
        assert_eq!(card.description, Some("A test agent".to_string()));
    }

    #[test]
    fn test_agent_skill() {
        let skill = AgentSkill::new("search", "Search the web")
            .with_description("Searches the web for information");

        assert_eq!(skill.id, "search");
        assert_eq!(skill.name, "Search the web");
        assert!(skill.description.is_some());
    }
}

