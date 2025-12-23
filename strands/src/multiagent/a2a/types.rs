//! A2A protocol type definitions.

use serde::{Deserialize, Serialize};

/// State of an A2A task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum A2ATaskState {
    Pending,
    Working,
    Completed,
    Failed,
    Cancelled,
}

/// An A2A task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2ATask {
    pub id: String,
    pub context_id: String,
    pub state: A2ATaskState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<A2AMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<Vec<A2AArtifact>>,
}

impl A2ATask {
    pub fn new(id: impl Into<String>, context_id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            context_id: context_id.into(),
            state: A2ATaskState::Pending,
            message: None,
            artifacts: None,
        }
    }
}

/// An artifact produced by a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2AArtifact {
    pub name: String,
    pub parts: Vec<A2APart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
}

/// An A2A message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2AMessage {
    pub role: String,
    pub parts: Vec<A2APart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

impl A2AMessage {
    pub fn user(parts: Vec<A2APart>) -> Self {
        Self {
            role: "user".to_string(),
            parts,
            context_id: None,
            task_id: None,
        }
    }

    pub fn agent(parts: Vec<A2APart>, context_id: Option<String>, task_id: Option<String>) -> Self {
        Self {
            role: "agent".to_string(),
            parts,
            context_id,
            task_id,
        }
    }
}

/// A part of an A2A message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum A2APart {
    Text { text: String },
    File { file: A2AFile },
    Data { data: serde_json::Value },
}

impl A2APart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn data(data: serde_json::Value) -> Self {
        Self::Data { data }
    }
}

/// A file in an A2A message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2AFile {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

/// A2A request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2ARequest {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl A2ARequest {
    pub fn new(id: impl Into<serde_json::Value>, method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            method: method.into(),
            params: None,
        }
    }

    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params);
        self
    }
}

/// A2A response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2AResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<A2AError>,
}

impl A2AResponse {
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: serde_json::Value, error: A2AError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// A2A error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl A2AError {
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: -32603,
            message: message.into(),
            data: None,
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: -32600,
            message: message.into(),
            data: None,
        }
    }

    pub fn method_not_found(message: impl Into<String>) -> Self {
        Self {
            code: -32601,
            message: message.into(),
            data: None,
        }
    }
}

/// Agent capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct A2ACapabilities {
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub push_notifications: bool,
    #[serde(default)]
    pub state_transition_history: bool,
}

/// Agent skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSkill {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<String>>,
}

impl AgentSkill {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            tags: None,
            examples: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Agent card describing the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub url: String,
    pub version: String,
    pub capabilities: A2ACapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<AgentSkill>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_input_modes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_output_modes: Option<Vec<String>>,
}

impl AgentCard {
    pub fn new(
        name: impl Into<String>,
        url: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            url: url.into(),
            version: version.into(),
            capabilities: A2ACapabilities::default(),
            skills: None,
            default_input_modes: None,
            default_output_modes: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.capabilities.streaming = streaming;
        self
    }

    pub fn with_skills(mut self, skills: Vec<AgentSkill>) -> Self {
        self.skills = Some(skills);
        self
    }
}

