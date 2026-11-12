//! Agent-to-Agent (A2A) protocol support.
//!
//! This module provides the A2A protocol implementation for inter-agent communication.
//! Docs: https://github.com/google/A2A

mod client;
mod executor;
mod server;
mod types;

pub use client::{task_status_to_status, task_to_text, A2AClient, DEFAULT_A2A_TIMEOUT_SECS};
pub use executor::StrandsA2AExecutor;
pub use server::{A2AServer, A2AServerConfig};
pub use types::{
    A2ACapabilities, A2AError, A2AMessage, A2APart, A2ARequest, A2AResponse, A2ATask,
    A2ATaskState, AgentCard, AgentSkill,
};

