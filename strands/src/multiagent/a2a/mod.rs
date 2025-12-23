//! Agent-to-Agent (A2A) protocol support.
//!
//! This module provides the A2A protocol implementation for inter-agent communication.
//! Docs: https://github.com/google/A2A

mod executor;
mod server;
mod types;

pub use executor::StrandsA2AExecutor;
pub use server::{A2AServer, A2AServerConfig};
pub use types::{
    A2ACapabilities, A2AError, A2AMessage, A2APart, A2ARequest, A2AResponse, A2ATask,
    A2ATaskState, AgentCard, AgentSkill,
};

