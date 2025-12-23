//! # Strands Agents - A Rust AI Agents SDK
//!
//! Strands is a model-agnostic SDK for building AI agents.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use strands_agents::{Agent, AgentBuilder};
//! use strands_agents::models::BedrockModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut agent = Agent::builder()
//!         .model(BedrockModel::new("anthropic.claude-3-sonnet-20240229-v1:0"))
//!         .system_prompt("You are a helpful assistant")
//!         .build()?;
//!     
//!     let result = agent.invoke_async("Hello!").await?;
//!     println!("{}", result);
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod async_utils;
pub mod conversation;
pub mod event_loop;
pub mod handlers;
pub mod hooks;
pub mod identifier;
pub mod models;
pub mod multiagent;
pub mod session;
pub mod streaming;
pub mod telemetry;
pub mod tools;
pub mod types;

pub use agent::{Agent, AgentBuilder, AgentResult, AgentState};
pub use multiagent::{Graph, GraphBuilder, Swarm, MultiAgentBase, MultiAgentResult};
pub use types::content::{ContentBlock, Message, Messages, Role};
pub use types::errors::StrandsError;
pub use types::tools::{ToolResult, ToolSpec, ToolUse};

#[cfg(feature = "macros")]
pub use strands_agents_macros::tool;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::agent::{Agent, AgentBuilder, AgentResult};
    pub use crate::models::Model;
    pub use crate::tools::AgentTool;
    pub use crate::types::content::{ContentBlock, Message, Role};
    pub use crate::types::errors::StrandsError;
    pub use crate::types::tools::{ToolContext, ToolResult, ToolSpec};

    #[cfg(feature = "macros")]
    pub use strands_agents_macros::tool;
}
