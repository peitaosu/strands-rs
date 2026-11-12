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
pub mod background_tasks;
pub mod bidi;
pub mod conversation;
pub mod decisions;
pub mod event_loop;
pub mod handlers;
pub mod hooks;
pub mod identifier;
pub mod interventions;
pub mod memory;
pub mod middleware;
pub mod models;
pub mod multiagent;
pub mod plugins;
pub mod sandbox;
pub mod session;
pub mod storage;
pub mod streaming;
pub mod telemetry;
pub mod testing;
pub mod tools;
pub mod types;
pub mod vended_tools;

pub use agent::{Agent, AgentBuilder, AgentResult, AgentState, Limits};
pub use interventions::{InterventionAction, InterventionHandler, InterventionRegistry};
pub use memory::{MemoryEntry, MemoryManager, MemoryStore};
pub use plugins::{apply_plugins, Plugin};
pub use middleware::{
    AgentStreamContext, AgentStreamMiddleware, ExecuteToolContext, ExecuteToolMiddleware,
    InvokeModelContext, InvokeModelMiddleware, MiddlewareRegistry, Stage,
};
pub use multiagent::{Graph, GraphBuilder, Swarm, MultiAgentBase, MultiAgentResult};
pub use session::{SaveLatestStrategy, SnapshotSessionManager};
pub use storage::{InMemoryStorage, LocalFileStorage, NamespacedStorage, Storage};
pub use testing::{MockModel, MockResponse};
pub use types::content::{ContentBlock, Message, Messages, Role};
pub use types::errors::StrandsError;
pub use types::snapshot::Snapshot;
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
