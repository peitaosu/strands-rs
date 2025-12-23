//! Multi-agent orchestration primitives.
//!
//! This module provides patterns for orchestrating multiple agents:
//! - Graph: Deterministic graph-based execution with dependency resolution
//! - Swarm: Collaborative agent teams with dynamic handoffs
//! - A2A: Agent-to-Agent protocol for inter-agent communication

pub mod a2a;
pub mod base;
pub mod graph;
pub mod swarm;

pub use a2a::{
    A2ACapabilities, A2AError, A2AMessage, A2APart, A2ARequest, A2AResponse, A2AServer,
    A2AServerConfig, A2ATask, A2ATaskState, AgentCard, AgentSkill, StrandsA2AExecutor,
};
pub use base::{
    Interrupt, InterruptState, InvocationState, MultiAgentBase, MultiAgentEvent,
    MultiAgentEventStream, MultiAgentInput, MultiAgentResult, NodeResult, NodeResultValue, Status,
};
pub use graph::{Graph, GraphBuilder, GraphConfig, GraphEdge, GraphNode, GraphResult, GraphState};
pub use swarm::{SharedContext, Swarm, SwarmConfig, SwarmNode, SwarmNodeResult, SwarmResult, SwarmState};
