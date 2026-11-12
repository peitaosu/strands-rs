//! Swarm-based multi-agent orchestration.
//!
//! Provides a collaborative agent orchestration system where agents work
//! together as a team to solve complex tasks, with shared context and
//! autonomous coordination.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::RwLock;

use super::base::{
    Interrupt, InterruptState, InvocationState, MultiAgentBase, MultiAgentEvent,
    MultiAgentEventStream, MultiAgentInput, MultiAgentResult, NodeResult, NodeResultValue, Status,
};
use crate::agent::Agent;
use crate::hooks::{
    AfterInvocationEvent, AfterToolCallEvent, BeforeInvocationEvent, BeforeToolCallEvent,
    HookEvent, HookRegistry,
};
use crate::types::tools::{ToolResult as ToolResultType, ToolUse};
use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::tools::ToolSpec;
use crate::types::errors::{Result, StrandsError};
use crate::types::streaming::{Metrics, Usage};
use crate::types::tools::ToolResultStatus;

/// Shared context between swarm nodes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SharedContext {
    context: HashMap<String, HashMap<String, serde_json::Value>>,
}

impl SharedContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds context for a node.
    pub fn add_context(
        &mut self,
        node_id: &str,
        key: impl Into<String>,
        value: impl Serialize,
    ) -> Result<()> {
        let key = key.into();
        if key.is_empty() {
            return Err(StrandsError::ConfigurationError {
                message: "Key cannot be empty".to_string(),
            });
        }

        let value = serde_json::to_value(value).map_err(|e| StrandsError::ConfigurationError {
            message: format!("Value is not JSON serializable: {e}"),
        })?;

        self.context
            .entry(node_id.to_string())
            .or_default()
            .insert(key, value);

        Ok(())
    }

    /// Gets context for a node.
    pub fn get_context(&self, node_id: &str) -> Option<&HashMap<String, serde_json::Value>> {
        self.context.get(node_id)
    }

    /// Gets all context.
    pub fn all(&self) -> &HashMap<String, HashMap<String, serde_json::Value>> {
        &self.context
    }
}

/// Represents a node (agent) in the swarm.
pub struct SwarmNode {
    pub node_id: String,
    pub agent: Agent,
    initial_messages: Vec<crate::types::content::Message>,
}

impl SwarmNode {
    pub fn new(node_id: impl Into<String>, agent: Agent) -> Self {
        let initial_messages = agent.messages().to_vec();
        Self {
            node_id: node_id.into(),
            agent,
            initial_messages,
        }
    }

    /// Resets the node state to initial state.
    pub fn reset(&mut self) {
        self.agent.clear_messages();
        for msg in &self.initial_messages {
            self.agent.add_message(msg.clone());
        }
    }
}

impl std::hash::Hash for SwarmNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.node_id.hash(state);
    }
}

impl PartialEq for SwarmNode {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for SwarmNode {}

/// Result from a swarm node execution.
#[derive(Debug, Clone)]
pub struct SwarmNodeResult {
    pub node_id: String,
    pub result: NodeResult,
}

/// Current state of swarm execution.
pub struct SwarmState {
    pub current_node_id: Option<String>,
    pub task: String,
    pub status: Status,
    pub shared_context: SharedContext,
    pub node_history: Vec<String>,
    pub start_time: Instant,
    pub results: HashMap<String, NodeResult>,
    pub accumulated_usage: Usage,
    pub accumulated_metrics: Metrics,
    pub execution_time_ms: u64,
    pub handoff_node_id: Option<String>,
    pub handoff_message: Option<String>,
}

impl Default for SwarmState {
    fn default() -> Self {
        Self {
            current_node_id: None,
            task: String::new(),
            status: Status::Pending,
            shared_context: SharedContext::new(),
            node_history: Vec::new(),
            start_time: Instant::now(),
            results: HashMap::new(),
            accumulated_usage: Usage::default(),
            accumulated_metrics: Metrics::default(),
            execution_time_ms: 0,
            handoff_node_id: None,
            handoff_message: None,
        }
    }
}

impl SwarmState {
    /// Check if the swarm should continue execution.
    pub fn should_continue(&self, config: &SwarmConfig) -> (bool, &'static str) {
        if self.node_history.len() >= config.max_handoffs {
            return (false, "Max handoffs reached");
        }

        if self.node_history.len() >= config.max_iterations {
            return (false, "Max iterations reached");
        }

        if let Some(timeout) = config.execution_timeout {
            if self.start_time.elapsed() > timeout {
                return (false, "Execution timed out");
            }
        }

        if config.repetitive_handoff_detection_window > 0
            && self.node_history.len() >= config.repetitive_handoff_detection_window
        {
            let recent: Vec<_> = self
                .node_history
                .iter()
                .rev()
                .take(config.repetitive_handoff_detection_window)
                .collect();
            let unique: std::collections::HashSet<_> = recent.iter().collect();
            if unique.len() < config.repetitive_handoff_min_unique_agents {
                return (false, "Repetitive handoff detected");
            }
        }

        (true, "Continuing")
    }
}

/// Result from swarm execution.
#[derive(Debug, Clone)]
pub struct SwarmResult {
    pub status: Status,
    pub results: HashMap<String, NodeResult>,
    pub node_history: Vec<String>,
    pub accumulated_usage: Usage,
    pub accumulated_metrics: Metrics,
    pub execution_time_ms: u64,
    pub interrupts: Vec<Interrupt>,
}

impl From<SwarmResult> for MultiAgentResult {
    fn from(sr: SwarmResult) -> Self {
        MultiAgentResult {
            status: sr.status,
            results: sr.results,
            accumulated_usage: sr.accumulated_usage,
            accumulated_metrics: sr.accumulated_metrics,
            execution_count: sr.node_history.len() as u32,
            execution_time_ms: sr.execution_time_ms,
            interrupts: sr.interrupts,
        }
    }
}

/// Configuration options for swarm execution.
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    pub max_handoffs: usize,
    pub max_iterations: usize,
    pub execution_timeout: Option<Duration>,
    pub node_timeout: Option<Duration>,
    pub repetitive_handoff_detection_window: usize,
    pub repetitive_handoff_min_unique_agents: usize,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            max_handoffs: 20,
            max_iterations: 20,
            execution_timeout: Some(Duration::from_secs(900)),
            node_timeout: Some(Duration::from_secs(300)),
            repetitive_handoff_detection_window: 0,
            repetitive_handoff_min_unique_agents: 0,
        }
    }
}

/// Handoff tool for coordinating agent-to-agent transfers.
struct HandoffTool {
    swarm_state: Arc<RwLock<HandoffState>>,
    available_agents: Vec<String>,
}

#[derive(Default)]
struct HandoffState {
    target_node_id: Option<String>,
    message: Option<String>,
    context: HashMap<String, serde_json::Value>,
}

#[async_trait]
impl AgentTool for HandoffTool {
    fn name(&self) -> &str {
        "handoff_to_agent"
    }

    fn description(&self) -> &str {
        "Transfer control to another agent in the swarm for specialized help"
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(
            "handoff_to_agent",
            "Transfer control to another agent in the swarm for specialized help",
        ).with_input_schema(json!({
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent to hand off to"
                },
                "message": {
                    "type": "string",
                    "description": "Message explaining what needs to be done and why you're handing off"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context to share with the next agent",
                    "additionalProperties": true
                }
            },
            "required": ["agent_name", "message"]
        }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let agent_name = input
            .get("agent_name")
            .and_then(|v| v.as_str())
            .ok_or("Missing agent_name")?;
        let message = input
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or("Missing message")?;
        let context = input
            .get("context")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        if !self.available_agents.contains(&agent_name.to_string()) {
            return Ok(ToolResult2 {
                status: ToolResultStatus::Error,
                content: vec![crate::types::tools::ToolResultContent::text(format!(
                    "Error: Agent '{}' not found in swarm. Available agents: {:?}",
                    agent_name, self.available_agents
                ))],
            });
        }

        let mut state = self.swarm_state.write().await;
        state.target_node_id = Some(agent_name.to_string());
        state.message = Some(message.to_string());
        state.context = context.into_iter().collect();

        Ok(ToolResult2 {
            status: ToolResultStatus::Success,
            content: vec![crate::types::tools::ToolResultContent::text(format!(
                "Handing off to {}: {}",
                agent_name, message
            ))],
        })
    }
}

/// Self-organizing collaborative agent teams with shared working memory.
pub struct Swarm {
    id: String,
    nodes: HashMap<String, SwarmNode>,
    entry_point_id: Option<String>,
    config: SwarmConfig,
    state: SwarmState,
    hooks: HookRegistry,
    interrupt_state: InterruptState,
    handoff_state: Arc<RwLock<HandoffState>>,
    resume_from_session: bool,
}

impl Swarm {
    /// Creates a new swarm with the given agents.
    pub fn new(
        agents: Vec<Agent>,
        entry_point: Option<&str>,
        config: SwarmConfig,
    ) -> Result<Self> {
        if agents.is_empty() {
            return Err(StrandsError::ConfigurationError {
                message: "Swarm must have at least one agent".to_string(),
            });
        }

        let mut nodes = HashMap::new();
        let mut node_names: Vec<String> = Vec::new();

        for (i, agent) in agents.into_iter().enumerate() {
            let node_id = agent.name().cloned().unwrap_or_else(|| format!("node_{i}"));

            if nodes.contains_key(&node_id) {
                return Err(StrandsError::ConfigurationError {
                    message: format!("Duplicate node ID: {node_id}"),
                });
            }

            node_names.push(node_id.clone());
            nodes.insert(node_id.clone(), SwarmNode::new(node_id, agent));
        }

        let entry_point_id = entry_point.map(|s| s.to_string()).or_else(|| {
            nodes.keys().next().cloned()
        });

        if let Some(ref ep) = entry_point_id {
            if !nodes.contains_key(ep) {
                return Err(StrandsError::ConfigurationError {
                    message: format!("Entry point '{ep}' not found in swarm nodes"),
                });
            }
        }

        let handoff_state = Arc::new(RwLock::new(HandoffState::default()));

        let mut swarm = Self {
            id: "default_swarm".to_string(),
            nodes,
            entry_point_id,
            config,
            state: SwarmState::default(),
            hooks: HookRegistry::new(),
            interrupt_state: InterruptState::new(),
            handoff_state,
            resume_from_session: false,
        };

        for node in swarm.nodes.values_mut() {
            let tool = HandoffTool {
                swarm_state: Arc::clone(&swarm.handoff_state),
                available_agents: node_names.iter().filter(|n| *n != &node.node_id).cloned().collect(),
            };
            node.agent.tool_registry_mut().register(Box::new(tool));
        }

        Ok(swarm)
    }

    /// Sets the swarm ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Sets the hook registry.
    pub fn with_hooks(mut self, hooks: HookRegistry) -> Self {
        self.hooks = hooks;
        self
    }

    /// Returns the swarm ID.
    pub fn swarm_id(&self) -> &str {
        &self.id
    }

    /// Returns the current state.
    pub fn state(&self) -> &SwarmState {
        &self.state
    }

    /// Returns an iterator over node IDs.
    pub fn node_ids(&self) -> impl Iterator<Item = &str> {
        self.nodes.keys().map(|s| s.as_str())
    }

    /// Returns a reference to the interrupt state.
    pub fn interrupt_state(&self) -> &InterruptState {
        &self.interrupt_state
    }

    /// Returns a mutable reference to the interrupt state.
    pub fn interrupt_state_mut(&mut self) -> &mut InterruptState {
        &mut self.interrupt_state
    }

    /// Activates interrupt state for a node.
    fn activate_interrupt(
        &mut self,
        node_id: &str,
        interrupts: Vec<Interrupt>,
    ) -> MultiAgentEvent {
        tracing::debug!("node=<{}> | node interrupted", node_id);
        self.state.status = Status::Interrupted;

        self.interrupt_state.context.insert(
            node_id.to_string(),
            serde_json::json!({
                "activated": true,
            }),
        );

        for interrupt in &interrupts {
            self.interrupt_state.add(interrupt.clone());
        }

        self.interrupt_state.activate();

        MultiAgentEvent::node_interrupt(node_id, interrupts)
    }

    /// Invokes the swarm synchronously.
    pub fn call(&mut self, task: impl Into<MultiAgentInput>) -> Result<SwarmResult> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.invoke_async(task.into(), None))
        })
    }

    /// Invokes the swarm asynchronously and returns the result.
    pub async fn invoke_async(
        &mut self,
        task: MultiAgentInput,
        invocation_state: Option<&InvocationState>,
    ) -> Result<SwarmResult> {
        let mut stream = self.stream_async(task, invocation_state);
        let mut final_result = None;

        while let Some(event) = stream.next().await {
            if let MultiAgentEvent::Result(result) = event {
                final_result = Some(result);
            }
        }

        drop(stream);

        final_result
            .map(|r| SwarmResult {
                status: r.status,
                results: r.results,
                node_history: self.state.node_history.clone(),
                accumulated_usage: r.accumulated_usage,
                accumulated_metrics: r.accumulated_metrics,
                execution_time_ms: r.execution_time_ms,
                interrupts: r.interrupts,
            })
            .ok_or_else(|| StrandsError::MultiAgentError {
                message: "Swarm execution completed without result".to_string(),
            })
    }

    /// Streams events during swarm execution.
    pub fn stream_async<'a>(
        &'a mut self,
        task: MultiAgentInput,
        _invocation_state: Option<&'a InvocationState>,
    ) -> MultiAgentEventStream<'a> {
        let task_str = task.to_string_lossy();

        Box::pin(async_stream::stream! {
            self.hooks.fire_and_forget(HookEvent::BeforeInvocation(BeforeInvocationEvent::default())).await;

            if self.resume_from_session || self.interrupt_state.activated {
                self.state.status = Status::Executing;
                self.state.start_time = Instant::now();
            } else {
                self.state = SwarmState {
                    current_node_id: self.entry_point_id.clone(),
                    task: task_str.clone(),
                    status: Status::Executing,
                    start_time: Instant::now(),
                    ..Default::default()
                };

                {
                    let mut handoff = self.handoff_state.write().await;
                    *handoff = HandoffState::default();
                }
            }

            while self.state.status == Status::Executing {
                let (should_continue, reason) = self.state.should_continue(&self.config);
                if !should_continue {
                    tracing::warn!("Swarm execution stopped: {reason}");
                    self.state.status = Status::Failed;
                    break;
                }

                let current_node_id = match &self.state.current_node_id {
                    Some(id) => id.clone(),
                    None => {
                        self.state.status = Status::Failed;
                        break;
                    }
                };

                if !self.nodes.contains_key(&current_node_id) {
                    tracing::error!("Node '{}' not found", current_node_id);
                    self.state.status = Status::Failed;
                    break;
                }

                yield MultiAgentEvent::node_start(&current_node_id, "agent");

                let (before_event, _) = self.hooks.fire(HookEvent::BeforeToolCall(BeforeToolCallEvent::new(
                    ToolUse::new(&current_node_id, &current_node_id, serde_json::json!({}))
                ))).await;
                let _ = before_event;

                let result = self.execute_node(&current_node_id, &task_str).await;

                match result {
                    Ok(node_result) => {
                        self.state.node_history.push(current_node_id.clone());
                        self.state.accumulated_usage.add(&node_result.accumulated_usage);
                        self.state.accumulated_metrics.latency_ms += node_result.accumulated_metrics.latency_ms;

                        yield MultiAgentEvent::node_stop(&current_node_id, node_result.clone());

                        if node_result.status == Status::Interrupted {
                            let interrupt_event = self.activate_interrupt(&current_node_id, node_result.interrupts.clone());
                            yield interrupt_event;
                            break;
                        }

                        self.interrupt_state.deactivate();

                        self.state.results.insert(current_node_id.clone(), node_result);

                        let handoff = {
                            let state = self.handoff_state.read().await;
                            (state.target_node_id.clone(), state.message.clone())
                        };

                        if let (Some(target_id), message) = handoff {
                            {
                                let mut state = self.handoff_state.write().await;
                                *state = HandoffState::default();
                            }

                            yield MultiAgentEvent::handoff(
                                vec![current_node_id.clone()],
                                vec![target_id.clone()],
                                message,
                            );

                            self.state.current_node_id = Some(target_id);
                        } else {
                            self.state.status = Status::Completed;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Node '{}' failed: {}", current_node_id, e);
                        let error_result = NodeResult::from_error(e.to_string(), 0);
                        yield MultiAgentEvent::node_stop(&current_node_id, error_result);
                        self.state.status = Status::Failed;
                    }
                }

                let _ = self.hooks.fire(HookEvent::AfterToolCall(AfterToolCallEvent::new(
                    ToolUse::new(&current_node_id, &current_node_id, serde_json::json!({})),
                    ToolResultType::success(&current_node_id, "completed")
                ))).await;
            }

            self.state.execution_time_ms = self.state.start_time.elapsed().as_millis() as u64;

            self.hooks.fire_and_forget(HookEvent::AfterInvocation(AfterInvocationEvent::new(None))).await;

            let result = MultiAgentResult {
                status: self.state.status,
                results: self.state.results.clone(),
                accumulated_usage: self.state.accumulated_usage.clone(),
                accumulated_metrics: self.state.accumulated_metrics.clone(),
                execution_count: self.state.node_history.len() as u32,
                execution_time_ms: self.state.execution_time_ms,
                interrupts: Vec::new(),
            };

            yield MultiAgentEvent::result(result);
        })
    }

    async fn execute_node(&mut self, node_id: &str, task: &str) -> Result<NodeResult> {
        let start = Instant::now();

        let input = self.build_node_input(node_id, task);

        let node = self.nodes.get_mut(node_id).ok_or_else(|| StrandsError::InternalError {
            message: format!("Node '{node_id}' not found"),
        })?;

        let agent_result = node.agent.invoke_async(input.as_str()).await?;
        let execution_time_ms = start.elapsed().as_millis() as u64;

        let usage = agent_result.usage().cloned().unwrap_or_default();

        Ok(NodeResult {
            result: NodeResultValue::Agent(agent_result),
            execution_time_ms,
            status: Status::Completed,
            accumulated_usage: usage,
            accumulated_metrics: Metrics { latency_ms: execution_time_ms, time_to_first_byte_ms: 0 },
            execution_count: 1,
            interrupts: Vec::new(),
        })
    }

    fn build_node_input(&self, target_node_id: &str, task: &str) -> String {
        let mut input = String::new();

        if let Some(ref message) = self.state.handoff_message {
            input.push_str(&format!("Handoff Message: {message}\n\n"));
        }

        input.push_str(&format!("User Request: {task}\n\n"));

        if !self.state.node_history.is_empty() {
            input.push_str(&format!(
                "Previous agents who worked on this: {}\n\n",
                self.state.node_history.join(" → ")
            ));
        }

        if !self.state.shared_context.context.is_empty() {
            input.push_str("Shared knowledge from previous agents:\n");
            for (node_name, context) in &self.state.shared_context.context {
                if !context.is_empty() {
                    input.push_str(&format!("• {node_name}: {:?}\n", context));
                }
            }
            input.push('\n');
        }

        let other_nodes: Vec<_> = self.nodes.keys()
            .filter(|id| *id != target_node_id)
            .collect();

        if !other_nodes.is_empty() {
            input.push_str("Other agents available for collaboration:\n");
            for node_id in other_nodes {
                input.push_str(&format!("Agent name: {node_id}.\n"));
            }
            input.push('\n');
        }

        input.push_str(
            "You have access to swarm coordination tools if you need help from other agents. \
             If you don't hand off to another agent, the swarm will consider the task complete."
        );

        input
    }
}

#[async_trait]
impl MultiAgentBase for Swarm {
    fn id(&self) -> &str {
        &self.id
    }

    async fn invoke_async(
        &mut self,
        task: MultiAgentInput,
        invocation_state: Option<&InvocationState>,
    ) -> Result<MultiAgentResult> {
        self.invoke_async(task, invocation_state).await.map(Into::into)
    }

    fn stream_async<'a>(
        &'a mut self,
        task: MultiAgentInput,
        invocation_state: Option<&'a InvocationState>,
    ) -> MultiAgentEventStream<'a> {
        self.stream_async(task, invocation_state)
    }

    fn serialize_state(&self) -> serde_json::Value {
        json!({
            "type": "swarm",
            "id": self.id,
            "status": format!("{:?}", self.state.status).to_lowercase(),
            "node_history": self.state.node_history,
            "current_node": self.state.current_node_id,
            "current_task": self.state.task,
            "shared_context": self.state.shared_context.context,
            "interrupt_state": self.interrupt_state.to_dict(),
        })
    }

    fn deserialize_state(&mut self, payload: &serde_json::Value) -> Result<()> {
        if let Some(status_str) = payload.get("status").and_then(|v| v.as_str()) {
            self.state.status = match status_str {
                "pending" => Status::Pending,
                "executing" => Status::Executing,
                "completed" => Status::Completed,
                "failed" => Status::Failed,
                "interrupted" => Status::Interrupted,
                _ => Status::Pending,
            };
        }

        if let Some(history) = payload.get("node_history").and_then(|v| v.as_array()) {
            self.state.node_history = history
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
        }

        if let Some(current) = payload.get("current_node").and_then(|v| v.as_str()) {
            self.state.current_node_id = Some(current.to_string());
        }

        if let Some(task) = payload.get("current_task").and_then(|v| v.as_str()) {
            self.state.task = task.to_string();
        }

        if let Some(interrupt_obj) = payload.get("interrupt_state").and_then(|v| v.as_object()) {
            let interrupt_map: std::collections::HashMap<String, serde_json::Value> = interrupt_obj
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            self.interrupt_state = InterruptState::from_dict(interrupt_map);
            self.resume_from_session = true;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_context() {
        let mut ctx = SharedContext::new();
        ctx.add_context("node1", "key1", "value1").unwrap();
        assert!(ctx.get_context("node1").is_some());
    }

    #[test]
    fn test_shared_context_empty_key() {
        let mut ctx = SharedContext::new();
        let result = ctx.add_context("node1", "", "value");
        assert!(result.is_err());
    }

    #[test]
    fn test_swarm_state_should_continue() {
        let config = SwarmConfig::default();
        let state = SwarmState::default();
        let (should_continue, _) = state.should_continue(&config);
        assert!(should_continue);
    }

    #[test]
    fn test_swarm_state_max_handoffs() {
        let config = SwarmConfig {
            max_handoffs: 2,
            ..Default::default()
        };
        let mut state = SwarmState::default();
        state.node_history = vec!["a".to_string(), "b".to_string()];
        let (should_continue, reason) = state.should_continue(&config);
        assert!(!should_continue);
        assert_eq!(reason, "Max handoffs reached");
    }
}
