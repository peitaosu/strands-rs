//! Graph-based multi-agent orchestration.
//!
//! Provides a deterministic graph-based agent orchestration system where
//! agents are nodes in a graph, executed according to edge dependencies,
//! with output from one node passed as input to connected nodes.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::StreamExt;

use super::base::{
    InterruptState, InvocationState, MultiAgentBase, MultiAgentEvent,
    MultiAgentEventStream, MultiAgentInput, MultiAgentResult, NodeResult, NodeResultValue, Status,
};
use crate::agent::Agent;
use crate::hooks::{
    AfterInvocationEvent, AfterToolCallEvent, BeforeInvocationEvent, BeforeToolCallEvent,
    HookEvent, HookRegistry,
};
use crate::types::tools::{ToolResult as ToolResultType, ToolUse};
use crate::types::errors::{Result, StrandsError};
use crate::types::streaming::{Metrics, Usage};

/// Type alias for edge conditions.
pub type EdgeCondition = Arc<dyn Fn(&GraphState) -> bool + Send + Sync>;

/// An edge connecting two nodes in the graph.
pub struct GraphEdge {
    pub from_node: String,
    pub to_node: String,
    pub condition: Option<EdgeCondition>,
}

impl GraphEdge {
    /// Creates an unconditional edge.
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from_node: from.into(),
            to_node: to.into(),
            condition: None,
        }
    }

    /// Creates a conditional edge.
    pub fn conditional(
        from: impl Into<String>,
        to: impl Into<String>,
        condition: impl Fn(&GraphState) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            from_node: from.into(),
            to_node: to.into(),
            condition: Some(Arc::new(condition)),
        }
    }

    /// Check if this edge should be traversed.
    pub fn should_traverse(&self, state: &GraphState) -> bool {
        match &self.condition {
            Some(cond) => cond(state),
            None => true,
        }
    }
}

/// A node in the agent graph.
pub struct GraphNode {
    pub node_id: String,
    pub agent: Agent,
    pub dependencies: HashSet<String>,
    pub status: Status,
    pub result: Option<NodeResult>,
    pub execution_time_ms: u64,
}

impl GraphNode {
    pub fn new(node_id: impl Into<String>, agent: Agent) -> Self {
        Self {
            node_id: node_id.into(),
            agent,
            dependencies: HashSet::new(),
            status: Status::Pending,
            result: None,
            execution_time_ms: 0,
        }
    }

    /// Resets the node state for re-execution.
    pub fn reset(&mut self) {
        self.status = Status::Pending;
        self.result = None;
        self.execution_time_ms = 0;
    }
}

/// State of graph execution.
#[derive(Debug, Clone, Default)]
pub struct GraphState {
    pub status: Status,
    pub task: String,
    pub completed_nodes: HashSet<String>,
    pub failed_nodes: HashSet<String>,
    pub execution_order: Vec<String>,
    pub results: HashMap<String, NodeResult>,
    pub accumulated_usage: Usage,
    pub accumulated_metrics: Metrics,
    pub execution_count: u32,
    pub execution_time_ms: u64,
    pub start_time: Option<Instant>,
    pub total_nodes: usize,
}

impl GraphState {
    /// Check if graph execution should continue.
    pub fn should_continue(
        &self,
        max_node_executions: Option<usize>,
        execution_timeout: Option<Duration>,
    ) -> (bool, &'static str) {
        if let Some(max) = max_node_executions {
            if self.execution_order.len() >= max {
                return (false, "Max node executions reached");
            }
        }

        if let (Some(timeout), Some(start)) = (execution_timeout, self.start_time) {
            if start.elapsed() > timeout {
                return (false, "Execution timed out");
            }
        }

        (true, "Continuing")
    }
}

/// Result from graph execution.
#[derive(Debug, Clone)]
pub struct GraphResult {
    pub status: Status,
    pub results: HashMap<String, NodeResult>,
    pub execution_order: Vec<String>,
    pub accumulated_usage: Usage,
    pub accumulated_metrics: Metrics,
    pub execution_time_ms: u64,
    pub total_nodes: usize,
    pub completed_nodes: usize,
    pub failed_nodes: usize,
    pub entry_points: Vec<String>,
}

impl From<GraphResult> for MultiAgentResult {
    fn from(gr: GraphResult) -> Self {
        MultiAgentResult {
            status: gr.status,
            results: gr.results,
            accumulated_usage: gr.accumulated_usage,
            accumulated_metrics: gr.accumulated_metrics,
            execution_count: gr.execution_order.len() as u32,
            execution_time_ms: gr.execution_time_ms,
            interrupts: Vec::new(),
        }
    }
}

/// Configuration options for graph execution.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    pub max_node_executions: Option<usize>,
    pub execution_timeout: Option<Duration>,
    pub node_timeout: Option<Duration>,
    pub reset_on_revisit: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_node_executions: Some(100),
            execution_timeout: Some(Duration::from_secs(900)),
            node_timeout: Some(Duration::from_secs(300)),
            reset_on_revisit: false,
        }
    }
}

/// Builder for constructing graphs.
pub struct GraphBuilder {
    nodes: HashMap<String, GraphNode>,
    edges: Vec<GraphEdge>,
    entry_points: HashSet<String>,
    config: GraphConfig,
    id: String,
    hooks: HookRegistry,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_points: HashSet::new(),
            config: GraphConfig::default(),
            id: "default_graph".to_string(),
            hooks: HookRegistry::new(),
        }
    }

    /// Sets the graph ID.
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Adds a node to the graph.
    pub fn add_node(mut self, node_id: impl Into<String>, agent: Agent) -> Self {
        let node_id = node_id.into();
        self.nodes.insert(node_id.clone(), GraphNode::new(node_id, agent));
        self
    }

    /// Adds an edge between two nodes.
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        let from = from.into();
        let to = to.into();

        if let Some(node) = self.nodes.get_mut(&to) {
            node.dependencies.insert(from.clone());
        }

        self.edges.push(GraphEdge::new(from, to));
        self
    }

    /// Adds a conditional edge between two nodes.
    pub fn add_conditional_edge<F>(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: F,
    ) -> Self
    where
        F: Fn(&GraphState) -> bool + Send + Sync + 'static,
    {
        let from = from.into();
        let to = to.into();

        if let Some(node) = self.nodes.get_mut(&to) {
            node.dependencies.insert(from.clone());
        }

        self.edges.push(GraphEdge::conditional(from, to, condition));
        self
    }

    /// Sets explicit entry points for the graph.
    pub fn set_entry_points(mut self, entry_points: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.entry_points = entry_points.into_iter().map(Into::into).collect();
        self
    }

    /// Sets a single entry point.
    pub fn set_entry_point(mut self, node_id: impl Into<String>) -> Self {
        self.entry_points.insert(node_id.into());
        self
    }

    /// Sets the graph configuration.
    pub fn config(mut self, config: GraphConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the maximum number of node executions.
    pub fn max_node_executions(mut self, max: usize) -> Self {
        self.config.max_node_executions = Some(max);
        self
    }

    /// Sets the execution timeout.
    pub fn execution_timeout(mut self, timeout: Duration) -> Self {
        self.config.execution_timeout = Some(timeout);
        self
    }

    /// Sets the node timeout.
    pub fn node_timeout(mut self, timeout: Duration) -> Self {
        self.config.node_timeout = Some(timeout);
        self
    }

    /// Enables reset on revisit.
    pub fn reset_on_revisit(mut self, enabled: bool) -> Self {
        self.config.reset_on_revisit = enabled;
        self
    }

    /// Sets the hook registry.
    pub fn hooks(mut self, hooks: HookRegistry) -> Self {
        self.hooks = hooks;
        self
    }

    /// Sets the graph ID, matching the Python builder name.
    pub fn set_graph_id(self, id: impl Into<String>) -> Self {
        self.id(id)
    }

    /// Sets the maximum number of node executions, matching the Python builder name.
    pub fn set_max_node_executions(self, max: usize) -> Self {
        self.max_node_executions(max)
    }

    /// Sets the execution timeout, matching the Python builder name.
    pub fn set_execution_timeout(self, timeout: Duration) -> Self {
        self.execution_timeout(timeout)
    }

    /// Sets the node timeout, matching the Python builder name.
    pub fn set_node_timeout(self, timeout: Duration) -> Self {
        self.node_timeout(timeout)
    }

    /// Sets the hook registry, matching the Python builder name.
    pub fn set_hook_providers(self, hooks: HookRegistry) -> Self {
        self.hooks(hooks)
    }

    /// Builds the graph.
    pub fn build(self) -> Result<Graph> {
        if self.nodes.is_empty() {
            return Err(StrandsError::ConfigurationError {
                message: "Graph must have at least one node".to_string(),
            });
        }

        let entry_points = if self.entry_points.is_empty() {
            self.nodes
                .values()
                .filter(|n| n.dependencies.is_empty())
                .map(|n| n.node_id.clone())
                .collect()
        } else {
            self.entry_points
        };

        if entry_points.is_empty() {
            return Err(StrandsError::ConfigurationError {
                message: "Graph has no entry points (all nodes have dependencies)".to_string(),
            });
        }

        Ok(Graph {
            id: self.id,
            nodes: self.nodes,
            edges: self.edges,
            entry_points,
            config: self.config,
            state: GraphState::default(),
            hooks: self.hooks,
            interrupt_state: InterruptState::new(),
        })
    }
}

/// A graph of agents for orchestrated execution.
pub struct Graph {
    id: String,
    nodes: HashMap<String, GraphNode>,
    edges: Vec<GraphEdge>,
    entry_points: HashSet<String>,
    config: GraphConfig,
    state: GraphState,
    hooks: HookRegistry,
    interrupt_state: InterruptState,
}

impl Graph {
    /// Creates a new graph builder.
    pub fn builder() -> GraphBuilder {
        GraphBuilder::new()
    }

    /// Returns the graph ID.
    pub fn graph_id(&self) -> &str {
        &self.id
    }

    /// Returns the current graph state.
    pub fn state(&self) -> &GraphState {
        &self.state
    }

    /// Returns an iterator over node IDs.
    pub fn node_ids(&self) -> impl Iterator<Item = &str> {
        self.nodes.keys().map(|s| s.as_str())
    }

    /// Returns the entry point node IDs.
    pub fn entry_points(&self) -> &HashSet<String> {
        &self.entry_points
    }

    /// Returns a reference to the interrupt state.
    pub fn interrupt_state(&self) -> &InterruptState {
        &self.interrupt_state
    }

    /// Returns a mutable reference to the interrupt state.
    pub fn interrupt_state_mut(&mut self) -> &mut InterruptState {
        &mut self.interrupt_state
    }


    /// Invokes the graph synchronously.
    pub fn call(&mut self, task: impl Into<MultiAgentInput>) -> Result<GraphResult> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.invoke_async(task.into(), None))
        })
    }

    /// Invokes the graph asynchronously and returns the result.
    pub async fn invoke_async(
        &mut self,
        task: MultiAgentInput,
        invocation_state: Option<&InvocationState>,
    ) -> Result<GraphResult> {
        let total_nodes = self.nodes.len();
        let entry_points_vec: Vec<String> = self.entry_points.iter().cloned().collect();

        let mut stream = self.stream_async(task, invocation_state);
        let mut final_result = None;

        while let Some(event) = stream.next().await {
            if let MultiAgentEvent::Result(result) = event {
                final_result = Some(result);
            }
        }

        drop(stream);

        final_result
            .map(|r| GraphResult {
                status: r.status,
                results: r.results,
                execution_order: self.state.execution_order.clone(),
                accumulated_usage: r.accumulated_usage,
                accumulated_metrics: r.accumulated_metrics,
                execution_time_ms: r.execution_time_ms,
                total_nodes,
                completed_nodes: self.state.completed_nodes.len(),
                failed_nodes: self.state.failed_nodes.len(),
                entry_points: entry_points_vec,
            })
            .ok_or_else(|| StrandsError::MultiAgentError {
                message: "Graph execution completed without result".to_string(),
            })
    }

    /// Streams events during graph execution.
    pub fn stream_async<'a>(
        &'a mut self,
        task: MultiAgentInput,
        _invocation_state: Option<&'a InvocationState>,
    ) -> MultiAgentEventStream<'a> {
        let task_str = task.to_string_lossy();

        Box::pin(async_stream::stream! {
            self.hooks.fire_and_forget(HookEvent::BeforeInvocation(BeforeInvocationEvent::default())).await;

            self.state = GraphState {
                status: Status::Executing,
                task: task_str.clone(),
                start_time: Some(Instant::now()),
                total_nodes: self.nodes.len(),
                ..Default::default()
            };

            let mut queue: VecDeque<String> = self.entry_points.iter().cloned().collect();
            let mut processed: HashSet<String> = HashSet::new();

            while let Some(node_id) = queue.pop_front() {
                if processed.contains(&node_id) {
                    continue;
                }

                let (should_continue, reason) = self.state.should_continue(
                    self.config.max_node_executions,
                    self.config.execution_timeout,
                );
                if !should_continue {
                    tracing::warn!("Graph execution stopped: {reason}");
                    self.state.status = Status::Failed;
                    break;
                }

                let deps_met = {
                    if let Some(node) = self.nodes.get(&node_id) {
                        node.dependencies.iter().all(|dep| self.state.completed_nodes.contains(dep))
                    } else {
                        false
                    }
                };

                if !deps_met {
                    queue.push_back(node_id);
                    continue;
                }

                if self.config.reset_on_revisit && self.state.completed_nodes.contains(&node_id) {
                    if let Some(node) = self.nodes.get_mut(&node_id) {
                        node.reset();
                    }
                    self.state.completed_nodes.remove(&node_id);
                }

                yield MultiAgentEvent::node_start(&node_id, "agent");

                let (before_event, _) = self.hooks.fire(HookEvent::BeforeToolCall(BeforeToolCallEvent::new(
                    ToolUse::new(&node_id, &node_id, serde_json::json!({}))
                ))).await;
                let _ = before_event;

                let result = self.execute_node(&node_id, &task_str).await;

                match result {
                    Ok(node_result) => {

                        if node_result.status == Status::Interrupted {
                            self.interrupt_state.deactivate();
                            tracing::error!("user raised interrupt from agent | interrupts are not yet supported in graphs");
                            self.state.status = Status::Failed;
                            yield MultiAgentEvent::node_stop(&node_id, node_result);
                            break;
                        }

                        self.state.completed_nodes.insert(node_id.clone());
                        self.state.execution_order.push(node_id.clone());
                        self.state.accumulated_usage.add(&node_result.accumulated_usage);
                        self.state.accumulated_metrics.latency_ms += node_result.accumulated_metrics.latency_ms;
                        self.state.execution_count += 1;

                        if let Some(node) = self.nodes.get_mut(&node_id) {
                            node.status = Status::Completed;
                            node.execution_time_ms = node_result.execution_time_ms;
                        }

                        yield MultiAgentEvent::node_stop(&node_id, node_result.clone());

                        self.state.results.insert(node_id.clone(), node_result);

                        let mut next_nodes = Vec::new();
                        for edge in &self.edges {
                            if edge.from_node == node_id && edge.should_traverse(&self.state) {
                                if !processed.contains(&edge.to_node) {
                                    next_nodes.push(edge.to_node.clone());
                                }
                            }
                        }

                        if !next_nodes.is_empty() {
                            yield MultiAgentEvent::handoff(
                                vec![node_id.clone()],
                                next_nodes.clone(),
                                None,
                            );
                            for next in next_nodes {
                                queue.push_back(next);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Node {node_id} failed: {e}");
                        self.state.failed_nodes.insert(node_id.clone());
                        if let Some(node) = self.nodes.get_mut(&node_id) {
                            node.status = Status::Failed;
                        }

                        let error_result = NodeResult::from_error(e.to_string(), 0);
                        yield MultiAgentEvent::node_stop(&node_id, error_result);
                    }
                }

                let _ = self.hooks.fire(HookEvent::AfterToolCall(AfterToolCallEvent::new(
                    ToolUse::new(&node_id, &node_id, serde_json::json!({})),
                    ToolResultType::success(&node_id, "completed")
                ))).await;
                processed.insert(node_id);
            }

            if self.state.failed_nodes.is_empty() && self.state.status == Status::Executing {
                self.state.status = Status::Completed;
            } else if !self.state.failed_nodes.is_empty() {
                self.state.status = Status::Failed;
            }

            self.state.execution_time_ms = self.state.start_time
                .map(|s| s.elapsed().as_millis() as u64)
                .unwrap_or(0);

            self.hooks.fire_and_forget(HookEvent::AfterInvocation(AfterInvocationEvent::new(None))).await;

            let result = MultiAgentResult {
                status: self.state.status,
                results: self.state.results.clone(),
                accumulated_usage: self.state.accumulated_usage.clone(),
                accumulated_metrics: self.state.accumulated_metrics.clone(),
                execution_count: self.state.execution_count,
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

        node.status = Status::Executing;

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

    fn build_node_input(&self, node_id: &str, task: &str) -> String {
        let mut input = String::new();

        let node = match self.nodes.get(node_id) {
            Some(n) => n,
            None => {
                input.push_str(&format!("Task: {task}"));
                return input;
            }
        };

        if node.dependencies.is_empty() {
            input.push_str(&format!("Task: {task}"));
        } else {
            input.push_str(&format!("Original Task: {task}\n\n"));
            input.push_str("Inputs from previous nodes:\n\n");

            for dep in &node.dependencies {
                if let Some(result) = self.state.results.get(dep) {
                    input.push_str(&format!("From {dep}:\n"));
                    for agent_result in result.get_agent_results() {
                        let text = agent_result.text();
                        input.push_str(&format!("  - Agent: {text}\n"));
                    }
                }
            }
        }

        input
    }
}

#[async_trait]
impl MultiAgentBase for Graph {
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
        serde_json::json!({
            "type": "graph",
            "id": self.id,
            "status": format!("{:?}", self.state.status).to_lowercase(),
            "completed_nodes": self.state.completed_nodes.iter().collect::<Vec<_>>(),
            "failed_nodes": self.state.failed_nodes.iter().collect::<Vec<_>>(),
            "execution_order": self.state.execution_order,
            "current_task": self.state.task,
        })
    }

    fn deserialize_state(&mut self, payload: &serde_json::Value) -> Result<()> {
        if let Some(status_str) = payload.get("status").and_then(|v| v.as_str()) {
            self.state.status = match status_str {
                "pending" => Status::Pending,
                "dispatched" => Status::Dispatched,
                "executing" => Status::Executing,
                "completed" => Status::Completed,
                "failed" => Status::Failed,
                "interrupted" => Status::Interrupted,
                _ => Status::Pending,
            };
        }

        if let Some(completed) = payload.get("completed_nodes").and_then(|v| v.as_array()) {
            self.state.completed_nodes = completed
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
        }

        if let Some(task) = payload.get("current_task").and_then(|v| v.as_str()) {
            self.state.task = task.to_string();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_no_nodes() {
        let result = Graph::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_state_should_continue() {
        let state = GraphState::default();
        let (should_continue, _) = state.should_continue(Some(10), None);
        assert!(should_continue);

        let mut state = GraphState::default();
        state.execution_order = vec!["a".to_string(); 10];
        let (should_continue, reason) = state.should_continue(Some(10), None);
        assert!(!should_continue);
        assert_eq!(reason, "Max node executions reached");
    }

    #[test]
    fn test_node_result() {
        let result = NodeResult::from_error("test error", 100);
        assert!(result.is_error());
        assert_eq!(result.execution_time_ms, 100);
    }

    #[test]
    fn test_status_default() {
        let status = Status::default();
        assert_eq!(status, Status::Pending);
    }
}
