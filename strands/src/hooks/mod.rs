//! Hook system for agent lifecycle events.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;

use crate::agent::{AgentInput, AgentResult};
use crate::tools::{AgentTool, InvocationState};
use crate::types::content::Message;
use crate::types::streaming::StopReason;
use crate::types::tools::{ToolResult, ToolUse};

pub use crate::types::interrupt::{Interrupt, InterruptState};

/// Named callback priorities matching the Python `HookOrder`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HookOrder;

impl HookOrder {
    pub const SDK_FIRST: f64 = -100.0;
    pub const INTERVENTION_OUTPUT: f64 = -90.0;
    pub const DEFAULT: f64 = 0.0;
    pub const MODEL_ROUTING: f64 = 50.0;
    pub const INTERVENTION_INPUT: f64 = 90.0;
    pub const SDK_LAST: f64 = 100.0;
}

/// Base trait for hook events.
pub trait HookEventBase: Send + Sync {
    /// Whether callbacks should be invoked in reverse order (for cleanup events).
    fn should_reverse_callbacks(&self) -> bool {
        false
    }

    /// Returns the event as Any for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns the event as mutable Any for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Trait for events that can generate interrupts.
///
/// Mirrors the Python `_Interruptible` protocol: a hook calls `interrupt`
/// and the invoking framework collects the pending interrupt after the
/// callback pass completes.
pub trait Interruptible {
    /// Generate a unique interrupt ID for this event.
    ///
    /// The ID should be deterministic based on the event context and name,
    /// allowing for consistent interrupt handling across sessions.
    fn interrupt_id(&self, name: &str) -> String;

    /// Requests an interrupt under `name` and returns its deterministic id.
    fn interrupt(&mut self, name: &str, reason: Option<serde_json::Value>) -> String {
        let id = self.interrupt_id(name);
        let mut interrupt = Interrupt::new(id.clone(), name);
        if let Some(reason) = reason {
            interrupt = interrupt.with_reason(reason);
        }
        self.set_interrupt(interrupt);
        id
    }

    /// Stores the pending interrupt on the event.
    fn set_interrupt(&mut self, interrupt: Interrupt);

    /// Takes the pending interrupt raised by a callback pass.
    fn take_interrupt(&mut self) -> Option<Interrupt>;
}

/// Event triggered when an agent has finished initialization.
#[derive(Debug, Clone, Default)]
pub struct AgentInitializedEvent;

impl HookEventBase for AgentInitializedEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered at the beginning of a new agent request.
#[derive(Debug, Clone, Default)]
pub struct BeforeInvocationEvent {
    pub invocation_state: Option<InvocationState>,
    /// When set, cancels the invocation before any work starts.
    pub cancel: Option<String>,
}

impl BeforeInvocationEvent {
    pub fn cancel(&mut self, message: impl Into<String>) {
        self.cancel = Some(message.into());
    }
}

impl HookEventBase for BeforeInvocationEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered at the end of an agent request.
#[derive(Debug, Clone, Default)]
pub struct AfterInvocationEvent {
    pub result: Option<AgentResult>,
    /// When set, the framework re-invokes the agent with this input.
    pub resume: Option<AgentInput>,
}

impl AfterInvocationEvent {
    pub fn new(result: Option<AgentResult>) -> Self {
        Self { result, resume: None }
    }
}

impl HookEventBase for AfterInvocationEvent {
    fn should_reverse_callbacks(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered when a message is added to the conversation.
#[derive(Debug, Clone)]
pub struct MessageAddedEvent {
    pub message: Message,
}

impl MessageAddedEvent {
    pub fn new(message: Message) -> Self {
        Self { message }
    }
}

impl HookEventBase for MessageAddedEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered when an existing message is replaced in the conversation.
#[derive(Debug, Clone)]
pub struct MessageUpdatedEvent {
    pub tracking_id: String,
    pub message: Message,
}

impl HookEventBase for MessageUpdatedEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered before tools are executed for a cycle.
#[derive(Debug, Clone, Default)]
pub struct BeforeToolsEvent {
    pub tool_uses: Vec<ToolUse>,
    pub invocation_state: Option<InvocationState>,
    /// When set, cancels every tool in the batch with this message.
    pub cancel: Option<String>,
    pub(crate) interrupt_slot: Option<Interrupt>,
}

impl BeforeToolsEvent {
    pub fn new(tool_uses: Vec<ToolUse>) -> Self {
        Self { tool_uses, invocation_state: None, cancel: None, interrupt_slot: None }
    }

    pub fn cancel(&mut self, message: impl Into<String>) {
        self.cancel = Some(message.into());
    }
}

impl Interruptible for BeforeToolsEvent {
    fn interrupt_id(&self, name: &str) -> String {
        let name_uuid = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, name.as_bytes());
        format!("v1:before_tools:{}:{}", self.cycle_index(), name_uuid)
    }

    fn set_interrupt(&mut self, interrupt: Interrupt) {
        self.interrupt_slot = Some(interrupt);
    }

    fn take_interrupt(&mut self) -> Option<Interrupt> {
        self.interrupt_slot.take()
    }
}

impl BeforeToolsEvent {
    fn cycle_index(&self) -> u64 {
        self.invocation_state
            .as_ref()
            .and_then(|s| s.get::<u64>("event_loop_cycle_id"))
            .unwrap_or(0)
    }
}

impl HookEventBase for BeforeToolsEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered after tools are executed for a cycle.
#[derive(Debug, Clone, Default)]
pub struct AfterToolsEvent {
    pub tool_results: Vec<ToolResult>,
    pub invocation_state: Option<InvocationState>,
    /// When set to true, forces the event loop to stop after this cycle.
    pub end_turn: bool,
}

impl HookEventBase for AfterToolsEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered before a tool is invoked.
pub struct BeforeToolCallEvent {
    pub tool_use: ToolUse,
    pub invocation_state: Option<InvocationState>,
    pub cancel_tool: Option<String>,
    /// Hook-replaceable tool implementation for the call.
    pub selected_tool: Option<Arc<dyn AgentTool>>,
    pub(crate) interrupt_slot: Option<Interrupt>,
}

impl std::fmt::Debug for BeforeToolCallEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BeforeToolCallEvent")
            .field("tool_use", &self.tool_use)
            .field("invocation_state", &self.invocation_state)
            .field("cancel_tool", &self.cancel_tool)
            .field("selected_tool", &self.selected_tool.as_ref().map(|t| t.name()))
            .field("interrupt_slot", &self.interrupt_slot)
            .finish()
    }
}

impl Clone for BeforeToolCallEvent {
    fn clone(&self) -> Self {
        Self {
            tool_use: self.tool_use.clone(),
            invocation_state: self.invocation_state.clone(),
            cancel_tool: self.cancel_tool.clone(),
            selected_tool: self.selected_tool.clone(),
            interrupt_slot: self.interrupt_slot.clone(),
        }
    }
}

impl BeforeToolCallEvent {
    pub fn new(tool_use: ToolUse) -> Self {
        Self {
            tool_use,
            invocation_state: None,
            cancel_tool: None,
            selected_tool: None,
            interrupt_slot: None,
        }
    }

    /// Cancel the tool call with a message.
    pub fn cancel(&mut self, message: impl Into<String>) {
        self.cancel_tool = Some(message.into());
    }
}

impl Interruptible for BeforeToolCallEvent {
    /// Generate a unique interrupt ID for before tool call events.
    ///
    /// Format: `v1:before_tool_call:{tool_use_id}:{uuid5(name)}`
    fn interrupt_id(&self, name: &str) -> String {
        use uuid::Uuid;
        let name_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, name.as_bytes());
        format!(
            "v1:before_tool_call:{}:{}",
            self.tool_use.tool_use_id, name_uuid
        )
    }

    fn set_interrupt(&mut self, interrupt: Interrupt) {
        self.interrupt_slot = Some(interrupt);
    }

    fn take_interrupt(&mut self) -> Option<Interrupt> {
        self.interrupt_slot.take()
    }
}

impl HookEventBase for BeforeToolCallEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered after a tool invocation completes.
#[derive(Debug, Clone)]
pub struct AfterToolCallEvent {
    pub tool_use: ToolUse,
    pub invocation_state: Option<InvocationState>,
    pub result: ToolResult,
    pub exception: Option<String>,
    pub cancel_message: Option<String>,
    /// Wall-clock duration of the tool call in milliseconds.
    pub duration: Option<u64>,
    /// When set, the executor re-runs the tool call.
    pub retry: bool,
}

impl AfterToolCallEvent {
    pub fn new(tool_use: ToolUse, result: ToolResult) -> Self {
        Self {
            tool_use,
            invocation_state: None,
            result,
            exception: None,
            cancel_message: None,
            duration: None,
            retry: false,
        }
    }

    pub fn with_exception(mut self, exception: String) -> Self {
        self.exception = Some(exception);
        self
    }

    pub fn request_retry(&mut self) {
        self.retry = true;
    }
}

impl HookEventBase for AfterToolCallEvent {
    fn should_reverse_callbacks(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered before the model is invoked.
#[derive(Debug, Clone, Default)]
pub struct BeforeModelCallEvent {
    pub invocation_state: Option<InvocationState>,
    /// Token estimate for the upcoming model call, when available.
    pub projected_input_tokens: Option<u64>,
    /// When set, cancels the model call with this message.
    pub cancel: Option<String>,
    pub(crate) interrupt_slot: Option<Interrupt>,
}

impl BeforeModelCallEvent {
    pub fn cancel(&mut self, message: impl Into<String>) {
        self.cancel = Some(message.into());
    }
}

impl Interruptible for BeforeModelCallEvent {
    fn interrupt_id(&self, name: &str) -> String {
        let name_uuid = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, name.as_bytes());
        format!("v1:before_model_call:{}:{}", self.cycle_index(), name_uuid)
    }

    fn set_interrupt(&mut self, interrupt: Interrupt) {
        self.interrupt_slot = Some(interrupt);
    }

    fn take_interrupt(&mut self) -> Option<Interrupt> {
        self.interrupt_slot.take()
    }
}

impl BeforeModelCallEvent {
    fn cycle_index(&self) -> u64 {
        self.invocation_state
            .as_ref()
            .and_then(|s| s.get::<u64>("event_loop_cycle_id"))
            .unwrap_or(0)
    }
}

impl HookEventBase for BeforeModelCallEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Model stop response data.
#[derive(Debug, Clone)]
pub struct ModelStopResponse {
    pub message: Message,
    pub stop_reason: StopReason,
}

/// Event triggered after the model invocation completes.
#[derive(Debug, Clone, Default)]
pub struct AfterModelCallEvent {
    pub stop_response: Option<ModelStopResponse>,
    pub exception: Option<String>,
    /// When set, the event loop re-issues the model call.
    pub retry: bool,
}

impl AfterModelCallEvent {
    pub fn success(message: Message, stop_reason: StopReason) -> Self {
        Self {
            stop_response: Some(ModelStopResponse {
                message,
                stop_reason,
            }),
            exception: None,
            retry: false,
        }
    }

    pub fn error(exception: String) -> Self {
        Self {
            stop_response: None,
            exception: Some(exception),
            retry: false,
        }
    }

    pub fn request_retry(&mut self) {
        self.retry = true;
    }
}

impl HookEventBase for AfterModelCallEvent {
    fn should_reverse_callbacks(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event emitted while the model retry strategy waits between attempts.
#[derive(Debug, Clone)]
pub struct EventLoopThrottleEvent {
    pub delay_seconds: f64,
    pub attempt: u32,
    pub max_attempts: u32,
    pub reason: String,
}

impl HookEventBase for EventLoopThrottleEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered when a multi-agent orchestrator has finished initialization.
#[derive(Debug, Clone, Default)]
pub struct MultiAgentInitializedEvent;

impl HookEventBase for MultiAgentInitializedEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered before a multi-agent node is invoked.
#[derive(Debug, Clone, Default)]
pub struct BeforeNodeCallEvent {
    pub node_id: String,
    pub invocation_state: Option<InvocationState>,
    /// When set, skips the node with this message.
    pub cancel_node: Option<String>,
    pub(crate) interrupt_slot: Option<Interrupt>,
}

impl Interruptible for BeforeNodeCallEvent {
    fn interrupt_id(&self, name: &str) -> String {
        let node_uuid = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, self.node_id.as_bytes());
        let name_uuid = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, name.as_bytes());
        format!("v1:before_node_call:{}:{}", node_uuid, name_uuid)
    }

    fn set_interrupt(&mut self, interrupt: Interrupt) {
        self.interrupt_slot = Some(interrupt);
    }

    fn take_interrupt(&mut self) -> Option<Interrupt> {
        self.interrupt_slot.take()
    }
}

impl HookEventBase for BeforeNodeCallEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered after a multi-agent node is invoked.
#[derive(Debug, Clone, Default)]
pub struct AfterNodeCallEvent {
    pub node_id: String,
    pub invocation_state: Option<InvocationState>,
    pub exception: Option<String>,
}

impl HookEventBase for AfterNodeCallEvent {
    fn should_reverse_callbacks(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered before a multi-agent orchestrator invocation.
#[derive(Debug, Clone, Default)]
pub struct BeforeMultiAgentInvocationEvent;

impl HookEventBase for BeforeMultiAgentInvocationEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered after a multi-agent orchestrator invocation.
#[derive(Debug, Clone, Default)]
pub struct AfterMultiAgentInvocationEvent;

impl HookEventBase for AfterMultiAgentInvocationEvent {
    fn should_reverse_callbacks(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Enum wrapper for all hook events.
#[derive(Debug, Clone)]
pub enum HookEvent {
    AgentInitialized(AgentInitializedEvent),
    BeforeInvocation(BeforeInvocationEvent),
    AfterInvocation(AfterInvocationEvent),
    MessageAdded(MessageAddedEvent),
    MessageUpdated(MessageUpdatedEvent),
    BeforeTools(BeforeToolsEvent),
    AfterTools(AfterToolsEvent),
    BeforeToolCall(BeforeToolCallEvent),
    AfterToolCall(AfterToolCallEvent),
    BeforeModelCall(BeforeModelCallEvent),
    AfterModelCall(AfterModelCallEvent),
    EventLoopThrottle(EventLoopThrottleEvent),
    MultiAgentInitialized(MultiAgentInitializedEvent),
    BeforeNodeCall(BeforeNodeCallEvent),
    AfterNodeCall(AfterNodeCallEvent),
    BeforeMultiAgentInvocation(BeforeMultiAgentInvocationEvent),
    AfterMultiAgentInvocation(AfterMultiAgentInvocationEvent),
}

impl HookEvent {
    pub fn should_reverse_callbacks(&self) -> bool {
        match self {
            Self::AfterInvocation(_)
            | Self::AfterToolCall(_)
            | Self::AfterModelCall(_)
            | Self::AfterTools(_)
            | Self::AfterNodeCall(_)
            | Self::AfterMultiAgentInvocation(_) => true,
            _ => false,
        }
    }

    /// Returns the pending interrupt carried by an interruptible event, if any.
    pub fn take_interrupt(&mut self) -> Option<Interrupt> {
        match self {
            Self::BeforeTools(e) => e.take_interrupt(),
            Self::BeforeToolCall(e) => e.take_interrupt(),
            Self::BeforeModelCall(e) => e.take_interrupt(),
            Self::BeforeNodeCall(e) => e.take_interrupt(),
            _ => None,
        }
    }
}

/// Trait for implementing hook providers.
#[async_trait]
pub trait HookProvider: Send + Sync {
    /// Called when a hook event occurs.
    async fn on_event(&self, event: &mut HookEvent);
}

/// Callback function type for hook events.
pub type HookCallback = Arc<dyn Fn(&mut HookEvent) + Send + Sync>;

/// Async callback function type for hook events.
pub type AsyncHookCallback = Arc<dyn Fn(&mut HookEvent) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>;

/// Ordered callback entry.
struct CallbackEntry<T> {
    order: f64,
    sequence: u64,
    callback: T,
}

/// Registry for managing hook providers and callbacks.
///
/// Callbacks are dispatched in `HookOrder` priority order; ties keep
/// registration order. Reverse events invert the order within each priority
/// group rather than reversing the whole list.
#[derive(Default)]
pub struct HookRegistry {
    providers: Vec<CallbackEntry<Arc<dyn HookProvider>>>,
    callbacks: Vec<CallbackEntry<HookCallback>>,
    async_callbacks: Vec<CallbackEntry<AsyncHookCallback>>,
    sequence: u64,
    should_reverse: bool,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Controls whether callback groups are dispatched in reverse order.
    pub fn set_reverse_callbacks(&mut self, should_reverse: bool) {
        self.should_reverse = should_reverse;
    }

    fn next_sequence(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }

    /// Add a hook provider.
    pub fn add_provider(&mut self, provider: impl HookProvider + 'static) {
        let callback: Arc<dyn HookProvider> = Arc::new(provider);
        self.push_provider(callback, HookOrder::DEFAULT);
    }

    /// Add a hook provider at an explicit priority.
    pub fn add_provider_at(&mut self, provider: impl HookProvider + 'static, order: f64) {
        let callback: Arc<dyn HookProvider> = Arc::new(provider);
        self.push_provider(callback, order);
    }

    /// Add a hook provider as Arc.
    pub fn add_provider_arc(&mut self, provider: Arc<dyn HookProvider>) {
        self.push_provider(provider, HookOrder::DEFAULT);
    }

    /// Add a hook provider Arc at an explicit priority.
    pub fn add_provider_arc_at(&mut self, provider: Arc<dyn HookProvider>, order: f64) {
        self.push_provider(provider, order);
    }

    fn push_provider(&mut self, callback: Arc<dyn HookProvider>, order: f64) {
        let entry = CallbackEntry {
            order,
            sequence: self.next_sequence(),
            callback,
        };
        self.providers.push(entry);
        Self::sort_entries(&mut self.providers);
    }

    /// Add a synchronous callback.
    pub fn add_callback<F>(&mut self, callback: F)
    where
        F: Fn(&mut HookEvent) + Send + Sync + 'static,
    {
        let callback: HookCallback = Arc::new(callback);
        self.push_callback(callback, HookOrder::DEFAULT);
    }

    /// Add a synchronous callback at an explicit priority.
    pub fn add_callback_at<F>(&mut self, callback: F, order: f64)
    where
        F: Fn(&mut HookEvent) + Send + Sync + 'static,
    {
        let callback: HookCallback = Arc::new(callback);
        self.push_callback(callback, order);
    }

    fn push_callback(&mut self, callback: HookCallback, order: f64) {
        let entry = CallbackEntry {
            order,
            sequence: self.next_sequence(),
            callback,
        };
        self.callbacks.push(entry);
        Self::sort_entries(&mut self.callbacks);
    }

    /// Add an async callback.
    pub fn add_async_callback<F, Fut>(&mut self, callback: F)
    where
        F: Fn(&mut HookEvent) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        self.add_async_callback_at(callback, HookOrder::DEFAULT);
    }

    /// Add an async callback at an explicit priority.
    pub fn add_async_callback_at<F, Fut>(&mut self, callback: F, order: f64)
    where
        F: Fn(&mut HookEvent) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let callback: AsyncHookCallback = Arc::new(move |event| {
            Box::pin(callback(event))
        });
        let entry = CallbackEntry {
            order,
            sequence: self.next_sequence(),
            callback,
        };
        self.async_callbacks.push(entry);
        Self::sort_entries(&mut self.async_callbacks);
    }

    fn sort_entries<T>(entries: &mut Vec<CallbackEntry<T>>) {
        entries.sort_by(|a, b| {
            a.order.partial_cmp(&b.order).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.sequence.cmp(&b.sequence))
        });
    }

    /// Invoke all callbacks for an event.
    ///
    /// Returns interrupts raised by interruptible events during the pass.
    pub async fn invoke(&self, event: &mut HookEvent) -> Vec<Interrupt> {
        let mut interrupts = Vec::new();

        let reverse = self.should_reverse || event.should_reverse_callbacks();

        self.dispatch(event, self.callbacks_list(), reverse).await;
        self.dispatch(event, self.async_callbacks_list(), reverse).await;
        self.dispatch(event, self.providers_list(), reverse).await;

        if let Some(interrupt) = event.take_interrupt() {
            interrupts.push(interrupt);
        }

        interrupts
    }

    /// Convenience wrapper that takes ownership of the event.
    ///
    /// Returns the mutated event along with any interrupts raised, so callers
    /// can read control fields set by hooks.
    pub async fn fire(&self, event: HookEvent) -> (HookEvent, Vec<Interrupt>) {
        let mut event = event;
        let interrupts = self.invoke(&mut event).await;
        (event, interrupts)
    }

    /// Convenience wrapper that takes ownership and discards the event.
    pub async fn fire_and_forget(&self, event: HookEvent) {
        let _ = self.fire(event).await;
    }

    /// Invoke callbacks synchronously (panics if async callbacks exist).
    pub fn invoke_sync(&self, event: &mut HookEvent) -> Vec<Interrupt> {
        if !self.async_callbacks.is_empty() {
            panic!("Cannot invoke sync with async callbacks registered");
        }

        let reverse = self.should_reverse || event.should_reverse_callbacks();

        self.dispatch_sync(event, self.callbacks_list(), reverse);

        event.take_interrupt().into_iter().collect()
    }
    async fn dispatch<T>(&self, event: &mut HookEvent, entries: Vec<T>, reverse: bool)
    where
        T: InvokeTarget,
    {
        if reverse {
            for target in entries.into_iter().rev() {
                target.invoke(event).await;
            }
        } else {
            for target in entries {
                target.invoke(event).await;
            }
        }
    }

    fn dispatch_sync<T>(&self, event: &mut HookEvent, entries: Vec<T>, reverse: bool)
    where
        T: SyncInvokeTarget,
    {
        if reverse {
            for target in entries.into_iter().rev() {
                target.invoke_sync(event);
            }
        } else {
            for target in entries {
                target.invoke_sync(event);
            }
        }
    }

    fn callbacks_list(&self) -> Vec<HookCallback> {
        self.callbacks.iter().map(|e| e.callback.clone()).collect()
    }

    fn async_callbacks_list(&self) -> Vec<AsyncHookCallback> {
        self.async_callbacks.iter().map(|e| e.callback.clone()).collect()
    }

    fn providers_list(&self) -> Vec<Arc<dyn HookProvider>> {
        self.providers.iter().map(|e| e.callback.clone()).collect()
    }

    pub fn has_callbacks(&self) -> bool {
        !self.providers.is_empty() || !self.callbacks.is_empty() || !self.async_callbacks.is_empty()
    }

    pub fn len(&self) -> usize {
        self.providers.len() + self.callbacks.len() + self.async_callbacks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
trait InvokeTarget {
    async fn invoke(&self, event: &mut HookEvent);
}

#[async_trait]
impl InvokeTarget for HookCallback {
    async fn invoke(&self, event: &mut HookEvent) {
        (self)(event);
    }
}

#[async_trait]
impl InvokeTarget for AsyncHookCallback {
    async fn invoke(&self, event: &mut HookEvent) {
        (self)(event).await;
    }
}

#[async_trait]
impl InvokeTarget for Arc<dyn HookProvider> {
    async fn invoke(&self, event: &mut HookEvent) {
        self.on_event(event).await;
    }
}

trait SyncInvokeTarget {
    fn invoke_sync(&self, event: &mut HookEvent);
}

impl SyncInvokeTarget for HookCallback {
    fn invoke_sync(&self, event: &mut HookEvent) {
        (self)(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    struct RecordingProvider {
        hits: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl HookProvider for RecordingProvider {
        async fn on_event(&self, _event: &mut HookEvent) {
            self.hits.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    #[tokio::test]
    async fn test_callback_ordering() {
        let mut registry = HookRegistry::new();

        let order: Arc<std::sync::Mutex<Vec<&'static str>>> = Arc::default();
        let sink = order.clone();
        registry.add_callback_at(
            move |_| {
                sink.lock().unwrap().push("late");
            },
            HookOrder::SDK_LAST,
        );
        let sink = order.clone();
        registry.add_callback_at(
            move |_| {
                sink.lock().unwrap().push("first");
            },
            HookOrder::SDK_FIRST,
        );

        registry.fire_and_forget(HookEvent::BeforeInvocation(BeforeInvocationEvent::default())).await;

        assert_eq!(*order.lock().unwrap(), vec!["first", "late"]);
    }

    #[tokio::test]
    async fn test_interrupt_collection() {
        let mut registry = HookRegistry::new();
        registry.add_callback(|event: &mut HookEvent| {
            if let HookEvent::BeforeToolCall(e) = event {
                e.interrupt("approval", Some(serde_json::json!({"action": "delete"})));
            }
        });

        let tool_use = ToolUse::new("risky_tool", "t1", serde_json::json!({}));
        let (_, interrupts) = registry
            .fire(HookEvent::BeforeToolCall(BeforeToolCallEvent::new(tool_use)))
            .await;

        assert_eq!(interrupts.len(), 1);
        assert_eq!(interrupts[0].name, "approval");
        assert!(interrupts[0].id.starts_with("v1:before_tool_call:t1:"));
    }

    #[tokio::test]
    async fn test_provider_invocation() {
        let mut registry = HookRegistry::new();
        let hits = Arc::new(AtomicUsize::new(0));
        registry.add_provider(RecordingProvider { hits: hits.clone() });

        registry.fire_and_forget(HookEvent::AgentInitialized(AgentInitializedEvent)).await;

        assert_eq!(hits.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_mutable_control_fields() {
        let mut registry = HookRegistry::new();
        registry.add_callback(|event: &mut HookEvent| {
            if let HookEvent::BeforeInvocation(e) = event {
                e.cancel("blocked by policy");
            }
        });

        let (event, _) = registry
            .fire(HookEvent::BeforeInvocation(BeforeInvocationEvent::default()))
            .await;

        match event {
            HookEvent::BeforeInvocation(e) => assert_eq!(e.cancel.as_deref(), Some("blocked by policy")),
            _ => panic!("unexpected event"),
        }
    }

    #[test]
    fn test_invoke_sync_no_async() {
        let mut registry = HookRegistry::new();
        registry.add_callback(|_| {});

        let mut event = HookEvent::AgentInitialized(AgentInitializedEvent);
        let interrupts = registry.invoke_sync(&mut event);
        assert!(interrupts.is_empty());
    }
}
