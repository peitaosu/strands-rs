//! Hook system for agent lifecycle events.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::agent::AgentResult;
use crate::types::content::Message;
use crate::types::streaming::StopReason;
use crate::types::tools::{ToolResult, ToolUse};

/// Interrupt for human-in-the-loop workflows.
#[derive(Debug, Clone)]
pub struct Interrupt {
    pub id: String,
    pub name: String,
    pub reason: Option<serde_json::Value>,
    pub response: Option<serde_json::Value>,
}

impl Interrupt {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            reason: None,
            response: None,
        }
    }

    pub fn with_reason(mut self, reason: serde_json::Value) -> Self {
        self.reason = Some(reason);
        self
    }
}

/// State for managing interrupts during agent execution.
#[derive(Debug, Clone, Default)]
pub struct InterruptState {
    pub interrupts: HashMap<String, Interrupt>,
}

impl InterruptState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_interrupt(&mut self, interrupt: Interrupt) {
        self.interrupts.insert(interrupt.id.clone(), interrupt);
    }

    pub fn get_response(&self, id: &str) -> Option<&serde_json::Value> {
        self.interrupts.get(id).and_then(|i| i.response.as_ref())
    }

    pub fn set_response(&mut self, id: &str, response: serde_json::Value) {
        if let Some(interrupt) = self.interrupts.get_mut(id) {
            interrupt.response = Some(response);
        }
    }
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

/// Event triggered when an agent has finished initialization.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct BeforeInvocationEvent;

impl HookEventBase for BeforeInvocationEvent {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Event triggered at the end of an agent request.
#[derive(Debug, Clone)]
pub struct AfterInvocationEvent {
    pub result: Option<AgentResult>,
}

impl AfterInvocationEvent {
    pub fn new(result: Option<AgentResult>) -> Self {
        Self { result }
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

/// Trait for events that can generate interrupts.
pub trait Interruptible {
    /// Generate a unique interrupt ID for this event.
    ///
    /// The ID should be deterministic based on the event context and name,
    /// allowing for consistent interrupt handling across sessions.
    fn interrupt_id(&self, name: &str) -> String;
}

/// Event triggered before a tool is invoked.
#[derive(Debug, Clone)]
pub struct BeforeToolCallEvent {
    pub tool_use: ToolUse,
    pub invocation_state: HashMap<String, serde_json::Value>,
    pub cancel_tool: Option<String>,
}

impl BeforeToolCallEvent {
    pub fn new(tool_use: ToolUse) -> Self {
        Self {
            tool_use,
            invocation_state: HashMap::new(),
            cancel_tool: None,
        }
    }

    pub fn with_state(mut self, state: HashMap<String, serde_json::Value>) -> Self {
        self.invocation_state = state;
        self
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
    pub invocation_state: HashMap<String, serde_json::Value>,
    pub result: ToolResult,
    pub exception: Option<String>,
    pub cancel_message: Option<String>,
}

impl AfterToolCallEvent {
    pub fn new(tool_use: ToolUse, result: ToolResult) -> Self {
        Self {
            tool_use,
            invocation_state: HashMap::new(),
            result,
            exception: None,
            cancel_message: None,
        }
    }

    pub fn with_exception(mut self, exception: String) -> Self {
        self.exception = Some(exception);
        self
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
#[derive(Debug, Clone)]
pub struct BeforeModelCallEvent;

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
#[derive(Debug, Clone)]
pub struct AfterModelCallEvent {
    pub stop_response: Option<ModelStopResponse>,
    pub exception: Option<String>,
}

impl AfterModelCallEvent {
    pub fn success(message: Message, stop_reason: StopReason) -> Self {
        Self {
            stop_response: Some(ModelStopResponse {
                message,
                stop_reason,
            }),
            exception: None,
        }
    }

    pub fn error(exception: String) -> Self {
        Self {
            stop_response: None,
            exception: Some(exception),
        }
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

/// Enum wrapper for all hook events.
#[derive(Debug, Clone)]
pub enum HookEvent {
    AgentInitialized(AgentInitializedEvent),
    BeforeInvocation(BeforeInvocationEvent),
    AfterInvocation(AfterInvocationEvent),
    MessageAdded(MessageAddedEvent),
    BeforeToolCall(BeforeToolCallEvent),
    AfterToolCall(AfterToolCallEvent),
    BeforeModelCall(BeforeModelCallEvent),
    AfterModelCall(AfterModelCallEvent),
}

impl HookEvent {
    pub fn should_reverse_callbacks(&self) -> bool {
        match self {
            Self::AfterInvocation(_) | Self::AfterToolCall(_) | Self::AfterModelCall(_) => true,
            _ => false,
        }
    }
}

/// Trait for implementing hook providers.
#[async_trait]
pub trait HookProvider: Send + Sync {
    /// Called when a hook event occurs.
    async fn on_event(&self, event: &HookEvent);
}

/// Callback function type for hook events.
pub type HookCallback = Arc<dyn Fn(&HookEvent) + Send + Sync>;

/// Async callback function type for hook events.
pub type AsyncHookCallback = Arc<dyn Fn(&HookEvent) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>;

/// Registry for managing hook providers and callbacks.
#[derive(Default)]
pub struct HookRegistry {
    providers: Vec<Arc<dyn HookProvider>>,
    callbacks: Vec<HookCallback>,
    async_callbacks: Vec<AsyncHookCallback>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a hook provider.
    pub fn add_provider(&mut self, provider: impl HookProvider + 'static) {
        self.providers.push(Arc::new(provider));
    }

    /// Add a hook provider as Arc.
    pub fn add_provider_arc(&mut self, provider: Arc<dyn HookProvider>) {
        self.providers.push(provider);
    }

    /// Add a synchronous callback.
    pub fn add_callback<F>(&mut self, callback: F)
    where
        F: Fn(&HookEvent) + Send + Sync + 'static,
    {
        self.callbacks.push(Arc::new(callback));
    }

    /// Add an async callback.
    pub fn add_async_callback<F, Fut>(&mut self, callback: F)
    where
        F: Fn(&HookEvent) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        self.async_callbacks.push(Arc::new(move |event| {
            Box::pin(callback(event))
        }));
    }

    /// Invoke all callbacks for an event.
    pub async fn invoke(&self, event: &HookEvent) -> Vec<Interrupt> {
        let interrupts = Vec::new();

        let reverse = event.should_reverse_callbacks();

        if reverse {
            for callback in self.callbacks.iter().rev() {
                callback(event);
            }
        } else {
            for callback in &self.callbacks {
                callback(event);
            }
        }

        if reverse {
            for callback in self.async_callbacks.iter().rev() {
                callback(event).await;
            }
        } else {
            for callback in &self.async_callbacks {
                callback(event).await;
            }
        }

        if reverse {
            for provider in self.providers.iter().rev() {
                provider.on_event(event).await;
            }
        } else {
            for provider in &self.providers {
                provider.on_event(event).await;
            }
        }

        interrupts
    }

    /// Invoke callbacks synchronously (panics if async callbacks exist).
    pub fn invoke_sync(&self, event: &HookEvent) -> Vec<Interrupt> {
        if !self.async_callbacks.is_empty() {
            panic!("Cannot invoke sync with async callbacks registered");
        }

        let interrupts = Vec::new();
        let reverse = event.should_reverse_callbacks();

        if reverse {
            for callback in self.callbacks.iter().rev() {
                callback(event);
            }
        } else {
            for callback in &self.callbacks {
                callback(event);
            }
        }

        interrupts
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
