//! Agent implementation for conversational AI.

mod builder;
mod limits;
mod result;
mod state;

use std::sync::Arc;

use futures::StreamExt;

use crate::conversation::ConversationManager;
use crate::hooks::HookRegistry;
use crate::models::Model;
use crate::telemetry::EventLoopMetrics;
use crate::tools::{InvocationState, ToolRegistry};
use crate::types::content::{
    ContentBlock, Message, MessageMetadata, Messages, ReasoningContentBlock, ReasoningTextBlock,
    Role,
};
use crate::types::errors::{Result, StrandsError};

pub use builder::AgentBuilder;
pub use limits::Limits;
pub use result::AgentResult;
pub use state::AgentState;

/// Input types for agent invocation.
#[derive(Debug, Clone, Default)]
pub enum AgentInput {
    Text(String),
    ContentBlocks(Vec<ContentBlock>),
    Messages(Messages),
    /// Interrupt responses resuming a paused invocation.
    InterruptResponses(Vec<crate::types::interrupt::InterruptResponseContent>),
    #[default]
    None,
}

impl From<&str> for AgentInput {
    fn from(s: &str) -> Self { AgentInput::Text(s.to_string()) }
}

impl From<String> for AgentInput {
    fn from(s: String) -> Self { AgentInput::Text(s) }
}

impl From<Vec<ContentBlock>> for AgentInput {
    fn from(blocks: Vec<ContentBlock>) -> Self { AgentInput::ContentBlocks(blocks) }
}

impl From<Vec<crate::types::interrupt::InterruptResponseContent>> for AgentInput {
    fn from(responses: Vec<crate::types::interrupt::InterruptResponseContent>) -> Self {
        AgentInput::InterruptResponses(responses)
    }
}

impl From<Messages> for AgentInput {
    fn from(messages: Messages) -> Self { AgentInput::Messages(messages) }
}

impl<T: Into<String>> From<Option<T>> for AgentInput {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(s) => AgentInput::Text(s.into()),
            None => AgentInput::None,
        }
    }
}

/// Tool caller for direct tool invocation.
pub struct ToolCaller<'a> {
    agent: &'a mut Agent,
}

impl<'a> ToolCaller<'a> {
    /// Invoke a tool by name with the given input.
    pub async fn invoke(
        &mut self,
        tool_name: &str,
        input: serde_json::Value,
    ) -> Result<crate::types::tools::ToolResult> {
        self.invoke_with_options(tool_name, input, None, None).await
    }

    /// Invoke a tool by name with options.
    pub async fn invoke_with_options(
        &mut self,
        tool_name: &str,
        input: serde_json::Value,
        user_message_override: Option<&str>,
        record_direct_tool_call: Option<bool>,
    ) -> Result<crate::types::tools::ToolResult> {
        use crate::types::tools::{ToolResult, ToolUse};
        use crate::tools::ToolContext;

        if self.agent.interrupt_state.activated {
            return Err(StrandsError::EventLoopError {
                message: "cannot directly call tool during interrupt".to_string(),
            });
        }

        let tool = self.agent.tool_registry.get(tool_name)
            .ok_or_else(|| StrandsError::ToolNotFound {
                tool_name: tool_name.to_string(),
            })?;

        let tool_id = format!("tooluse_{}_{}", tool_name, uuid::Uuid::new_v4());
        let tool_use = ToolUse {
            name: tool_name.to_string(),
            tool_use_id: tool_id.clone(),
            input: input.clone(),
            reasoning_signature: None,
        };

        let context =
            ToolContext::new(tool_use.clone()).with_cancel_signal(self.agent.cancel_signal.clone());
        let result = match tool.invoke(input.clone(), &context).await {
            Ok(r) => ToolResult {
                tool_use_id: tool_id.clone(),
                status: r.status,
                content: r.content,
            },
            Err(e) => ToolResult::error(&tool_id, e),
        };

        let should_record = record_direct_tool_call
            .unwrap_or(self.agent.record_direct_tool_call);

        if should_record {
            self.record_tool_execution(&tool_use, &result, user_message_override).await?;
        }

        self.agent.conversation_manager.apply_management(&mut self.agent.messages);

        Ok(result)
    }

    async fn record_tool_execution(
        &mut self,
        tool_use: &crate::types::tools::ToolUse,
        tool_result: &crate::types::tools::ToolResult,
        user_message_override: Option<&str>,
    ) -> Result<()> {
        let input_json = serde_json::to_string(&tool_use.input)
            .unwrap_or_else(|_| "<<non-serializable>>".to_string());

        let mut user_content = Vec::new();
        if let Some(msg) = user_message_override {
            user_content.push(ContentBlock::text(format!("{}\n", msg)));
        }
        user_content.push(ContentBlock::text(format!(
            "agent.tool.{} direct tool call.\nInput parameters: {}\n",
            tool_use.name, input_json
        )));

        let user_msg = Message::new(Role::User, user_content);
        let tool_use_msg = Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_use(tool_use.clone())],
        );
        let tool_result_msg = Message::new(
            Role::User,
            vec![ContentBlock::tool_result(tool_result.clone())],
        );
        let assistant_msg = Message::new(
            Role::Assistant,
            vec![ContentBlock::text(format!("agent.tool.{} was called.", tool_use.name))],
        );

        self.agent.messages.push(user_msg);
        self.agent.messages.push(tool_use_msg);
        self.agent.messages.push(tool_result_msg);
        self.agent.messages.push(assistant_msg);

        Ok(())
    }
}

/// The main agent struct for conversational AI.
pub struct Agent {
    pub(crate) model: Arc<dyn Model>,
    pub(crate) messages: Messages,
    pub(crate) system_prompt: Option<String>,
    pub(crate) tool_registry: ToolRegistry,
    agent_name: Option<String>,
    pub agent_id: String,
    pub description: Option<String>,
    pub state: AgentState,
    pub(crate) hooks: HookRegistry,
    pub(crate) conversation_manager: Box<dyn ConversationManager>,
    interrupt_state: crate::types::interrupt::InterruptState,
    /// Whether to record direct tool calls in message history.
    pub record_direct_tool_call: bool,
    /// Custom trace attributes for OpenTelemetry.
    pub trace_attributes: std::collections::HashMap<String, String>,
    /// Maximum number of tool calls per cycle (None = unlimited).
    pub max_tool_calls: Option<usize>,
    /// Structured output context for typed responses.
    pub(crate) structured_output_context: Option<crate::tools::structured_output::StructuredOutputContext>,
    /// Shared cancellation signal for the in-flight invocation.
    pub(crate) cancel_signal: crate::async_utils::CancelSignal,
    /// Set when cancel(after_current_tools=true) defers cancellation to the
    /// end of the current tool cycle.
    pub(crate) deferred_cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Message reported when the invocation is cancelled.
    pub(crate) cancel_message: Arc<std::sync::Mutex<Option<String>>>,
    /// Invocation limits checked before each turn.
    pub(crate) limits: Limits,
    /// Retry configuration applied to model calls.
    pub(crate) retry_strategy: crate::event_loop::ModelRetryStrategy,
    /// Middleware handlers for the three execution stages.
    pub(crate) middleware: crate::middleware::MiddlewareRegistry,
}

impl Agent {
    /// Creates a new agent builder.
    pub fn builder() -> AgentBuilder { AgentBuilder::new() }

    /// Returns the agent name.
    pub fn name(&self) -> Option<&String> { self.agent_name.as_ref() }

    /// Sets the agent name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.agent_name = Some(name.into());
    }

    /// Returns the system prompt.
    pub fn system_prompt(&self) -> Option<&str> { self.system_prompt.as_deref() }

    /// Sets the system prompt.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = Some(prompt.into());
    }

    /// Returns the current messages.
    pub fn messages(&self) -> &Messages { &self.messages }

    /// Adds a message to the conversation.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Clears all messages.
    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }

    /// Returns the tool registry.
    pub fn tool_registry(&self) -> &ToolRegistry { &self.tool_registry }

    /// Returns a mutable reference to the tool registry.
    pub fn tool_registry_mut(&mut self) -> &mut ToolRegistry { &mut self.tool_registry }

    /// Returns the names of all registered tools.
    pub fn tool_names(&self) -> Vec<&str> { self.tool_registry.tool_names() }

    /// Returns the agent ID.
    pub fn agent_id(&self) -> Option<&str> { Some(&self.agent_id) }

    /// Returns the hooks registry.
    pub fn hooks(&self) -> &HookRegistry { &self.hooks }

    /// Returns a mutable reference to the hooks registry.
    pub fn hooks_mut(&mut self) -> &mut HookRegistry { &mut self.hooks }

    /// Returns the conversation manager.
    pub fn conversation_manager(&self) -> &dyn ConversationManager { self.conversation_manager.as_ref() }

    /// Returns a mutable reference to the conversation manager.
    pub fn conversation_manager_mut(&mut self) -> &mut dyn ConversationManager { self.conversation_manager.as_mut() }

    /// Returns a reference to the agent state.
    pub fn state(&self) -> &AgentState { &self.state }

    /// Returns a mutable reference to the agent state.
    pub fn state_mut(&mut self) -> &mut AgentState { &mut self.state }

    /// Returns a reference to the interrupt state.
    pub fn interrupt_state(&self) -> &crate::types::interrupt::InterruptState { &self.interrupt_state }

    /// Returns a mutable reference to the interrupt state.
    pub fn interrupt_state_mut(&mut self) -> &mut crate::types::interrupt::InterruptState { &mut self.interrupt_state }

    /// Sets the interrupt state.
    pub fn set_interrupt_state(&mut self, state: crate::types::interrupt::InterruptState) {
        self.interrupt_state = state;
    }

    /// Returns whether the agent is currently in an interrupt state.
    pub fn is_interrupted(&self) -> bool { self.interrupt_state.activated }

    /// Sets the messages directly.
    pub fn set_messages(&mut self, messages: Messages) {
        self.messages = messages;
    }

    /// Returns a tool caller for direct tool invocation.
    ///
    /// This enables the pattern: `agent.tool().invoke("tool_name", input).await`
    pub fn tool(&mut self) -> ToolCaller<'_> {
        ToolCaller { agent: self }
    }

    /// Returns the trace attributes.
    pub fn trace_attributes(&self) -> &std::collections::HashMap<String, String> {
        &self.trace_attributes
    }

    /// Sets a trace attribute.
    pub fn set_trace_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.trace_attributes.insert(key.into(), value.into());
    }

    /// Returns the max tool calls setting.
    pub fn max_tool_calls(&self) -> Option<usize> {
        self.max_tool_calls
    }

    /// Sets the max tool calls.
    pub fn set_max_tool_calls(&mut self, max: Option<usize>) {
        self.max_tool_calls = max;
    }

    /// Invokes the agent synchronously (blocking).
    pub fn call(&mut self, prompt: impl Into<AgentInput>) -> Result<AgentResult> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.invoke_async(prompt))
        })
    }

    /// Clears stale interrupt bookkeeping before a fresh (non-resume) input.
    ///
    /// A stored pending tool batch survives even when every interrupt has an
    /// answer, so the next invocation can still replay it.
    fn clear_stale_interrupt_state(&mut self) {
        if !self.interrupt_state.has_pending()
            && self.interrupt_state.pending_tool_execution.is_none()
        {
            self.interrupt_state.end_tool_cycle();
        }
    }

    /// Invokes the agent asynchronously.
    pub async fn invoke_async(&mut self, prompt: impl Into<AgentInput>) -> Result<AgentResult> {
        let input = prompt.into();
        let is_resume = matches!(input, AgentInput::InterruptResponses(_));
        if !is_resume {
            self.clear_stale_interrupt_state();
        }

        self.run_event_loop(input).await
    }

    /// Returns a stream of events from the agent.
    pub async fn stream_async(
        &mut self,
        prompt: impl Into<AgentInput>,
    ) -> impl futures::Stream<Item = Result<crate::event_loop::TypedEvent>> + '_ {
        let input = prompt.into();
        let is_resume = matches!(input, AgentInput::InterruptResponses(_));
        if !is_resume {
            self.clear_stale_interrupt_state();
        }

        async_stream::stream! {
            match self.run_event_loop(input).await {
                Ok(result) => yield Ok(crate::event_loop::TypedEvent::agent_result(result)),
                Err(e) => yield Err(e),
            }
        }
    }

    fn convert_input_to_messages(&mut self, input: AgentInput) -> Result<Messages> {
        match input {
            AgentInput::Text(text) => Ok(vec![Message::user(text)]),
            AgentInput::ContentBlocks(blocks) => Ok(vec![Message::new(Role::User, blocks)]),
            AgentInput::Messages(messages) => Ok(messages),
            AgentInput::InterruptResponses(responses) => {
                if !self.interrupt_state.activated {
                    return Err(StrandsError::InvalidInput {
                        message: "interrupt responses provided but the agent is not in an interrupt state"
                            .to_string(),
                    });
                }
                self.interrupt_state.resume(responses).map_err(|e| {
                    StrandsError::Interrupted { message: e }
                })?;
                Ok(vec![])
            }
            AgentInput::None => Ok(vec![]),
        }
    }

    /// Appends a message to the conversation, assigning a tracking id and
    /// emitting `MessageAddedEvent`.
    pub async fn append_message(&mut self, mut message: Message) {
        message.ensure_tracking_id();
        self.messages.push(message.clone());
        self.hooks
            .fire_and_forget(crate::hooks::HookEvent::MessageAdded(
                crate::hooks::MessageAddedEvent::new(message),
            ))
            .await;
    }

    /// Requests cancellation of the in-flight invocation.
    ///
    /// With `after_current_tools`, cancellation takes effect after the current
    /// tool cycle completes; otherwise in-flight model streams and tools are
    /// stopped as soon as possible.
    pub fn cancel(&self, message: impl Into<String>, after_current_tools: bool) {
        *self.cancel_message.lock().unwrap() = Some(message.into());
        if after_current_tools {
            self.deferred_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        } else {
            self.cancel_signal.set();
        }
    }

    /// Returns the shared cancellation signal.
    pub fn cancel_signal(&self) -> crate::async_utils::CancelSignal {
        self.cancel_signal.clone()
    }
    /// Returns true when cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_signal.is_set() || self.deferred_cancel.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Returns the middleware registry.
    pub fn middleware(&self) -> &crate::middleware::MiddlewareRegistry { &self.middleware }

    /// Returns the invocation limits.
    pub fn limits(&self) -> Limits { self.limits }

    /// Sets the invocation limits.
    pub fn set_limits(&mut self, limits: Limits) { self.limits = limits; }

    /// Captures a versioned snapshot of the agent's current state.
    pub fn take_snapshot(&self, scope: &str) -> Result<crate::types::snapshot::Snapshot> {
        let data = serde_json::json!({
            "messages": self.messages,
            "state": serde_json::to_value(&self.state).unwrap_or(serde_json::Value::Null),
            "system_prompt": self.system_prompt,
            "interrupt_state": serde_json::to_value(&self.interrupt_state)
                .unwrap_or(serde_json::Value::Null),
        });
        Ok(crate::types::snapshot::Snapshot::new(scope.to_string(), data))
    }

    /// Restores agent state from a snapshot.
    pub fn load_snapshot(
        &mut self,
        snapshot: crate::types::snapshot::Snapshot,
    ) -> Result<()> {
        let data = snapshot.data;

        let messages: Messages = serde_json::from_value(
            data.get("messages").cloned().unwrap_or(serde_json::Value::Null),
        )
        .map_err(|e| StrandsError::SnapshotException {
            message: format!("invalid snapshot messages: {}", e),
        })?;

        let state: AgentState = serde_json::from_value(
            data.get("state").cloned().unwrap_or(serde_json::Value::Null),
        )
        .map_err(|e| StrandsError::SnapshotException {
            message: format!("invalid snapshot state: {}", e),
        })?;

        self.system_prompt = data
            .get("system_prompt")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if let Some(interrupt_state) = data.get("interrupt_state") {
            if !interrupt_state.is_null() {
                self.interrupt_state =
                    serde_json::from_value(interrupt_state.clone()).map_err(|e| {
                        StrandsError::SnapshotException {
                            message: format!("invalid snapshot interrupt state: {}", e),
                        }
                    })?;
            }
        }

        self.messages = messages;
        self.state = state;
        Ok(())
    }

    fn cancel_message_text(&self) -> String {
        self.cancel_message
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| "Cancelled by user".to_string())
    }

    fn cancelled_result(&self, message: &str, invocation_state: &InvocationState) -> AgentResult {
        AgentResult {
            stop_reason: crate::types::streaming::StopReason::Cancelled,
            message: Message::assistant(message),
            metrics: EventLoopMetrics::default(),
            state: invocation_state.clone(),
            interrupts: None,
            structured_output: None,
            checkpoint: None,
        }
    }

    async fn run_event_loop(&mut self, input: AgentInput) -> Result<AgentResult> {
        use crate::hooks::{BeforeInvocationEvent, AfterInvocationEvent, HookEvent};

        let mut invocation_state = InvocationState::new();
        let (before_event, _) = self
            .hooks
            .fire(HookEvent::BeforeInvocation(BeforeInvocationEvent {
                invocation_state: Some(invocation_state.clone()),
                cancel: None,
            }))
            .await;

        if let HookEvent::BeforeInvocation(event) = before_event {
            if let Some(message) = event.cancel {
                self.reset_cancellation();
                return Ok(self.cancelled_result(&message, &invocation_state));
            }
        }

        let new_messages = self.convert_input_to_messages(input)?;
        for message in new_messages {
            self.append_message(message).await;
        }

        let mut result = self.invoke_cycle(&mut invocation_state).await;

        loop {
            let resume_input = {
                let agent_result = result.as_ref().ok().cloned();
                let (after_event, _) = self
                    .hooks
                    .fire(HookEvent::AfterInvocation(AfterInvocationEvent::new(agent_result)))
                    .await;
                match after_event {
                    HookEvent::AfterInvocation(event) => event.resume,
                    _ => None,
                }
            };

            match resume_input {
                Some(input) => {
                    let new_messages = self.convert_input_to_messages(input)?;
                    for message in new_messages {
                        self.append_message(message).await;
                    }
                    result = self.invoke_cycle(&mut invocation_state).await;
                }
                None => break,
            }
        }

        // The cancel signal clears when an invocation completes, so the
        // agent is ready for the next request. Invocation-scoped interrupts
        // release the same way when the state is not activated.
        self.reset_cancellation();
        if !self.interrupt_state.activated {
            self.interrupt_state.end_interrupt_cycle();
        }

        result
    }

    /// Clears cancellation state after an invocation completes.
    fn reset_cancellation(&mut self) {
        self.cancel_signal = crate::async_utils::CancelSignal::new();
        self.deferred_cancel
            .store(false, std::sync::atomic::Ordering::SeqCst);
        *self.cancel_message.lock().unwrap() = None;
    }

    async fn invoke_cycle(
        &mut self,
        invocation_state: &mut InvocationState,
    ) -> Result<AgentResult> {
        let mut structured_output_ctx = self.structured_output_context.clone();

        if let Some(ref ctx) = structured_output_ctx {
            ctx.register_tool(&mut self.tool_registry);
        }

        let result = self
            .event_loop_inner(invocation_state, &mut structured_output_ctx)
            .await;

        if let Some(ref ctx) = structured_output_ctx {
            ctx.cleanup(&mut self.tool_registry);
        }

        self.conversation_manager.apply_management(&mut self.messages);
        result
    }

    async fn event_loop_inner(
        &mut self,
        invocation_state: &mut InvocationState,
        structured_output_ctx: &mut Option<crate::tools::structured_output::StructuredOutputContext>,
    ) -> Result<AgentResult> {
        use crate::event_loop::recover_message_on_max_tokens_reached;
        use crate::hooks::{
            AfterModelCallEvent, BeforeModelCallEvent, EventLoopThrottleEvent, HookEvent,
        };
        use crate::types::streaming::StopReason;

        let mut retry_strategy = self.retry_strategy.clone();
        let mut metrics = EventLoopMetrics::default();
        let mut cycle_id: u64 = 0;
        let mut total_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;
        let mut structured_output_attempts: u32 = 0;

        loop {
            cycle_id += 1;
            invocation_state.set("event_loop_cycle_id", cycle_id);

            if let Some(limit_stop) = self
                .limits
                .check(cycle_id as u32, total_tokens, output_tokens, structured_output_attempts)
            {
                return Ok(AgentResult {
                    stop_reason: limit_stop,
                    message: Message::assistant(format!(
                        "Invocation stopped: {} limit reached",
                        limit_stop
                    )),
                    metrics: metrics.clone(),
                    state: invocation_state.clone(),
                    interrupts: None,
                    structured_output: None,
                    checkpoint: None,
                });
            }

            if self.cancel_signal.is_set() {
                return Ok(self.cancelled_result(&self.cancel_message_text(), invocation_state));
            }

            if self.interrupt_state.activated {
                if let Some(pending) = self.interrupt_state.pending_tool_execution.clone() {
                    let completed_ids: std::collections::HashSet<&str> = pending
                        .completed_tool_results
                        .iter()
                        .map(|result| result.tool_use_id.as_str())
                        .collect();

                    let remaining_content: Vec<ContentBlock> = pending
                        .assistant_message
                        .content
                        .iter()
                        .filter(|block| {
                            block
                                .as_tool_use()
                                .map(|tool_use| !completed_ids.contains(tool_use.tool_use_id.as_str()))
                                .unwrap_or(false)
                        })
                        .cloned()
                        .collect();

                    if !remaining_content.is_empty() {
                        let (tool_results, extracted_output, interrupts, end_turn_forced) = self
                            .execute_tools_with_structured_output(
                                &remaining_content,
                                invocation_state,
                                structured_output_ctx,
                            )
                            .await?;

                        if let Some(tool_interrupts) = interrupts {
                            self.interrupt_state.activated = true;
                            for interrupt in &tool_interrupts {
                                self.interrupt_state.add(interrupt.clone());
                            }
                            return Ok(AgentResult {
                                stop_reason: StopReason::Interrupt,
                                message: pending.assistant_message,
                                metrics: metrics.clone(),
                                state: invocation_state.clone(),
                                interrupts: Some(tool_interrupts),
                                structured_output: None,
                                checkpoint: None,
                            });
                        }

                        let tool_result_message = Message::new(
                            Role::User,
                            pending
                                .completed_tool_results
                                .iter()
                                .chain(tool_results.iter())
                                .map(|result| ContentBlock::tool_result(result.clone()))
                                .collect(),
                        );
                        self.append_message(tool_result_message).await;
                        self.interrupt_state.end_tool_cycle();

                        if end_turn_forced {
                            return Ok(AgentResult {
                                stop_reason: StopReason::EndTurn,
                                message: pending.assistant_message,
                                metrics: metrics.clone(),
                                state: invocation_state.clone(),
                                interrupts: None,
                                structured_output: None,
                                checkpoint: None,
                            });
                        }

                        if extracted_output.is_some() {
                            return Ok(AgentResult {
                                stop_reason: StopReason::EndTurn,
                                message: pending.assistant_message,
                                metrics: metrics.clone(),
                                state: invocation_state.clone(),
                                interrupts: None,
                                structured_output: extracted_output,
                                checkpoint: None,
                            });
                        }
                        continue;
                    }
                }
            }

            let tool_specs = self.tool_registry.get_all_tool_specs();
            let messages_snapshot: Messages =
                self.messages.iter().map(|m| m.to_model_message()).collect();
            let system_prompt_snapshot = self.system_prompt.clone();

            let mut before_event = HookEvent::BeforeModelCall(BeforeModelCallEvent {
                invocation_state: Some(invocation_state.clone()),
                projected_input_tokens: None,
                cancel: None,
                interrupt_slot: None,
            });
            let before_interrupts = self.hooks.invoke(&mut before_event).await;

            if let HookEvent::BeforeModelCall(ref event) = before_event {
                if let Some(message) = &event.cancel {
                    return Ok(self.cancelled_result(message, invocation_state));
                }
            }
            if let Some(interrupt) = before_interrupts.into_iter().next() {
                self.interrupt_state.activated = true;
                self.interrupt_state.add(interrupt.clone());
                return Ok(AgentResult {
                    stop_reason: StopReason::Interrupt,
                    message: Message::assistant("Agent paused for interrupt."),
                    metrics: metrics.clone(),
                    state: invocation_state.clone(),
                    interrupts: Some(vec![interrupt]),
                    structured_output: None,
                    checkpoint: None,
                });
            }

            let middleware_context = crate::middleware::InvokeModelContext {
                messages: messages_snapshot.clone(),
                system_prompt: system_prompt_snapshot.clone(),
                tool_specs: tool_specs.clone(),
                tool_choice: None,
                invocation_state: invocation_state.clone(),
                model: None,
                projected_input_tokens: None,
            };

            let outcome = loop {
                let this = &*self;
                let attempt = self.middleware.run_invoke_model(
                    middleware_context.clone(),
                    |context| async move {
                        let specs: Option<&[crate::types::tools::ToolSpec]> = if context
                            .tool_specs
                            .is_empty()
                        {
                            None
                        } else {
                            Some(&context.tool_specs)
                        };
                        this.stream_model_once(
                            &context.messages,
                            specs,
                            context.system_prompt.as_deref(),
                            context.model.as_ref(),
                        )
                        .await
                        .map(|(content, stop_reason, usage)| {
                            crate::middleware::InvokeModelResult {
                                stop_reason,
                                message: Message::new(Role::Assistant, content),
                                usage,
                            }
                        })
                    },
                ).await;

                match attempt {
                    Ok(result) => {
                        retry_strategy.reset();
                        break Ok((result.message.content, result.stop_reason, result.usage));
                    }
                    Err(error) => {
                        if retry_strategy.is_retryable(&error) {
                            if let Some(delay) = retry_strategy.next_delay() {
                                self.hooks
                                    .fire_and_forget(HookEvent::EventLoopThrottle(
                                        EventLoopThrottleEvent {
                                            delay_seconds: delay.as_secs_f64(),
                                            attempt: retry_strategy.attempts(),
                                            max_attempts: retry_strategy.max_attempts(),
                                            reason: error.to_string(),
                                        },
                                    ))
                                    .await;
                                tokio::time::sleep(delay).await;
                                continue;
                            }
                        }
                        break Err(error);
                    }
                }
            };

            let (response_content, stop_reason, usage) = match outcome {
                Ok(outcome) => outcome,
                Err(error) => return Err(error),
            };

            let mut cycle_trace = metrics.start_cycle();
            metrics.update_usage(&usage);
            metrics.update_metrics(&crate::types::streaming::Metrics {
                latency_ms: 0,
                time_to_first_byte_ms: 0,
            });
            metrics.end_cycle(&mut cycle_trace);

            total_tokens += usage.total_tokens as u64;
            output_tokens += usage.output_tokens as u64;

            let mut assistant_message = Message::new(Role::Assistant, response_content);
            assistant_message.ensure_tracking_id();
            assistant_message.metadata = Some(MessageMetadata {
                usage: Some(usage),
                metrics: None,
                custom: std::collections::HashMap::new(),
            });

            let mut after_event = HookEvent::AfterModelCall(AfterModelCallEvent::success(
                assistant_message.clone(),
                stop_reason,
            ));
            let _ = self.hooks.invoke(&mut after_event).await;
            if let HookEvent::AfterModelCall(event) = after_event {
                if event.retry {
                    // A retried call must not leave the failed turn in
                    // history; the message is appended only once the loop
                    // commits to keeping it.
                    continue;
                }
            }

            self.append_message(assistant_message.clone()).await;

            match stop_reason {
                StopReason::EndTurn | StopReason::StopSequence => {
                    if let Some(ref mut ctx) = structured_output_ctx {
                        if ctx.is_enabled() {
                            if ctx.force_attempted {
                                structured_output_attempts += 1;

                                if let Some(output) = ctx.result_from_text(&assistant_message) {
                                    return Ok(AgentResult {
                                        stop_reason,
                                        message: assistant_message,
                                        metrics: metrics.clone(),
                                        state: invocation_state.clone(),
                                        interrupts: None,
                                        structured_output: Some(output),
                                        checkpoint: None,
                                    });
                                }

                                if let Some(limit_stop) = self.limits.check(
                                    cycle_id as u32,
                                    total_tokens,
                                    output_tokens,
                                    structured_output_attempts,
                                ) {
                                    return Ok(AgentResult {
                                        stop_reason: limit_stop,
                                        message: assistant_message,
                                        metrics: metrics.clone(),
                                        state: invocation_state.clone(),
                                        interrupts: None,
                                        structured_output: None,
                                        checkpoint: None,
                                    });
                                }

                                return Err(StrandsError::StructuredOutputException {
                                    message: "The model failed to invoke the structured output tool even after it was forced.".to_string(),
                                });
                            }

                            ctx.set_forced_mode();
                            tracing::debug!("Forcing structured output tool");

                            let force_message = Message::user(
                                crate::tools::structured_output::DEFAULT_STRUCTURED_OUTPUT_PROMPT,
                            );
                            self.append_message(force_message).await;

                            continue;
                        }
                    }

                    return Ok(AgentResult {
                        stop_reason,
                        message: assistant_message,
                        metrics: metrics.clone(),
                        state: invocation_state.clone(),
                        interrupts: None,
                        structured_output: None,
                        checkpoint: None,
                    });
                }
                StopReason::ToolUse => {
                    let (tool_results, extracted_output, interrupts, end_turn_forced) = self
                        .execute_tools_with_structured_output(
                            &assistant_message.content,
                            invocation_state,
                            structured_output_ctx,
                        )
                        .await?;

                    if let Some(tool_interrupts) = interrupts {
                        self.interrupt_state.activated = true;
                        for interrupt in &tool_interrupts {
                            self.interrupt_state.add(interrupt.clone());
                        }
                        return Ok(AgentResult {
                            stop_reason: StopReason::Interrupt,
                            message: assistant_message,
                            metrics: metrics.clone(),
                            state: invocation_state.clone(),
                            interrupts: Some(tool_interrupts),
                            structured_output: None,
                            checkpoint: None,
                        });
                    }

                    let tool_result_message = Message::new(
                        Role::User,
                        tool_results
                            .into_iter()
                            .map(ContentBlock::tool_result)
                            .collect(),
                    );
                    self.append_message(tool_result_message).await;
                    self.interrupt_state.end_tool_cycle();

                    if end_turn_forced {
                        return Ok(AgentResult {
                            stop_reason: StopReason::EndTurn,
                            message: assistant_message,
                            metrics: metrics.clone(),
                            state: invocation_state.clone(),
                            interrupts: None,
                            structured_output: None,
                            checkpoint: None,
                        });
                    }

                    if self
                        .deferred_cancel
                        .load(std::sync::atomic::Ordering::SeqCst)
                    {
                        self.cancel_signal.set();
                        return Ok(self.cancelled_result(&self.cancel_message_text(), invocation_state));
                    }

                    let should_stop = invocation_state.should_stop()
                        || structured_output_ctx
                            .as_ref()
                            .map(|c| c.stop_loop)
                            .unwrap_or(false);

                    if should_stop {
                        return Ok(AgentResult {
                            stop_reason: StopReason::EndTurn,
                            message: assistant_message,
                            metrics: metrics.clone(),
                            state: invocation_state.clone(),
                            interrupts: None,
                            structured_output: extracted_output,
                            checkpoint: None,
                        });
                    }
                }
                StopReason::MaxTokens => {
                    let recovered = recover_message_on_max_tokens_reached(assistant_message);
                    if let Some(last) = self.messages.last_mut() {
                        *last = recovered.clone();
                    }
                    self.hooks
                        .fire_and_forget(HookEvent::MessageUpdated(
                            crate::hooks::MessageUpdatedEvent {
                                tracking_id: recovered.tracking_id.clone().unwrap_or_default(),
                                message: recovered.clone(),
                            },
                        ))
                        .await;
                    return Err(StrandsError::MaxTokensReached {
                        message: "Maximum token limit reached".to_string(),
                    });
                }
                StopReason::ContentFiltered => {
                    return Err(StrandsError::ContentFiltered {
                        message: "Content was filtered".to_string(),
                    })
                }
                StopReason::GuardrailIntervened => {
                    return Err(StrandsError::GuardrailIntervention {
                        message: "Guardrail intervention".to_string(),
                    })
                }
                StopReason::Interrupt
                | StopReason::Cancelled
                | StopReason::Checkpoint
                | StopReason::LimitTurns
                | StopReason::LimitTotalTokens
                | StopReason::LimitOutputTokens
                | StopReason::LimitStructuredOutputAttempts => {
                    return Ok(AgentResult {
                        stop_reason,
                        message: assistant_message,
                        metrics: metrics.clone(),
                        state: invocation_state.clone(),
                        interrupts: None,
                        structured_output: None,
                        checkpoint: None,
                    });
                }
            }
        }
    }

/// Appends the in-progress reasoning block, if any, to the response content.
///
/// Mirrors the Python streaming accumulator: reasoning text and signature
/// accumulate into a single `reasoningContent` block that is finalized as
/// soon as non-reasoning content starts.
fn finalize_reasoning(
    response_content: &mut Vec<ContentBlock>,
    current_reasoning: &mut Option<(String, Option<String>)>,
    reasoning_redacted: &mut Option<Vec<u8>>,
) {
    if let Some((text, signature)) = current_reasoning.take() {
        if !text.is_empty() {
            response_content.push(ContentBlock {
                reasoning_content: Some(ReasoningContentBlock {
                    reasoning_text: Some(ReasoningTextBlock { text, signature }),
                    redacted_content: None,
                }),
                ..Default::default()
            });
        }
    }
    if let Some(redacted) = reasoning_redacted.take() {
        response_content.push(ContentBlock {
            reasoning_content: Some(ReasoningContentBlock {
                reasoning_text: None,
                redacted_content: Some(redacted),
            }),
            ..Default::default()
        });
    }
}

/// Streams one model call, accumulating content and handling cancellation
/// and redaction.
    async fn stream_model_once(
        &self,
        messages: &[Message],
        tool_specs: Option<&[crate::types::tools::ToolSpec]>,
        system_prompt: Option<&str>,
        model_override: Option<&Arc<dyn Model>>,
    ) -> Result<(Vec<ContentBlock>, crate::types::streaming::StopReason, crate::types::streaming::Usage)> {
        use crate::types::streaming::{StopReason, Usage};

        let model: &Arc<dyn Model> = model_override.unwrap_or(&self.model);
        let stream = model.stream(messages, tool_specs, system_prompt, None, None);

        let mut response_content: Vec<ContentBlock> = Vec::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut usage = Usage::default();
        // Tool uses accumulate per content-block index, matching the Python
        // streaming accumulator: providers may interleave blocks.
        let mut tool_uses: std::collections::HashMap<u32, (crate::types::tools::ToolUse, String)> =
            std::collections::HashMap::new();
        let mut tool_use_order: Vec<u32> = Vec::new();
        let mut current_reasoning: Option<(String, Option<String>)> = None;
        let mut reasoning_redacted: Option<Vec<u8>> = None;
        let mut cancelled = false;

        futures::pin_mut!(stream);
        while let Some(event_result) = stream.next().await {
            if self.cancel_signal.is_set() {
                cancelled = true;
                break;
            }

            let event = event_result?;

            if let Some(ref redact_event) = event.redact_content {
                if let Some(ref redacted) = redact_event.redact_assistant_content_message {
                    response_content.clear();
                    response_content.push(ContentBlock::text(redacted.clone()));
                }
            }

            if let Some(ref delta_event) = event.content_block_delta {
                if let Some(ref delta) = delta_event.delta {
                    if let Some(ref reasoning) = delta.reasoning_content {
                        if let Some(ref text) = reasoning.text {
                            let entry =
                                current_reasoning.get_or_insert_with(|| (String::new(), None));
                            entry.0.push_str(text);
                        }
                        if let Some(ref signature) = reasoning.signature {
                            if let Some(entry) = current_reasoning.as_mut() {
                                entry.1 = Some(signature.clone());
                            }
                        }
                        if let Some(ref redacted) = reasoning.redacted_content {
                            reasoning_redacted.get_or_insert_with(Vec::new).extend(redacted.iter().copied());
                        }
                    }
                    if let Some(ref text) = delta.text {
                        Self::finalize_reasoning(&mut response_content, &mut current_reasoning, &mut reasoning_redacted);
                        match response_content.last_mut() {
                            Some(block) if block.text.is_some() => {
                                if let Some(existing) = block.text.as_mut() {
                                    existing.push_str(text);
                                }
                            }
                            _ => response_content.push(ContentBlock::text(text)),
                        }
                    }
                    if let Some(ref tool_delta) = delta.tool_use {
                        if let Some(index) = delta_event.content_block_index {
                            if let Some((_, buffer)) = tool_uses.get_mut(&index) {
                                buffer.push_str(&tool_delta.input);
                            }
                        }
                    }
                }
            }

            if let Some(ref start_event) = event.content_block_start {
                if let Some(ref start) = start_event.start {
                    if let Some(ref tu) = start.tool_use {
                        let signature = current_reasoning
                            .as_ref()
                            .and_then(|(_, signature)| signature.clone());
                        let index = start_event.content_block_index.unwrap_or_default();
                        tool_uses.insert(
                            index,
                            (
                                crate::types::tools::ToolUse {
                                    name: tu.name.clone(),
                                    tool_use_id: tu.tool_use_id.clone(),
                                    input: serde_json::Value::Null,
                                    reasoning_signature: signature,
                                },
                                String::new(),
                            ),
                        );
                        if !tool_use_order.contains(&index) {
                            tool_use_order.push(index);
                        }
                    }
                }
            }

            if let Some(ref stop_event) = event.content_block_stop {
                if let Some(index) = stop_event.content_block_index {
                    if let Some((mut tool_use, buffer)) = tool_uses.remove(&index) {
                        tool_use.input =
                            serde_json::from_str(&buffer).unwrap_or(serde_json::Value::Null);
                        response_content.push(ContentBlock::tool_use(tool_use));
                    }
                }
                Self::finalize_reasoning(&mut response_content, &mut current_reasoning, &mut reasoning_redacted);
            }

            if let Some(ref stop_event) = event.message_stop {
                if let Some(sr) = stop_event.stop_reason {
                    stop_reason = sr;
                }
            }

            if let Some(ref meta) = event.metadata {
                if let Some(ref u) = meta.usage {
                    usage = u.clone();
                }
            }
        }

        // Finalize any tool blocks the stream ended without stopping, and
        // reasoning still in progress.
        for index in &tool_use_order {
            if let Some((mut tool_use, buffer)) = tool_uses.remove(index) {
                tool_use.input =
                    serde_json::from_str(&buffer).unwrap_or(serde_json::Value::Null);
                response_content.push(ContentBlock::tool_use(tool_use));
            }
        }
        Self::finalize_reasoning(&mut response_content, &mut current_reasoning, &mut reasoning_redacted);

        if cancelled {
            response_content.clear();
            response_content.push(ContentBlock::text(self.cancel_message_text()));
            stop_reason = StopReason::Cancelled;
        }

        Ok((response_content, stop_reason, usage))
    }

    async fn execute_tools_with_structured_output(
        &mut self,
        content: &[ContentBlock],
        invocation_state: &InvocationState,
        structured_output_ctx: &mut Option<crate::tools::structured_output::StructuredOutputContext>,
    ) -> Result<(
        Vec<crate::types::tools::ToolResult>,
        Option<serde_json::Value>,
        Option<Vec<crate::types::interrupt::Interrupt>>,
        bool,
    )> {        use crate::hooks::{AfterToolCallEvent, BeforeToolCallEvent, BeforeToolsEvent, AfterToolsEvent, HookEvent};
        use crate::types::tools::ToolResult;
        use crate::tools::ToolContext;

        let tool_uses: Vec<&crate::types::tools::ToolUse> = content
            .iter()
            .filter_map(|block| block.as_tool_use())
            .collect();

        let mut before_event = HookEvent::BeforeTools(BeforeToolsEvent {
            tool_uses: tool_uses.iter().map(|t| (*t).clone()).collect(),
            invocation_state: Some(invocation_state.clone()),
            cancel: None,
            interrupt_slot: None,
        });
        let before_interrupts = self.hooks.invoke(&mut before_event).await;

        if let HookEvent::BeforeTools(ref event) = before_event {
            if let Some(message) = &event.cancel {
                let results = tool_uses
                    .iter()
                    .map(|t| ToolResult::error(&t.tool_use_id, message.clone()))
                    .collect::<Vec<_>>();
                let mut after = HookEvent::AfterTools(AfterToolsEvent {
                    tool_results: results.clone(),
                    invocation_state: Some(invocation_state.clone()),
                    end_turn: false,
                });
                let _ = self.hooks.invoke(&mut after).await;
                return Ok((results, None, None, false));
            }
        }
        if let Some(interrupt) = before_interrupts.into_iter().next() {
            // No tool ran: record the full batch as pending so a resume
            // replays it without a new model call.
            self.interrupt_state.set_pending_tool_execution(
                crate::types::interrupt::PendingToolExecution {
                    assistant_message: Message::new(Role::Assistant, content.to_vec()),
                    completed_tool_results: Vec::new(),
                },
            );
            return Ok((Vec::new(), None, Some(vec![interrupt]), false));
        }

        let mut results: Vec<ToolResult> = Vec::new();
        let mut extracted_output: Option<serde_json::Value> = None;
        let mut raised_interrupts: Vec<crate::types::interrupt::Interrupt> = Vec::new();

        let expected_tool_name = structured_output_ctx
            .as_ref()
            .and_then(|ctx| ctx.expected_tool_name().map(|s| s.to_string()));

        'tools: for tool_use in tool_uses {
            if self.cancel_signal.is_set()
                && !self
                    .deferred_cancel
                    .load(std::sync::atomic::Ordering::SeqCst)
            {
                results.push(ToolResult::error(&tool_use.tool_use_id, self.cancel_message_text()));
                continue;
            }

            if !crate::tools::is_valid_tool_name(&tool_use.name) {
                results.push(ToolResult::error(
                    &tool_use.tool_use_id,
                    format!("Invalid tool name: {}", tool_use.name),
                ));
                continue;
            }

            let tool = match self.tool_registry.get(&tool_use.name) {
                Some(tool) => tool,
                None => {
                    results.push(ToolResult::error(
                        &tool_use.tool_use_id,
                        format!("Tool not found: {}", tool_use.name),
                    ));
                    continue;
                }
            };

            let is_structured_output_tool = expected_tool_name
                .as_ref()
                .map(|expected| expected == &tool_use.name)
                .unwrap_or(false);

            let mut result: Option<ToolResult>;
            let mut attempt_error: Option<String> = None;

            loop {
                let mut before_tool_event = HookEvent::BeforeToolCall(BeforeToolCallEvent {
                    tool_use: tool_use.clone(),
                    invocation_state: Some(invocation_state.clone()),
                    cancel_tool: None,
                    selected_tool: Some(tool.clone()),
                    interrupt_slot: None,
                });
                let tool_interrupts = self.hooks.invoke(&mut before_tool_event).await;

                let (cancel_message, raised, hook_selected_tool) = match &before_tool_event {
                    HookEvent::BeforeToolCall(event) => (
                        event.cancel_tool.clone(),
                        event.interrupt_slot.clone(),
                        event.selected_tool.clone(),
                    ),
                    _ => (None, None, None),
                };

                let effective_tool = hook_selected_tool.unwrap_or_else(|| tool.clone());

                if let Some(interrupt) = raised.or_else(|| tool_interrupts.into_iter().next()) {
                    self.interrupt_state.set_pending_tool_execution(
                        crate::types::interrupt::PendingToolExecution {
                            assistant_message: Message::new(Role::Assistant, content.to_vec()),
                            completed_tool_results: results.clone(),
                        },
                    );
                    raised_interrupts.push(interrupt);
                    // The interrupted tool gets no result entry: its use
                    // stays in the pending batch and replays on resume.
                    break 'tools;
                }

                if let Some(message) = cancel_message {
                    result = Some(ToolResult::error(&tool_use.tool_use_id, message));
                    break;
                }

                let context = ToolContext::with_state(
                    tool_use.clone(),
                    invocation_state.clone(),
                )
                .with_cancel_signal(self.cancel_signal.clone());

                let exec_context = crate::middleware::ExecuteToolContext {
                    tool_name: Some(tool_use.name.clone()),
                    tool_use: tool_use.clone(),
                    invocation_state: invocation_state.clone(),
                    cancel_signal: Some(self.cancel_signal.clone()),
                };

                let start = std::time::Instant::now();
                let tool = effective_tool.clone();
                let context = context.clone();
                let this = &*self;
                let execution = this.middleware.run_execute_tool(
                    exec_context,
                    |ctx| async move {
                        match tool.invoke(ctx.tool_use.input.clone(), &context).await {
                            Ok(r) => Ok(crate::middleware::ExecuteToolResult {
                                result: ToolResult {
                                    tool_use_id: ctx.tool_use.tool_use_id.clone(),
                                    status: r.status,
                                    content: r.content,
                                },
                            }),
                            Err(e) => Ok(crate::middleware::ExecuteToolResult {
                                result: ToolResult::error(&ctx.tool_use.tool_use_id, e),
                            }),
                        }
                    },
                ).await;
                let duration_ms = start.elapsed().as_millis() as u64;

                let tool_result = match execution {
                    Ok(executed) => {
                        let mut tool_result = executed.result;
                        tool_result.tool_use_id = tool_use.tool_use_id.clone();
                        if is_structured_output_tool && tool_result.is_success() {
                            if let Some(ref mut ctx) = structured_output_ctx {
                                ctx.store_result(&tool_use.tool_use_id, tool_use.input.clone());
                                ctx.stop_loop = true;
                                extracted_output = Some(tool_use.input.clone());
                                tracing::debug!(
                                    "Extracted structured output for tool: {}",
                                    tool_use.name
                                );
                            }
                        }
                        tool_result
                    }
                    Err(e) => {
                        attempt_error = Some(e.to_string());
                        ToolResult::error(&tool_use.tool_use_id, e.to_string())
                    }
                };

                let mut after_tool_event = HookEvent::AfterToolCall(AfterToolCallEvent {
                    tool_use: tool_use.clone(),
                    invocation_state: Some(invocation_state.clone()),
                    result: tool_result.clone(),
                    exception: attempt_error.clone(),
                    cancel_message: None,
                    duration: Some(duration_ms),
                    retry: false,
                });
                let _ = self.hooks.invoke(&mut after_tool_event).await;

                let (retry, hook_cancelled) = match &after_tool_event {
                    HookEvent::AfterToolCall(e) => {
                        (e.retry, e.cancel_message.clone())
                    }
                    _ => (false, None),
                };

                result = Some(match hook_cancelled {
                    Some(message) => ToolResult::error(&tool_use.tool_use_id, message),
                    None => tool_result,
                });

                if !retry {
                    break;
                }
            }

            if let Some(result) = result {
                results.push(result);
            }
        }

        let mut after_event = HookEvent::AfterTools(AfterToolsEvent {
            tool_results: results.clone(),
            invocation_state: Some(invocation_state.clone()),
            end_turn: false,
        });
        self.hooks.invoke(&mut after_event).await;
        let end_turn_forced =
            matches!(&after_event, HookEvent::AfterTools(e) if e.end_turn);

        let interrupts = if raised_interrupts.is_empty() {
            None
        } else {
            Some(raised_interrupts)
        };

        Ok((results, extracted_output, interrupts, end_turn_forced))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_utils::CancelSignal;
    use crate::event_loop::ModelRetryStrategy;
    use crate::hooks::Interruptible;
    use crate::testing::{MockModel, MockResponse};
    use crate::tools::AgentTool;
    use crate::types::streaming::StopReason;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    struct EchoTool {
        hits: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl AgentTool for EchoTool {
        fn name(&self) -> &str { "echo" }
        fn description(&self) -> &str { "Echoes input" }
        fn tool_spec(&self) -> crate::types::tools::ToolSpec {
            crate::types::tools::ToolSpec::new("echo", "Echoes input")
        }

        async fn invoke(
            &self,
            _input: serde_json::Value,
            _context: &crate::tools::ToolContext,
        ) -> std::result::Result<crate::tools::ToolResult2, String> {
            self.hits.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(crate::tools::ToolResult2::success("echoed"))
        }
    }

    fn agent_with(model: MockModel) -> Agent {
        Agent::builder().model_arc(Arc::new(model)).build().unwrap()
    }

    #[tokio::test]
    async fn test_basic_invocation() {
        let model = MockModel::new(vec![MockResponse::text("Hello there")]);
        let mut agent = agent_with(model);
        let result = agent.invoke_async("hi").await.unwrap();

        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.text(), "Hello there");
        assert!(result.usage().is_some());
    }

    #[tokio::test]
    async fn test_max_tokens_recovers_partial_message() {
        let events = vec![
            crate::types::streaming::StreamEvent::message_start(crate::types::content::Role::Assistant),
            crate::types::streaming::StreamEvent::tool_use_start(0, "echo", "t1"),
            crate::types::streaming::StreamEvent::tool_use_delta(0, "{}"),
            crate::types::streaming::StreamEvent::content_block_stop(0),
            crate::types::streaming::StreamEvent::message_stop(StopReason::MaxTokens),
        ];
        let model = MockModel::new(vec![MockResponse::Events(events)]);
        let mut agent = agent_with(model);

        let error = agent.invoke_async("hi").await.unwrap_err();
        assert!(matches!(error, StrandsError::MaxTokensReached { .. }));

        let last = agent.messages().last().unwrap();
        assert!(last.tool_uses().is_empty());
        assert!(last.text_content().contains("echo"));
        assert!(last.text_content().contains("maximum token limits"));
    }

    #[tokio::test]
    async fn test_throttle_retry_then_success() {
        let model = MockModel::new(vec![
            MockResponse::throttled(),
            MockResponse::throttled(),
            MockResponse::text("recovered"),
        ]);
        let throttle_events = Arc::new(AtomicUsize::new(0));
        let sink = throttle_events.clone();

        let mut agent = Agent::builder()
            .model_arc(Arc::new(model.clone()))
            .retry_strategy(ModelRetryStrategy::new(
                6,
                Duration::from_millis(1),
                Duration::from_millis(2),
            ))
            .build()
            .unwrap();
        agent.hooks_mut().add_callback(move |event| {
            if matches!(event, crate::hooks::HookEvent::EventLoopThrottle(_)) {
                sink.fetch_add(1, AtomicOrdering::SeqCst);
            }
        });

        let result = agent.invoke_async("hi").await.unwrap();
        assert_eq!(result.text(), "recovered");
        assert_eq!(model.calls(), 3);
        assert_eq!(throttle_events.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_retry_exhaustion() {
        let model = MockModel::new(vec![MockResponse::throttled()]);
        let mut agent = Agent::builder()
            .model_arc(Arc::new(model.clone()))
            .retry_strategy(ModelRetryStrategy::new(
                3,
                Duration::from_millis(1),
                Duration::from_millis(1),
            ))
            .build()
            .unwrap();

        let error = agent.invoke_async("hi").await.unwrap_err();
        assert!(matches!(error, StrandsError::ModelThrottled { .. }));
        assert_eq!(model.calls(), 3);
    }

    #[tokio::test]
    async fn test_cancel_before_invocation() {
        let model = MockModel::new(vec![]);
        let mut agent = agent_with(model.clone());
        agent.cancel("stopped by test", false);

        let result = agent.invoke_async("hi").await.unwrap();
        assert_eq!(result.stop_reason, StopReason::Cancelled);
        assert_eq!(result.text(), "stopped by test");
        assert_eq!(model.calls(), 0);
    }

    #[tokio::test]
    async fn test_model_retry_hook_does_not_duplicate_history() {
        let model = MockModel::new(vec![
            MockResponse::text("bad first answer"),
            MockResponse::text("good second answer"),
        ]);

        let mut agent = Agent::builder()
            .model_arc(Arc::new(model.clone()))
            .build()
            .unwrap();
        agent.hooks_mut().add_callback(|event| {
            if let crate::hooks::HookEvent::AfterModelCall(e) = event {
                if !e.retry {
                    // Retry only the first model call.
                    if let Some(response) = &e.stop_response {
                        if response.message.text_content() == "bad first answer" {
                            e.request_retry();
                        }
                    }
                }
            }
        });

        let result = agent.invoke_async("go").await.unwrap();
        assert_eq!(result.text(), "good second answer");
        assert_eq!(model.calls(), 2);

        let assistant_messages: Vec<&Message> = agent
            .messages()
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .collect();
        assert_eq!(assistant_messages.len(), 1);
        assert_eq!(assistant_messages[0].text_content(), "good second answer");
    }

    #[tokio::test]
    async fn test_tool_interrupt_stops_remaining_batch() {
        struct NamedEchoTool {
            name: String,
            hits: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl AgentTool for NamedEchoTool {
            fn name(&self) -> &str {
                &self.name
            }
            fn description(&self) -> &str {
                "Echoes input"
            }
            fn tool_spec(&self) -> crate::types::tools::ToolSpec {
                crate::types::tools::ToolSpec::new(&self.name, "Echoes input")
            }
            async fn invoke(
                &self,
                _input: serde_json::Value,
                _context: &crate::tools::ToolContext,
            ) -> std::result::Result<crate::tools::ToolResult2, String> {
                self.hits.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(crate::tools::ToolResult2::success("ran"))
            }
        }

        // The assistant message carries both tool uses in one batch.
        let model = MockModel::new(vec![
            MockResponse::Events(vec![
                crate::types::streaming::StreamEvent::message_start(crate::types::content::Role::Assistant),
                crate::types::streaming::StreamEvent::tool_use_start(0, "gated", "t1"),
                crate::types::streaming::StreamEvent::tool_use_delta(0, "{}"),
                crate::types::streaming::StreamEvent::tool_use_start(1, "worker", "t2"),
                crate::types::streaming::StreamEvent::tool_use_delta(1, "{}"),
                crate::types::streaming::StreamEvent::message_stop(StopReason::ToolUse),
            ]),
            MockResponse::text("after"),
        ]);

        let worker_hits = Arc::new(AtomicUsize::new(0));
        let gated_hits = Arc::new(AtomicUsize::new(0));
        let interrupted = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hook_interrupted = interrupted.clone();

        let mut agent = Agent::builder()
            .model_arc(Arc::new(model.clone()))
            .tool(NamedEchoTool {
                name: "gated".to_string(),
                hits: gated_hits.clone(),
            })
            .unwrap()
            .tool(NamedEchoTool {
                name: "worker".to_string(),
                hits: worker_hits.clone(),
            })
            .unwrap()
            .build()
            .unwrap();

        agent.hooks_mut().add_callback(move |event| {
            if let crate::hooks::HookEvent::BeforeToolCall(e) = event {
                if e.tool_use.name == "gated"
                    && !hook_interrupted.swap(true, AtomicOrdering::SeqCst)
                {
                    e.interrupt("gate", None);
                }
            }
        });

        let first = agent.invoke_async("go").await.unwrap();
        assert_eq!(first.stop_reason, StopReason::Interrupt);
        assert_eq!(gated_hits.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(
            worker_hits.load(AtomicOrdering::SeqCst),
            0,
            "tools after an interrupted one must not run"
        );

        // Resume replays the whole batch, running both tools.
        let second = agent
            .invoke_async(crate::agent::AgentInput::Text("resume".to_string()))
            .await
            .unwrap();
        assert_eq!(second.stop_reason, StopReason::EndTurn);
        assert_eq!(gated_hits.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(worker_hits.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(model.calls(), 2);
    }

    #[tokio::test]
    async fn test_before_tools_interrupt_replays_batch_on_resume() {
        let model = MockModel::new(vec![
            MockResponse::tool_use("echo", "t1", serde_json::json!({"text": "x"})),
            MockResponse::text("completed after resume"),
        ]);
        let hits = Arc::new(AtomicUsize::new(0));
        let interrupted = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hook_interrupted = interrupted.clone();

        let mut agent = Agent::builder()
            .model_arc(Arc::new(model.clone()))
            .tool(EchoTool { hits: hits.clone() })
            .unwrap()
            .build()
            .unwrap();
        agent.hooks_mut().add_callback(move |event| {
            match event {
                crate::hooks::HookEvent::BeforeTools(e) => {
                    if !hook_interrupted.swap(true, AtomicOrdering::SeqCst) {
                        e.interrupt("batch-hold", None);
                    }
                }
                _ => {}
            }
        });

        let first = agent.invoke_async("go").await.unwrap();
        assert_eq!(first.stop_reason, StopReason::Interrupt);
        assert_eq!(hits.load(AtomicOrdering::SeqCst), 0);

        // Resume with a user response; the batch replays without a model call.
        let second = agent
            .invoke_async(crate::agent::AgentInput::Text("resume".to_string()))
            .await
            .unwrap();
        assert_eq!(second.stop_reason, StopReason::EndTurn);
        assert_eq!(hits.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_turn_limit_stops_loop() {
        let model = MockModel::new(vec![MockResponse::tool_use("echo", "t1", serde_json::json!({}))]);
        let mut agent = Agent::builder()
            .model_arc(Arc::new(model.clone()))
            .tool(EchoTool { hits: Arc::new(AtomicUsize::new(0)) })
            .unwrap()
            .limits(Limits { turns: Some(2), ..Default::default() })
            .build()
            .unwrap();

        let result = agent.invoke_async("hi").await.unwrap();
        assert_eq!(result.stop_reason, StopReason::LimitTurns);
        assert_eq!(model.calls(), 2);
    }

    #[tokio::test]
    async fn test_turn_limit_of_one_allows_single_call() {
        let model = MockModel::new(vec![MockResponse::text("only turn")]);
        let mut agent = Agent::builder()
            .model_arc(Arc::new(model.clone()))
            .limits(Limits { turns: Some(1), ..Default::default() })
            .build()
            .unwrap();

        let result = agent.invoke_async("hi").await.unwrap();
        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(model.calls(), 1);
    }

    #[tokio::test]
    async fn test_cancel_does_not_leak_into_next_invocation() {
        let model = MockModel::new(vec![MockResponse::text("fresh run")]);
        let mut agent = agent_with(model.clone());

        agent.cancel("stopped", false);
        let cancelled = agent.invoke_async("first").await.unwrap();
        assert_eq!(cancelled.stop_reason, StopReason::Cancelled);

        let second = agent.invoke_async("second").await.unwrap();
        assert_eq!(second.stop_reason, StopReason::EndTurn);
        assert_eq!(second.text(), "fresh run");
        assert_eq!(model.calls(), 1);
    }

    #[tokio::test]
    async fn test_structured_output_text_fallback() {
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Out {
            name: String,
        }

        let model = MockModel::new(vec![
            MockResponse::text("let me format that"),
            MockResponse::text("{\"name\": \"structured\"}"),
        ]);
        let mut agent = Agent::builder()
            .model_arc(Arc::new(model))
            .structured_output_model::<Out>()
            .build()
            .unwrap();

        let result = agent.invoke_async("hi").await.unwrap();
        assert_eq!(result.structured_output.as_ref().unwrap()["name"], "structured");
    }

    #[tokio::test]
    async fn test_before_tool_call_interrupt_skips_execution() {
        let model = MockModel::new(vec![MockResponse::tool_use("echo", "t1", serde_json::json!({}))]);
        let hits = Arc::new(AtomicUsize::new(0));

        let mut agent = Agent::builder()
            .model_arc(Arc::new(model))
            .tool(EchoTool { hits: hits.clone() })
            .unwrap()
            .build()
            .unwrap();
        agent.hooks_mut().add_callback(|event| {
            if let crate::hooks::HookEvent::BeforeToolCall(e) = event {
                e.interrupt("approval", Some(serde_json::json!({"confirm": false})));
            }
        });

        let result = agent.invoke_async("hi").await.unwrap();
        assert_eq!(result.stop_reason, StopReason::Interrupt);
        assert!(result.has_interrupts());
        assert_eq!(hits.load(AtomicOrdering::SeqCst), 0);
        assert!(agent.interrupt_state().pending_tool_execution.is_some());
    }

    #[tokio::test]
    async fn test_interrupt_resume_replays_tool_execution() {
        let model = MockModel::new(vec![
            MockResponse::tool_use("echo", "t1", serde_json::json!({})),
            MockResponse::text("resumed"),
        ]);
        let hits = Arc::new(AtomicUsize::new(0));
        let interrupted = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hook_hits = hits.clone();
        let hook_interrupted = interrupted.clone();

        let mut agent = Agent::builder()
            .model_arc(Arc::new(model.clone()))
            .tool(EchoTool { hits: hits.clone() })
            .unwrap()
            .build()
            .unwrap();
        agent.hooks_mut().add_callback(move |event| {
            if let crate::hooks::HookEvent::BeforeToolCall(e) = event {
                if !hook_interrupted.swap(true, AtomicOrdering::SeqCst)
                    && hook_hits.load(AtomicOrdering::SeqCst) == 0
                {
                    e.interrupt("approval", None);
                }
            }
        });

        let first = agent.invoke_async("hi").await.unwrap();
        assert_eq!(first.stop_reason, StopReason::Interrupt);

        let interrupt_id = agent
            .interrupt_state()
            .interrupts
            .values()
            .find(|interrupt| !interrupt.has_response())
            .map(|interrupt| interrupt.id.clone())
            .expect("unanswered interrupt recorded");

        let responses = vec![crate::types::interrupt::InterruptResponseContent {
            interrupt_response: crate::types::interrupt::InterruptResponse {
                interrupt_id,
                response: serde_json::json!("approved"),
            },
        }];

        let second = agent.invoke_async(responses).await.unwrap();
        assert_eq!(second.stop_reason, StopReason::EndTurn);
        assert_eq!(hits.load(AtomicOrdering::SeqCst), 1);

        let tool_results: Vec<_> = agent
            .messages()
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| b.as_tool_result())
            .collect();
        assert_eq!(tool_results.len(), 1);
        assert_eq!(
            tool_results[0].content[0].text.as_deref(),
            Some("echoed")
        );
    }

    #[tokio::test]
    async fn test_after_tools_end_turn() {
        let model = MockModel::new(vec![MockResponse::tool_use("echo", "t1", serde_json::json!({}))]);
        let mut agent = Agent::builder()
            .model_arc(Arc::new(model.clone()))
            .tool(EchoTool { hits: Arc::new(AtomicUsize::new(0)) })
            .unwrap()
            .build()
            .unwrap();
        agent.hooks_mut().add_callback(|event| {
            if let crate::hooks::HookEvent::AfterTools(e) = event {
                e.end_turn = true;
            }
        });

        let result = agent.invoke_async("hi").await.unwrap();
        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(model.calls(), 1);
    }

    #[tokio::test]
    async fn test_unknown_tool_error_result() {
        let model = MockModel::new(vec![
            MockResponse::tool_use("missing_tool", "t1", serde_json::json!({})),
            MockResponse::text("done"),
        ]);
        let mut agent = agent_with(model);

        let result = agent.invoke_async("hi").await.unwrap();
        assert_eq!(result.stop_reason, StopReason::EndTurn);

        let tool_results: Vec<_> = agent
            .messages()
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| b.as_tool_result())
            .collect();
        assert_eq!(tool_results.len(), 1);
        assert_eq!(tool_results[0].status, crate::types::tools::ToolResultStatus::Error);
        assert!(tool_results[0].content[0].text.as_deref().unwrap().contains("Tool not found"));
    }

    #[tokio::test]
    async fn test_message_tracking_ids_assigned() {
        let model = MockModel::new(vec![MockResponse::text("hi")]);
        let mut agent = agent_with(model);

        agent.invoke_async("hello").await.unwrap();

        for message in agent.messages() {
            assert!(message.tracking_id.is_some());
        }
    }

    #[test]
    fn test_cancel_signal_wiring() {
        let signal = CancelSignal::new();
        assert!(!signal.is_set());
        signal.set();
        assert!(signal.is_set());
    }
}
