//! Agent implementation for conversational AI.

mod builder;
mod result;
mod state;

use std::sync::Arc;

use futures::StreamExt;

use crate::conversation::ConversationManager;
use crate::hooks::HookRegistry;
use crate::models::Model;
use crate::telemetry::EventLoopMetrics;
use crate::tools::{InvocationState, ToolRegistry};
use crate::types::content::{ContentBlock, Message, Messages, Role};
use crate::types::errors::{Result, StrandsError};

pub use builder::AgentBuilder;
pub use result::AgentResult;
pub use state::AgentState;

/// Input types for agent invocation.
pub enum AgentInput {
    Text(String),
    ContentBlocks(Vec<ContentBlock>),
    Messages(Messages),
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
        };

        let context = ToolContext::with_state(InvocationState::new());
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

        let user_msg = Message { role: Role::User, content: user_content };
        let tool_use_msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::tool_use(tool_use.clone())],
        };
        let tool_result_msg = Message {
            role: Role::User,
            content: vec![ContentBlock::tool_result(tool_result.clone())],
        };
        let assistant_msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::text(format!("agent.tool.{} was called.", tool_use.name))],
        };

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

    /// Invokes the agent asynchronously.
    pub async fn invoke_async(&mut self, prompt: impl Into<AgentInput>) -> Result<AgentResult> {
        let input = prompt.into();
        let new_messages = self.convert_input_to_messages(input)?;

        for msg in new_messages {
            self.messages.push(msg);
        }

        self.run_event_loop().await
    }

    /// Returns a stream of events from the agent.
    pub async fn stream_async(
        &mut self,
        prompt: impl Into<AgentInput>,
    ) -> impl futures::Stream<Item = Result<crate::event_loop::TypedEvent>> + '_ {
        let input = prompt.into();

        async_stream::stream! {
            let new_messages = match self.convert_input_to_messages(input) {
                Ok(msgs) => msgs,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            for msg in new_messages {
                self.messages.push(msg);
            }

            match self.run_event_loop().await {
                Ok(result) => yield Ok(crate::event_loop::TypedEvent::agent_result(result)),
                Err(e) => yield Err(e),
            }
        }
    }

    fn convert_input_to_messages(&self, input: AgentInput) -> Result<Messages> {
        match input {
            AgentInput::Text(text) => Ok(vec![Message { role: Role::User, content: vec![ContentBlock::text(text)] }]),
            AgentInput::ContentBlocks(blocks) => Ok(vec![Message { role: Role::User, content: blocks }]),
            AgentInput::Messages(messages) => Ok(messages),
            AgentInput::None => Ok(vec![]),
        }
    }

    async fn run_event_loop(&mut self) -> Result<AgentResult> {
        use crate::hooks::{BeforeInvocationEvent, AfterInvocationEvent, HookEvent};
        let invocation_state = InvocationState::new();
        self.hooks.invoke(&HookEvent::BeforeInvocation(BeforeInvocationEvent)).await;

        let mut structured_output_ctx = self.structured_output_context.clone();

        if let Some(ref ctx) = structured_output_ctx {
            ctx.register_tool(&mut self.tool_registry);
        }

        let result = self.event_loop_inner(&invocation_state, &mut structured_output_ctx).await;

        if let Some(ref ctx) = structured_output_ctx {
            ctx.cleanup(&mut self.tool_registry);
        }

        self.conversation_manager.apply_management(&mut self.messages);
        let agent_result = result.as_ref().ok().cloned();
        self.hooks.invoke(&HookEvent::AfterInvocation(AfterInvocationEvent::new(agent_result))).await;
        result
    }

    async fn event_loop_inner(
        &mut self,
        invocation_state: &InvocationState,
        structured_output_ctx: &mut Option<crate::tools::structured_output::StructuredOutputContext>,
    ) -> Result<AgentResult> {
        use crate::hooks::{
            BeforeModelCallEvent, AfterModelCallEvent, HookEvent, MessageAddedEvent,
        };
        use crate::types::streaming::{StopReason, Usage};

        loop {
            let tool_specs = self.tool_registry.get_all_tool_specs();
            let messages_snapshot = self.messages.clone();
            let system_prompt_snapshot = self.system_prompt.clone();

            let tool_specs_ref: Option<&[_]> = if tool_specs.is_empty() { None } else { Some(&tool_specs) };

            self.hooks.invoke(&HookEvent::BeforeModelCall(BeforeModelCallEvent)).await;

            let stream = self.model.stream(
                &messages_snapshot,
                tool_specs_ref,
                system_prompt_snapshot.as_deref(),
                None,
                None,
            );

            let mut response_content: Vec<ContentBlock> = Vec::new();
            let mut stop_reason = StopReason::EndTurn;
            let mut usage = Usage::default();
            let mut current_tool_use: Option<crate::types::tools::ToolUse> = None;
            let mut tool_input_buffer = String::new();

            futures::pin_mut!(stream);
            while let Some(event_result) = stream.next().await {
                let event = event_result?;

                if let Some(ref delta_event) = event.content_block_delta {
                    if let Some(ref delta) = delta_event.delta {
                        if let Some(ref text) = delta.text {
                            if let Some(block) = response_content.last_mut() {
                                if block.text.is_some() {
                                    block.text.as_mut().unwrap().push_str(text);
                                } else {
                                    response_content.push(ContentBlock::text(text));
                                }
                            } else {
                                response_content.push(ContentBlock::text(text));
                            }
                        }
                        if let Some(ref tool_delta) = delta.tool_use {
                            tool_input_buffer.push_str(&tool_delta.input);
                        }
                    }
                }

                if let Some(ref start_event) = event.content_block_start {
                    if let Some(ref start) = start_event.start {
                        if let Some(ref tu) = start.tool_use {
                            current_tool_use = Some(crate::types::tools::ToolUse {
                                name: tu.name.clone(),
                                tool_use_id: tu.tool_use_id.clone(),
                                input: serde_json::Value::Null,
                            });
                            tool_input_buffer.clear();
                        }
                    }
                }

                if event.content_block_stop.is_some() {
                    if let Some(mut tu) = current_tool_use.take() {
                        tu.input = serde_json::from_str(&tool_input_buffer).unwrap_or(serde_json::Value::Null);
                        response_content.push(ContentBlock::tool_use(tu));
                        tool_input_buffer.clear();
                    }
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

            let assistant_message = Message { role: Role::Assistant, content: response_content.clone() };
            self.messages.push(assistant_message.clone());

            self.hooks.invoke(&HookEvent::MessageAdded(MessageAddedEvent::new(assistant_message.clone()))).await;

            self.hooks.invoke(&HookEvent::AfterModelCall(AfterModelCallEvent::success(
                assistant_message.clone(),
                stop_reason.clone(),
            ))).await;

            match stop_reason {
                StopReason::EndTurn | StopReason::StopSequence => {
                    if let Some(ref mut ctx) = structured_output_ctx {
                        if ctx.is_enabled() {
                            if ctx.force_attempted {
                                return Err(StrandsError::StructuredOutputError {
                                    message: "The model failed to invoke the structured output tool even after it was forced.".to_string(),
                                });
                            }

                            ctx.set_forced_mode();
                            tracing::debug!("Forcing structured output tool");

                            let force_message = Message {
                                role: Role::User,
                                content: vec![ContentBlock::text("You must format the previous response as structured output.")],
                            };
                            self.messages.push(force_message);

                            continue;
                        }
                    }

                    return Ok(AgentResult {
                        stop_reason,
                        message: assistant_message,
                        usage,
                        metrics: EventLoopMetrics::default(),
                        state: invocation_state.clone(),
                        interrupts: None,
                        structured_output: None,
                    });
                }
                StopReason::ToolUse => {
                    let (tool_results, extracted_output) = self.execute_tools_with_structured_output(
                        &response_content,
                        invocation_state,
                        structured_output_ctx,
                    ).await?;

                    let tool_result_message = Message {
                        role: Role::User,
                        content: tool_results.into_iter().map(ContentBlock::tool_result).collect(),
                    };
                    self.messages.push(tool_result_message.clone());

                    self.hooks.invoke(&HookEvent::MessageAdded(MessageAddedEvent::new(tool_result_message))).await;

                    let should_stop = invocation_state.stop_event_loop
                        || structured_output_ctx.as_ref().map(|c| c.stop_loop).unwrap_or(false);

                    if should_stop {
                        return Ok(AgentResult {
                            stop_reason: StopReason::EndTurn,
                            message: assistant_message,
                            usage,
                            metrics: EventLoopMetrics::default(),
                            state: invocation_state.clone(),
                            interrupts: None,
                            structured_output: extracted_output,
                        });
                    }
                }
                StopReason::MaxTokens => return Err(StrandsError::MaxTokensReached),
                StopReason::ContentFiltered => return Err(StrandsError::ContentFiltered { message: "Content was filtered".to_string() }),
                StopReason::GuardrailIntervention => return Err(StrandsError::GuardrailIntervention { message: "Guardrail intervention".to_string() }),
                StopReason::Interrupt => return Err(StrandsError::Interrupted { message: "Agent was interrupted".to_string() }),
            }
        }
    }

    async fn execute_tools_with_structured_output(
        &self,
        content: &[ContentBlock],
        invocation_state: &InvocationState,
        structured_output_ctx: &mut Option<crate::tools::structured_output::StructuredOutputContext>,
    ) -> Result<(Vec<crate::types::tools::ToolResult>, Option<serde_json::Value>)> {
        use crate::types::tools::ToolResult;
        use crate::tools::ToolContext;

        let mut results = Vec::new();
        let mut extracted_output: Option<serde_json::Value> = None;

        let expected_tool_name = structured_output_ctx
            .as_ref()
            .and_then(|ctx| ctx.expected_tool_name().map(|s| s.to_string()));

        for block in content {
            if let Some(ref tool_use) = block.tool_use {
                let tool = self.tool_registry.get(&tool_use.name);

                let is_structured_output_tool = expected_tool_name
                    .as_ref()
                    .map(|expected| expected == &tool_use.name)
                    .unwrap_or(false);

                let result = match tool {
                    Some(tool) => {
                        let context = ToolContext::with_state(invocation_state.clone());
                        match tool.invoke(tool_use.input.clone(), &context).await {
                            Ok(r) => {
                                if is_structured_output_tool {
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

                                ToolResult {
                                    tool_use_id: tool_use.tool_use_id.clone(),
                                    status: r.status,
                                    content: r.content,
                                }
                            }
                            Err(e) => ToolResult::error(&tool_use.tool_use_id, e),
                        }
                    }
                    None => ToolResult::error(&tool_use.tool_use_id, format!("Tool not found: {}", tool_use.name)),
                };

                results.push(result);
            }
        }

        Ok((results, extracted_output))
    }
}
