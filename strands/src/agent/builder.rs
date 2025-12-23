//! Agent builder for fluent agent construction.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::conversation::ConversationManager;
use crate::hooks::HookRegistry;
use crate::models::Model;
use crate::tools::structured_output::StructuredOutputContext;
use crate::tools::{AgentTool, ToolRegistry};
use crate::types::content::Messages;
use crate::types::errors::{Result, StrandsError};

use super::{Agent, AgentState};

/// Builder for creating Agent instances.
pub struct AgentBuilder {
    model: Option<Arc<dyn Model>>,
    messages: Messages,
    system_prompt: Option<String>,
    tool_registry: ToolRegistry,
    agent_name: Option<String>,
    agent_id: String,
    description: Option<String>,
    state: AgentState,
    hooks: HookRegistry,
    conversation_manager: Option<Box<dyn ConversationManager>>,
    record_direct_tool_call: bool,
    trace_attributes: std::collections::HashMap<String, String>,
    max_tool_calls: Option<usize>,
    structured_output_context: Option<StructuredOutputContext>,
}

impl Default for AgentBuilder {
    fn default() -> Self { Self::new() }
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            messages: Vec::new(),
            system_prompt: None,
            tool_registry: ToolRegistry::new(),
            agent_name: None,
            agent_id: "default".to_string(),
            description: None,
            state: AgentState::new(),
            hooks: HookRegistry::new(),
            conversation_manager: None,
            record_direct_tool_call: false,
            trace_attributes: std::collections::HashMap::new(),
            max_tool_calls: None,
            structured_output_context: None,
        }
    }

    /// Sets the model for the agent.
    pub fn model(mut self, model: impl Model + 'static) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    /// Sets the model using an Arc.
    pub fn model_arc(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the initial messages.
    pub fn messages(mut self, messages: Messages) -> Self {
        self.messages = messages;
        self
    }

    /// Sets the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Adds a tool to the agent.
    pub fn tool(mut self, tool: impl AgentTool + 'static) -> Result<Self> {
        self.tool_registry.register_typed(tool)?;
        Ok(self)
    }

    /// Adds multiple tools to the agent.
    pub fn tools(mut self, tools: impl IntoIterator<Item = impl AgentTool + 'static>) -> Result<Self> {
        for tool in tools {
            self.tool_registry.register_typed(tool)?;
        }
        Ok(self)
    }

    /// Sets the tool registry.
    pub fn tool_registry(mut self, registry: ToolRegistry) -> Self {
        self.tool_registry = registry;
        self
    }

    /// Sets the agent name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.agent_name = Some(name.into());
        self
    }

    /// Sets the agent ID.
    pub fn agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = id.into();
        self
    }

    /// Sets the agent description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the agent state.
    pub fn state(mut self, state: AgentState) -> Self {
        self.state = state;
        self
    }

    /// Sets the hook registry.
    pub fn hooks(mut self, hooks: HookRegistry) -> Self {
        self.hooks = hooks;
        self
    }

    /// Sets the conversation manager.
    pub fn conversation_manager(mut self, manager: impl ConversationManager + 'static) -> Self {
        self.conversation_manager = Some(Box::new(manager));
        self
    }

    /// Sets whether to record direct tool calls in message history.
    pub fn record_direct_tool_call(mut self, record: bool) -> Self {
        self.record_direct_tool_call = record;
        self
    }

    /// Sets a trace attribute.
    pub fn trace_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.trace_attributes.insert(key.into(), value.into());
        self
    }

    /// Sets multiple trace attributes.
    pub fn trace_attributes(mut self, attrs: std::collections::HashMap<String, String>) -> Self {
        self.trace_attributes = attrs;
        self
    }

    /// Sets the maximum number of tool calls per cycle.
    pub fn max_tool_calls(mut self, max: usize) -> Self {
        self.max_tool_calls = Some(max);
        self
    }

    /// Sets the structured output model type.
    ///
    /// This configures the agent to enforce responses matching the schema of type `T`.
    /// The structured output tool will be dynamically registered at invocation time
    /// and cleaned up afterward.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(JsonSchema, Deserialize)]
    /// struct MyOutput {
    ///     name: String,
    ///     count: i32,
    /// }
    ///
    /// let agent = Agent::builder()
    ///     .model(model)
    ///     .structured_output_model::<MyOutput>()
    ///     .build()?;
    /// ```
    pub fn structured_output_model<T: JsonSchema + DeserializeOwned + 'static>(mut self) -> Self {
        let context = StructuredOutputContext::with_type::<T>();
        self.structured_output_context = Some(context);
        self
    }

    /// Sets a custom structured output context.
    pub fn structured_output_context(mut self, context: StructuredOutputContext) -> Self {
        self.structured_output_context = Some(context);
        self
    }

    /// Builds the agent.
    pub fn build(self) -> Result<Agent> {
        let model = self.model.ok_or_else(|| StrandsError::ConfigurationError {
            message: "Model is required".to_string(),
        })?;

        Ok(Agent {
            model,
            messages: self.messages,
            system_prompt: self.system_prompt,
            tool_registry: self.tool_registry,
            agent_name: self.agent_name,
            agent_id: self.agent_id,
            description: self.description,
            state: self.state,
            hooks: self.hooks,
            conversation_manager: self.conversation_manager.unwrap_or_else(|| {
                Box::new(crate::conversation::SlidingWindowConversationManager::default())
            }),
            interrupt_state: crate::types::interrupt::InterruptState::new(),
            record_direct_tool_call: self.record_direct_tool_call,
            trace_attributes: self.trace_attributes,
            max_tool_calls: self.max_tool_calls,
            structured_output_context: self.structured_output_context,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::BedrockModel;

    #[test]
    fn test_builder_basic() {
        let agent = Agent::builder()
            .model(BedrockModel::default())
            .system_prompt("Test prompt")
            .name("TestAgent")
            .build()
            .unwrap();

        assert_eq!(agent.name(), Some(&"TestAgent".to_string()));
        assert_eq!(agent.system_prompt(), Some("Test prompt"));
    }

    #[test]
    fn test_builder_no_model() {
        let result = Agent::builder().build();
        assert!(result.is_err());
    }
}
