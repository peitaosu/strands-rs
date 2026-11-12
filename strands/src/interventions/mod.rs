//! Interventions: guardrails wired into the hook registry.
//!
//! Skeletal port of the Python `strands.interventions` package: an
//! `InterventionHandler` inspects lifecycle events and returns typed
//! actions; the `InterventionRegistry` bridges those actions into the
//! hook events' control fields at the `HookOrder` intervention bands.

use std::sync::Arc;

use async_trait::async_trait;

use crate::hooks::{HookEvent, HookOrder, HookProvider};
use crate::types::interrupt::Interrupt;

/// What an intervention asks the framework to do.
#[derive(Debug, Clone, PartialEq)]
pub enum InterventionAction {
    /// Continue normally.
    Proceed,
    /// Reject the call; the string explains why.
    Deny(String),
    /// Let the call proceed but steer the model with guidance.
    Guide(String),
}

/// Inspects agent lifecycle events and returns intervention actions.
#[async_trait]
pub trait InterventionHandler: Send + Sync {
    async fn before_invocation(&self, _input: Option<&str>) -> InterventionAction {
        InterventionAction::Proceed
    }

    async fn before_tool_call(&self, _tool_name: &str, _input: &serde_json::Value) -> InterventionAction {
        InterventionAction::Proceed
    }

    async fn after_tool_call(&self, _tool_name: &str, _result: &str) -> InterventionAction {
        InterventionAction::Proceed
    }

    async fn before_model_call(&self) -> InterventionAction {
        InterventionAction::Proceed
    }

    async fn after_invocation(&self, _result_text: &str) -> InterventionAction {
        InterventionAction::Proceed
    }
}

/// Bridges `InterventionHandler`s into the hook registry.
pub struct InterventionRegistry {
    handlers: Vec<Arc<dyn InterventionHandler>>,
}

impl InterventionRegistry {
    pub fn new() -> Self {
        Self { handlers: Vec::new() }
    }

    pub fn add_handler(&mut self, handler: Arc<dyn InterventionHandler>) {
        self.handlers.push(handler);
    }

    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Hooks into a registry: input-side decisions at `INTERVENTION_INPUT`,
    /// output-side at `INTERVENTION_OUTPUT`.
    pub fn register(self, hooks: &mut crate::hooks::HookRegistry) {
        if self.is_empty() {
            return;
        }
        let input = Arc::new(InterventionHook {
            handlers: self.handlers.clone(),
            side: Side::Input,
        });
        let output = Arc::new(InterventionHook {
            handlers: self.handlers,
            side: Side::Output,
        });
        hooks.add_provider_arc_at(input, HookOrder::INTERVENTION_INPUT);
        hooks.add_provider_arc_at(output, HookOrder::INTERVENTION_OUTPUT);
    }
}

impl Default for InterventionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Input,
    Output,
}

struct InterventionHook {
    handlers: Vec<Arc<dyn InterventionHandler>>,
    side: Side,
}

#[async_trait]
impl HookProvider for InterventionHook {
    async fn on_event(&self, event: &mut HookEvent) {
        match (self.side, event) {
            (Side::Input, HookEvent::BeforeInvocation(before)) => {
                for handler in &self.handlers {
                    if let InterventionAction::Deny(reason) = handler.before_invocation(None).await {
                        before.cancel = Some(reason);
                        return;
                    }
                }
            }
            (Side::Input, HookEvent::BeforeToolCall(before)) => {
                for handler in &self.handlers {
                    if let InterventionAction::Deny(reason) =
                        handler.before_tool_call(&before.tool_use.name, &before.tool_use.input).await
                    {
                        before.cancel_tool = Some(reason);
                        return;
                    }
                }
            }
            (Side::Input, HookEvent::BeforeModelCall(before)) => {
                for handler in &self.handlers {
                    if let InterventionAction::Deny(reason) = handler.before_model_call().await {
                        before.cancel = Some(reason);
                        return;
                    }
                }
            }
            (Side::Output, HookEvent::AfterToolCall(after)) => {
                let result_text = after
                    .result
                    .content
                    .iter()
                    .filter_map(|content| content.text.as_deref())
                    .collect::<Vec<_>>()
                    .join("");
                for handler in &self.handlers {
                    if let InterventionAction::Deny(reason) =
                        handler.after_tool_call(&after.tool_use.name, &result_text).await
                    {
                        after.cancel_message = Some(reason);
                        return;
                    }
                }
            }
            (Side::Output, HookEvent::AfterInvocation(after)) => {
                let result_text = after
                    .result
                    .as_ref()
                    .map(|result| result.text())
                    .unwrap_or_default();
                for handler in &self.handlers {
                    if let InterventionAction::Deny(reason) =
                        handler.after_invocation(&result_text).await
                    {
                        after.resume = Some(crate::agent::AgentInput::Text(reason));
                        return;
                    }
                }
            }
            _ => {}
        }
    }
}

/// Raised when an intervention denies a call, as an interrupt.
pub fn intervention_interrupt(name: &str, reason: &str) -> Interrupt {
    Interrupt::new(
        format!("v1:intervention:{}", name),
        name,
    )
    .with_reason(serde_json::json!({ "reason": reason }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::BeforeToolCallEvent;

    struct DenyAll;

    #[async_trait]
    impl InterventionHandler for DenyAll {
        async fn before_tool_call(&self, _tool_name: &str, _input: &serde_json::Value) -> InterventionAction {
            InterventionAction::Deny("blocked by policy".to_string())
        }
    }

    #[tokio::test]
    async fn test_intervention_denies_tool_call() {
        let mut hooks = crate::hooks::HookRegistry::new();
        let mut interventions = InterventionRegistry::new();
        interventions.add_handler(Arc::new(DenyAll));
        interventions.register(&mut hooks);

        let mut event = HookEvent::BeforeToolCall(BeforeToolCallEvent::new(
            crate::types::tools::ToolUse::new("risky", "t1", serde_json::json!({})),
        ));
        hooks.invoke(&mut event).await;

        if let HookEvent::BeforeToolCall(e) = event {
            assert_eq!(e.cancel_tool.as_deref(), Some("blocked by policy"));
        } else {
            panic!("unexpected event");
        }
    }

    #[test]
    fn test_intervention_interrupt_shape() {
        let interrupt = intervention_interrupt("approval", "needs human");
        assert!(interrupt.id.starts_with("v1:intervention:"));
        assert!(interrupt.reason.is_some());
    }
}