//! Plugins: bundle hooks and tools into a reusable agent extension.
//!
//! Skeletal port of the Python `strands.plugins` package: a `Plugin`
//! contributes hook providers and tools, and `AgentBuilder::plugin`
//! registers them at build time.

use std::sync::Arc;

use crate::hooks::HookRegistry;
use crate::tools::ToolRegistry;
use crate::types::errors::{Result, StrandsError};

/// A reusable agent extension.
#[async_trait::async_trait]
pub trait Plugin: Send + Sync {
    /// The plugin name.
    fn name(&self) -> &str;

    /// Contributes hook providers to the registry.
    fn register_hooks(&self, _hooks: &mut HookRegistry) {}

    /// Contributes tools to the registry.
    fn register_tools(&self, _tools: &mut ToolRegistry) -> Result<()> {
        Ok(())
    }

    /// Applies plugin configuration to a built agent.
    fn init_agent(&self, _agent: &mut crate::agent::Agent) {}
}

/// Registers every plugin with the hook and tool registries, then runs
/// `init_agent` on the built agent.
pub fn apply_plugins(
    plugins: &[Arc<dyn Plugin>],
    hooks: &mut HookRegistry,
    tools: &mut ToolRegistry,
    agent: &mut crate::agent::Agent,
) -> Result<()> {
    for plugin in plugins {
        plugin.register_hooks(hooks);
        plugin.register_tools(tools).map_err(|e| StrandsError::ConfigurationError {
            message: format!("plugin '{}' failed to register tools: {}", plugin.name(), e),
        })?;
        plugin.init_agent(agent);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::HookEvent;
    use crate::types::tools::ToolSpec;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingPlugin {
        name: &'static str,
        invocations: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Plugin for CountingPlugin {
        fn name(&self) -> &str {
            self.name
        }

        fn register_hooks(&self, hooks: &mut HookRegistry) {
            let invocations = self.invocations.clone();
            hooks.add_provider(CountingHook { invocations });
        }

        fn register_tools(&self, tools: &mut ToolRegistry) -> Result<()> {
            tools.register_typed(CountingTool)?;
            Ok(())
        }
    }

    struct CountingHook {
        invocations: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl crate::hooks::HookProvider for CountingHook {
        async fn on_event(&self, _event: &mut HookEvent) {
            self.invocations.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct CountingTool;

    #[async_trait::async_trait]
    impl crate::tools::AgentTool for CountingTool {
        fn name(&self) -> &str {
            "counting_tool"
        }
        fn description(&self) -> &str {
            "Counts invocations"
        }
        fn tool_spec(&self) -> ToolSpec {
            ToolSpec::new(self.name(), self.description())
        }
        async fn invoke(
            &self,
            _input: serde_json::Value,
            _context: &crate::tools::ToolContext,
        ) -> std::result::Result<crate::tools::ToolResult2, String> {
            Ok(crate::tools::ToolResult2::success("counted"))
        }
    }

    #[tokio::test]
    async fn test_plugin_registration() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let plugin: Arc<dyn Plugin> = Arc::new(CountingPlugin {
            name: "counter",
            invocations: invocations.clone(),
        });

        let mut hooks = HookRegistry::new();
        let mut tools = ToolRegistry::new();
        let mut agent = crate::agent::Agent::builder()
            .model(crate::testing::MockModel::new(vec![]))
            .build()
            .unwrap();

        apply_plugins(&[plugin], &mut hooks, &mut tools, &mut agent).unwrap();

        assert!(tools.get("counting_tool").is_some());

        let mut event = HookEvent::AgentInitialized(crate::hooks::AgentInitializedEvent);
        hooks.invoke(&mut event).await;
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_plugin_tool_failure_surfaces() {
        struct BrokenPlugin;

        #[async_trait::async_trait]
        impl Plugin for BrokenPlugin {
            fn name(&self) -> &str {
                "broken"
            }
            fn register_tools(&self, _tools: &mut ToolRegistry) -> Result<()> {
                Err(StrandsError::ToolProviderError {
                    message: "boom".to_string(),
                })
            }
        }

        let mut hooks = HookRegistry::new();
        let mut tools = ToolRegistry::new();
        let mut agent = crate::agent::Agent::builder()
            .model(crate::testing::MockModel::new(vec![]))
            .build()
            .unwrap();

        let error =
            apply_plugins(&[Arc::new(BrokenPlugin)], &mut hooks, &mut tools, &mut agent).unwrap_err();
        assert!(error.to_string().contains("boom"));
    }
}
