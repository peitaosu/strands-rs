//! Tool registry for managing agent tools.

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::types::errors::StrandsError;
use crate::types::tools::ToolSpec;

use super::mcp::ToolProvider;
use super::AgentTool;

/// A tool input that can be processed by the registry.
pub enum ToolInput {
    /// A single tool instance.
    Tool(Box<dyn AgentTool>),
    /// A tool provider.
    Provider(Arc<dyn ToolProvider>),
    /// Multiple tool inputs (nested).
    Multiple(Vec<ToolInput>),
}

impl ToolInput {
    /// Create a tool input from an AgentTool.
    pub fn tool(tool: impl AgentTool + 'static) -> Self {
        Self::Tool(Box::new(tool))
    }

    /// Create a tool input from a ToolProvider.
    pub fn provider(provider: impl ToolProvider + 'static) -> Self {
        Self::Provider(Arc::new(provider))
    }

    /// Create a nested collection of tool inputs.
    pub fn multiple(inputs: impl IntoIterator<Item = ToolInput>) -> Self {
        Self::Multiple(inputs.into_iter().collect())
    }
}

/// Registry for managing agent tools.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn AgentTool>>,
    dynamic_tools: HashMap<String, Arc<dyn AgentTool>>,
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    registry_id: String,
}

impl Default for ToolRegistry {
    fn default() -> Self { Self::new() }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            dynamic_tools: HashMap::new(),
            tool_providers: Vec::new(),
            registry_id: Uuid::new_v4().to_string(),
        }
    }

    /// Process a list of tools.
    ///
    /// This is the Rust equivalent of Python's `process_tools` method.
    /// It handles:
    /// - AgentTool instances
    /// - ToolProvider instances
    /// - Nested collections of tools
    ///
    /// Note: Unlike Python, Rust cannot dynamically load modules from file paths.
    /// For file-based tools, use MCP servers or pre-compiled tool modules.
    ///
    /// Returns the list of tool names that were registered.
    pub async fn process_tools(&mut self, inputs: Vec<ToolInput>) -> Result<Vec<String>, StrandsError> {
        let mut tool_names = Vec::new();
        self.process_tools_recursive(inputs, &mut tool_names).await?;
        Ok(tool_names)
    }

    /// Internal recursive helper for process_tools.
    async fn process_tools_recursive(
        &mut self,
        inputs: Vec<ToolInput>,
        tool_names: &mut Vec<String>,
    ) -> Result<(), StrandsError> {
        for input in inputs {
            match input {
                ToolInput::Tool(tool) => {
                    let name = tool.tool_name().to_string();
                    if tool.is_dynamic() {
                        self.dynamic_tools.insert(name.clone(), Arc::from(tool));
                    } else {
                        self.tools.insert(name.clone(), Arc::from(tool));
                    }
                    tool_names.push(name);
                }
                ToolInput::Provider(provider) => {
                    provider.add_consumer(&self.registry_id);
                    let provider_tools = provider.load_tools().await
                        .map_err(|e| StrandsError::ToolError {
                            tool_name: "provider".to_string(),
                            message: format!("Failed to load tools from provider: {}", e),
                        })?;

                    for tool in provider_tools {
                        let name = tool.tool_name().to_string();
                        self.tools.insert(name.clone(), tool);
                        tool_names.push(name);
                    }

                    self.tool_providers.push(provider);
                }
                ToolInput::Multiple(nested) => {
                    Box::pin(self.process_tools_recursive(nested, tool_names)).await?;
                }
            }
        }
        Ok(())
    }

    /// Process tools synchronously (blocking version).
    ///
    /// This is useful when you don't have an async runtime.
    pub fn process_tools_sync(&mut self, inputs: Vec<ToolInput>) -> Result<Vec<String>, StrandsError> {
        crate::async_utils::run_async(self.process_tools(inputs))
    }

    /// Registers a tool with the registry.
    pub fn register(&mut self, tool: Box<dyn AgentTool>) {
        let name = tool.tool_name().to_string();
        self.tools.insert(name, Arc::from(tool));
    }

    /// Registers a tool with the registry (typed version).
    ///
    /// Same-name re-registration is allowed when the existing tool supports
    /// hot reload, mirroring the Python registry contract.
    pub fn register_typed(&mut self, tool: impl AgentTool + 'static) -> Result<(), StrandsError> {
        let name = tool.tool_name().to_string();

        if let Some(existing) = self.tools.get(&name) {
            if existing.supports_hot_reload() {
                self.tools.insert(name, Arc::new(tool));
                return Ok(());
            }
            return Err(StrandsError::ConfigurationError {
                message: format!("Tool '{name}' already exists"),
            });
        }

        let normalized_name = name.replace('-', "_");
        for existing_name in self.tools.keys() {
            if existing_name.replace('-', "_") == normalized_name && *existing_name != name {
                return Err(StrandsError::ConfigurationError {
                    message: format!(
                        "Tool '{name}' conflicts with existing tool '{existing_name}' (differ only by - vs _)"
                    ),
                });
            }
        }

        self.tools.insert(name, Arc::new(tool));
        Ok(())
    }

    /// Registers multiple tools with the registry.
    pub fn register_all(
        &mut self,
        tools: impl IntoIterator<Item = impl AgentTool + 'static>,
    ) {
        for tool in tools {
            self.tools.insert(tool.tool_name().to_string(), Arc::new(tool));
        }
    }

    /// Gets a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn AgentTool>> {
        self.tools.get(name).or_else(|| self.dynamic_tools.get(name)).cloned()
    }

    /// Returns a list of all tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().chain(self.dynamic_tools.keys()).map(|s| s.as_str()).collect()
    }

    /// Returns all tool specifications, normalized per the Python contract.
    ///
    /// Specs failing validation or normalization are dropped with a warning
    /// instead of failing the call, matching `ToolRegistry.get_all_tools_config`.
    pub fn get_all_tool_specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .chain(self.dynamic_tools.values())
            .filter_map(|tool| {
                let spec = tool.tool_spec();
                match super::validator::normalize_tool_spec(&spec) {
                    Ok(normalized) => Some(normalized),
                    Err(e) => {
                        tracing::warn!(
                            tool_name = %tool.name(),
                            error = %e,
                            "Invalid tool specification, skipping"
                        );
                        None
                    }
                }
            })
            .collect()
    }

    /// Returns all tools as a configuration map.
    pub fn get_all_tools_config(&self) -> HashMap<String, ToolSpec> {
        self.tools.iter().chain(self.dynamic_tools.iter()).map(|(n, t)| (n.clone(), t.tool_spec())).collect()
    }

    pub fn len(&self) -> usize { self.tools.len() + self.dynamic_tools.len() }
    pub fn is_empty(&self) -> bool { self.tools.is_empty() && self.dynamic_tools.is_empty() }

    /// Registers a dynamic tool.
    pub fn register_dynamic(&mut self, tool: impl AgentTool + 'static) -> Result<(), StrandsError> {
        let name = tool.tool_name().to_string();

        if self.tools.contains_key(&name) || self.dynamic_tools.contains_key(&name) {
            return Err(StrandsError::ConfigurationError {
                message: format!("Tool '{name}' already exists"),
            });
        }

        self.dynamic_tools.insert(name, Arc::new(tool));
        Ok(())
    }

    /// Registers a tool from a ToolSpec using StructuredOutputAgentTool.
    pub fn register_spec(&mut self, spec: ToolSpec) -> Result<(), StrandsError> {
        let tool = super::structured_output::StructuredOutputAgentTool::from_spec(spec);
        self.register_typed(tool)
    }

    /// Removes a dynamic tool by name.
    pub fn remove_dynamic(&mut self, name: &str) -> bool {
        self.dynamic_tools.remove(name).is_some()
    }

    /// Replaces an existing tool.
    pub fn replace(&mut self, tool: impl AgentTool + 'static) -> Result<(), StrandsError> {
        let name = tool.tool_name().to_string();
        let tool_arc = Arc::new(tool);

        if let Some(entry) = self.tools.get_mut(&name) {
            *entry = tool_arc;
            Ok(())
        } else if let Some(entry) = self.dynamic_tools.get_mut(&name) {
            *entry = tool_arc;
            Ok(())
        } else {
            Err(StrandsError::ToolNotFound { tool_name: name })
        }
    }

    /// Clears all tools from the registry.
    pub fn clear(&mut self) {
        self.tools.clear();
        self.dynamic_tools.clear();
    }

    /// Cleans up the registry and all tool providers.
    ///
    /// This removes all consumers from tool providers and clears the registry.
    /// Errors from individual providers are logged but don't stop cleanup.
    pub fn cleanup(&mut self) {

        for provider in &self.tool_providers {
            provider.remove_consumer(&self.registry_id);
            tracing::debug!("provider cleanup | removed consumer");
        }
        self.tool_providers.clear();
        self.clear();
    }

    /// Get the registry ID.
    pub fn registry_id(&self) -> &str {
        &self.registry_id
    }

    /// Reloads a tool by name.
    ///
    /// In Python, this reloads a tool module from disk for hot-reloading during development.
    /// In Rust, compiled tools cannot be dynamically reloaded. This method is provided for
    /// API compatibility and will:
    /// 1. Log that a reload was requested
    /// 2. Return success without any action for compiled tools
    ///
    /// For dynamic tool implementations (e.g., MCP tools, external process tools),
    /// subclasses or custom implementations may override this behavior.
    pub fn reload_tool(&mut self, name: &str) -> Result<(), StrandsError> {
        if !self.tools.contains_key(name) && !self.dynamic_tools.contains_key(name) {
            return Err(StrandsError::ToolNotFound {
                tool_name: name.to_string(),
            });
        }

        tracing::info!(
            "tool_name=<{}> | reload requested (compiled Rust tools do not support hot reload)",
            name
        );
        Ok(())
    }

    /// Gets the directories being watched for tools.
    ///
    /// Returns the standard tool directories that would be scanned for tools.
    /// In Python, tools can be loaded from these directories at runtime.
    /// In Rust, this is provided for API compatibility and returns the current
    /// working directory's `./tools/` path if it exists.
    pub fn get_tools_dirs(&self) -> Vec<std::path::PathBuf> {
        let mut dirs = Vec::new();

        if let Ok(cwd) = std::env::current_dir() {
            let tools_dir = cwd.join("tools");
            if tools_dir.exists() && tools_dir.is_dir() {
                tracing::debug!("tools_dir=<{}> | found tools directory", tools_dir.display());
                dirs.push(tools_dir);
            }
        }

        dirs
    }

    /// Discovers available tool modules in all tools directories.
    ///
    /// Returns a map of tool names to their full paths.
    /// In Python, this scans for `.py` files in the tools directories.
    /// In Rust, this is provided for API compatibility and scans for common
    /// tool configuration files (JSON, YAML, WASM).
    pub fn discover_tool_modules(&self) -> HashMap<String, std::path::PathBuf> {
        let mut tool_modules = HashMap::new();

        for tools_dir in self.get_tools_dirs() {
            tracing::debug!("tools_dir=<{}> | scanning", tools_dir.display());

            let entries = match std::fs::read_dir(&tools_dir) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("tools_dir=<{}> | failed to read: {}", tools_dir.display(), e);
                    continue;
                }
            };

            let valid_extensions = ["json", "yaml", "yml", "wasm"];

            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }

                let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if !valid_extensions.contains(&extension) {
                    continue;
                }

                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if stem.starts_with('_') {
                        continue;
                    }

                    tracing::debug!(
                        "tools_dir=<{}>, module_name=<{}> | discovered tool",
                        tools_dir.display(),
                        stem
                    );
                    tool_modules.insert(stem.to_string(), path);
                }
            }
        }

        tracing::debug!("tool_modules=<{:?}> | discovered", tool_modules.keys().collect::<Vec<_>>());
        tool_modules
    }

    /// Validates a tool specification.
    pub fn validate_spec(spec: &ToolSpec) -> Result<(), StrandsError> {
        if spec.name.is_empty() {
            return Err(StrandsError::ToolValidationError {
                message: "Tool name cannot be empty".to_string(),
            });
        }

        if spec.description.is_empty() {
            return Err(StrandsError::ToolValidationError {
                message: format!("Tool '{}' has an empty description", spec.name),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::tools::{ToolContext, ToolResult2};

    struct DummyTool { name: String }

    impl DummyTool {
        fn new(name: &str) -> Self { Self { name: name.to_string() } }
    }

    #[async_trait]
    impl AgentTool for DummyTool {
        fn name(&self) -> &str { &self.name }
        fn description(&self) -> &str { "A dummy tool" }
        fn tool_spec(&self) -> ToolSpec { ToolSpec::new(&self.name, "A dummy tool") }

        async fn invoke(
            &self,
            _input: serde_json::Value,
            _context: &ToolContext,
        ) -> std::result::Result<ToolResult2, String> {
            Ok(ToolResult2::success("dummy result"))
        }
    }

    #[test]
    fn test_registry_register() {
        let mut registry = ToolRegistry::new();
        registry.register_typed(DummyTool::new("test")).unwrap();
        assert_eq!(registry.len(), 1);
        assert!(registry.get("test").is_some());
    }

    #[test]
    fn test_registry_duplicate() {
        let mut registry = ToolRegistry::new();
        registry.register_typed(DummyTool::new("test")).unwrap();
        let result = registry.register_typed(DummyTool::new("test"));
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_normalized_conflict() {
        let mut registry = ToolRegistry::new();
        registry.register_typed(DummyTool::new("my_tool")).unwrap();
        let result = registry.register_typed(DummyTool::new("my-tool"));
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_get_all_specs() {
        let mut registry = ToolRegistry::new();
        registry.register_typed(DummyTool::new("tool1")).unwrap();
        registry.register_typed(DummyTool::new("tool2")).unwrap();
        let specs = registry.get_all_tool_specs();
        assert_eq!(specs.len(), 2);
    }
}
