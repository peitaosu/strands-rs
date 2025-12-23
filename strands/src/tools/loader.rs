//! Tool loading utilities for dynamic tool discovery.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::tools::AgentTool;
use crate::types::errors::{Result, StrandsError};

/// Configuration for loading tools from a directory.
#[derive(Debug, Clone)]
pub struct ToolLoaderConfig {
    pub tool_dirs: Vec<PathBuf>,
    pub recursive: bool,
    pub file_patterns: Vec<String>,
}

impl Default for ToolLoaderConfig {
    fn default() -> Self {
        Self {
            tool_dirs: Vec::new(),
            recursive: false,
            file_patterns: vec!["*.rs".to_string()],
        }
    }
}

impl ToolLoaderConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.tool_dirs.push(dir.into());
        self
    }

    pub fn recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }
}

/// Tool loader for discovering and loading tools from directories.
pub struct ToolLoader {
    config: ToolLoaderConfig,
    loaded_tools: HashMap<String, Arc<dyn AgentTool>>,
    tool_paths: HashMap<String, PathBuf>,
}

impl ToolLoader {
    pub fn new(config: ToolLoaderConfig) -> Self {
        Self {
            config,
            loaded_tools: HashMap::new(),
            tool_paths: HashMap::new(),
        }
    }

    /// Get the list of tool directories being watched.
    pub fn tool_dirs(&self) -> &[PathBuf] {
        &self.config.tool_dirs
    }

    /// Get all loaded tools.
    pub fn tools(&self) -> Vec<Arc<dyn AgentTool>> {
        self.loaded_tools.values().cloned().collect()
    }

    /// Get a specific tool by name.
    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn AgentTool>> {
        self.loaded_tools.get(name).cloned()
    }

    /// Check if a tool is loaded.
    pub fn has_tool(&self, name: &str) -> bool {
        self.loaded_tools.contains_key(name)
    }

    /// Register a tool.
    pub fn register_tool(&mut self, tool: Arc<dyn AgentTool>, path: Option<PathBuf>) {
        let name = tool.tool_name().to_string();
        self.loaded_tools.insert(name.clone(), tool);
        if let Some(p) = path {
            self.tool_paths.insert(name, p);
        }
    }

    /// Unregister a tool.
    pub fn unregister_tool(&mut self, name: &str) -> Option<Arc<dyn AgentTool>> {
        self.tool_paths.remove(name);
        self.loaded_tools.remove(name)
    }

    /// Get the path for a tool.
    pub fn tool_path(&self, name: &str) -> Option<&PathBuf> {
        self.tool_paths.get(name)
    }

    /// Scan directories for tool files.
    pub fn scan_directories(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for dir in &self.config.tool_dirs {
            if !dir.exists() {
                continue;
            }

            self.scan_directory(dir, &mut files)?;
        }

        Ok(files)
    }

    fn scan_directory(&self, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| StrandsError::InternalError {
            message: format!("Failed to read directory {}: {}", dir.display(), e),
        })?;

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() && self.config.recursive {
                self.scan_directory(&path, files)?;
            } else if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "rs" {
                        files.push(path);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Callback type for tool reload events.
pub type ReloadCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// Tool watcher for monitoring tool changes during development.
///
/// This implementation provides a callback-based notification system for tool changes.
/// Unlike the Python SDK which uses `watchdog` for automatic file system monitoring,
/// Rust tools are typically compiled and don't support runtime hot-reloading.
///
/// This watcher is useful for:
/// - Notifying when tools are programmatically reloaded
/// - Integration with external file watchers (e.g., `notify` crate)
/// - Development-time tooling that manages tool lifecycle
///
/// For automatic file watching, integrate with the `notify` crate and call
/// `notify_modified` when file changes are detected.
pub struct ToolWatcher {
    loader: ToolLoader,
    on_reload: Option<ReloadCallback>,
}

impl ToolWatcher {
    pub fn new(loader: ToolLoader) -> Self {
        Self {
            loader,
            on_reload: None,
        }
    }

    pub fn on_reload(mut self, callback: ReloadCallback) -> Self {
        self.on_reload = Some(callback);
        self
    }

    /// Get the tool loader.
    pub fn loader(&self) -> &ToolLoader {
        &self.loader
    }

    /// Get mutable reference to the tool loader.
    pub fn loader_mut(&mut self) -> &mut ToolLoader {
        &mut self.loader
    }

    /// Notify that a tool has been modified.
    pub fn notify_modified(&self, tool_name: &str) {
        if let Some(ref callback) = self.on_reload {
            callback(tool_name);
        }
    }

    /// Get the directories being watched.
    pub fn watched_dirs(&self) -> &[PathBuf] {
        self.loader.tool_dirs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_loader_config() {
        let config = ToolLoaderConfig::new()
            .add_dir("/tmp/tools")
            .recursive(true);

        assert_eq!(config.tool_dirs.len(), 1);
        assert!(config.recursive);
    }

    #[test]
    fn test_tool_loader_creation() {
        let config = ToolLoaderConfig::new();
        let loader = ToolLoader::new(config);

        assert!(loader.tools().is_empty());
        assert!(loader.tool_dirs().is_empty());
    }
}

