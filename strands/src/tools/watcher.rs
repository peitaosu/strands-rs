//! Tool watcher for hot reloading tools during development.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::tools::registry::ToolRegistry;
use crate::types::errors::Result;

/// Watches tool directories for changes and reloads tools when modified.
pub struct ToolWatcher {
    tool_registry: Arc<RwLock<ToolRegistry>>,
    change_handler: Arc<ToolChangeHandler>,
    watched_dirs: HashSet<PathBuf>,
    running: Arc<Mutex<bool>>,
}

impl ToolWatcher {
    /// Create a new tool watcher for the given tool registry.
    pub fn new(tool_registry: Arc<RwLock<ToolRegistry>>) -> Self {
        let change_handler = Arc::new(ToolChangeHandler::new(tool_registry.clone()));
        Self {
            tool_registry,
            change_handler,
            watched_dirs: HashSet::new(),
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Add a directory to watch.
    pub fn watch_dir(&mut self, dir: PathBuf) {
        self.watched_dirs.insert(dir);
    }

    /// Get the tool registry.
    pub fn tool_registry(&self) -> &Arc<RwLock<ToolRegistry>> {
        &self.tool_registry
    }

    /// Get the change handler.
    pub fn change_handler(&self) -> &Arc<ToolChangeHandler> {
        &self.change_handler
    }

    /// Start watching for changes using polling.
    pub fn start(&self) -> Result<()> {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Ok(());
        }
        *running = true;

        let running_flag = self.running.clone();
        let handler = self.change_handler.clone();
        let dirs: Vec<PathBuf> = self.watched_dirs.iter().cloned().collect();

        tokio::spawn(async move {
            while *running_flag.lock().unwrap() {
                for dir in &dirs {
                    let changed = handler.poll_changes(dir);
                    for path in changed {
                        handler.on_modified(&path);
                    }
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });

        tracing::debug!("Tool watcher started for {} directories", self.watched_dirs.len());

        Ok(())
    }

    /// Stop watching for changes.
    pub fn stop(&self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
        tracing::debug!("Tool watcher stopped");
    }

    /// Check if the watcher is running.
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }

    /// Get the watched directories.
    pub fn watched_dirs(&self) -> &HashSet<PathBuf> {
        &self.watched_dirs
    }
}

/// Change handler for tool file modifications.
pub struct ToolChangeHandler {
    tool_registry: Arc<RwLock<ToolRegistry>>,
    file_timestamps: Mutex<HashMap<PathBuf, std::time::SystemTime>>,
}

impl ToolChangeHandler {
    pub fn new(tool_registry: Arc<RwLock<ToolRegistry>>) -> Self {
        Self {
            tool_registry,
            file_timestamps: Mutex::new(HashMap::new()),
        }
    }

    /// Handle a file modification event.
    pub fn on_modified(&self, path: &PathBuf) {
        if let Some(ext) = path.extension() {
            if ext != "rs" {
                return;
            }
        } else {
            return;
        }

        if let Some(stem) = path.file_stem() {
            let stem_str = stem.to_string_lossy();
            if stem_str == "mod" || stem_str.starts_with("_") {
                return;
            }

            tracing::debug!("Tool change detected: {}", stem_str);

            if let Ok(mut registry) = self.tool_registry.write() {
                if let Err(e) = registry.reload_tool(&stem_str) {
                    tracing::error!("Failed to reload tool {}: {}", stem_str, e);
                }
            }
        }
    }

    /// Check for modifications using polling.
    pub fn poll_changes(&self, dir: &PathBuf) -> Vec<PathBuf> {
        let mut changed = Vec::new();
        let mut timestamps = self.file_timestamps.lock().unwrap();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "rs").unwrap_or(false) {
                    if let Ok(metadata) = std::fs::metadata(&path) {
                        if let Ok(modified) = metadata.modified() {
                            let prev = timestamps.get(&path).copied();
                            if prev.map(|p| modified > p).unwrap_or(true) {
                                timestamps.insert(path.clone(), modified);
                                if prev.is_some() {
                                    changed.push(path);
                                }
                            }
                        }
                    }
                }
            }
        }

        changed
    }
}

/// Master handler that delegates to all registered handlers.
pub struct MasterChangeHandler {
    dir_path: PathBuf,
    handlers: Arc<RwLock<HashMap<String, Arc<ToolChangeHandler>>>>,
}

impl MasterChangeHandler {
    pub fn new(dir_path: PathBuf) -> Self {
        Self {
            dir_path,
            handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a handler for a registry.
    pub fn add_handler(&self, registry_id: String, handler: Arc<ToolChangeHandler>) {
        if let Ok(mut handlers) = self.handlers.write() {
            handlers.insert(registry_id, handler);
        }
    }

    /// Remove a handler.
    pub fn remove_handler(&self, registry_id: &str) {
        if let Ok(mut handlers) = self.handlers.write() {
            handlers.remove(registry_id);
        }
    }

    /// Handle a file modification.
    pub fn on_modified(&self, path: &PathBuf) {
        if let Ok(handlers) = self.handlers.read() {
            for handler in handlers.values() {
                handler.on_modified(path);
            }
        }
    }

    /// Get the directory path.
    pub fn dir_path(&self) -> &PathBuf {
        &self.dir_path
    }
}

/// Polling-based watcher that periodically checks for file changes.
pub struct PollingWatcher {
    watched_dirs: Vec<PathBuf>,
    interval: Duration,
    running: Arc<Mutex<bool>>,
    handler: Arc<ToolChangeHandler>,
}

impl PollingWatcher {
    /// Create a new polling watcher.
    pub fn new(tool_registry: Arc<RwLock<ToolRegistry>>, interval: Duration) -> Self {
        let handler = Arc::new(ToolChangeHandler::new(tool_registry));
        Self {
            watched_dirs: Vec::new(),
            interval,
            running: Arc::new(Mutex::new(false)),
            handler,
        }
    }

    /// Add a directory to watch.
    pub fn watch_dir(&mut self, dir: PathBuf) {
        self.watched_dirs.push(dir);
    }

    /// Start the polling loop in a background task.
    pub fn start(&self) {
        let mut running = self.running.lock().unwrap();
        if *running {
            return;
        }
        *running = true;

        let running_flag = self.running.clone();
        let handler = self.handler.clone();
        let dirs = self.watched_dirs.clone();
        let interval = self.interval;

        tokio::spawn(async move {
            while *running_flag.lock().unwrap() {
                for dir in &dirs {
                    let changed = handler.poll_changes(dir);
                    for path in changed {
                        handler.on_modified(&path);
                    }
                }
                tokio::time::sleep(interval).await;
            }
        });

        tracing::info!("Polling watcher started with {:?} interval", self.interval);
    }

    /// Stop the polling loop.
    pub fn stop(&self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
        tracing::info!("Polling watcher stopped");
    }

    /// Check if running.
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_watcher_creation() {
        let registry = Arc::new(RwLock::new(ToolRegistry::new()));
        let mut watcher = ToolWatcher::new(registry);
        
        watcher.watch_dir(PathBuf::from("/tmp/tools"));
        assert_eq!(watcher.watched_dirs().len(), 1);
    }

    #[test]
    fn test_tool_change_handler() {
        let registry = Arc::new(RwLock::new(ToolRegistry::new()));
        let handler = ToolChangeHandler::new(registry);
        
        handler.on_modified(&PathBuf::from("/tmp/test.txt"));
    }

    #[test]
    fn test_master_change_handler() {
        let handler = MasterChangeHandler::new(PathBuf::from("/tmp/tools"));
        assert_eq!(handler.dir_path(), &PathBuf::from("/tmp/tools"));
    }
}

