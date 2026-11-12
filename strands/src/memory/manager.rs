//! Memory manager: fan-out search across stores and tool exposure.

use std::sync::Arc;

use async_trait::async_trait;

use super::{MemoryEntry, MemoryStore};
use crate::hooks::{HookEvent, HookProvider};
use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::errors::StrandsError;
use crate::types::tools::ToolSpec;

/// Aggregates memory stores behind search and (optional) write tools.
///
/// Mirrors the Python `MemoryManager`: search fans out across every store,
/// write operations target writable stores, and message ingestion is
/// delegated to stores that support server-side extraction.
pub struct MemoryManager {
    stores: Vec<Arc<dyn MemoryStore>>,
    max_search_results: usize,
}

impl MemoryManager {
    /// Creates a manager over `stores`.
    pub fn new(stores: Vec<Arc<dyn MemoryStore>>) -> Self {
        Self {
            stores,
            max_search_results: 5,
        }
    }

    /// Sets the per-store search result cap (Python default: 5).
    pub fn with_max_search_results(mut self, max: usize) -> Self {
        self.max_search_results = max;
        self
    }

    pub fn stores(&self) -> &[Arc<dyn MemoryStore>] {
        &self.stores
    }

    /// Fans a query out across all stores, aggregating results.
    ///
    /// Store failures are aggregated into a single `AggregateMemoryError`
    /// when every store failed; partial success returns the successes.
    pub async fn search(&self, query: &str) -> Result<Vec<MemoryEntry>, StrandsError> {
        let mut results = Vec::new();
        let mut failures = Vec::new();

        for store in &self.stores {
            match store.search(query, self.max_search_results).await {
                Ok(entries) => results.extend(entries),
                Err(e) => failures.push(format!("{}: {}", store.name(), e)),
            }
        }

        if results.is_empty() && !failures.is_empty() {
            return Err(StrandsError::AggregateMemoryError {
                message: failures.join("; "),
            });
        }

        Ok(results)
    }

    /// Writes an entry to every writable store.
    pub async fn add(&self, content: impl Into<String>, metadata: serde_json::Value) -> Result<(), StrandsError> {
        let content = content.into();
        let mut failures = Vec::new();

        for store in &self.stores {
            if store.is_writable() {
                if let Err(e) = store.add(content.clone(), metadata.clone()).await {
                    failures.push(format!("{}: {}", store.name(), e));
                }
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(StrandsError::AggregateMemoryError {
                message: failures.join("; "),
            })
        }
    }

    /// Feeds messages to stores with server-side extraction.
    pub async fn add_messages(&self, messages: &[crate::types::content::Message]) -> Result<(), StrandsError> {
        let mut failures = Vec::new();

        for store in &self.stores {
            if store.supports_add_messages() {
                if let Err(e) = store.add_messages(messages).await {
                    failures.push(format!("{}: {}", store.name(), e));
                }
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(StrandsError::AggregateMemoryError {
                message: failures.join("; "),
            })
        }
    }

    /// Registers the `search_memory` tool on the registry.
    pub fn register_search_tool(&self, registry: &mut crate::tools::ToolRegistry) -> Result<(), StrandsError> {
        let tool = SearchMemoryTool {
            manager: Arc::new(self.snapshot()),
            store_names: self.stores.iter().map(|s| s.name().to_string()).collect(),
        };
        registry.register_dynamic(tool).map_err(|e| {
            StrandsError::ToolProviderError {
                message: format!("failed to register search_memory tool: {}", e),
            }
        })
    }

    fn snapshot(&self) -> MemoryManagerHandle {
        MemoryManagerHandle {
            stores: self.stores.clone(),
            max_search_results: self.max_search_results,
        }
    }
}

/// Cloneable handle used by the registered tool.
struct MemoryManagerHandle {
    stores: Vec<Arc<dyn MemoryStore>>,
    max_search_results: usize,
}

impl MemoryManagerHandle {
    async fn search(&self, query: &str) -> Result<Vec<MemoryEntry>, StrandsError> {
        let manager = MemoryManager {
            stores: self.stores.clone(),
            max_search_results: self.max_search_results,
        };
        manager.search(query).await
    }

    async fn add_messages(&self, messages: &[crate::types::content::Message]) -> Result<(), StrandsError> {
        let manager = MemoryManager {
            stores: self.stores.clone(),
            max_search_results: self.max_search_results,
        };
        manager.add_messages(messages).await
    }
}

/// The `search_memory` agent tool exposed by `MemoryManager`.
struct SearchMemoryTool {
    manager: Arc<MemoryManagerHandle>,
    store_names: Vec<String>,
}

#[async_trait]
impl AgentTool for SearchMemoryTool {
    fn name(&self) -> &str {
        "search_memory"
    }

    fn description(&self) -> &str {
        "Search the agent's long-term memory. Returns matching memory entries."
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description()).with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        match self.manager.search(&query).await {
            Ok(entries) => {
                let payload = serde_json::json!({
                    "stores": self.store_names,
                    "entries": entries,
                });
                Ok(ToolResult2::success_json(payload))
            }
            Err(e) => Err(e.to_string()),
        }
    }
}

/// Hook provider feeding appended messages to extraction-capable stores.
pub struct MemoryHook {
    manager: Arc<MemoryManagerHandle>,
}

impl MemoryHook {
    /// Creates a hook provider from a manager.
    pub fn new(manager: &MemoryManager) -> Self {
        Self {
            manager: Arc::new(manager.snapshot()),
        }
    }
}

#[async_trait]
impl HookProvider for MemoryHook {
    async fn on_event(&self, event: &mut HookEvent) {
        if let HookEvent::MessageAdded(added) = event {
            if let Err(e) = self.manager.add_messages(std::slice::from_ref(&added.message)).await {
                tracing::debug!("memory extraction failed: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemoryMemoryStore;
    use crate::tools::ToolRegistry;
    use crate::types::tools::ToolResultStatus;

    fn manager() -> MemoryManager {
        let a: Arc<dyn MemoryStore> = Arc::new(InMemoryMemoryStore::new("a"));
        let b: Arc<dyn MemoryStore> = Arc::new(InMemoryMemoryStore::new("b"));
        MemoryManager::new(vec![a, b])
    }

    #[tokio::test]
    async fn test_search_fan_out() {
        let manager = manager();
        manager.add("rust likes cargo", serde_json::json!({})).await.unwrap();

        let results = manager.search("cargo").await.unwrap();
        assert_eq!(results.len(), 2);
        let mut store_names: Vec<&str> = results.iter().map(|e| e.store_name.as_str()).collect();
        store_names.sort();
        assert_eq!(store_names, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn test_search_all_stores_fail_aggregates() {
        struct Broken;

        #[async_trait]
        impl MemoryStore for Broken {
            fn name(&self) -> &str { "broken" }
            async fn search(&self, _query: &str, _max: usize) -> Result<Vec<MemoryEntry>, StrandsError> {
                Err(StrandsError::StorageError { message: "down".to_string() })
            }
        }

        let manager = MemoryManager::new(vec![Arc::new(Broken)]);
        let error = manager.search("q").await.unwrap_err();
        assert!(matches!(error, StrandsError::AggregateMemoryError { .. }));
    }

    #[tokio::test]
    async fn test_search_memory_tool() {
        let manager = manager();
        manager.add("secret token abc", serde_json::json!({})).await.unwrap();

        let mut registry = ToolRegistry::new();
        manager.register_search_tool(&mut registry).unwrap();

        let tool = registry.get("search_memory").expect("tool registered");
        let context = ToolContext::new(crate::types::tools::ToolUse::new(
            "search_memory",
            "t1",
            serde_json::json!({}),
        ));
        let result = tool
            .invoke(serde_json::json!({"query": "token"}), &context)
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        let text = result.content[0].json.as_ref().expect("json payload");
        assert!(text["entries"].as_array().unwrap().iter().any(|e| e["content"]
            .as_str()
            .unwrap()
            .contains("secret token abc")));
    }

    #[tokio::test]
    async fn test_memory_hook_feeds_extraction() {
        let manager = manager();
        let hook = MemoryHook::new(&manager);

        let mut event = HookEvent::MessageAdded(crate::hooks::MessageAddedEvent::new(
            crate::types::content::Message::assistant("remember this"),
        ));
        hook.on_event(&mut event).await;
    }
}
