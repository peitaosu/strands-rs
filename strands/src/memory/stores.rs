//! Reference memory store implementations.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use super::{MemoryEntry, MemoryResult, MemoryStore};
use crate::types::errors::StrandsError;

/// Simple in-memory vector of entries with substring search.
#[derive(Clone, Default)]
pub struct InMemoryMemoryStore {
    name: String,
    entries: Arc<RwLock<Vec<MemoryEntry>>>,
    writable: bool,
}

impl InMemoryMemoryStore {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entries: Arc::new(RwLock::new(Vec::new())),
            writable: true,
        }
    }

    pub fn read_only(mut self) -> Self {
        self.writable = false;
        self
    }
}

#[async_trait]
impl MemoryStore for InMemoryMemoryStore {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "In-memory memory store with substring search"
    }

    async fn search(&self, query: &str, max_results: usize) -> MemoryResult<Vec<MemoryEntry>> {
        let query_lower = query.to_lowercase();
        let entries = self.entries.read().await;
        let matched = entries
            .iter()
            .filter(|entry| entry.content.to_lowercase().contains(&query_lower))
            .take(max_results)
            .cloned()
            .collect();
        Ok(matched)
    }

    async fn add(&self, content: String, metadata: serde_json::Value) -> MemoryResult<()> {
        if !self.writable {
            return Err(StrandsError::StorageError {
                message: format!("memory store '{}' is not writable", self.name),
            });
        }
        self.entries
            .write()
            .await
            .push(MemoryEntry::new(self.name.clone(), content).with_metadata(metadata));
        Ok(())
    }

    fn is_writable(&self) -> bool {
        self.writable
    }
}

/// File-backed store: one markdown document per key, keyword-scored search.
///
/// Skeletal port of the Python `FileMemoryStore` (design 0017): documents
/// live in the `knowledge/` namespace of a `Storage` backend; search ranks
/// documents by keyword hits.
pub struct FileMemoryStore {
    name: String,
    storage: crate::storage::NamespacedStorage,
}

impl FileMemoryStore {
    /// Creates a store rooted at the `knowledge/` namespace of `backend`.
    pub fn new(
        name: impl Into<String>,
        backend: Arc<dyn crate::storage::Storage>,
        namespace: &str,
    ) -> Self {
        Self {
            name: name.into(),
            storage: crate::storage::NamespacedStorage::new(backend, namespace)
                .nested("knowledge"),
        }
    }

    /// Writes a knowledge document.
    pub async fn write_document(&self, key: &str, content: &str) -> MemoryResult<()> {
        self.storage
            .write(key, content.as_bytes())
            .await
            .map_err(|e| StrandsError::StorageError {
                message: format!("failed to write memory document: {}", e),
            })
    }
}

#[async_trait]
impl MemoryStore for FileMemoryStore {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "File-backed knowledge store with keyword search"
    }

    async fn search(&self, query: &str, max_results: usize) -> MemoryResult<Vec<MemoryEntry>> {
        let keys = self
            .storage
            .list("")
            .await
            .map_err(|e| StrandsError::StorageError {
                message: format!("failed to list memory documents: {}", e),
            })?;

        let keywords: Vec<String> = query
            .split_whitespace()
            .map(|word| word.to_lowercase())
            .filter(|word| !word.is_empty())
            .collect();

        let mut scored: BTreeMap<u64, Vec<MemoryEntry>> = BTreeMap::new();
        for key in keys {
            if !key.ends_with(".md") {
                continue;
            }
            let content = match self.storage.read(&key).await {
                Ok(Some(bytes)) => String::from_utf8_lossy(&bytes).to_string(),
                Ok(None) => continue,
                Err(e) => {
                    return Err(StrandsError::StorageError {
                        message: format!("failed to read memory document '{}': {}", key, e),
                    });
                }
            };
            let content_lower = content.to_lowercase();
            let score = keywords
                .iter()
                .filter(|keyword| content_lower.contains(keyword.as_str()))
                .count() as u64;
            if score > 0 {
                let entry =
                    MemoryEntry::new(self.name.clone(), content.clone()).with_metadata(
                        serde_json::json!({ "key": key, "score": score }),
                    );
                scored.entry(score).or_default().push(entry);
            }
        }

        let mut results = Vec::new();
        for (_, mut entries) in scored.into_iter().rev() {
            for entry in entries.drain(..) {
                if results.len() >= max_results {
                    return Ok(results);
                }
                results.push(entry);
            }
        }
        Ok(results)
    }

    fn is_writable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalFileStorage;
    use crate::types::content::Message;

    #[tokio::test]
    async fn test_in_memory_search_and_add() {
        let store = InMemoryMemoryStore::new("notes");
        store.add("likes coffee".to_string(), serde_json::json!({})).await.unwrap();
        store.add("likes tea".to_string(), serde_json::json!({})).await.unwrap();

        let results = store.search("coffee", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "likes coffee");
        assert_eq!(results[0].store_name, "notes");
    }

    #[tokio::test]
    async fn test_in_memory_read_only() {
        let store = InMemoryMemoryStore::new("ro").read_only();
        assert!(store.add("x".to_string(), serde_json::json!({})).await.is_err());
        assert!(!store.is_writable());
    }

    #[tokio::test]
    async fn test_file_memory_store_keyword_search() {
        let root = std::env::temp_dir().join(format!("strands-mem-{}", uuid::Uuid::new_v4()));
        let backend = Arc::new(LocalFileStorage::new(&root).unwrap());
        let store = FileMemoryStore::new("kb", backend, "agent-1");

        store.write_document("coffee.md", "Supe prefers espresso").await.unwrap();
        store.write_document("team.md", "The team meets on Tuesdays").await.unwrap();

        let results = store.search("espresso", 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("espresso"));
        assert_eq!(results[0].metadata["key"], "coffee.md");

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn test_default_add_messages_unsupported() {
        let store = InMemoryMemoryStore::new("notes");
        let messages = vec![Message::user("hello")];
        assert!(!store.supports_add_messages());
        assert!(store.add_messages(&messages).await.is_err());
    }
}
