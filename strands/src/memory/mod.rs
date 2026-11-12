//! Cross-session memory subsystem.
//!
//! Skeletal port of the Python `strands.memory` package: a `MemoryStore`
//! protocol with fan-out search aggregation in `MemoryManager`, plus
//! in-memory and file-backed reference stores.

pub mod manager;
pub mod stores;

pub use manager::MemoryManager;
pub use stores::{FileMemoryStore, InMemoryMemoryStore};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::content::Message;
use crate::types::errors::StrandsError;

/// Result type for memory operations.
pub type MemoryResult<T> = std::result::Result<T, StrandsError>;

/// A single memory record returned from a store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub content: String,
    pub store_name: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

impl MemoryEntry {
    pub fn new(store_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            store_name: store_name.into(),
            metadata: Value::Null,
        }
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// A pluggable memory backend.
///
/// `search` is required; write methods are optional capabilities that stores
/// may reject. Stores supporting server-side extraction implement
/// `add_messages`; stores supporting direct writes implement `add`.
#[async_trait::async_trait]
pub trait MemoryStore: Send + Sync {
    /// Unique store name used in results and tool naming.
    fn name(&self) -> &str;

    /// Human-readable description for tool specs.
    fn description(&self) -> &str {
        "Memory store"
    }

    /// Searches the store, returning at most `max_results` entries.
    async fn search(&self, query: &str, max_results: usize) -> MemoryResult<Vec<MemoryEntry>>;

    /// Writes a single entry. Stores that are read-only return an error.
    async fn add(&self, content: String, metadata: Value) -> MemoryResult<()> {
        let _ = (content, metadata);
        Err(StrandsError::StorageError {
            message: format!("memory store '{}' is not writable", self.name()),
        })
    }

    /// Server-side extraction from messages. Stores without it return an error.
    async fn add_messages(&self, messages: &[Message]) -> MemoryResult<()> {
        let _ = messages;
        Err(StrandsError::StorageError {
            message: format!("memory store '{}' does not support add_messages", self.name()),
        })
    }

    /// Returns true when the store accepts direct writes.
    fn is_writable(&self) -> bool {
        false
    }

    /// Returns true when the store extracts facts from messages server-side.
    fn supports_add_messages(&self) -> bool {
        false
    }
}
