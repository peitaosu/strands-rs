//! Unified byte-blob storage protocol.
//!
//! Mirrors the Python `strands.storage` subsystem: a small key/value blob
//! API (`read`/`write`/`delete`/`list`) with namespace-scoped views, backed
//! by pluggable stores. `SnapshotSessionManager` and memory stores build on
//! this protocol.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

/// A storage backend for byte blobs addressed by string keys.
#[async_trait]
pub trait Storage: Send + Sync {
    /// Reads the blob stored at `key`, if present.
    async fn read(&self, key: &str) -> std::io::Result<Option<Vec<u8>>>;

    /// Writes the blob at `key`.
    async fn write(&self, key: &str, data: &[u8]) -> std::io::Result<()>;

    /// Deletes the blob at `key`, if present.
    async fn delete(&self, key: &str) -> std::io::Result<()>;

    /// Lists keys sharing `prefix`, in lexical order.
    async fn list(&self, prefix: &str) -> std::io::Result<Vec<String>>;
}

/// A storage view rooted at a namespace prefix.
#[derive(Clone)]
pub struct NamespacedStorage {
    backend: Arc<dyn Storage>,
    namespace: String,
}

impl NamespacedStorage {
    /// Creates a namespaced view over `backend`.
    pub fn new(backend: Arc<dyn Storage>, namespace: &str) -> Self {
        let namespace = namespace.trim_matches('/').to_string();
        Self { backend, namespace }
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}/{}", self.namespace, key.trim_start_matches('/'))
    }

    /// Returns a further-nested view.
    pub fn nested(&self, namespace: &str) -> Self {
        Self::new(
            self.backend.clone(),
            &format!("{}/{}", self.namespace, namespace.trim_matches('/')),
        )
    }

    pub async fn read(&self, key: &str) -> std::io::Result<Option<Vec<u8>>> {
        self.backend.read(&self.full_key(key)).await
    }

    pub async fn write(&self, key: &str, data: &[u8]) -> std::io::Result<()> {
        self.backend.write(&self.full_key(key), data).await
    }

    pub async fn delete(&self, key: &str) -> std::io::Result<()> {
        self.backend.delete(&self.full_key(key)).await
    }

    pub async fn list(&self, prefix: &str) -> std::io::Result<Vec<String>> {
        let full_prefix = self.full_key(prefix);
        let keys = self.backend.list(&full_prefix).await?;
        let strip = self.namespace.len() + 1;
        Ok(keys.into_iter().map(|k| k[strip.min(k.len())..].to_string()).collect())
    }
}

/// In-memory storage backend.
#[derive(Clone, Default)]
pub struct InMemoryStorage {
    blobs: Arc<tokio::sync::RwLock<BTreeMap<String, Vec<u8>>>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Storage for InMemoryStorage {
    async fn read(&self, key: &str) -> std::io::Result<Option<Vec<u8>>> {
        Ok(self.blobs.read().await.get(key).cloned())
    }

    async fn write(&self, key: &str, data: &[u8]) -> std::io::Result<()> {
        self.blobs
            .write()
            .await
            .insert(key.to_string(), data.to_vec());
        Ok(())
    }

    async fn delete(&self, key: &str) -> std::io::Result<()> {
        self.blobs.write().await.remove(key);
        Ok(())
    }

    async fn list(&self, prefix: &str) -> std::io::Result<Vec<String>> {
        let blobs = self.blobs.read().await;
        Ok(blobs
            .range(prefix.to_string()..)
            .take_while(|(key, _)| key.starts_with(prefix))
            .map(|(key, _)| key.clone())
            .collect())
    }
}

/// Local filesystem storage backend.
#[derive(Clone, Debug)]
pub struct LocalFileStorage {
    root: PathBuf,
}

impl LocalFileStorage {
    /// Creates storage rooted at `root`, creating the directory if needed.
    pub fn new(root: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let canonical = root.canonicalize()?;
        Ok(Self { root: canonical })
    }

    fn resolve(&self, key: &str) -> std::io::Result<PathBuf> {
        let relative = key.trim_start_matches('/');
        if relative.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "empty storage key",
            ));
        }
        let path = self.root.join(relative);
        let parent = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.root.clone());
        let canonical_parent = parent.canonicalize().unwrap_or(parent);
        if !canonical_parent.starts_with(&self.root) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("key escapes storage root: {}", key),
            ));
        }
        Ok(path)
    }
}

#[async_trait]
impl Storage for LocalFileStorage {
    async fn read(&self, key: &str) -> std::io::Result<Option<Vec<u8>>> {
        let path = self.resolve(key)?;
        match tokio::fs::read(&path).await {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn write(&self, key: &str, data: &[u8]) -> std::io::Result<()> {
        let path = self.resolve(key)?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, data).await
    }

    async fn delete(&self, key: &str) -> std::io::Result<()> {
        let path = self.resolve(key)?;
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    async fn list(&self, prefix: &str) -> std::io::Result<Vec<String>> {
        let prefix_path = self.resolve(prefix)?;
        let mut keys = Vec::new();
        if prefix_path.is_file() {
            keys.push(prefix.trim_start_matches('/').to_string());
            return Ok(keys);
        }
        let mut stack = vec![prefix_path.clone()];
        while let Some(dir) = stack.pop() {
            let mut entries = match tokio::fs::read_dir(&dir).await {
                Ok(entries) => entries,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            };
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(relative) = path.strip_prefix(&self.root) {
                    keys.push(relative.to_string_lossy().to_string());
                }
            }
        }
        keys.sort();
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_roundtrip() {
        let storage = InMemoryStorage::new();
        storage.write("a/b.txt", b"hello").await.unwrap();
        assert_eq!(storage.read("a/b.txt").await.unwrap().unwrap(), b"hello");
        assert_eq!(storage.list("a/").await.unwrap(), vec!["a/b.txt".to_string()]);
        storage.delete("a/b.txt").await.unwrap();
        assert!(storage.read("a/b.txt").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_namespaced_view() {
        let storage = InMemoryStorage::new();
        let ns = NamespacedStorage::new(Arc::new(storage.clone()), "sessions");
        ns.write("abc/state.json", b"{}").await.unwrap();

        assert_eq!(
            storage.read("sessions/abc/state.json").await.unwrap().unwrap(),
            b"{}"
        );
        assert_eq!(ns.list("abc/").await.unwrap(), vec!["abc/state.json".to_string()]);
    }

    #[tokio::test]
    async fn test_local_file_roundtrip() {
        let root = std::env::temp_dir().join(format!("strands-storage-test-{}", uuid::Uuid::new_v4()));
        let storage = LocalFileStorage::new(&root).unwrap();

        storage.write("ns/key.bin", b"data").await.unwrap();
        assert_eq!(storage.read("ns/key.bin").await.unwrap().unwrap(), b"data");
        assert_eq!(storage.list("ns/").await.unwrap(), vec!["ns/key.bin".to_string()]);

        storage.delete("ns/key.bin").await.unwrap();
        assert!(storage.read("ns/key.bin").await.unwrap().is_none());

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn test_local_file_rejects_escape() {
        let root = std::env::temp_dir().join(format!("strands-storage-esc-{}", uuid::Uuid::new_v4()));
        let storage = LocalFileStorage::new(&root).unwrap();
        assert!(storage.resolve("../outside").is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
