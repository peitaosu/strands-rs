//! Agent state management.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Persistent state for an agent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentState {
    data: HashMap<String, serde_json::Value>,
}

impl AgentState {
    pub fn new() -> Self { Self { data: HashMap::new() } }

    pub fn from_map(data: HashMap<String, serde_json::Value>) -> Self { Self { data } }

    /// Gets a value by key, deserializing to the requested type.
    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.data.get(key).and_then(|v| T::deserialize(v).ok())
    }

    /// Gets the raw JSON value by key.
    pub fn get_raw(&self, key: &str) -> Option<&serde_json::Value> { self.data.get(key) }

    /// Sets a value by key.
    pub fn set<T: Serialize>(&mut self, key: impl Into<String>, value: T) {
        if let Ok(v) = serde_json::to_value(value) {
            self.data.insert(key.into(), v);
        }
    }

    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> { self.data.remove(key) }
    pub fn contains(&self, key: &str) -> bool { self.data.contains_key(key) }
    pub fn keys(&self) -> impl Iterator<Item = &str> { self.data.keys().map(|s| s.as_str()) }
    pub fn clear(&mut self) { self.data.clear(); }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn into_inner(self) -> HashMap<String, serde_json::Value> { self.data }

    /// Returns a clone of all state data.
    pub fn get_all(&self) -> HashMap<String, serde_json::Value> { self.data.clone() }

    /// Merges another state into this one.
    pub fn merge(&mut self, other: AgentState) { self.data.extend(other.data); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_basic() {
        let mut state = AgentState::new();
        state.set("count", 42);
        state.set("name", "test");
        assert_eq!(state.get::<i32>("count"), Some(42));
        assert_eq!(state.get::<String>("name"), Some("test".to_string()));
        assert_eq!(state.get::<i32>("missing"), None);
    }

    #[test]
    fn test_state_remove() {
        let mut state = AgentState::new();
        state.set("key", "value");
        assert!(state.contains("key"));
        state.remove("key");
        assert!(!state.contains("key"));
    }

    #[test]
    fn test_state_merge() {
        let mut state1 = AgentState::new();
        state1.set("a", 1);
        let mut state2 = AgentState::new();
        state2.set("b", 2);
        state1.merge(state2);
        assert_eq!(state1.get::<i32>("a"), Some(1));
        assert_eq!(state1.get::<i32>("b"), Some(2));
    }
}
