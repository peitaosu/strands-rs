//! Tracing type definitions for the SDK.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Attribute value for tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    String(String),
    Bool(bool),
    Float(f64),
    Int(i64),
    StringList(Vec<String>),
    BoolList(Vec<bool>),
    FloatList(Vec<f64>),
    IntList(Vec<i64>),
}

impl From<String> for AttributeValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for AttributeValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<bool> for AttributeValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<f64> for AttributeValue {
    fn from(f: f64) -> Self {
        Self::Float(f)
    }
}

impl From<i64> for AttributeValue {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

impl From<i32> for AttributeValue {
    fn from(i: i32) -> Self {
        Self::Int(i as i64)
    }
}

impl From<Vec<String>> for AttributeValue {
    fn from(v: Vec<String>) -> Self {
        Self::StringList(v)
    }
}

impl From<Vec<bool>> for AttributeValue {
    fn from(v: Vec<bool>) -> Self {
        Self::BoolList(v)
    }
}

impl From<Vec<f64>> for AttributeValue {
    fn from(v: Vec<f64>) -> Self {
        Self::FloatList(v)
    }
}

impl From<Vec<i64>> for AttributeValue {
    fn from(v: Vec<i64>) -> Self {
        Self::IntList(v)
    }
}

/// Attributes mapping for tracing.
pub type Attributes = Option<HashMap<String, AttributeValue>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_value_from_string() {
        let attr: AttributeValue = "test".into();
        assert!(matches!(attr, AttributeValue::String(_)));
    }

    #[test]
    fn test_attribute_value_from_bool() {
        let attr: AttributeValue = true.into();
        assert!(matches!(attr, AttributeValue::Bool(true)));
    }

    #[test]
    fn test_attribute_value_from_int() {
        let attr: AttributeValue = 42i64.into();
        assert!(matches!(attr, AttributeValue::Int(42)));
    }

    #[test]
    fn test_attributes_type() {
        let mut attrs: Attributes = Some(HashMap::new());
        if let Some(ref mut map) = attrs {
            map.insert("key".to_string(), "value".into());
        }
        assert!(attrs.is_some());
    }
}

