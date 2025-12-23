//! Collection type definitions.

use serde::{Deserialize, Serialize};

/// A paginated list of items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedList<T> {
    pub items: Vec<T>,
    pub total: usize,
    pub offset: usize,
    pub limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

impl<T> PaginatedList<T> {
    pub fn new(items: Vec<T>, total: usize, offset: usize, limit: Option<usize>) -> Self {
        Self {
            items,
            total,
            offset,
            limit,
            next_token: None,
        }
    }

    pub fn with_next_token(mut self, token: impl Into<String>) -> Self {
        self.next_token = Some(token.into());
        self
    }

    pub fn has_more(&self) -> bool {
        self.offset + self.items.len() < self.total
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

impl<T> Default for PaginatedList<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            total: 0,
            offset: 0,
            limit: None,
            next_token: None,
        }
    }
}

impl<T> IntoIterator for PaginatedList<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a PaginatedList<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

