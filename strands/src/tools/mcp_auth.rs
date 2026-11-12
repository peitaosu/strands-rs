//! OAuth client-credentials authentication for MCP HTTP transports.
//!
//! Port of the Python `MCPClientCredentials` + `_build_client_credentials_auth`:
//! fetches a bearer token from an OAuth token endpoint via the
//! `client_credentials` grant and attaches it as `Authorization: Bearer`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::types::errors::StrandsError;

/// OAuth client-credentials configuration for MCP servers.
#[derive(Debug, Clone)]
pub struct MCPClientCredentials {
    pub client_id: String,
    pub client_secret: String,
    pub token_url: String,
    pub scopes: Vec<String>,
}

impl MCPClientCredentials {
    pub fn new(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        token_url: impl Into<String>,
    ) -> Self {
        Self {
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            token_url: token_url.into(),
            scopes: Vec::new(),
        }
    }

    pub fn with_scopes(mut self, scopes: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.scopes = scopes.into_iter().map(Into::into).collect();
        self
    }
}

/// Token response from an OAuth token endpoint.
#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

/// Caches a bearer token and refreshes it shortly before expiry.
#[derive(Clone)]
pub struct BearerTokenSource {
    credentials: Arc<MCPClientCredentials>,
    cached: Arc<tokio::sync::RwLock<Option<CachedToken>>>,
    client: reqwest::Client,
}

#[derive(Clone)]
struct CachedToken {
    token: String,
    fetched_at: Instant,
    ttl: Option<Duration>,
}

impl BearerTokenSource {
    pub fn new(credentials: MCPClientCredentials) -> Self {
        Self {
            credentials: Arc::new(credentials),
            cached: Arc::new(tokio::sync::RwLock::new(None)),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("valid HTTP client"),
        }
    }

    pub fn credentials(&self) -> &MCPClientCredentials {
        &self.credentials
    }

    /// Returns a valid bearer token, fetching a new one when missing or stale.
    pub async fn token(&self) -> Result<String, StrandsError> {
        const SKEW: Duration = Duration::from_secs(60);

        let cached = self.cached.read().await.clone();
        if let Some(entry) = cached {
            let lifetime = entry.ttl.map(|ttl| ttl.saturating_sub(SKEW));
            let still_valid = match lifetime {
                Some(lifetime) => entry.fetched_at.elapsed() < lifetime,
                None => entry.fetched_at.elapsed() < Duration::from_secs(300),
            };
            if still_valid {
                return Ok(entry.token);
            }
        }

        self.fetch().await
    }

    async fn fetch(&self) -> Result<String, StrandsError> {
        let mut form = vec![
            ("grant_type", "client_credentials".to_string()),
            ("client_id", self.credentials.client_id.clone()),
            ("client_secret", self.credentials.client_secret.clone()),
        ];
        if !self.credentials.scopes.is_empty() {
            form.push(("scope", self.credentials.scopes.join(" ")));
        }

        let response = self
            .client
            .post(&self.credentials.token_url)
            .form(&form)
            .send()
            .await
            .map_err(|e| StrandsError::NetworkError(format!("token request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(StrandsError::NetworkError(format!(
                "token endpoint returned HTTP {}: {}",
                status, body
            )));
        }

        let token_response: TokenResponse = response
            .json()
            .await
            .map_err(|e| StrandsError::NetworkError(format!("failed to parse token response: {}", e)))?;

        *self.cached.write().await = Some(CachedToken {
            token: token_response.access_token.clone(),
            fetched_at: Instant::now(),
            ttl: token_response.expires_in.map(Duration::from_secs),
        });

        Ok(token_response.access_token)
    }

    /// Builds the `Authorization` headers for an HTTP MCP transport.
    pub async fn auth_headers(&self) -> Result<HashMap<String, String>, StrandsError> {
        let token = self.token().await?;
        Ok(HashMap::from([(
            "Authorization".to_string(),
            format!("Bearer {}", token),
        )]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials_construction() {
        let credentials = MCPClientCredentials::new("id", "secret", "https://auth.example.com/token")
            .with_scopes(["mcp:read", "mcp:write"]);

        assert_eq!(credentials.client_id, "id");
        assert_eq!(credentials.scopes, vec!["mcp:read", "mcp:write"]);
    }

    #[tokio::test]
    async fn test_token_fetch_and_cache() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/token")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"access_token": "tok-1", "expires_in": 3600}"#)
            .create_async()
            .await;

        let source = BearerTokenSource::new(
            MCPClientCredentials::new("id", "secret", format!("{}/token", server.url())),
        );

        let first = source.token().await.unwrap();
        assert_eq!(first, "tok-1");

        let headers = source.auth_headers().await.unwrap();
        assert_eq!(
            headers.get("Authorization").map(|v| v.as_str()),
            Some("Bearer tok-1")
        );
    }

    #[tokio::test]
    async fn test_token_fetch_failure_is_error() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/token")
            .with_status(403)
            .create_async()
            .await;

        let source = BearerTokenSource::new(
            MCPClientCredentials::new("id", "secret", format!("{}/token", server.url())),
        );
        assert!(source.token().await.is_err());
    }
}
