//! HTTP vended tools: raw requests and web fetching.

use std::time::Duration;

use async_trait::async_trait;

use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::tools::ToolSpec;

/// Maximum response size returned to the model, in bytes.
pub const MAX_RESPONSE_BYTES: usize = 512 * 1024;

fn https_url_spec(name: &str, description: &str, extra_properties: serde_json::Value) -> ToolSpec {
    let mut properties = extra_properties;
    properties["url"] = serde_json::json!({
        "type": "string",
        "description": "HTTPS URL to request"
    });
    ToolSpec::new(name, description).with_input_schema(serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": ["url"]
    }))
}

/// Validates that the URL is a well-formed HTTPS address.
fn require_https(url: &str) -> Result<(), String> {
    if !url.starts_with("https://") {
        return Err(format!("only https:// URLs are allowed, got: {}", url));
    }
    Ok(())
}

/// Raw HTTPS requests with method, headers, and body.
pub struct HttpRequestTool {
    client: reqwest::Client,
}

impl HttpRequestTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("valid HTTP client"),
        }
    }
}

impl Default for HttpRequestTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentTool for HttpRequestTool {
    fn name(&self) -> &str {
        "http_request"
    }

    fn description(&self) -> &str {
        "Perform an HTTPS request and return the raw response body."
    }

    fn tool_spec(&self) -> ToolSpec {
        https_spec(
            self.name(),
            self.description(),
            serde_json::json!({
                "method": {"type": "string", "description": "HTTP method", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]},
                "headers": {"type": "object", "description": "Request headers as key/value pairs"},
                "body": {"type": "string", "description": "Request body"}
            }),
        )
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let url = input.get("url").and_then(|u| u.as_str()).ok_or("missing url")?;
        require_https(url)?;

        let method = input
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("GET")
            .to_uppercase();

        let mut request = self
            .client
            .request(
                reqwest::Method::from_bytes(method.as_bytes())
                    .map_err(|e| format!("invalid method: {}", e))?,
                url,
            )
            .timeout(Duration::from_secs(60));

        if let Some(headers) = input.get("headers").and_then(|h| h.as_object()) {
            for (key, value) in headers {
                let value = value.as_str().unwrap_or_default();
                request = request.header(key, value);
            }
        }

        if let Some(body) = input.get("body").and_then(|b| b.as_str()) {
            request = request.body(body.to_string());
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("request failed: {}", e))?;

        let status = response.status().as_u16();
        let body = response
            .bytes()
            .await
            .map_err(|e| format!("failed to read response: {}", e))?;
        let truncated: &[u8] = if body.len() > MAX_RESPONSE_BYTES {
            &body[..MAX_RESPONSE_BYTES]
        } else {
            &body[..]
        };

        Ok(ToolResult2::success_json(serde_json::json!({
            "status": status,
            "body": String::from_utf8_lossy(truncated),
            "truncated": body.len() > MAX_RESPONSE_BYTES,
        })))
    }
}

/// Fetches a URL and returns cleaned text extracted from the HTML.
pub struct WebFetchTool {
    client: reqwest::Client,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("valid HTTP client"),
        }
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Strips HTML tags, script/style bodies, and collapses whitespace.
pub fn extract_text(html: &str) -> String {
    let without_scripts = strip_tag_blocks(html, &["script", "style"]);
    let without_tags = strip_tags(&without_scripts);
    let decoded = decode_entities(&without_tags);
    decoded
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Strips the full bodies of the given tags (and their close tags) from HTML.
fn strip_tag_blocks(html: &str, tags: &[&str]) -> String {
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len());
    let mut active_tag: Option<&str> = None;
    let mut rest = html;
    let mut lower_rest = lower.as_str();

    while !rest.is_empty() {
        if let Some(tag) = active_tag {
            let close = format!("</{}", tag);
            let skip = match lower_rest.find(&close) {
                Some(pos) => {
                    let after_close = pos + close.len();
                    match lower_rest[after_close..].find('>') {
                        Some(rel_gt) => after_close + rel_gt + 1,
                        None => lower_rest.len(),
                    }
                }
                None => lower_rest.len(),
            };
            rest = &rest[skip..];
            lower_rest = &lower_rest[skip..];
            active_tag = None;
        } else {
            match lower_rest.find('<') {
                Some(lt) => {
                    out.push_str(&rest[..lt]);
                    let mut opening: Option<&str> = None;
                    for tag in tags {
                        let open = format!("<{}", tag);
                        if lower_rest.len() - lt >= open.len()
                            && lower_rest[lt..].starts_with(&open)
                        {
                            let next = lower_rest[lt + open.len()..].chars().next();
                            if matches!(next, Some('>') | Some(' ') | Some('/')) {
                                opening = Some(tag);
                                break;
                            }
                        }
                    }
                    match opening {
                        Some(tag) => {
                            active_tag = Some(tag);
                            rest = &rest[lt + 1..];
                            lower_rest = &lower_rest[lt + 1..];
                        }
                        None => {
                            out.push('<');
                            rest = &rest[lt + 1..];
                            lower_rest = &lower_rest[lt + 1..];
                        }
                    }
                }
                None => {
                    out.push_str(rest);
                    break;
                }
            }
        }
    }

    out
}

fn strip_tags(html: &str) -> String {
    const BLOCK_TAGS: [&str; 8] = ["p", "div", "br", "h1", "h2", "h3", "h4", "li"];

    let lower = html.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut result = String::with_capacity(html.len());
    let mut inside_tag = false;
    let mut tag_start = 0usize;
    let mut copy_from = 0usize;

    for (index, byte) in bytes.iter().enumerate() {
        match byte {
            b'<' if !inside_tag => {
                result.push_str(&html[copy_from..index]);
                inside_tag = true;
                tag_start = index;
            }
            b'>' if inside_tag => {
                let tag = &lower[tag_start + 1..index];
                let name: String = tag
                    .trim_start_matches('/')
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric())
                    .collect();
                if BLOCK_TAGS.contains(&name.as_str()) {
                    result.push('\n');
                }
                inside_tag = false;
                copy_from = index + 1;
            }
            _ => {}
        }
    }
    result.push_str(&html[copy_from.min(html.len())..]);

    result
}

fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

#[async_trait]
impl AgentTool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch an HTTPS URL and return cleaned text extracted from the response."
    }

    fn tool_spec(&self) -> ToolSpec {
        https_spec(
            self.name(),
            self.description(),
            serde_json::json!({}),
        )
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let url = input.get("url").and_then(|u| u.as_str()).ok_or("missing url")?;
        require_https(url)?;

        let response = self
            .client
            .get(url)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| format!("fetch failed: {}", e))?;

        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        let body = response
            .text()
            .await
            .map_err(|e| format!("failed to read response: {}", e))?;

        let text = if content_type.contains("html") {
            extract_text(&body)
        } else {
            body
        };
        let text: String = text.chars().take(MAX_RESPONSE_BYTES).collect();

        Ok(ToolResult2::success_json(serde_json::json!({
            "status": status,
            "content_type": content_type,
            "text": text,
        })))
    }
}

fn https_spec(name: &str, description: &str, properties: serde_json::Value) -> ToolSpec {
    https_url_spec(name, description, properties)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tools::ToolUse;

    fn context() -> ToolContext {
        ToolContext::new(ToolUse::new("t", "t1", serde_json::json!({})))
    }

    #[test]
    fn test_require_https() {
        assert!(require_https("https://example.com").is_ok());
        assert!(require_https("http://example.com").is_err());
        assert!(require_https("ftp://example.com").is_err());
    }

    #[test]
    fn test_extract_text_strips_scripts_and_tags() {
        let html = "<html><head><script>evil()</script><style>.x{}</style></head>\
                    <body><h1>Title</h1><p>Hello &amp; welcome</p></body></html>";
        assert_eq!(extract_text(html), "Title\nHello & welcome");
    }

    #[tokio::test]
    async fn test_http_request_rejects_http_url() {
        let tool = HttpRequestTool::new();
        let error = tool
            .invoke(serde_json::json!({"url": "http://example.com"}), &context())
            .await
            .unwrap_err();
        assert!(error.contains("https"));
    }

    #[tokio::test]
    async fn test_web_fetch_rejects_http_url() {
        let tool = WebFetchTool::new();
        let error = tool
            .invoke(serde_json::json!({"url": "http://example.com"}), &context())
            .await
            .unwrap_err();
        assert!(error.contains("https"));
    }
}
