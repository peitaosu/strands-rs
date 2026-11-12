//! MCP router vended tool: agent-callable connections to an allowlisted
//! set of MCP servers.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::tools::mcp::{MCPClient, MCPServerConfig};
use crate::tools::{AgentTool, ToolContext, ToolResult2};
use crate::types::tools::{ToolResultStatus, ToolSpec};

/// Maximum connections when none is configured.
pub const DEFAULT_MAX_CONNECTIONS: usize = 10;

/// Router commands accepted in the tool input.
pub const COMMANDS: [&str; 5] = [
    "connect",
    "list_connections",
    "list_tools",
    "call_tool",
    "disconnect",
];

/// Agent-callable MCP router bound to a developer-set server allowlist.
///
/// The agent, not the developer, decides which allowlisted server to talk
/// to at runtime: it opens named connections, discovers tools, invokes
/// them, and closes connections.
pub struct MCPRouterTool {
    name: String,
    servers: HashMap<String, MCPServerConfig>,
    max_connections: usize,
    connections: tokio::sync::Mutex<HashMap<String, Arc<MCPClient>>>,
}

impl MCPRouterTool {
    /// Creates a router over a developer-set allowlist.
    pub fn new(servers: HashMap<String, MCPServerConfig>) -> Self {
        Self {
            name: "mcp_router".to_string(),
            servers,
            max_connections: DEFAULT_MAX_CONNECTIONS,
            connections: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Overrides the tool name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Caps the number of simultaneous connections.
    pub fn with_max_connections(mut self, max_connections: usize) -> Self {
        self.max_connections = max_connections;
        self
    }

    pub fn allowlisted_servers(&self) -> Vec<&str> {
        self.servers.keys().map(|s| s.as_str()).collect()
    }

    async fn connected_client(&self, server: &str) -> Result<Arc<MCPClient>, String> {
        let connections = self.connections.lock().await;
        connections
            .get(server)
            .cloned()
            .ok_or_else(|| format!("no connection to server '{}'; call connect first", server))
    }

    async fn command_connect(&self, server: &str) -> Result<ToolResult2, String> {
        let config = self
            .servers
            .get(server)
            .ok_or_else(|| format!("server '{}' is not in the allowlist", server))?
            .clone();

        let mut connections = self.connections.lock().await;
        if connections.contains_key(server) {
            return Ok(ToolResult2::success(format!(
                "Already connected to '{}'",
                server
            )));
        }
        if connections.len() >= self.max_connections {
            return Err(format!(
                "connection limit of {} reached; disconnect a server first",
                self.max_connections
            ));
        }

        let client = Arc::new(MCPClient::new(config));
        client
            .connect()
            .await
            .map_err(|e| format!("failed to connect to '{}': {}", server, e))?;

        let tool_count = client.tools().await.len();
        connections.insert(server.to_string(), client);

        Ok(ToolResult2::success(format!(
            "Connected to '{}'; {} tools available",
            server, tool_count
        )))
    }

    async fn command_list_tools(&self, server: &str) -> Result<ToolResult2, String> {
        let client = self.connected_client(server).await?;
        let tools = client.tools().await;
        let names: Vec<&str> = tools.iter().map(|tool| tool.name()).collect();
        Ok(ToolResult2::success_json(serde_json::json!({
            "server": server,
            "tools": names,
        })))
    }

    async fn command_call_tool(
        &self,
        server: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolResult2, String> {
        let client = self.connected_client(server).await?;
        let tool = client
            .tools()
            .await
            .into_iter()
            .find(|tool| tool.name() == tool_name)
            .ok_or_else(|| format!("tool '{}' not found on server '{}'", tool_name, server))?;

        let tool_use = crate::types::tools::ToolUse::new(tool_name, "router-call", arguments);
        let context = ToolContext::new(tool_use);
        let result = tool
            .invoke(serde_json::Value::Null, &context)
            .await
            .map_err(|e| format!("tool '{}' failed: {}", tool_name, e))?;

        let status = result.status;
        let text = result
            .content
            .iter()
            .filter_map(|content| {
                content
                    .text
                    .clone()
                    .or_else(|| content.json.as_ref().map(|json| json.to_string()))
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ToolResult2 {
            status,
            content: vec![crate::types::tools::ToolResultContent::text(text)],
        })
    }

    async fn command_disconnect(&self, server: &str) -> Result<ToolResult2, String> {
        let client = self
            .connections
            .lock()
            .await
            .remove(server)
            .ok_or_else(|| format!("no connection to server '{}'", server))?;

        client
            .disconnect()
            .await
            .map_err(|e| format!("failed to disconnect '{}': {}", server, e))?;

        Ok(ToolResult2::success(format!("Disconnected from '{}'", server)))
    }
}

#[async_trait]
impl AgentTool for MCPRouterTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Connect to MCP servers from the approved list, discover their tools, \
         invoke them, and close connections. Commands: connect, list_connections, \
         list_tools, call_tool, disconnect."
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description()).with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Router command",
                    "enum": COMMANDS,
                },
                "server": {"type": "string", "description": "Allowlisted server name"},
                "tool_name": {"type": "string", "description": "Tool to invoke"},
                "arguments": {"type": "object", "description": "Arguments for call_tool"}
            },
            "required": ["command"]
        }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        let command = input
            .get("command")
            .and_then(|c| c.as_str())
            .ok_or("missing command")?;

        match command {
            "connect" => {
                let server = input.get("server").and_then(|s| s.as_str()).ok_or("missing server")?;
                self.command_connect(server).await
            }
            "list_connections" => {
                let connections = self.connections.lock().await;
                let names: Vec<&String> = connections.keys().collect();
                Ok(ToolResult2::success_json(serde_json::json!({
                    "connections": names,
                })))
            }
            "list_tools" => {
                let server = input.get("server").and_then(|s| s.as_str()).ok_or("missing server")?;
                self.command_list_tools(server).await
            }
            "call_tool" => {
                let server = input.get("server").and_then(|s| s.as_str()).ok_or("missing server")?;
                let tool_name = input
                    .get("tool_name")
                    .and_then(|t| t.as_str())
                    .ok_or("missing tool_name")?;
                let arguments = input.get("arguments").cloned().unwrap_or(serde_json::json!({}));
                self.command_call_tool(server, tool_name, arguments).await
            }
            "disconnect" => {
                let server = input.get("server").and_then(|s| s.as_str()).ok_or("missing server")?;
                self.command_disconnect(server).await
            }
            other => Err(format!(
                "unknown router command '{}'; expected one of {:?}",
                other, COMMANDS
            )),
        }
    }
}

#[allow(dead_code)]
fn status_marker() -> ToolResultStatus {
    ToolResultStatus::Success
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::mcp::{MCPTransport, ToolFilters};

    fn allowlist() -> HashMap<String, MCPServerConfig> {
        HashMap::from([
            (
                "echo-server".to_string(),
                MCPServerConfig {
                    name: "echo-server".to_string(),
                    transport: MCPTransport::StreamableHttp {
                        url: "http://127.0.0.1:1/mcp".to_string(),
                        headers: Default::default(),
                    },
                    timeout_secs: 1,
                },
            ),
            (
                "secondary".to_string(),
                MCPServerConfig {
                    name: "secondary".to_string(),
                    transport: MCPTransport::StreamableHttp {
                        url: "http://127.0.0.1:2/mcp".to_string(),
                        headers: Default::default(),
                    },
                    timeout_secs: 1,
                },
            ),
        ])
    }

    fn context() -> ToolContext {
        ToolContext::new(crate::types::tools::ToolUse::new(
            "mcp_router",
            "t1",
            serde_json::json!({}),
        ))
    }

    #[test]
    fn test_allowlist_only() {
        let router = MCPRouterTool::new(allowlist());
        assert_eq!(router.allowlisted_servers().len(), 2);
        assert!(router.allowlisted_servers().contains(&"echo-server"));
    }

    #[tokio::test]
    async fn test_connect_rejects_unknown_server() {
        let router = MCPRouterTool::new(allowlist());
        let error = router
            .command_connect("rogue")
            .await
            .unwrap_err();
        assert!(error.contains("not in the allowlist"));
    }

    #[tokio::test]
    async fn test_list_tools_requires_connection() {
        let router = MCPRouterTool::new(allowlist());
        let error = router.command_list_tools("echo-server").await.unwrap_err();
        assert!(error.contains("call connect first"));
    }

    #[tokio::test]
    async fn test_disconnect_requires_connection() {
        let router = MCPRouterTool::new(allowlist());
        let error = router.command_disconnect("echo-server").await.unwrap_err();
        assert!(error.contains("no connection"));
    }

    #[tokio::test]
    async fn test_connection_limit() {
        let router = MCPRouterTool::new(allowlist()).with_max_connections(0);
        let error = router.command_connect("echo-server").await.unwrap_err();
        assert!(error.contains("connection limit"));
    }

    #[tokio::test]
    async fn test_unknown_command_rejected() {
        let router = MCPRouterTool::new(allowlist());
        let error = router
            .invoke(serde_json::json!({"command": "reboot"}), &context())
            .await
            .unwrap_err();
        assert!(error.contains("unknown router command"));
    }

    #[allow(dead_code)]
    fn filters_marker() -> ToolFilters {
        ToolFilters::default()
    }
}
