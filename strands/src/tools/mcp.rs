//! Model Context Protocol (MCP) tool provider.
//!
//! This module provides integration with MCP servers for accessing
//! remote tools through the Model Context Protocol.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Command, Child};
use tokio::sync::RwLock;
use tokio::time::timeout;

use crate::types::errors::{Result, StrandsError};
use crate::types::tools::ToolSpec;
use crate::types::{ToolResultContent, ToolResultStatus};

use super::{AgentTool, ToolContext, ToolResult2};

/// Configuration for an MCP server connection.
#[derive(Debug, Clone)]
pub struct MCPServerConfig {
    /// Name identifier for this server.
    pub name: String,
    /// Transport configuration.
    pub transport: MCPTransport,
    /// Connection timeout in seconds.
    pub timeout_secs: u64,
}

/// Transport configuration for MCP connections.
#[derive(Debug, Clone)]
pub enum MCPTransport {
    /// Standard I/O transport (subprocess).
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    },
    /// SSE transport (HTTP).
    Sse {
        url: String,
        headers: HashMap<String, String>,
    },
}

/// Filters for controlling which MCP tools are loaded.
#[derive(Debug, Clone, Default)]
pub struct ToolFilters {
    /// Tools matching these patterns are included.
    pub allowed: Vec<String>,
    /// Tools matching these patterns are excluded.
    pub rejected: Vec<String>,
}

impl ToolFilters {
    /// Checks if a tool should be included based on filters.
    pub fn should_include(&self, tool_name: &str) -> bool {
        if !self.allowed.is_empty() && !self.allowed.iter().any(|p| p == tool_name) {
            return false;
        }
        if self.rejected.iter().any(|p| p == tool_name) {
            return false;
        }
        true
    }
}

/// Tool specification from MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolSpec {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: Option<String>,
    /// Input schema.
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
    /// Output schema (optional).
    #[serde(rename = "outputSchema")]
    pub output_schema: Option<serde_json::Value>,
}

/// Result of an MCP tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolResult {
    /// Status of the call.
    pub status: String,
    /// Tool use ID.
    #[serde(rename = "toolUseId")]
    pub tool_use_id: String,
    /// Result content.
    pub content: Vec<MCPResultContent>,
    /// Optional structured content.
    #[serde(rename = "structuredContent")]
    pub structured_content: Option<serde_json::Value>,
    /// Optional metadata.
    pub metadata: Option<serde_json::Value>,
}

/// Content from MCP tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MCPResultContent {
    /// Text content.
    Text { text: String },
    /// Image content.
    Image { image: MCPImageContent },
}

/// Image content from MCP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPImageContent {
    /// Image format.
    pub format: String,
    /// Image source.
    pub source: MCPImageSource,
}

/// Image source from MCP.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MCPImageSource {
    /// Base64 encoded bytes.
    Bytes { bytes: Vec<u8> },
    /// URL reference.
    Url { url: String },
}

/// Trait for tool providers with lifecycle management.
#[async_trait]
pub trait ToolProvider: Send + Sync {
    /// Loads and returns the tools in this provider.
    async fn load_tools(&self) -> Result<Vec<Arc<dyn AgentTool>>>;

    /// Adds a consumer to this tool provider.
    fn add_consumer(&self, consumer_id: &str);

    /// Removes a consumer from this tool provider.
    fn remove_consumer(&self, consumer_id: &str);
}

/// State of the MCP connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Not connected.
    Disconnected,
    /// Currently connecting.
    Connecting,
    /// Connected and ready.
    Connected,
    /// Connection failed.
    Failed,
}

/// Shared stdio handles for MCP stdio transport.
#[derive(Clone)]
pub(crate) struct StdioHandles {
    stdin: Arc<tokio::sync::Mutex<tokio::process::ChildStdin>>,
    stdout: Arc<tokio::sync::Mutex<BufReader<tokio::process::ChildStdout>>>,
    timeout_secs: u64,
}

/// An MCP client that provides tools from an MCP server.
pub struct MCPClient {
    config: MCPServerConfig,
    tools: RwLock<HashMap<String, Arc<MCPAgentTool>>>,
    state: RwLock<ConnectionState>,
    consumers: RwLock<std::collections::HashSet<String>>,
    filters: Option<ToolFilters>,
    prefix: Option<String>,
    stdio_process: RwLock<Option<Child>>,
    stdio_handles: RwLock<Option<StdioHandles>>,
}

impl MCPClient {
    /// Creates a new MCP client with the given configuration.
    pub fn new(config: MCPServerConfig) -> Self {
        Self {
            config,
            tools: RwLock::new(HashMap::new()),
            state: RwLock::new(ConnectionState::Disconnected),
            consumers: RwLock::new(std::collections::HashSet::new()),
            filters: None,
            prefix: None,
            stdio_process: RwLock::new(None),
            stdio_handles: RwLock::new(None),
        }
    }

    /// Creates an MCP client with stdio transport.
    pub fn stdio(name: impl Into<String>, command: impl Into<String>, args: Vec<String>) -> Self {
        Self::new(MCPServerConfig {
            name: name.into(),
            transport: MCPTransport::Stdio {
                command: command.into(),
                args,
                env: HashMap::new(),
            },
            timeout_secs: 30,
        })
    }

    /// Creates an MCP client with SSE transport.
    pub fn sse(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self::new(MCPServerConfig {
            name: name.into(),
            transport: MCPTransport::Sse {
                url: url.into(),
                headers: HashMap::new(),
            },
            timeout_secs: 30,
        })
    }

    /// Sets tool filters for this client.
    pub fn with_filters(mut self, filters: ToolFilters) -> Self {
        self.filters = Some(filters);
        self
    }

    /// Sets a prefix for tool names.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Sets the connection timeout.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.config.timeout_secs = timeout_secs;
        self
    }

    /// Returns the server name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Checks if the client is connected.
    pub async fn is_connected(&self) -> bool {
        *self.state.read().await == ConnectionState::Connected
    }

    /// Returns the current connection state.
    pub async fn connection_state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Connects to the MCP server and discovers available tools.
    pub async fn connect(&self) -> Result<()> {
        {
            let mut state = self.state.write().await;
            if *state == ConnectionState::Connected {
                return Ok(());
            }
            *state = ConnectionState::Connecting;
        }

        let result = self.do_connect().await;

        {
            let mut state = self.state.write().await;
            *state = if result.is_ok() {
                ConnectionState::Connected
            } else {
                ConnectionState::Failed
            };
        }

        result
    }

    async fn do_connect(&self) -> Result<()> {
        match &self.config.transport {
            MCPTransport::Stdio { command, args, env } => {
                self.connect_stdio(command, args, env).await
            }
            MCPTransport::Sse { url, headers } => {
                self.connect_sse(url, headers).await
            }
        }
    }

    async fn connect_stdio(
        &self,
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<()> {
        use tracing::debug;

        debug!(
            command = %command,
            args = ?args,
            "Starting MCP stdio transport"
        );

        let mut cmd = Command::new(command);
        cmd.args(args);
        cmd.envs(env);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            StrandsError::MCPClientInitializationError {
                message: format!("Failed to spawn MCP server process: {}", e),
            }
        })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            StrandsError::MCPClientInitializationError {
                message: "Failed to acquire stdin handle".to_string(),
            }
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            StrandsError::MCPClientInitializationError {
                message: "Failed to acquire stdout handle".to_string(),
            }
        })?;

        let stdin_handle = Arc::new(tokio::sync::Mutex::new(stdin));
        let stdout_handle = Arc::new(tokio::sync::Mutex::new(BufReader::new(stdout)));

        {
            let mut process = self.stdio_process.write().await;
            *process = Some(child);
        }

        {
            let mut handles = self.stdio_handles.write().await;
            *handles = Some(StdioHandles {
                stdin: stdin_handle.clone(),
                stdout: stdout_handle.clone(),
                timeout_secs: self.config.timeout_secs,
            });
        }

        let mut line_buf = String::new();

        let init_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "strands-rs",
                    "version": "0.1.0"
                }
            }
        });

        let init_json = serde_json::to_string(&init_request).map_err(|e| {
            StrandsError::MCPClientInitializationError {
                message: format!("Failed to serialize init request: {}", e),
            }
        })?;

        {
            let mut stdin_guard = stdin_handle.lock().await;
            stdin_guard
                .write_all(format!("{}\n", init_json).as_bytes())
                .await
                .map_err(|e| StrandsError::MCPClientInitializationError {
                    message: format!("Failed to write init request: {}", e),
                })?;

            stdin_guard.flush().await.map_err(|e| {
                StrandsError::MCPClientInitializationError {
                    message: format!("Failed to flush stdin: {}", e),
                }
            })?;
        }

        let read_result = {
            let mut stdout_guard = stdout_handle.lock().await;
            timeout(
                Duration::from_secs(self.config.timeout_secs),
                stdout_guard.read_line(&mut line_buf),
            )
            .await
        };

        match read_result {
            Ok(Ok(0)) | Err(_) => {
                return Err(StrandsError::MCPClientInitializationError {
                    message: "Timeout or EOF while waiting for initialize response".to_string(),
                });
            }
            Ok(Ok(_)) => {
                let init_response: serde_json::Value = serde_json::from_str(&line_buf)
                    .map_err(|e| StrandsError::MCPClientInitializationError {
                        message: format!("Failed to parse init response: {}", e),
                    })?;

                debug!(response = ?init_response, "Received initialize response");
            }
            Ok(Err(e)) => {
                return Err(StrandsError::MCPClientInitializationError {
                    message: format!("Failed to read init response: {}", e),
                });
            }
        }

        let initialized_notification = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });

        let initialized_json = serde_json::to_string(&initialized_notification).map_err(|e| {
            StrandsError::MCPClientInitializationError {
                message: format!("Failed to serialize initialized notification: {}", e),
            }
        })?;

        {
            let mut stdin_guard = stdin_handle.lock().await;
            stdin_guard
                .write_all(format!("{}\n", initialized_json).as_bytes())
                .await
                .map_err(|e| StrandsError::MCPClientInitializationError {
                    message: format!("Failed to write initialized notification: {}", e),
                })?;

            stdin_guard.flush().await.map_err(|e| {
                StrandsError::MCPClientInitializationError {
                    message: format!("Failed to flush stdin: {}", e),
                }
            })?;
        }

        let tools_list_request = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        });

        let tools_list_json = serde_json::to_string(&tools_list_request).map_err(|e| {
            StrandsError::MCPClientInitializationError {
                message: format!("Failed to serialize tools/list request: {}", e),
            }
        })?;

        line_buf.clear();
        {
            let mut stdin_guard = stdin_handle.lock().await;
            stdin_guard
                .write_all(format!("{}\n", tools_list_json).as_bytes())
                .await
                .map_err(|e| StrandsError::MCPClientInitializationError {
                    message: format!("Failed to write tools/list request: {}", e),
                })?;

            stdin_guard.flush().await.map_err(|e| {
                StrandsError::MCPClientInitializationError {
                    message: format!("Failed to flush stdin: {}", e),
                }
            })?;
        }

        let read_result = {
            let mut stdout_guard = stdout_handle.lock().await;
            timeout(
                Duration::from_secs(self.config.timeout_secs),
                stdout_guard.read_line(&mut line_buf),
            )
            .await
        };

        match read_result {
            Ok(Ok(0)) | Err(_) => {
                return Err(StrandsError::MCPClientInitializationError {
                    message: "Timeout or EOF while waiting for tools/list response".to_string(),
                });
            }
            Ok(Ok(_)) => {
                let tools_response: serde_json::Value = serde_json::from_str(&line_buf)
                    .map_err(|e| StrandsError::MCPClientInitializationError {
                        message: format!("Failed to parse tools/list response: {}", e),
                    })?;

                debug!(response = ?tools_response, "Received tools/list response");

                if let Some(result) = tools_response.get("result") {
                    if let Some(tools) = result.get("tools").and_then(|t| t.as_array()) {
                        let mut tools_map = self.tools.write().await;
                        for tool_value in tools {
                            if let Ok(tool_spec) = serde_json::from_value::<MCPToolSpec>(tool_value.clone()) {
                                let tool_name = if let Some(prefix) = &self.prefix {
                                    format!("{}_{}", prefix, tool_spec.name)
                                } else {
                                    tool_spec.name.clone()
                                };

                                if let Some(ref filters) = self.filters {
                                    if !filters.should_include(&tool_spec.name) {
                                        continue;
                                    }
                                }

                                let handles = self.stdio_handles.read().await.clone();
                                let mcp_tool = Arc::new(MCPAgentTool::new_stdio(
                                    tool_spec.clone(),
                                    handles,
                                    self.prefix.clone(),
                                ));

                                tools_map.insert(tool_name, mcp_tool);
                            }
                        }
                    }
                }
            }
            Ok(Err(e)) => {
                return Err(StrandsError::MCPClientInitializationError {
                    message: format!("Failed to read tools/list response: {}", e),
                });
            }
        }

        let tool_count = self.tools.read().await.len();
        debug!(
            tool_count = tool_count,
            "MCP stdio transport connected and tools loaded"
        );

        Ok(())
    }

    async fn connect_sse(&self, url: &str, headers: &HashMap<String, String>) -> Result<()> {
        use reqwest::Client;

        let client = Client::builder()
            .timeout(Duration::from_secs(self.config.timeout_secs))
            .build()
            .map_err(|e| StrandsError::NetworkError(e.to_string()))?;

        let mut request = client.get(format!("{}/tools/list", url.trim_end_matches('/')));

        for (key, value) in headers {
            request = request.header(key, value);
        }

        let response = request
            .send()
            .await
            .map_err(|e| StrandsError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(StrandsError::NetworkError(format!(
                "MCP server returned status: {}",
                response.status()
            )));
        }

        #[derive(Deserialize)]
        struct ListToolsResponse {
            tools: Vec<MCPToolSpec>,
        }

        let list_response: ListToolsResponse = response
            .json()
            .await
            .map_err(|e| StrandsError::NetworkError(format!("Failed to parse response: {e}")))?;

        let mut tools = self.tools.write().await;
        tools.clear();

        for mcp_spec in list_response.tools {
            let tool_name = if let Some(ref prefix) = self.prefix {
                format!("{}_{}", prefix, mcp_spec.name)
            } else {
                mcp_spec.name.clone()
            };

            if let Some(ref filters) = self.filters {
                if !filters.should_include(&tool_name) {
                    continue;
                }
            }

            let agent_tool = MCPAgentTool::new(
                mcp_spec,
                url.to_string(),
                headers.clone(),
                self.config.timeout_secs,
                self.prefix.clone(),
            );

            tools.insert(tool_name, Arc::new(agent_tool));
        }

        Ok(())
    }

    /// Disconnects from the MCP server.
    pub async fn disconnect(&self) -> Result<()> {
        let mut state = self.state.write().await;
        *state = ConnectionState::Disconnected;

        let mut tools = self.tools.write().await;
        tools.clear();

        Ok(())
    }

    /// Returns the list of available tools from the MCP server.
    pub async fn tools(&self) -> Vec<Arc<dyn AgentTool>> {
        let tools = self.tools.read().await;
        tools.values().map(|t| t.clone() as Arc<dyn AgentTool>).collect()
    }

    /// Calls a tool on the MCP server.
    pub async fn call_tool(
        &self,
        tool_use_id: &str,
        name: &str,
        arguments: &serde_json::Value,
    ) -> Result<MCPToolResult> {
        if !self.is_connected().await {
            return Err(StrandsError::ConfigurationError {
                message: "MCP client is not connected".to_string(),
            });
        }

        let tools = self.tools.read().await;
        let tool = tools.get(name).ok_or_else(|| StrandsError::ToolNotFound {
            tool_name: name.to_string(),
        })?;

        tool.call_mcp(tool_use_id, arguments).await
    }
}

#[async_trait]
impl ToolProvider for MCPClient {
    async fn load_tools(&self) -> Result<Vec<Arc<dyn AgentTool>>> {
        if !self.is_connected().await {
            self.connect().await?;
        }
        Ok(self.tools().await)
    }

    fn add_consumer(&self, consumer_id: &str) {
        if let Ok(mut consumers) = self.consumers.try_write() {
            consumers.insert(consumer_id.to_string());
        }
    }

    fn remove_consumer(&self, consumer_id: &str) {
        if let Ok(mut consumers) = self.consumers.try_write() {
            consumers.remove(consumer_id);
        }
    }
}

/// An MCP tool wrapped as an AgentTool.
pub struct MCPAgentTool {
    mcp_spec: MCPToolSpec,
    server_url: String,
    headers: HashMap<String, String>,
    timeout_secs: u64,
    name_override: Option<String>,
    stdio_handles: Option<StdioHandles>,
}

impl MCPAgentTool {
    /// Creates a new MCP agent tool for SSE transport.
    pub fn new(
        mcp_spec: MCPToolSpec,
        server_url: String,
        headers: HashMap<String, String>,
        timeout_secs: u64,
        prefix: Option<String>,
    ) -> Self {
        let name_override = prefix.map(|p| format!("{}_{}", p, mcp_spec.name));
        Self {
            mcp_spec,
            server_url,
            headers,
            timeout_secs,
            name_override,
            stdio_handles: None,
        }
    }

    /// Creates a new MCP agent tool for stdio transport.
    pub(crate) fn new_stdio(
        mcp_spec: MCPToolSpec,
        stdio_handles: Option<StdioHandles>,
        prefix: Option<String>,
    ) -> Self {
        let name_override = prefix.map(|p| format!("{}_{}", p, mcp_spec.name));
        let timeout_secs = stdio_handles.as_ref().map(|h| h.timeout_secs).unwrap_or(30);
        Self {
            mcp_spec,
            server_url: String::new(),
            headers: HashMap::new(),
            timeout_secs,
            name_override,
            stdio_handles,
        }
    }

    /// Calls the MCP server to execute this tool.
    pub async fn call_mcp(
        &self,
        tool_use_id: &str,
        arguments: &serde_json::Value,
    ) -> Result<MCPToolResult> {
        if let Some(ref handles) = self.stdio_handles {
            return self.call_mcp_stdio(tool_use_id, arguments, handles).await;
        }

        self.call_mcp_sse(tool_use_id, arguments).await
    }

    /// Calls the MCP server via stdio transport.
    async fn call_mcp_stdio(
        &self,
        tool_use_id: &str,
        arguments: &serde_json::Value,
        handles: &StdioHandles,
    ) -> Result<MCPToolResult> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static REQUEST_ID: AtomicU64 = AtomicU64::new(1000);

        let request_id = REQUEST_ID.fetch_add(1, Ordering::SeqCst);

        let call_request = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": self.mcp_spec.name,
                "arguments": arguments
            }
        });

        let request_json = serde_json::to_string(&call_request).map_err(|e| {
            StrandsError::ToolProviderError {
                message: format!("Failed to serialize tool call request: {}", e),
            }
        })?;

        {
            let mut stdin_guard = handles.stdin.lock().await;
            stdin_guard
                .write_all(format!("{}\n", request_json).as_bytes())
                .await
                .map_err(|e| StrandsError::ToolProviderError {
                    message: format!("Failed to write tool call request: {}", e),
                })?;

            stdin_guard.flush().await.map_err(|e| {
                StrandsError::ToolProviderError {
                    message: format!("Failed to flush stdin: {}", e),
                }
            })?;
        }

        let mut line_buf = String::new();
        let read_result = {
            let mut stdout_guard = handles.stdout.lock().await;
            timeout(
                Duration::from_secs(handles.timeout_secs),
                stdout_guard.read_line(&mut line_buf),
            )
            .await
        };

        match read_result {
            Ok(Ok(0)) | Err(_) => {
                return Ok(MCPToolResult {
                    status: "error".to_string(),
                    tool_use_id: tool_use_id.to_string(),
                    content: vec![MCPResultContent::Text {
                        text: "Timeout or EOF while waiting for tool call response".to_string(),
                    }],
                    structured_content: None,
                    metadata: None,
                });
            }
            Ok(Ok(_)) => {
                let response: serde_json::Value = serde_json::from_str(&line_buf).map_err(|e| {
                    StrandsError::ToolProviderError {
                        message: format!("Failed to parse tool call response: {}", e),
                    }
                })?;

                if let Some(error) = response.get("error") {
                    return Ok(MCPToolResult {
                        status: "error".to_string(),
                        tool_use_id: tool_use_id.to_string(),
                        content: vec![MCPResultContent::Text {
                            text: format!("MCP error: {}", error),
                        }],
                        structured_content: None,
                        metadata: None,
                    });
                }

                if let Some(result) = response.get("result") {
                    #[derive(Deserialize)]
                    struct CallToolResult {
                        content: Vec<MCPResultContent>,
                        #[serde(rename = "isError")]
                        is_error: Option<bool>,
                        #[serde(rename = "structuredContent")]
                        structured_content: Option<serde_json::Value>,
                        #[serde(rename = "meta")]
                        metadata: Option<serde_json::Value>,
                    }

                    if let Ok(call_result) = serde_json::from_value::<CallToolResult>(result.clone()) {
                        return Ok(MCPToolResult {
                            status: if call_result.is_error.unwrap_or(false) {
                                "error"
                            } else {
                                "success"
                            }
                            .to_string(),
                            tool_use_id: tool_use_id.to_string(),
                            content: call_result.content,
                            structured_content: call_result.structured_content,
                            metadata: call_result.metadata,
                        });
                    }
                }

                Ok(MCPToolResult {
                    status: "error".to_string(),
                    tool_use_id: tool_use_id.to_string(),
                    content: vec![MCPResultContent::Text {
                        text: "Invalid response format from MCP server".to_string(),
                    }],
                    structured_content: None,
                    metadata: None,
                })
            }
            Ok(Err(e)) => {
                return Err(StrandsError::ToolProviderError {
                    message: format!("Failed to read tool call response: {}", e),
                });
            }
        }
    }

    /// Calls the MCP server via SSE transport.
    async fn call_mcp_sse(
        &self,
        tool_use_id: &str,
        arguments: &serde_json::Value,
    ) -> Result<MCPToolResult> {
        use reqwest::Client;

        let client = Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| StrandsError::NetworkError(e.to_string()))?;

        #[derive(Serialize)]
        struct CallToolRequest<'a> {
            name: &'a str,
            arguments: &'a serde_json::Value,
        }

        let request_body = CallToolRequest {
            name: &self.mcp_spec.name,
            arguments,
        };

        let mut request = client
            .post(format!("{}/tools/call", self.server_url.trim_end_matches('/')))
            .json(&request_body);

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        let response = request
            .send()
            .await
            .map_err(|e| StrandsError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Ok(MCPToolResult {
                status: "error".to_string(),
                tool_use_id: tool_use_id.to_string(),
                content: vec![MCPResultContent::Text {
                    text: format!("MCP server returned status: {}", response.status()),
                }],
                structured_content: None,
                metadata: None,
            });
        }

        #[derive(Deserialize)]
        struct CallToolResponse {
            content: Vec<MCPResultContent>,
            #[serde(rename = "isError")]
            is_error: Option<bool>,
            #[serde(rename = "structuredContent")]
            structured_content: Option<serde_json::Value>,
            #[serde(rename = "meta")]
            metadata: Option<serde_json::Value>,
        }

        let call_response: CallToolResponse = response
            .json()
            .await
            .map_err(|e| StrandsError::NetworkError(format!("Failed to parse response: {e}")))?;

        Ok(MCPToolResult {
            status: if call_response.is_error.unwrap_or(false) {
                "error"
            } else {
                "success"
            }
            .to_string(),
            tool_use_id: tool_use_id.to_string(),
            content: call_response.content,
            structured_content: call_response.structured_content,
            metadata: call_response.metadata,
        })
    }

}

#[async_trait]
impl AgentTool for MCPAgentTool {
    fn name(&self) -> &str {
        self.name_override.as_deref().unwrap_or(&self.mcp_spec.name)
    }

    fn description(&self) -> &str {
        self.mcp_spec.description.as_deref().unwrap_or("MCP tool")
    }

    fn tool_spec(&self) -> ToolSpec {
        let description = self
            .mcp_spec
            .description
            .clone()
            .unwrap_or_else(|| format!("Tool which performs {}", self.mcp_spec.name));

        let mut spec = ToolSpec::new(self.name(), &description)
            .with_input_schema(self.mcp_spec.input_schema.clone());

        if let Some(ref output_schema) = self.mcp_spec.output_schema {
            spec = spec.with_output_schema(output_schema.clone());
        }

        spec
    }

    fn tool_type(&self) -> &str {
        "mcp"
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &ToolContext,
    ) -> std::result::Result<ToolResult2, String> {
        use reqwest::Client;

        let client = Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| e.to_string())?;

        #[derive(Serialize)]
        struct CallToolRequest<'a> {
            name: &'a str,
            arguments: &'a serde_json::Value,
        }

        let request_body = CallToolRequest {
            name: &self.mcp_spec.name,
            arguments: &input,
        };

        let mut request = client
            .post(format!("{}/tools/call", self.server_url.trim_end_matches('/')))
            .json(&request_body);

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        let response = request.send().await.map_err(|e| e.to_string())?;

        if !response.status().is_success() {
            return Err(format!("MCP server returned status: {}", response.status()));
        }

        #[derive(Deserialize)]
        struct CallToolResponse {
            content: Vec<MCPResultContent>,
            #[serde(rename = "isError")]
            is_error: Option<bool>,
        }

        let call_response: CallToolResponse = response.json().await.map_err(|e| e.to_string())?;

        let content: Vec<ToolResultContent> = call_response
            .content
            .into_iter()
            .map(|c| match c {
                MCPResultContent::Text { text } => ToolResultContent::text(text),
                MCPResultContent::Image { image } => ToolResultContent::json(serde_json::json!({
                    "type": "image",
                    "format": image.format,
                })),
            })
            .collect();

        let status = if call_response.is_error.unwrap_or(false) {
            ToolResultStatus::Error
        } else {
            ToolResultStatus::Success
        };

        Ok(ToolResult2 { status, content })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_client_creation() {
        let client = MCPClient::stdio("test", "echo", vec!["hello".to_string()]);
        assert_eq!(client.name(), "test");
    }

    #[test]
    fn test_mcp_sse_client() {
        let client = MCPClient::sse("test", "http://localhost:8080");
        match client.config.transport {
            MCPTransport::Sse { url, .. } => assert_eq!(url, "http://localhost:8080"),
            _ => panic!("expected SSE transport"),
        }
    }

    #[test]
    fn test_tool_filters() {
        let filters = ToolFilters {
            allowed: vec!["tool_a".to_string(), "tool_b".to_string()],
            rejected: vec!["tool_b".to_string()],
        };

        assert!(filters.should_include("tool_a"));
        assert!(!filters.should_include("tool_b"));
        assert!(!filters.should_include("tool_c"));
    }

    #[test]
    fn test_mcp_client_with_options() {
        let client = MCPClient::sse("test", "http://localhost:8080")
            .with_prefix("my_prefix")
            .with_timeout(60)
            .with_filters(ToolFilters::default());

        assert_eq!(client.config.timeout_secs, 60);
        assert_eq!(client.prefix, Some("my_prefix".to_string()));
    }
}
