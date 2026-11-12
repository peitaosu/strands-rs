//! Vended tool families bundled with the SDK.
//!
//! Skeletal port of the Python `strands.vended_tools` package: each tool is
//! an `AgentTool` the developer registers on the tool registry.

pub mod a2a_client;
pub mod file_tools;
pub mod http_tools;
pub mod loop_control;
pub mod mcp_router;
pub mod multiagent_tools;
pub mod notebook;
pub mod shell_tools;
pub mod subagent;

pub use a2a_client::A2AClientTool;
pub use file_tools::{EditFileTool, ReadFileTool, WriteFileTool};
pub use mcp_router::MCPRouterTool;
pub use multiagent_tools::MultiAgentTool;
pub use http_tools::{HttpRequestTool, WebFetchTool};
pub use loop_control::{HandoffToUserTool, SleepTool, StopTool};
pub use notebook::NotebookTool;
pub use shell_tools::{PythonReplTool, ShellTool};
pub use subagent::SubagentTool;
