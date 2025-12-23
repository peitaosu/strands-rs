# Strands Agents

A Rust implementation of the [Strands Agents SDK](https://strandsagents.com/) for building AI agents with model-driven orchestration.

[![Crates.io](https://img.shields.io/crates/v/strands-agents.svg)](https://crates.io/crates/strands-agents)
[![Documentation](https://docs.rs/strands-agents/badge.svg)](https://docs.rs/strands-agents)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **Model-Driven Orchestration** — Agents use LLM reasoning to plan, execute tools, and iterate
- **Multi-Provider Support** — AWS Bedrock, OpenAI, Anthropic, Ollama, and more
- **Type-Safe Tools** — Define tools with the `#[tool]` macro
- **Multi-Agent Patterns** — Graph and Swarm orchestration
- **Async-First** — Built on Tokio with streaming support
- **OpenTelemetry Integration** — Built-in tracing and metrics

## Installation

```toml
[dependencies]
strands-agents = "0.1"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

```rust
use strands_agents::prelude::*;
use strands_agents::models::BedrockModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = Agent::builder()
        .model(BedrockModel::new("anthropic.claude-3-sonnet-20240229-v1:0").await?)
        .system_prompt("You are a helpful assistant")
        .build()?;

    let result = agent.invoke_async("Hello!").await?;
    println!("{}", result);
    Ok(())
}
```

### Tools

```rust
use strands_agents::{tool, Agent};
use strands_agents::models::BedrockModel;

#[tool]
/// Get current weather for a location.
async fn get_weather(location: String) -> String {
    format!("Weather in {location}: 72°F, Sunny")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = Agent::builder()
        .model(BedrockModel::default().await?)
        .tool(GetWeatherTool::new())?
        .build()?;

    let result = agent.invoke_async("What's the weather in Seattle?").await?;
    println!("{}", result);
    Ok(())
}
```

## Cargo Features

| Feature | Description |
|---------|-------------|
| `macros` | `#[tool]` procedural macro (default) |
| `s3-session` | S3-based session persistence |
| `otel-stdout` | OpenTelemetry stdout exporter |
| `otel-otlp` | OpenTelemetry OTLP exporter |
| `otel` | All OpenTelemetry exporters |
| `full` | All features |

## Modules

| Module | Description |
|--------|-------------|
| `agent` | Core Agent interface |
| `models` | Model providers (Bedrock, OpenAI, Anthropic, Ollama, etc.) |
| `tools` | Tool definition, execution, and MCP support |
| `multiagent` | Graph and Swarm patterns |
| `hooks` | Lifecycle hooks |
| `session` | Session persistence |
| `conversation` | Context window management |
| `telemetry` | Metrics and tracing |

## Testing

```bash
cargo test
```

## License

Licensed under the [MIT License](LICENSE).

## Credits

Rust port of [Strands Agents SDK](https://github.com/strands-agents/sdk-python) by AWS.
