# Strands Agents

[![Awesome Strands Agents](https://img.shields.io/badge/Awesome-Strands%20Agents-00FF77?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjkwIiBoZWlnaHQ9IjQ2MyIgdmlld0JveD0iMCAwIDI5MCA0NjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik05Ny4yOTAyIDUyLjc4ODRDODUuMDY3NCA0OS4xNjY3IDcyLjIyMzQgNTYuMTM4OSA2OC42MDE3IDY4LjM2MTZDNjQuOTgwMSA4MC41ODQzIDcxLjk1MjQgOTMuNDI4MyA4NC4xNzQ5IDk3LjA1MDFMMjM1LjExNyAxMzkuNzc1QzI0NS4yMjMgMTQyLjc2OSAyNDYuMzU3IDE1Ni42MjggMjM2Ljg3NCAxNjEuMjI2TDMyLjU0NiAyNjAuMjkxQy0xNC45NDM5IDI4My4zMTYgLTkuMTYxMDcgMzUyLjc0IDQxLjQ4MzUgMzY3LjU5MUwxODkuNTUxIDQxMS4wMDlMMTkwLjEyNSA0MTEuMTY5QzIwMi4xODMgNDE0LjM3NiAyMTQuNjY1IDQwNy4zOTYgMjE4LjE5NiAzOTUuMzU1QzIyMS43ODQgMzgzLjEyMiAyMTQuNzc0IDM3MC4yOTYgMjAyLjU0MSAzNjYuNzA5TDU0LjQ3MzggMzIzLjI5MUM0NC4zNDQ3IDMyMC4zMjEgNDMuMTg3OSAzMDYuNDM2IDUyLjY4NTcgMzAxLjgzMUwyNTcuMDE0IDIwMi43NjZDMzA0LjQzMiAxNzkuNzc2IDI5OC43NTggMTEwLjQ4MyAyNDguMjMzIDk1LjUxMkw5Ny4yOTAyIDUyLjc4ODRaIiBmaWxsPSIjRkZGRkZGIi8+CjxwYXRoIGQ9Ik0yNTkuMTQ3IDAuOTgxODEyQzI3MS4zODkgLTIuNTc0OTggMjg0LjE5NyA0LjQ2NTcxIDI4Ny43NTQgMTYuNzA3NEMyOTEuMzExIDI4Ljk0OTIgMjg0LjI3IDQxLjc1NyAyNzIuMDI4IDQ1LjMxMzhMNzEuMTcyNyAxMDMuNjcxQzQwLjcxNDIgMTEyLjUyMSAzNy4xOTc2IDE1NC4yNjIgNjUuNzQ1OSAxNjguMDgzTDI0MS4zNDMgMjUzLjA5M0MzMDcuODcyIDI4NS4zMDIgMjk5Ljc5NCAzODIuNTQ2IDIyOC44NjIgNDAzLjMzNkwzMC40MDQxIDQ2MS41MDJDMTguMTcwNyA0NjUuMDg4IDUuMzQ3MDggNDU4LjA3OCAxLjc2MTUzIDQ0NS44NDRDLTEuODIzOSA0MzMuNjExIDUuMTg2MzcgNDIwLjc4NyAxNy40MTk3IDQxNy4yMDJMMjE1Ljg3OCAzNTkuMDM1QzI0Ni4yNzcgMzUwLjEyNSAyNDkuNzM5IDMwOC40NDkgMjIxLjIyNiAyOTQuNjQ1TDQ1LjYyOTcgMjA5LjYzNUMtMjAuOTgzNCAxNzcuMzg2IC0xMi43NzcyIDc5Ljk4OTMgNTguMjkyOCA1OS4zNDAyTDI1OS4xNDcgMC45ODE4MTJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=&logoColor=white)](https://github.com/cagataycali/awesome-strands-agents)

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
