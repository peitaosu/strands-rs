//! Basic agent example.
//!
//! This example demonstrates how to create a simple agent and invoke it.

use strands::models::BedrockModel;
use strands::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    // Create an agent with a Bedrock model
    let mut agent = Agent::builder()
        .model(BedrockModel::new("anthropic.claude-3-sonnet-20240229-v1:0"))
        .system_prompt("You are a helpful assistant. Be concise in your responses.")
        .name("BasicAgent")
        .build()?;

    println!("Agent created: {}", agent.name);
    println!("System prompt: {:?}", agent.system_prompt());
    println!();

    // Invoke the agent
    println!("User: Hello! What can you help me with?");
    let result = agent.invoke_async("Hello! What can you help me with?").await?;

    println!("Assistant: {}", result);
    println!();
    println!("Stop reason: {:?}", result.stop_reason);
    println!(
        "Tokens: {} input, {} output",
        result.usage.input_tokens, result.usage.output_tokens
    );

    Ok(())
}

