//! Wire-level SSE streaming tests against a local mock server.

use futures::StreamExt;

use crate::models::{AnthropicModel, Model, OpenAIModel, OpenAIResponsesModel};
use crate::types::content::Message;
use crate::types::errors::StrandsError;
use crate::types::streaming::{StopReason, StreamEvent};

async fn collect(model: &dyn Model) -> Vec<Result<StreamEvent, StrandsError>> {
    let messages = vec![Message::user("hi")];
    let mut stream = model.stream(&messages, None, None, None, None);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

fn ok_events(events: &[Result<StreamEvent, StrandsError>]) -> Vec<StreamEvent> {
    events.iter().map(|e| e.as_ref().unwrap().clone()).collect()
}

fn find_stop_reason(events: &[StreamEvent]) -> Option<StopReason> {
    events
        .iter()
        .find_map(|event| event.stop_reason())
}

fn concatenated_text(events: &[StreamEvent]) -> String {
    events
        .iter()
        .filter_map(|event| event.as_text_delta())
        .collect()
}

fn sse_body(lines: &[&str]) -> String {
    lines
        .iter()
        .map(|line| format!("data: {}\n\n", line))
        .collect()
}

mod anthropic_tests {
    use super::*;

    fn model(server_url: &str) -> AnthropicModel {
        AnthropicModel::new("test-key", 1024).with_base_url(server_url)
    }

    #[tokio::test]
    async fn test_text_stream_with_stop_reason_and_usage() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body(&[
                r#"{"type":"message_start"}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello "}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}"#,
                r#"{"type":"message_delta","delta":{"type":"text_delta","stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":5}}"#,
                r#"{"type":"message_stop"}"#,
            ]))
            .create_async()
            .await;

        let events = ok_events(&collect(&model(&server.url())).await);

        assert_eq!(concatenated_text(&events), "Hello world");
        assert_eq!(find_stop_reason(&events), Some(StopReason::EndTurn));

        let usage = events
            .iter()
            .find_map(|event| event.metadata.as_ref().and_then(|m| m.usage.clone()))
            .expect("usage metadata emitted");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[tokio::test]
    async fn test_max_tokens_stop_reason_from_message_delta() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body(&[
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#,
                r#"{"type":"message_delta","delta":{"type":"text_delta","stop_reason":"max_tokens"}}"#,
                r#"{"type":"message_stop"}"#,
            ]))
            .create_async()
            .await;

        let events = ok_events(&collect(&model(&server.url())).await);

        assert_eq!(find_stop_reason(&events), Some(StopReason::MaxTokens));
    }

    #[tokio::test]
    async fn test_tool_use_stream() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body(&[
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool-1","name":"calc"}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"1}"}}"#,
                r#"{"type":"content_block_stop","index":0}"#,
                r#"{"type":"message_delta","delta":{"type":"text_delta","stop_reason":"tool_use"}}"#,
                r#"{"type":"message_stop"}"#,
            ]))
            .create_async()
            .await;

        let events = ok_events(&collect(&model(&server.url())).await);

        assert_eq!(find_stop_reason(&events), Some(StopReason::ToolUse));

        let tool_use = events
            .iter()
            .find_map(|event| event.content_block_start.as_ref().and_then(|start| start.start.as_ref()).and_then(|start| start.tool_use.clone()))
            .expect("tool use start");
        assert_eq!(tool_use.name, "calc");
        assert_eq!(tool_use.tool_use_id, "tool-1");

        let json_input = events
            .iter()
            .filter_map(|event| event.content_block_delta.as_ref().and_then(|d| d.delta.as_ref()))
            .filter_map(|delta| delta.tool_use.as_ref())
            .map(|tool_use| tool_use.input.clone())
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(json_input, r#"{"a":1}"#);
    }

    #[tokio::test]
    async fn test_thinking_deltas_stream() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body(&[
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thinking hard"}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}"#,
                r#"{"type":"message_stop"}"#,
            ]))
            .create_async()
            .await;

        let events = ok_events(&collect(&model(&server.url())).await);

        let reasoning = events
            .iter()
            .find_map(|event| event.content_block_delta.as_ref().and_then(|d| d.delta.as_ref()).and_then(|d| d.reasoning_content.clone()))
            .expect("reasoning delta");
        assert_eq!(reasoning.text.as_deref(), Some("thinking hard"));
    }

    #[tokio::test]
    async fn test_error_status_maps_to_throttling() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/")
            .with_status(429)
            .with_body("rate limited")
            .create_async()
            .await;

        let events = collect(&model(&server.url())).await;
        assert!(matches!(events[0], Err(StrandsError::ModelThrottled { .. })));
    }
}

mod openai_tests {
    use super::*;

    fn model(server_url: &str) -> OpenAIModel {
        OpenAIModel::new("test-key").with_base_url(format!("{}/v1/chat/completions", server_url))
    }

    #[tokio::test]
    async fn test_text_stream_with_usage() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body(&[
                r#"{"choices":[{"delta":{"content":"Hey"}}]}"#,
                r#"{"choices":[{"delta":{"content":" there"}}]}"#,
                r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3}}"#,
            ]))
            .create_async()
            .await;

        let events = ok_events(&collect(&model(&server.url())).await);

        assert_eq!(concatenated_text(&events), "Hey there");
        assert_eq!(find_stop_reason(&events), Some(StopReason::EndTurn));
    }

    #[tokio::test]
    async fn test_reasoning_content_deltas() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body(&[
                r#"{"choices":[{"delta":{"reasoning_content":"pondering"}}]}"#,
                r#"{"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}"#,
                r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            ]))
            .create_async()
            .await;

        let events = ok_events(&collect(&model(&server.url())).await);

        let reasoning = events
            .iter()
            .find_map(|event| event.content_block_delta.as_ref().and_then(|d| d.delta.as_ref()).and_then(|d| d.reasoning_content.clone()))
            .expect("reasoning delta");
        assert_eq!(reasoning.text.as_deref(), Some("pondering"));
        assert_eq!(concatenated_text(&events), "answer");
    }

    #[tokio::test]
    async fn test_tool_call_accumulation() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body(&[
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"calc"}}]}}]}"#,
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":"}}]}}]}"#,
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}"#,
                r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            ]))
            .create_async()
            .await;

        let events = ok_events(&collect(&model(&server.url())).await);

        assert_eq!(find_stop_reason(&events), Some(StopReason::ToolUse));

        let tool_start = events
            .iter()
            .find_map(|event| event.content_block_start.as_ref().and_then(|start| start.start.as_ref()).and_then(|start| start.tool_use.clone()))
            .expect("tool start");
        assert_eq!(tool_start.name, "calc");
        assert_eq!(tool_start.tool_use_id, "call-1");
    }
}

mod openai_responses_tests {
    use super::*;

    fn model(server_url: &str) -> OpenAIResponsesModel {
        OpenAIResponsesModel::new("test-key").with_base_url(format!("{}/v1/responses", server_url))
    }

    #[tokio::test]
    async fn test_full_response_event_stream() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/responses")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body(&[
                r#"{"type":"response.created","response":{"id":"resp_9"}}"#,
                r#"{"type":"response.reasoning_summary_text.delta","delta":"thinking out loud"}"#,
                r#"{"type":"response.output_text.delta","delta":"final answer"}"#,
                r#"{"type":"response.output_item.added","item":{"type":"function_call","call_id":"call-7","name":"lookup","id":"item-1"}}"#,
                r#"{"type":"response.function_call_arguments.delta","item_id":"item-1","delta":"{\"q\":"}"#,
                r#"{"type":"response.function_call_arguments.done","item_id":"item-1","arguments":"{\"q\":\"x\"}"}"#,
                r#"{"type":"response.completed","response":{"usage":{"input_tokens":11,"output_tokens":4,"input_tokens_details":{"cached_tokens":3}}}}"#,
            ]))
            .create_async()
            .await;

        let responses_model = model(&server.url());
        let events = ok_events(&collect(&responses_model).await);

        assert_eq!(concatenated_text(&events), "final answer");
        assert_eq!(find_stop_reason(&events), Some(StopReason::ToolUse));

        let tool_start = events
            .iter()
            .find_map(|event| event.content_block_start.as_ref().and_then(|start| start.start.as_ref()).and_then(|start| start.tool_use.clone()))
            .expect("tool call emitted");
        assert_eq!(tool_start.name, "lookup");
        assert_eq!(tool_start.tool_use_id, "call-7");

        let usage = events
            .iter()
            .find_map(|event| event.metadata.as_ref().and_then(|m| m.usage.clone()))
            .expect("usage metadata");
        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 4);
        assert_eq!(usage.cache_read_input_tokens, 3);
    }
}

mod ollama_tests {
    use super::*;
    use crate::models::OllamaModel;

    fn model(server_url: &str) -> OllamaModel {
        OllamaModel::new("llama3").with_host(server_url)
    }

    #[tokio::test]
    async fn test_stream_with_usage_mapping() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/api/chat")
            .with_status(200)
            .with_header("content-type", "application/x-ndjson")
            .with_body(
                "{\"message\":{\"role\":\"assistant\",\"content\":\"Hello\"}}\n\
                 {\"done\":true,\"message\":{\"role\":\"assistant\",\"content\":\"!\"},\"prompt_eval_count\":9,\"eval_count\":4}\n",
            )
            .create_async()
            .await;

        let events = ok_events(&collect(&model(&server.url())).await);

        assert_eq!(concatenated_text(&events), "Hello!");
        assert_eq!(find_stop_reason(&events), Some(StopReason::EndTurn));

        let usage = events
            .iter()
            .find_map(|event| event.metadata.as_ref().and_then(|m| m.usage.clone()))
            .expect("usage from eval counts");
        assert_eq!(usage.input_tokens, 9);
        assert_eq!(usage.output_tokens, 4);
    }

    #[tokio::test]
    async fn test_context_overflow_maps_error() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/api/chat")
            .with_status(400)
            .with_body("input length exceeds context length")
            .create_async()
            .await;

        let events = collect(&model(&server.url())).await;
        assert!(matches!(events[0], Err(StrandsError::ContextWindowOverflow { .. })));
    }
}
