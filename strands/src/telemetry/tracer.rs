//! OpenTelemetry integration.
//!
//! This module provides tracing capabilities using OpenTelemetry,
//! enabling trace data to be sent to OTLP endpoints.

use std::collections::HashMap;
use std::sync::OnceLock;

use chrono::Utc;
use opentelemetry::trace::{SpanKind, Status, TraceContextExt, Tracer as OtelTracer};
use opentelemetry::{global, Context, KeyValue};
use serde::Serialize;

use crate::agent::AgentResult;
use crate::types::content::{ContentBlock, Message, Messages};
use crate::types::streaming::{Metrics, StopReason, Usage};
use crate::types::tools::{ToolResult, ToolUse};

/// Type alias for trace attribute values.
pub type AttributeValue = opentelemetry::Value;

/// Type alias for trace attributes.
pub type Attributes = HashMap<String, AttributeValue>;

/// Custom JSON encoder that handles non-serializable types.
fn serialize_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let processed: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), serialize_value(v)))
                .collect();
            serde_json::Value::Object(processed)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(serialize_value).collect())
        }
        other => other.clone(),
    }
}

/// Serialize an object to JSON with consistent settings.
pub fn serialize<T: Serialize + ?Sized>(obj: &T) -> String {
    match serde_json::to_value(obj) {
        Ok(value) => {
            let processed = serialize_value(&value);
            serde_json::to_string(&processed).unwrap_or_else(|_| "<serialization_error>".to_string())
        }
        Err(_) => "<serialization_error>".to_string(),
    }
}

/// Handles OpenTelemetry tracing.
///
/// This struct provides a simple interface for creating and managing traces,
/// with support for sending to OTLP endpoints.
///
/// When the OTEL_EXPORTER_OTLP_ENDPOINT environment variable is set, traces
/// are sent to the OTLP endpoint.
///
/// Both attributes are controlled by including "gen_ai_latest_experimental" or "gen_ai_tool_definitions",
/// respectively, in the OTEL_SEMCONV_STABILITY_OPT_IN environment variable.
pub struct Tracer {
    service_name: String,
    use_latest_genai_conventions: bool,
    include_tool_definitions: bool,
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracer {
    /// Initialize the tracer.
    pub fn new() -> Self {
        let opt_in_values = Self::parse_semconv_opt_in();

        Self {
            service_name: "strands-agents".to_string(),
            use_latest_genai_conventions: opt_in_values.contains("gen_ai_latest_experimental"),
            include_tool_definitions: opt_in_values.contains("gen_ai_tool_definitions"),
        }
    }

    /// Parse the OTEL_SEMCONV_STABILITY_OPT_IN environment variable.
    fn parse_semconv_opt_in() -> std::collections::HashSet<String> {
        std::env::var("OTEL_SEMCONV_STABILITY_OPT_IN")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Get common attributes for a span.
    fn get_common_attributes(&self, operation_name: &str) -> Vec<KeyValue> {
        let mut attrs = vec![KeyValue::new("gen_ai.operation.name", operation_name.to_string())];

        if self.use_latest_genai_conventions {
            attrs.push(KeyValue::new("gen_ai.provider.name", "strands-agents"));
        } else {
            attrs.push(KeyValue::new("gen_ai.system", "strands-agents"));
        }

        attrs
    }

    /// Get the service name for this tracer.
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Start a new span with common attributes.
    fn start_span(
        &self,
        span_name: &str,
        parent_context: Option<&Context>,
        attributes: Vec<KeyValue>,
        span_kind: SpanKind,
    ) -> Context {



        let tracer = global::tracer(self.service_name.clone());
        
        let mut builder = tracer.span_builder(span_name.to_string());
        builder = builder.with_kind(span_kind);
        builder = builder.with_attributes(attributes);

        let context = parent_context.cloned().unwrap_or_else(Context::current);
        let span = builder.start_with_context(&tracer, &context);
        let new_context = context.with_span(span);


        let span_ref = new_context.span();
        span_ref.set_attribute(KeyValue::new(
            "gen_ai.event.start_time",
            Utc::now().to_rfc3339(),
        ));

        new_context
    }

    /// End a span with optional attributes and error.
    fn end_span_internal(
        &self,
        context: &Context,
        attributes: Option<Vec<KeyValue>>,
        error: Option<&dyn std::error::Error>,
    ) {
        let span = context.span();


        span.set_attribute(KeyValue::new(
            "gen_ai.event.end_time",
            Utc::now().to_rfc3339(),
        ));


        if let Some(attrs) = attributes {
            for attr in attrs {
                span.set_attribute(attr);
            }
        }


        if let Some(err) = error {
            span.set_status(Status::error(err.to_string()));
            span.record_error(err);
        } else {
            span.set_status(Status::Ok);
        }

        span.end();





    }

    /// Add an event to a span.
    fn add_event(&self, context: &Context, event_name: &str, attributes: Vec<KeyValue>) {
        let span = context.span();
        span.add_event(event_name.to_string(), attributes);
    }

    /// Get the event name for a message.
    fn get_event_name_for_message(&self, message: &Message) -> String {

        for content in &message.content {
            if content.tool_result.is_some() {
                return "gen_ai.tool.message".to_string();
            }
        }

        format!("gen_ai.{}.message", message.role.as_str())
    }

    /// Add optional usage and metrics attributes.
    fn add_optional_usage_and_metrics_attributes(
        &self,
        attributes: &mut Vec<KeyValue>,
        usage: &Usage,
        metrics: &Metrics,
    ) {
        if usage.cache_read_input_tokens > 0 {
            attributes.push(KeyValue::new(
                "gen_ai.usage.cache_read_input_tokens",
                usage.cache_read_input_tokens as i64,
            ));
        }

        if usage.cache_write_input_tokens > 0 {
            attributes.push(KeyValue::new(
                "gen_ai.usage.cache_write_input_tokens",
                usage.cache_write_input_tokens as i64,
            ));
        }

        if metrics.time_to_first_byte_ms > 0 {
            attributes.push(KeyValue::new(
                "gen_ai.server.time_to_first_token",
                metrics.time_to_first_byte_ms as i64,
            ));
        }

        if metrics.latency_ms > 0 {
            attributes.push(KeyValue::new(
                "gen_ai.server.request.duration",
                metrics.latency_ms as i64,
            ));
        }
    }

    /// Start a new span for a model invocation.
    pub fn start_model_invoke_span(
        &self,
        messages: &Messages,
        parent_context: Option<&Context>,
        model_id: Option<&str>,
        custom_trace_attributes: Option<&HashMap<String, String>>,
    ) -> Context {
        let mut attributes = self.get_common_attributes("chat");

        if let Some(attrs) = custom_trace_attributes {
            for (k, v) in attrs {
                attributes.push(KeyValue::new(k.clone(), v.clone()));
            }
        }

        if let Some(id) = model_id {
            attributes.push(KeyValue::new("gen_ai.request.model", id.to_string()));
        }

        let context = self.start_span("chat", parent_context, attributes, SpanKind::Internal);
        self.add_event_messages(&context, messages);

        context
    }

    /// End a model invocation span with results and metrics.
    pub fn end_model_invoke_span(
        &self,
        context: &Context,
        message: &Message,
        usage: &Usage,
        metrics: &Metrics,
        stop_reason: &StopReason,
        error: Option<&dyn std::error::Error>,
    ) {
        let mut attributes = vec![
            KeyValue::new("gen_ai.usage.prompt_tokens", usage.input_tokens as i64),
            KeyValue::new("gen_ai.usage.input_tokens", usage.input_tokens as i64),
            KeyValue::new("gen_ai.usage.completion_tokens", usage.output_tokens as i64),
            KeyValue::new("gen_ai.usage.output_tokens", usage.output_tokens as i64),
            KeyValue::new("gen_ai.usage.total_tokens", usage.total_tokens as i64),
        ];

        self.add_optional_usage_and_metrics_attributes(&mut attributes, usage, metrics);

        if self.use_latest_genai_conventions {
            let output_message = serde_json::json!([{
                "role": message.role.as_str(),
                "parts": self.map_content_blocks_to_otel_parts(&message.content),
                "finish_reason": stop_reason.as_str(),
            }]);
            self.add_event(
                context,
                "gen_ai.client.inference.operation.details",
                vec![KeyValue::new("gen_ai.output.messages", serialize(&output_message))],
            );
        } else {
            self.add_event(
                context,
                "gen_ai.choice",
                vec![
                    KeyValue::new("finish_reason", stop_reason.as_str().to_string()),
                    KeyValue::new("message", serialize(&message.content)),
                ],
            );
        }

        self.end_span_internal(context, Some(attributes), error);
    }

    /// Start a new span for a tool call.
    pub fn start_tool_call_span(
        &self,
        tool: &ToolUse,
        parent_context: Option<&Context>,
        custom_trace_attributes: Option<&HashMap<String, String>>,
    ) -> Context {
        let mut attributes = self.get_common_attributes("execute_tool");
        attributes.push(KeyValue::new("gen_ai.tool.name", tool.name.clone()));
        attributes.push(KeyValue::new("gen_ai.tool.call.id", tool.tool_use_id.clone()));

        if let Some(attrs) = custom_trace_attributes {
            for (k, v) in attrs {
                attributes.push(KeyValue::new(k.clone(), v.clone()));
            }
        }

        let span_name = format!("execute_tool {}", tool.name);
        let context = self.start_span(&span_name, parent_context, attributes, SpanKind::Internal);

        if self.use_latest_genai_conventions {
            let input_message = serde_json::json!([{
                "role": "tool",
                "parts": [{
                    "type": "tool_call",
                    "name": tool.name,
                    "id": tool.tool_use_id,
                    "arguments": tool.input,
                }],
            }]);
            self.add_event(
                &context,
                "gen_ai.client.inference.operation.details",
                vec![KeyValue::new("gen_ai.input.messages", serialize(&input_message))],
            );
        } else {
            self.add_event(
                &context,
                "gen_ai.tool.message",
                vec![
                    KeyValue::new("role", "tool"),
                    KeyValue::new("content", serialize(&tool.input)),
                    KeyValue::new("id", tool.tool_use_id.clone()),
                ],
            );
        }

        context
    }

    /// End a tool call span with results.
    pub fn end_tool_call_span(
        &self,
        context: &Context,
        tool_result: Option<&ToolResult>,
        error: Option<&dyn std::error::Error>,
    ) {
        let mut attributes = Vec::new();

        if let Some(result) = tool_result {
            let status_str = result.status.as_str();
            attributes.push(KeyValue::new("gen_ai.tool.status", status_str.to_string()));

            if self.use_latest_genai_conventions {
                let output_message = serde_json::json!([{
                    "role": "tool",
                    "parts": [{
                        "type": "tool_call_response",
                        "id": result.tool_use_id,
                        "response": result.content,
                    }],
                }]);
                self.add_event(
                    context,
                    "gen_ai.client.inference.operation.details",
                    vec![KeyValue::new("gen_ai.output.messages", serialize(&output_message))],
                );
            } else {
                self.add_event(
                    context,
                    "gen_ai.choice",
                    vec![
                        KeyValue::new("message", serialize(&result.content)),
                        KeyValue::new("id", result.tool_use_id.clone()),
                    ],
                );
            }
        }

        self.end_span_internal(context, Some(attributes), error);
    }

    /// Start a new span for an event loop cycle.
    pub fn start_event_loop_cycle_span(
        &self,
        event_loop_cycle_id: &str,
        messages: &Messages,
        parent_context: Option<&Context>,
        custom_trace_attributes: Option<&HashMap<String, String>>,
    ) -> Context {
        let mut attributes = vec![
            KeyValue::new("event_loop.cycle_id", event_loop_cycle_id.to_string()),
        ];

        if let Some(attrs) = custom_trace_attributes {
            for (k, v) in attrs {
                attributes.push(KeyValue::new(k.clone(), v.clone()));
            }
        }

        let context = self.start_span(
            "execute_event_loop_cycle",
            parent_context,
            attributes,
            SpanKind::Internal,
        );
        self.add_event_messages(&context, messages);

        context
    }

    /// End an event loop cycle span with results.
    pub fn end_event_loop_cycle_span(
        &self,
        context: &Context,
        message: &Message,
        tool_result_message: Option<&Message>,
        error: Option<&dyn std::error::Error>,
    ) {
        let attributes: Vec<KeyValue> = Vec::new();

        if let Some(tool_msg) = tool_result_message {
            if self.use_latest_genai_conventions {
                let output_message = serde_json::json!([{
                    "role": tool_msg.role.as_str(),
                    "parts": self.map_content_blocks_to_otel_parts(&tool_msg.content),
                }]);
                self.add_event(
                    context,
                    "gen_ai.client.inference.operation.details",
                    vec![KeyValue::new("gen_ai.output.messages", serialize(&output_message))],
                );
            } else {
                self.add_event(
                    context,
                    "gen_ai.choice",
                    vec![
                        KeyValue::new("message", serialize(&message.content)),
                        KeyValue::new("tool.result", serialize(&tool_msg.content)),
                    ],
                );
            }
        }

        self.end_span_internal(context, Some(attributes), error);
    }

    /// Start a new span for an agent invocation.
    pub fn start_agent_span(
        &self,
        messages: &Messages,
        agent_name: &str,
        model_id: Option<&str>,
        tools: Option<&[String]>,
        custom_trace_attributes: Option<&HashMap<String, String>>,
        tools_config: Option<&HashMap<String, serde_json::Value>>,
    ) -> Context {
        let mut attributes = self.get_common_attributes("invoke_agent");
        attributes.push(KeyValue::new("gen_ai.agent.name", agent_name.to_string()));

        if let Some(id) = model_id {
            attributes.push(KeyValue::new("gen_ai.request.model", id.to_string()));
        }

        if let Some(t) = tools {
            attributes.push(KeyValue::new("gen_ai.agent.tools", serialize(t)));
        }

        if self.include_tool_definitions {
            if let Some(config) = tools_config {
                let tool_definitions: Vec<serde_json::Value> = config
                    .iter()
                    .map(|(name, spec)| {
                        serde_json::json!({
                            "name": name,
                            "description": spec.get("description"),
                            "inputSchema": spec.get("inputSchema"),
                            "outputSchema": spec.get("outputSchema"),
                        })
                    })
                    .collect();
                attributes.push(KeyValue::new("gen_ai.tool.definitions", serialize(&tool_definitions)));
            }
        }

        if let Some(attrs) = custom_trace_attributes {
            for (k, v) in attrs {
                attributes.push(KeyValue::new(k.clone(), v.clone()));
            }
        }

        let span_name = format!("invoke_agent {}", agent_name);
        let context = self.start_span(&span_name, None, attributes, SpanKind::Internal);
        self.add_event_messages(&context, messages);

        context
    }

    /// End an agent span with results and metrics.
    pub fn end_agent_span(
        &self,
        context: &Context,
        response: Option<&AgentResult>,
        error: Option<&dyn std::error::Error>,
    ) {
        let mut attributes: Vec<KeyValue> = Vec::new();

        if let Some(resp) = response {
            if self.use_latest_genai_conventions {
                let output_message = serde_json::json!([{
                    "role": "assistant",
                    "parts": [{"type": "text", "content": resp.to_string()}],
                    "finish_reason": resp.stop_reason.as_str(),
                }]);
                self.add_event(
                    context,
                    "gen_ai.client.inference.operation.details",
                    vec![KeyValue::new("gen_ai.output.messages", serialize(&output_message))],
                );
            } else {
                self.add_event(
                    context,
                    "gen_ai.choice",
                    vec![
                        KeyValue::new("message", resp.to_string()),
                        KeyValue::new("finish_reason", resp.stop_reason.as_str().to_string()),
                    ],
                );
            }


            let usage = resp.usage().cloned().unwrap_or_default();
            attributes.extend(vec![
                KeyValue::new("gen_ai.usage.prompt_tokens", usage.input_tokens as i64),
                KeyValue::new("gen_ai.usage.completion_tokens", usage.output_tokens as i64),
                KeyValue::new("gen_ai.usage.input_tokens", usage.input_tokens as i64),
                KeyValue::new("gen_ai.usage.output_tokens", usage.output_tokens as i64),
                KeyValue::new("gen_ai.usage.total_tokens", usage.total_tokens as i64),
                KeyValue::new("gen_ai.usage.cache_read_input_tokens", usage.cache_read_input_tokens as i64),
                KeyValue::new("gen_ai.usage.cache_write_input_tokens", usage.cache_write_input_tokens as i64),
            ]);
        }

        self.end_span_internal(context, Some(attributes), error);
    }

    /// Start a new span for swarm/multiagent invocation.
    pub fn start_multiagent_span(
        &self,
        task: &str,
        instance: &str,
        custom_trace_attributes: Option<&HashMap<String, String>>,
    ) -> Context {
        let operation = format!("invoke_{}", instance);
        let mut attributes = self.get_common_attributes(&operation);
        attributes.push(KeyValue::new("gen_ai.agent.name", instance.to_string()));

        if let Some(attrs) = custom_trace_attributes {
            for (k, v) in attrs {
                attributes.push(KeyValue::new(k.clone(), v.clone()));
            }
        }

        let context = self.start_span(&operation, None, attributes, SpanKind::Client);

        if self.use_latest_genai_conventions {
            let input_message = serde_json::json!([{
                "role": "user",
                "parts": [{"type": "text", "content": task}],
            }]);
            self.add_event(
                &context,
                "gen_ai.client.inference.operation.details",
                vec![KeyValue::new("gen_ai.input.messages", serialize(&input_message))],
            );
        } else {
            self.add_event(
                &context,
                "gen_ai.user.message",
                vec![KeyValue::new("content", task.to_string())],
            );
        }

        context
    }

    /// End a swarm span with results.
    pub fn end_swarm_span(
        &self,
        context: &Context,
        result: Option<&str>,
    ) {
        if let Some(res) = result {
            if self.use_latest_genai_conventions {
                let output_message = serde_json::json!([{
                    "role": "assistant",
                    "parts": [{"type": "text", "content": res}],
                }]);
                self.add_event(
                    context,
                    "gen_ai.client.inference.operation.details",
                    vec![KeyValue::new("gen_ai.output.messages", serialize(&output_message))],
                );
            } else {
                self.add_event(
                    context,
                    "gen_ai.choice",
                    vec![KeyValue::new("message", res.to_string())],
                );
            }
        }

        self.end_span_internal(context, None, None);
    }

    /// Add messages as events to the provided span.
    fn add_event_messages(&self, context: &Context, messages: &Messages) {
        if self.use_latest_genai_conventions {
            let input_messages: Vec<serde_json::Value> = messages
                .iter()
                .map(|msg| {
                    serde_json::json!({
                        "role": msg.role.as_str(),
                        "parts": self.map_content_blocks_to_otel_parts(&msg.content),
                    })
                })
                .collect();
            self.add_event(
                context,
                "gen_ai.client.inference.operation.details",
                vec![KeyValue::new("gen_ai.input.messages", serialize(&input_messages))],
            );
        } else {
            for message in messages {
                let event_name = self.get_event_name_for_message(message);
                self.add_event(
                    context,
                    &event_name,
                    vec![KeyValue::new("content", serialize(&message.content))],
                );
            }
        }
    }

    /// Map content blocks to OpenTelemetry parts format.
    fn map_content_blocks_to_otel_parts(&self, content_blocks: &[ContentBlock]) -> Vec<serde_json::Value> {
        let mut parts = Vec::new();

        for block in content_blocks {
            if let Some(text) = &block.text {
                parts.push(serde_json::json!({
                    "type": "text",
                    "content": text,
                }));
            } else if let Some(tool_use) = &block.tool_use {
                parts.push(serde_json::json!({
                    "type": "tool_call",
                    "name": tool_use.name,
                    "id": tool_use.tool_use_id,
                    "arguments": tool_use.input,
                }));
            } else if let Some(tool_result) = &block.tool_result {
                parts.push(serde_json::json!({
                    "type": "tool_call_response",
                    "id": tool_result.tool_use_id,
                    "response": tool_result.content,
                }));
            } else if let Some(image) = &block.image {
                parts.push(serde_json::json!({
                    "type": "image",
                    "content": image,
                }));
            } else if let Some(document) = &block.document {
                parts.push(serde_json::json!({
                    "type": "document",
                    "content": document,
                }));
            }
        }

        parts
    }

    /// End a span with error status.
    pub fn end_span_with_error(
        &self,
        context: &Context,
        error_message: &str,
    ) {
        #[derive(Debug)]
        struct SimpleError(String);
        impl std::fmt::Display for SimpleError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        impl std::error::Error for SimpleError {}

        let error = SimpleError(error_message.to_string());
        self.end_span_internal(context, None, Some(&error));
    }
}


static TRACER_INSTANCE: OnceLock<Tracer> = OnceLock::new();

/// Get or create the global tracer.
pub fn get_tracer() -> &'static Tracer {
    TRACER_INSTANCE.get_or_init(Tracer::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize() {
        let data = serde_json::json!({
            "name": "test",
            "value": 42,
        });
        let result = serialize(&data);
        assert!(result.contains("test"));
        assert!(result.contains("42"));
    }

    #[test]
    fn test_tracer_creation() {
        let tracer = Tracer::new();
        assert_eq!(tracer.service_name, "strands-agents");
    }

    #[test]
    fn test_get_tracer_singleton() {
        let tracer1 = get_tracer();
        let tracer2 = get_tracer();
        assert_eq!(tracer1.service_name, tracer2.service_name);
    }
}
