//! Telemetry and metrics for agent execution.

pub mod config;
pub mod tracer;

pub use config::{OtelResource, StrandsTelemetry, StrandsTelemetryBuilder};
pub use tracer::{get_tracer, serialize, Tracer, AttributeValue, Attributes};

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::types::content::Message;
use crate::types::streaming::{Metrics, Usage};
use crate::types::tools::ToolUse;

/// A trace representing a single operation or step in the execution flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub id: String,
    pub name: String,
    pub raw_name: Option<String>,
    pub parent_id: Option<String>,
    pub start_time: f64,
    pub end_time: Option<f64>,
    pub children: Vec<Trace>,
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(skip)]
    start_instant: Option<Instant>,
}

impl Trace {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            raw_name: None,
            parent_id: None,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            end_time: None,
            children: Vec::new(),
            metadata: HashMap::new(),
            start_instant: Some(Instant::now()),
        }
    }

    pub fn child(name: impl Into<String>, parent_id: impl Into<String>) -> Self {
        let mut trace = Self::new(name);
        trace.parent_id = Some(parent_id.into());
        trace
    }

    pub fn end(&mut self) {
        self.end_time = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        );
    }

    pub fn add_child(&mut self, child: Trace) {
        self.children.push(child);
    }

    pub fn duration(&self) -> Option<f64> {

        if self.end_time.is_some() {
            if let Some(instant) = self.start_instant {
                return Some(instant.elapsed().as_secs_f64());
            }
        }

        self.end_time.map(|end| end - self.start_time)
    }

    pub fn duration_ms(&self) -> Option<u64> {
        self.duration().map(|d| (d * 1000.0) as u64)
    }

    pub fn add_message(&mut self, _message: &Message) {

    }

    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "name": self.name,
            "raw_name": self.raw_name,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "children": self.children.iter().map(|c| c.to_dict()).collect::<Vec<_>>(),
            "metadata": self.metadata,
        })
    }
}

/// Metrics for a specific tool's usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolMetrics {
    pub tool_name: String,
    pub tool_use_id: String,
    pub call_count: u32,
    pub success_count: u32,
    pub error_count: u32,
    pub total_time: f64,
}

impl ToolMetrics {
    pub fn new(tool: &ToolUse) -> Self {
        Self {
            tool_name: tool.name.clone(),
            tool_use_id: tool.tool_use_id.clone(),
            call_count: 0,
            success_count: 0,
            error_count: 0,
            total_time: 0.0,
        }
    }

    pub fn add_call(&mut self, tool: &ToolUse, duration: f64, success: bool) {
        self.tool_name = tool.name.clone();
        self.tool_use_id = tool.tool_use_id.clone();
        self.call_count += 1;
        self.total_time += duration;
        if success {
            self.success_count += 1;
        } else {
            self.error_count += 1;
        }
    }

    pub fn average_time(&self) -> f64 {
        if self.call_count > 0 {
            self.total_time / self.call_count as f64
        } else {
            0.0
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.call_count > 0 {
            self.success_count as f64 / self.call_count as f64
        } else {
            0.0
        }
    }
}

/// Metrics collected during event loop execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventLoopMetrics {
    pub cycle_count: u32,
    pub tool_metrics: HashMap<String, ToolMetrics>,
    pub cycle_durations: Vec<f64>,
    pub traces: Vec<Trace>,
    pub accumulated_usage: Usage,
    pub accumulated_metrics: Metrics,
    #[serde(skip)]
    cycle_start: Option<Instant>,
}

impl EventLoopMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a new event loop cycle.
    pub fn start_cycle(&mut self) -> Trace {
        self.cycle_start = Some(Instant::now());
        self.cycle_count += 1;
        let trace = Trace::new(format!("Cycle {}", self.cycle_count));
        self.traces.push(trace.clone());
        trace
    }

    /// End the current cycle.
    pub fn end_cycle(&mut self, cycle_trace: &mut Trace) {
        if let Some(start) = self.cycle_start.take() {
            let duration = start.elapsed().as_secs_f64();
            self.cycle_durations.push(duration);
            cycle_trace.end();
        }
    }

    /// Record tool usage metrics.
    pub fn add_tool_usage(
        &mut self,
        tool: &ToolUse,
        duration: f64,
        tool_trace: &mut Trace,
        success: bool,
        message: &Message,
    ) {
        tool_trace.metadata.insert(
            "toolUseId".to_string(),
            serde_json::Value::String(tool.tool_use_id.clone()),
        );
        tool_trace.metadata.insert(
            "tool_name".to_string(),
            serde_json::Value::String(tool.name.clone()),
        );
        tool_trace.raw_name = Some(format!("{} - {}", tool.name, tool.tool_use_id));
        tool_trace.add_message(message);

        self.tool_metrics
            .entry(tool.name.clone())
            .or_insert_with(|| ToolMetrics::new(tool))
            .add_call(tool, duration, success);

        tool_trace.end();
    }

    /// Update accumulated token usage.
    pub fn update_usage(&mut self, usage: &Usage) {
        self.accumulated_usage.add(usage);
    }

    /// Update accumulated performance metrics.
    pub fn update_metrics(&mut self, metrics: &Metrics) {
        self.accumulated_metrics.latency_ms += metrics.latency_ms;
    }

    /// Get total duration of all cycles.
    pub fn total_duration(&self) -> f64 {
        self.cycle_durations.iter().sum()
    }

    /// Get average cycle time.
    pub fn average_cycle_time(&self) -> f64 {
        if self.cycle_count > 0 {
            self.total_duration() / self.cycle_count as f64
        } else {
            0.0
        }
    }

    /// Generate a comprehensive summary.
    pub fn get_summary(&self) -> serde_json::Value {
        serde_json::json!({
            "total_cycles": self.cycle_count,
            "total_duration": self.total_duration(),
            "average_cycle_time": self.average_cycle_time(),
            "tool_usage": self.tool_metrics.iter().map(|(name, metrics)| {
                (name.clone(), serde_json::json!({
                    "tool_info": {
                        "tool_use_id": metrics.tool_use_id,
                        "name": metrics.tool_name,
                    },
                    "execution_stats": {
                        "call_count": metrics.call_count,
                        "success_count": metrics.success_count,
                        "error_count": metrics.error_count,
                        "total_time": metrics.total_time,
                        "average_time": metrics.average_time(),
                        "success_rate": metrics.success_rate(),
                    }
                }))
            }).collect::<HashMap<_, _>>(),
            "traces": self.traces.iter().map(|t| t.to_dict()).collect::<Vec<_>>(),
            "accumulated_usage": {
                "inputTokens": self.accumulated_usage.input_tokens,
                "outputTokens": self.accumulated_usage.output_tokens,
                "totalTokens": self.accumulated_usage.total_tokens,
                "cacheReadInputTokens": self.accumulated_usage.cache_read_input_tokens,
                "cacheWriteInputTokens": self.accumulated_usage.cache_write_input_tokens,
            },
            "accumulated_metrics": {
                "latencyMs": self.accumulated_metrics.latency_ms,
            },
        })
    }

    pub fn total_input_tokens(&self) -> u32 {
        self.accumulated_usage.input_tokens
    }

    pub fn total_output_tokens(&self) -> u32 {
        self.accumulated_usage.output_tokens
    }

    pub fn total_latency_ms(&self) -> u64 {
        self.accumulated_metrics.latency_ms
    }
}

/// Convert metrics to a formatted string.
pub fn metrics_to_string(metrics: &EventLoopMetrics) -> String {
    let summary = metrics.get_summary();
    let mut lines = Vec::new();

    lines.push("Event Loop Metrics Summary:".to_string());
    lines.push(format!(
        "├─ Cycles: total={}, avg_time={:.3}s, total_time={:.3}s",
        summary["total_cycles"],
        summary["average_cycle_time"].as_f64().unwrap_or(0.0),
        summary["total_duration"].as_f64().unwrap_or(0.0)
    ));

    let usage = &summary["accumulated_usage"];
    let mut token_parts = vec![
        format!("in={}", usage["inputTokens"]),
        format!("out={}", usage["outputTokens"]),
        format!("total={}", usage["totalTokens"]),
    ];

    if let Some(cache_read) = usage["cacheReadInputTokens"].as_u64() {
        if cache_read > 0 {
            token_parts.push(format!("cache_read={}", cache_read));
        }
    }
    if let Some(cache_write) = usage["cacheWriteInputTokens"].as_u64() {
        if cache_write > 0 {
            token_parts.push(format!("cache_write={}", cache_write));
        }
    }

    lines.push(format!("├─ Tokens: {}", token_parts.join(", ")));
    lines.push(format!(
        "├─ Latency: {}ms",
        summary["accumulated_metrics"]["latencyMs"]
    ));

    lines.push("├─ Tool Usage:".to_string());
    if let Some(tool_usage) = summary["tool_usage"].as_object() {
        for (tool_name, data) in tool_usage {
            let stats = &data["execution_stats"];
            lines.push(format!("   └─ {}:", tool_name));
            lines.push(format!(
                "      ├─ Stats: calls={}, success={}, errors={}, success_rate={:.1}%",
                stats["call_count"],
                stats["success_count"],
                stats["error_count"],
                stats["success_rate"].as_f64().unwrap_or(0.0) * 100.0
            ));
            lines.push(format!(
                "      └─ Timing: avg={:.3}s, total={:.3}s",
                stats["average_time"].as_f64().unwrap_or(0.0),
                stats["total_time"].as_f64().unwrap_or(0.0)
            ));
        }
    }

    lines.join("\n")
}

/// Metrics constants matching Python SDK.
pub mod constants {
    pub const STRANDS_EVENT_LOOP_CYCLE_COUNT: &str = "strands.event_loop.cycle_count";
    pub const STRANDS_EVENT_LOOP_START_CYCLE: &str = "strands.event_loop.start_cycle";
    pub const STRANDS_EVENT_LOOP_END_CYCLE: &str = "strands.event_loop.end_cycle";
    pub const STRANDS_EVENT_LOOP_CYCLE_DURATION: &str = "strands.event_loop.cycle_duration";
    pub const STRANDS_EVENT_LOOP_LATENCY: &str = "strands.event_loop.latency";
    pub const STRANDS_EVENT_LOOP_INPUT_TOKENS: &str = "strands.event_loop.input.tokens";
    pub const STRANDS_EVENT_LOOP_OUTPUT_TOKENS: &str = "strands.event_loop.output.tokens";
    pub const STRANDS_EVENT_LOOP_CACHE_READ_INPUT_TOKENS: &str =
        "strands.event_loop.cache_read.input.tokens";
    pub const STRANDS_EVENT_LOOP_CACHE_WRITE_INPUT_TOKENS: &str =
        "strands.event_loop.cache_write.input.tokens";
    pub const STRANDS_MODEL_TIME_TO_FIRST_TOKEN: &str = "strands.model.time_to_first_token";
    pub const STRANDS_TOOL_CALL_COUNT: &str = "strands.tool.call_count";
    pub const STRANDS_TOOL_SUCCESS_COUNT: &str = "strands.tool.success_count";
    pub const STRANDS_TOOL_ERROR_COUNT: &str = "strands.tool.error_count";
    pub const STRANDS_TOOL_DURATION: &str = "strands.tool.duration";
}

use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry::KeyValue;

/// Global singleton instance for MetricsClient.
static METRICS_CLIENT_INSTANCE: std::sync::OnceLock<MetricsClient> = std::sync::OnceLock::new();

/// Singleton client for managing OpenTelemetry metrics instruments.
///
/// The actual metrics export destination (console, OTLP endpoint, etc.) is configured
/// through OpenTelemetry SDK configuration by users, not by this client.
pub struct MetricsClient {
    meter: Meter,

    event_loop_cycle_count: Counter<u64>,
    event_loop_start_cycle: Counter<u64>,
    event_loop_end_cycle: Counter<u64>,
    tool_call_count: Counter<u64>,
    tool_success_count: Counter<u64>,
    tool_error_count: Counter<u64>,

    event_loop_cycle_duration: Histogram<f64>,
    event_loop_latency: Histogram<f64>,
    event_loop_input_tokens: Histogram<u64>,
    event_loop_output_tokens: Histogram<u64>,
    event_loop_cache_read_input_tokens: Histogram<u64>,
    event_loop_cache_write_input_tokens: Histogram<u64>,
    model_time_to_first_token: Histogram<f64>,
    tool_duration: Histogram<f64>,
}

impl MetricsClient {
    /// Create a new MetricsClient with the given meter.
    fn new(meter: Meter) -> Self {
        tracing::info!("Creating Strands MetricsClient with OpenTelemetry instruments");

        Self {
            event_loop_cycle_count: meter
                .u64_counter(constants::STRANDS_EVENT_LOOP_CYCLE_COUNT)
                .with_description("Number of event loop cycles")
                .with_unit("count")
                .build(),
            event_loop_start_cycle: meter
                .u64_counter(constants::STRANDS_EVENT_LOOP_START_CYCLE)
                .with_description("Event loop cycle starts")
                .with_unit("count")
                .build(),
            event_loop_end_cycle: meter
                .u64_counter(constants::STRANDS_EVENT_LOOP_END_CYCLE)
                .with_description("Event loop cycle ends")
                .with_unit("count")
                .build(),
            tool_call_count: meter
                .u64_counter(constants::STRANDS_TOOL_CALL_COUNT)
                .with_description("Number of tool calls")
                .with_unit("count")
                .build(),
            tool_success_count: meter
                .u64_counter(constants::STRANDS_TOOL_SUCCESS_COUNT)
                .with_description("Number of successful tool calls")
                .with_unit("count")
                .build(),
            tool_error_count: meter
                .u64_counter(constants::STRANDS_TOOL_ERROR_COUNT)
                .with_description("Number of failed tool calls")
                .with_unit("count")
                .build(),
            event_loop_cycle_duration: meter
                .f64_histogram(constants::STRANDS_EVENT_LOOP_CYCLE_DURATION)
                .with_description("Duration of event loop cycles")
                .with_unit("s")
                .build(),
            event_loop_latency: meter
                .f64_histogram(constants::STRANDS_EVENT_LOOP_LATENCY)
                .with_description("Latency of model requests")
                .with_unit("ms")
                .build(),
            event_loop_input_tokens: meter
                .u64_histogram(constants::STRANDS_EVENT_LOOP_INPUT_TOKENS)
                .with_description("Number of input tokens")
                .with_unit("token")
                .build(),
            event_loop_output_tokens: meter
                .u64_histogram(constants::STRANDS_EVENT_LOOP_OUTPUT_TOKENS)
                .with_description("Number of output tokens")
                .with_unit("token")
                .build(),
            event_loop_cache_read_input_tokens: meter
                .u64_histogram(constants::STRANDS_EVENT_LOOP_CACHE_READ_INPUT_TOKENS)
                .with_description("Number of cache read input tokens")
                .with_unit("token")
                .build(),
            event_loop_cache_write_input_tokens: meter
                .u64_histogram(constants::STRANDS_EVENT_LOOP_CACHE_WRITE_INPUT_TOKENS)
                .with_description("Number of cache write input tokens")
                .with_unit("token")
                .build(),
            model_time_to_first_token: meter
                .f64_histogram(constants::STRANDS_MODEL_TIME_TO_FIRST_TOKEN)
                .with_description("Time to first token from model")
                .with_unit("ms")
                .build(),
            tool_duration: meter
                .f64_histogram(constants::STRANDS_TOOL_DURATION)
                .with_description("Duration of tool execution")
                .with_unit("s")
                .build(),
            meter,
        }
    }

    /// Get the singleton MetricsClient instance.
    ///
    /// Uses the global OpenTelemetry meter provider. Users should configure
    /// the meter provider before calling this method.
    pub fn global() -> &'static MetricsClient {
        METRICS_CLIENT_INSTANCE.get_or_init(|| {
            let meter = opentelemetry::global::meter("strands");
            MetricsClient::new(meter)
        })
    }

    /// Convert HashMap attributes to OpenTelemetry KeyValue pairs.
    fn to_key_values(attributes: &HashMap<String, String>) -> Vec<KeyValue> {
        attributes
            .iter()
            .map(|(k, v)| KeyValue::new(k.clone(), v.clone()))
            .collect()
    }

    /// Record event loop cycle count.
    pub fn record_cycle_count(&self, count: u64, attributes: &HashMap<String, String>) {
        self.event_loop_cycle_count
            .add(count, &Self::to_key_values(attributes));
    }

    /// Record event loop start cycle.
    pub fn record_start_cycle(&self, attributes: &HashMap<String, String>) {
        self.event_loop_start_cycle
            .add(1, &Self::to_key_values(attributes));
    }

    /// Record event loop end cycle.
    pub fn record_end_cycle(&self, attributes: &HashMap<String, String>) {
        self.event_loop_end_cycle
            .add(1, &Self::to_key_values(attributes));
    }

    /// Record cycle duration.
    pub fn record_cycle_duration(&self, duration_secs: f64, attributes: &HashMap<String, String>) {
        self.event_loop_cycle_duration
            .record(duration_secs, &Self::to_key_values(attributes));
    }

    /// Record latency.
    pub fn record_latency(&self, latency_ms: u64, attributes: &HashMap<String, String>) {
        self.event_loop_latency
            .record(latency_ms as f64, &Self::to_key_values(attributes));
    }

    /// Record input tokens.
    pub fn record_input_tokens(&self, tokens: u32, attributes: &HashMap<String, String>) {
        self.event_loop_input_tokens
            .record(tokens as u64, &Self::to_key_values(attributes));
    }

    /// Record output tokens.
    pub fn record_output_tokens(&self, tokens: u32, attributes: &HashMap<String, String>) {
        self.event_loop_output_tokens
            .record(tokens as u64, &Self::to_key_values(attributes));
    }

    /// Record cache read input tokens.
    pub fn record_cache_read_input_tokens(&self, tokens: u32, attributes: &HashMap<String, String>) {
        self.event_loop_cache_read_input_tokens
            .record(tokens as u64, &Self::to_key_values(attributes));
    }

    /// Record cache write input tokens.
    pub fn record_cache_write_input_tokens(&self, tokens: u32, attributes: &HashMap<String, String>) {
        self.event_loop_cache_write_input_tokens
            .record(tokens as u64, &Self::to_key_values(attributes));
    }

    /// Record model time to first token.
    pub fn record_time_to_first_token(&self, time_ms: u64, attributes: &HashMap<String, String>) {
        self.model_time_to_first_token
            .record(time_ms as f64, &Self::to_key_values(attributes));
    }

    /// Record tool call count.
    pub fn record_tool_call_count(&self, count: u64, attributes: &HashMap<String, String>) {
        self.tool_call_count
            .add(count, &Self::to_key_values(attributes));
    }

    /// Record tool success count.
    pub fn record_tool_success_count(&self, count: u64, attributes: &HashMap<String, String>) {
        self.tool_success_count
            .add(count, &Self::to_key_values(attributes));
    }

    /// Record tool error count.
    pub fn record_tool_error_count(&self, count: u64, attributes: &HashMap<String, String>) {
        self.tool_error_count
            .add(count, &Self::to_key_values(attributes));
    }

    /// Record tool duration.
    pub fn record_tool_duration(&self, duration_secs: f64, attributes: &HashMap<String, String>) {
        self.tool_duration
            .record(duration_secs, &Self::to_key_values(attributes));
    }

    /// Get the underlying meter for creating custom metrics.
    pub fn meter(&self) -> &Meter {
        &self.meter
    }
}
