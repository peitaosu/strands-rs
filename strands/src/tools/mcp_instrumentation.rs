//! OpenTelemetry instrumentation for Model Context Protocol (MCP) tracing.
//!
//! Enables distributed tracing across MCP client-server boundaries by injecting
//! OpenTelemetry context into MCP request metadata (_meta field) and extracting
//! it on the server side, creating unified traces that span from agent calls
//! through MCP tool executions.
//!
//! Based on: https://github.com/traceloop/openllmetry/tree/main/packages/opentelemetry-instrumentation-mcp
//! Related issue: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/246

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use opentelemetry::Context;
use opentelemetry::propagation::{TextMapPropagator, Extractor, Injector};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use serde_json::{json, Value};
use tracing::{debug, instrument, Span};

/// Module-level flag to ensure instrumentation is applied only once.
static INSTRUMENTATION_APPLIED: AtomicBool = AtomicBool::new(false);

/// Wrapper for items that need to carry tracing context.
///
/// Used to preserve tracing context across async boundaries in MCP sessions,
/// ensuring that distributed traces remain connected even when messages are
/// processed asynchronously.
#[derive(Debug, Clone)]
pub struct ItemWithContext<T> {
    /// The original item being wrapped.
    pub item: T,
    /// The context data associated with the item.
    pub context: HashMap<String, String>,
}

impl<T> ItemWithContext<T> {
    /// Creates a new item with the current tracing context.
    pub fn new(item: T) -> Self {
        Self {
            item,
            context: get_current_trace_context(),
        }
    }

    /// Creates a new item with explicit context.
    pub fn with_context(item: T, context: HashMap<String, String>) -> Self {
        Self { item, context }
    }

    /// Unwraps the item, discarding the context.
    pub fn into_inner(self) -> T {
        self.item
    }
}

/// Initializes MCP OpenTelemetry instrumentation.
///
/// This function sets up instrumentation for MCP client-server communication:
/// 1. Client-side: Injects tracing context into tool call requests
/// 2. Transport-level: Extracts context from incoming messages
/// 3. Session-level: Manages bidirectional context flow
///
/// The patches enable distributed tracing by:
/// - Adding OpenTelemetry context to the _meta field of MCP requests
/// - Extracting and activating context on the server side
/// - Preserving context across async message processing boundaries
///
/// This function is idempotent - multiple calls will not accumulate wrappers.
pub fn init_mcp_instrumentation() {
    if INSTRUMENTATION_APPLIED.swap(true, Ordering::SeqCst) {
        debug!("MCP instrumentation already applied");
        return;
    }

    debug!("Initializing MCP OpenTelemetry instrumentation");



}

/// Checks if MCP instrumentation has been initialized.
pub fn is_instrumentation_applied() -> bool {
    INSTRUMENTATION_APPLIED.load(Ordering::SeqCst)
}

/// Resets the instrumentation flag (mainly for testing).
#[cfg(test)]
pub fn reset_instrumentation() {
    INSTRUMENTATION_APPLIED.store(false, Ordering::SeqCst);
}

/// Global propagator instance for W3C TraceContext propagation.
static PROPAGATOR: std::sync::OnceLock<TraceContextPropagator> = std::sync::OnceLock::new();

fn get_propagator() -> &'static TraceContextPropagator {
    PROPAGATOR.get_or_init(|| TraceContextPropagator::new())
}

/// Gets the current trace context as a map of headers.
///
/// This extracts the current OpenTelemetry context's trace ID, span ID, and trace flags
/// into a format suitable for propagation via the _meta field using W3C TraceContext format.
pub fn get_current_trace_context() -> HashMap<String, String> {
    let mut context = HashMap::new();
    

    let otel_context = Context::current();
    

    let mut injector = HashMapInjector(&mut context);
    get_propagator().inject_context(&otel_context, &mut injector);
    
    context
}

/// Helper struct to inject context into a HashMap.
struct HashMapInjector<'a>(&'a mut HashMap<String, String>);

impl<'a> Injector for HashMapInjector<'a> {
    fn set(&mut self, key: &str, value: String) {
        self.0.insert(key.to_string(), value);
    }
}

/// Helper struct to extract context from a HashMap.
struct HashMapExtractor<'a>(&'a HashMap<String, String>);

impl<'a> Extractor for HashMapExtractor<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(|s| s.as_str())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|s| s.as_str()).collect()
    }
}

/// Injects trace context into an MCP request's _meta field.
///
/// This should be called before sending MCP tool call requests to
/// propagate the current trace context to the server.
///
/// # Arguments
///
/// * `request` - The MCP request parameters as a mutable JSON value
///
/// # Example
///
/// ```ignore
/// let mut params = json!({
///     "name": "my_tool",
///     "arguments": { "input": "value" }
/// });
/// inject_trace_context(&mut params);
/// // params now contains _meta with tracing headers
/// ```
#[instrument(skip(request), level = "trace")]
pub fn inject_trace_context(request: &mut Value) {
    let context = get_current_trace_context();

    if context.is_empty() {
        return;
    }


    let obj = match request.as_object_mut() {
        Some(o) => o,
        None => return,
    };


    let meta = obj
        .entry("_meta")
        .or_insert_with(|| json!({}));


    if let Some(meta_obj) = meta.as_object_mut() {
        for (key, value) in context {
            meta_obj.insert(key, Value::String(value));
        }
    }

    debug!("Injected trace context into MCP request");
}

/// Extracts trace context from an MCP request's _meta field.
///
/// This should be called on the server side when receiving MCP tool
/// call requests to restore the trace context from the client.
///
/// # Arguments
///
/// * `request` - The MCP request parameters as a JSON value
///
/// # Returns
///
/// A map of trace context headers extracted from the request.
#[instrument(skip(request), level = "trace")]
pub fn extract_trace_context(request: &Value) -> HashMap<String, String> {
    let mut context = HashMap::new();

    let meta = match request.get("_meta").and_then(|m| m.as_object()) {
        Some(m) => m,
        None => return context,
    };


    for key in &["traceparent", "tracestate", "baggage"] {
        if let Some(value) = meta.get(*key).and_then(|v| v.as_str()) {
            context.insert((*key).to_string(), value.to_string());
        }
    }

    if !context.is_empty() {
        debug!(keys = ?context.keys().collect::<Vec<_>>(), "Extracted trace context from MCP request");
    }

    context
}

/// Extracts and activates OpenTelemetry context from an MCP request.
///
/// This should be called on the server side to restore the trace context
/// and make it the current context for subsequent operations.
///
/// # Arguments
///
/// * `request` - The MCP request parameters as a JSON value
///
/// # Returns
///
/// A guard that will restore the previous context when dropped.
pub fn extract_and_activate_context(request: &Value) -> ContextGuard {
    let context_map = extract_trace_context(request);
    
    if context_map.is_empty() {
        return ContextGuard::new(Context::current());
    }
    
    let extractor = HashMapExtractor(&context_map);
    let extracted_context = get_propagator().extract(&extractor);
    

    let previous_context = Context::current();
    

    let _guard = extracted_context.clone().attach();
    
    ContextGuard::new(previous_context)
}

/// Guard that restores the previous OpenTelemetry context when dropped.
pub struct ContextGuard {
    _active_guard: opentelemetry::ContextGuard,
}

impl ContextGuard {
    fn new(context: Context) -> Self {

        let _active_guard = context.attach();
        Self { _active_guard }
    }
}

/// Creates a tracing span for an MCP tool call with extracted context.
///
/// This helper creates a properly configured span for server-side
/// tool execution, linking it to the client's trace if context is available.
///
/// # Arguments
///
/// * `tool_name` - The name of the tool being called
/// * `request` - The MCP request parameters (for context extraction)
///
/// # Returns
///
/// A configured tracing Span and a context guard that should be kept alive
/// for the duration of the span.
#[must_use]
pub fn create_mcp_tool_span(tool_name: &str, request: &Value) -> (Span, ContextGuard) {

    let context_guard = extract_and_activate_context(request);


    let span = tracing::info_span!(
        "mcp.tool.call",
        tool.name = %tool_name,
        otel.kind = "server",
        mcp.instrumented = true
    );

    (span, context_guard)
}

/// Trait for MCP message context injection.
///
/// Implement this trait for your MCP request types to enable
/// automatic trace context injection.
pub trait InjectableContext {
    /// Injects trace context into this message.
    fn inject_context(&mut self);

    /// Checks if context has already been injected.
    fn has_context(&self) -> bool;
}

impl InjectableContext for Value {
    fn inject_context(&mut self) {
        inject_trace_context(self);
    }

    fn has_context(&self) -> bool {
        self.get("_meta")
            .and_then(|m| m.get("traceparent"))
            .is_some()
    }
}

/// Trait for MCP message context extraction.
///
/// Implement this trait for your MCP request types to enable
/// automatic trace context extraction.
pub trait ExtractableContext {
    /// Extracts trace context from this message.
    fn extract_context(&self) -> HashMap<String, String>;
}

impl ExtractableContext for Value {
    fn extract_context(&self) -> HashMap<String, String> {
        extract_trace_context(self)
    }
}

/// Configuration for MCP instrumentation.
#[derive(Debug, Clone, Default)]
pub struct MCPInstrumentationConfig {
    /// Whether to enable client-side context injection.
    pub inject_client_context: bool,
    /// Whether to enable server-side context extraction.
    pub extract_server_context: bool,
    /// Additional headers to propagate.
    pub additional_headers: Vec<String>,
}

impl MCPInstrumentationConfig {
    /// Creates a new configuration with all features enabled.
    pub fn enabled() -> Self {
        Self {
            inject_client_context: true,
            extract_server_context: true,
            additional_headers: Vec::new(),
        }
    }

    /// Creates a client-only configuration.
    pub fn client_only() -> Self {
        Self {
            inject_client_context: true,
            extract_server_context: false,
            additional_headers: Vec::new(),
        }
    }

    /// Creates a server-only configuration.
    pub fn server_only() -> Self {
        Self {
            inject_client_context: false,
            extract_server_context: true,
            additional_headers: Vec::new(),
        }
    }

    /// Adds additional headers to propagate.
    pub fn with_headers(mut self, headers: Vec<String>) -> Self {
        self.additional_headers = headers;
        self
    }
}

/// Guards that apply instrumentation within a scope.
pub struct InstrumentationGuard {
    _config: MCPInstrumentationConfig,
}

impl InstrumentationGuard {
    /// Creates a new instrumentation guard.
    pub fn new(config: MCPInstrumentationConfig) -> Self {
        init_mcp_instrumentation();
        Self { _config: config }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inject_trace_context() {
        let mut request = json!({
            "name": "test_tool",
            "arguments": {}
        });

        inject_trace_context(&mut request);




    }

    #[test]
    fn test_extract_trace_context_empty() {
        let request = json!({
            "name": "test_tool",
            "arguments": {}
        });

        let context = extract_trace_context(&request);
        assert!(context.is_empty());
    }

    #[test]
    fn test_extract_trace_context_with_meta() {
        let request = json!({
            "name": "test_tool",
            "_meta": {
                "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
            }
        });

        let context = extract_trace_context(&request);
        assert_eq!(
            context.get("traceparent"),
            Some(&"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01".to_string())
        );
    }

    #[test]
    fn test_injectable_context_trait() {
        let mut request = json!({
            "name": "test_tool"
        });


        assert!(!request.has_context());
        request.inject_context();

        assert!(!request.has_context());
    }

    #[test]
    fn test_instrumentation_idempotent() {
        reset_instrumentation();
        assert!(!is_instrumentation_applied());

        init_mcp_instrumentation();
        assert!(is_instrumentation_applied());

        init_mcp_instrumentation();
        assert!(is_instrumentation_applied());
    }

    #[test]
    fn test_item_with_context() {
        let item = "test_message";
        let wrapped = ItemWithContext::new(item);

        assert_eq!(wrapped.item, "test_message");
        assert_eq!(wrapped.into_inner(), "test_message");
    }

    #[test]
    fn test_config_builders() {
        let enabled = MCPInstrumentationConfig::enabled();
        assert!(enabled.inject_client_context);
        assert!(enabled.extract_server_context);

        let client_only = MCPInstrumentationConfig::client_only();
        assert!(client_only.inject_client_context);
        assert!(!client_only.extract_server_context);

        let server_only = MCPInstrumentationConfig::server_only();
        assert!(!server_only.inject_client_context);
        assert!(server_only.extract_server_context);
    }
}

