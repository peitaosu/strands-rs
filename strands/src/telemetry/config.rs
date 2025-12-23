//! OpenTelemetry configuration and setup utilities for Strands agents.
//!
//! This module provides centralized configuration and initialization functionality
//! for OpenTelemetry components and other telemetry infrastructure shared across
//! Strands applications.

use opentelemetry::global;
use opentelemetry::trace::TracerProvider;
use opentelemetry_sdk::{
    propagation::TraceContextPropagator,
    trace::SdkTracerProvider,
    Resource,
};

/// Resource information for telemetry.
#[derive(Debug, Clone)]
pub struct OtelResource {
    pub service_name: String,
    pub service_version: String,
    pub sdk_name: String,
    pub sdk_language: String,
}

impl Default for OtelResource {
    fn default() -> Self {
        Self {
            service_name: "strands-agents".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            sdk_name: "opentelemetry".to_string(),
            sdk_language: "rust".to_string(),
        }
    }
}

impl OtelResource {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get resource attributes as a map.
    pub fn attributes(&self) -> std::collections::HashMap<String, String> {
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("service.name".to_string(), self.service_name.clone());
        attrs.insert("service.version".to_string(), self.service_version.clone());
        attrs.insert("telemetry.sdk.name".to_string(), self.sdk_name.clone());
        attrs.insert("telemetry.sdk.language".to_string(), self.sdk_language.clone());
        attrs
    }

    /// Convert to OpenTelemetry SDK Resource.
    fn to_otel_resource(&self) -> Resource {
        Resource::builder()
            .with_service_name(self.service_name.clone())
            .with_attributes([
                opentelemetry::KeyValue::new("service.version", self.service_version.clone()),
                opentelemetry::KeyValue::new("telemetry.sdk.name", self.sdk_name.clone()),
                opentelemetry::KeyValue::new("telemetry.sdk.language", self.sdk_language.clone()),
            ])
            .build()
    }
}

/// OpenTelemetry configuration and setup for Strands applications.
///
/// Automatically initializes a tracer provider with text map propagators.
/// Trace exporters (console, OTLP) can be set up individually using dedicated methods
/// that support method chaining for convenient configuration.
///
/// # Examples
///
/// Quick setup with method chaining:
/// ```ignore
/// StrandsTelemetry::new().setup_console_exporter().setup_otlp_exporter();
/// ```
///
/// Using a custom tracer provider:
/// ```ignore
/// StrandsTelemetry::with_tracer_provider(my_provider).setup_console_exporter();
/// ```
///
/// Step-by-step configuration:
/// ```ignore
/// let telemetry = StrandsTelemetry::new();
/// telemetry.setup_console_exporter();
/// telemetry.setup_otlp_exporter();
/// ```
///
/// To setup global meter provider:
/// ```ignore
/// telemetry.setup_meter(true, true); // enable_console, enable_otlp
/// ```
pub struct StrandsTelemetry {
    pub resource: OtelResource,
    tracer_provider: Option<SdkTracerProvider>,
    console_enabled: bool,
    otlp_enabled: bool,
    meter_enabled: bool,
}

impl Default for StrandsTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl StrandsTelemetry {
    /// Create a new StrandsTelemetry instance.
    ///
    /// This initializes a new tracer provider and sets it as the global provider.
    pub fn new() -> Self {
        let mut instance = Self {
            resource: OtelResource::default(),
            tracer_provider: None,
            console_enabled: false,
            otlp_enabled: false,
            meter_enabled: false,
        };
        instance.initialize_tracer();
        instance
    }

    /// Create a new StrandsTelemetry instance with a pre-configured tracer provider.
    pub fn with_tracer_provider(tracer_provider: SdkTracerProvider) -> Self {
        Self {
            resource: OtelResource::default(),
            tracer_provider: Some(tracer_provider),
            console_enabled: false,
            otlp_enabled: false,
            meter_enabled: false,
        }
    }

    /// Initialize the OpenTelemetry tracer.
    fn initialize_tracer(&mut self) {
        tracing::info!("Initializing tracer");

        let resource = self.resource.to_otel_resource();
        let tracer_provider = SdkTracerProvider::builder()
            .with_resource(resource)
            .build();

        global::set_tracer_provider(tracer_provider.clone());

        let propagator = TraceContextPropagator::new();
        global::set_text_map_propagator(propagator);

        self.tracer_provider = Some(tracer_provider);
    }

    /// Set up console exporter for traces.
    ///
    /// This method configures a SimpleSpanProcessor with a ConsoleSpanExporter,
    /// allowing trace data to be output to the console.
    #[cfg(feature = "otel-stdout")]
    pub fn setup_console_exporter(mut self) -> Self {
        use opentelemetry_stdout::SpanExporter as StdoutSpanExporter;

        tracing::info!("Enabling console export");
        self.console_enabled = true;

        let exporter = StdoutSpanExporter::default();
        let new_provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter)
            .with_resource(self.resource.to_otel_resource())
            .build();

        global::set_tracer_provider(new_provider.clone());
        self.tracer_provider = Some(new_provider);

        self
    }

    /// Set up console exporter for traces (no-op when feature not enabled).
    #[cfg(not(feature = "otel-stdout"))]
    pub fn setup_console_exporter(mut self) -> Self {
        tracing::warn!("otel-stdout feature not enabled, console exporter not available");
        self.console_enabled = false;
        self
    }

    /// Set up OTLP exporter for traces.
    ///
    /// This method configures a BatchSpanProcessor with an OTLPSpanExporter,
    /// allowing trace data to be exported to an OTLP endpoint.
    #[cfg(feature = "otel-otlp")]
    pub fn setup_otlp_exporter(mut self) -> Self {
        use opentelemetry_otlp::SpanExporter;

        tracing::info!("Enabling OTLP export");
        self.otlp_enabled = true;

        match SpanExporter::builder().with_tonic().build() {
            Ok(exporter) => {
                let new_provider = SdkTracerProvider::builder()
                    .with_batch_exporter(exporter)
                    .with_resource(self.resource.to_otel_resource())
                    .build();

                global::set_tracer_provider(new_provider.clone());
                self.tracer_provider = Some(new_provider);
                tracing::info!("OTLP exporter configured");
            }
            Err(e) => {
                tracing::error!("error=<{}> | Failed to configure OTLP exporter", e);
            }
        }

        self
    }

    /// Set up OTLP exporter for traces (no-op when feature not enabled).
    #[cfg(not(feature = "otel-otlp"))]
    pub fn setup_otlp_exporter(mut self) -> Self {
        tracing::warn!("otel-otlp feature not enabled, OTLP exporter not available");
        self.otlp_enabled = false;
        self
    }

    /// Set up the meter provider for metrics.
    #[cfg(all(feature = "otel-stdout", feature = "otel-otlp"))]
    pub fn setup_meter(mut self, enable_console: bool, enable_otlp: bool) -> Self {
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        tracing::info!("Initializing meter");
        self.meter_enabled = true;

        let resource = self.resource.to_otel_resource();
        let mut builder = SdkMeterProvider::builder().with_resource(resource);

        if enable_console {
            tracing::info!("Enabling console metrics exporter");
            let exporter = opentelemetry_stdout::MetricExporter::default();
            builder = builder.with_periodic_exporter(exporter);
        }

        if enable_otlp {
            tracing::info!("Enabling OTLP metrics exporter");
            match opentelemetry_otlp::MetricExporter::builder()
                .with_tonic()
                .build()
            {
                Ok(exporter) => {
                    builder = builder.with_periodic_exporter(exporter);
                }
                Err(e) => {
                    tracing::error!("error=<{}> | Failed to configure OTLP metrics exporter", e);
                }
            }
        }

        let meter_provider = builder.build();
        opentelemetry::global::set_meter_provider(meter_provider);
        tracing::info!("Strands Meter configured");

        self
    }

    /// Set up the meter provider for metrics (console only).
    #[cfg(all(feature = "otel-stdout", not(feature = "otel-otlp")))]
    pub fn setup_meter(mut self, enable_console: bool, _enable_otlp: bool) -> Self {
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        tracing::info!("Initializing meter");
        self.meter_enabled = true;

        let resource = self.resource.to_otel_resource();
        let mut builder = SdkMeterProvider::builder().with_resource(resource);

        if enable_console {
            tracing::info!("Enabling console metrics exporter");
            let exporter = opentelemetry_stdout::MetricExporter::default();
            builder = builder.with_periodic_exporter(exporter);
        }

        let meter_provider = builder.build();
        opentelemetry::global::set_meter_provider(meter_provider);
        tracing::info!("Strands Meter configured");

        self
    }

    /// Set up the meter provider for metrics (OTLP only).
    #[cfg(all(not(feature = "otel-stdout"), feature = "otel-otlp"))]
    pub fn setup_meter(mut self, _enable_console: bool, enable_otlp: bool) -> Self {
        use opentelemetry_sdk::metrics::SdkMeterProvider;

        tracing::info!("Initializing meter");
        self.meter_enabled = true;

        let resource = self.resource.to_otel_resource();
        let mut builder = SdkMeterProvider::builder().with_resource(resource);

        if enable_otlp {
            tracing::info!("Enabling OTLP metrics exporter");
            match opentelemetry_otlp::MetricExporter::builder()
                .with_tonic()
                .build()
            {
                Ok(exporter) => {
                    builder = builder.with_periodic_exporter(exporter);
                }
                Err(e) => {
                    tracing::error!("error=<{}> | Failed to configure OTLP metrics exporter", e);
                }
            }
        }

        let meter_provider = builder.build();
        opentelemetry::global::set_meter_provider(meter_provider);
        tracing::info!("Strands Meter configured");

        self
    }

    /// Set up the meter provider for metrics (no-op when features not enabled).
    #[cfg(not(any(feature = "otel-stdout", feature = "otel-otlp")))]
    pub fn setup_meter(mut self, _enable_console: bool, _enable_otlp: bool) -> Self {
        tracing::warn!("Neither otel-stdout nor otel-otlp features enabled, meter not available");
        self.meter_enabled = false;
        self
    }

    /// Check if console exporter is enabled.
    pub fn is_console_enabled(&self) -> bool {
        self.console_enabled
    }

    /// Check if OTLP exporter is enabled.
    pub fn is_otlp_enabled(&self) -> bool {
        self.otlp_enabled
    }

    /// Check if meter is enabled.
    pub fn is_meter_enabled(&self) -> bool {
        self.meter_enabled
    }

    /// Get the resource.
    pub fn resource(&self) -> &OtelResource {
        &self.resource
    }

    /// Get the tracer provider.
    pub fn tracer_provider(&self) -> Option<&SdkTracerProvider> {
        self.tracer_provider.as_ref()
    }

    /// Get a tracer from the provider.
    pub fn tracer(&self, name: &'static str) -> Option<opentelemetry_sdk::trace::Tracer> {
        self.tracer_provider.as_ref().map(|p| p.tracer(name))
    }
}

/// Builder for StrandsTelemetry.
pub struct StrandsTelemetryBuilder {
    resource: OtelResource,
    enable_console: bool,
    enable_otlp: bool,
    enable_console_metrics: bool,
    enable_otlp_metrics: bool,
}

impl Default for StrandsTelemetryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StrandsTelemetryBuilder {
    pub fn new() -> Self {
        Self {
            resource: OtelResource::default(),
            enable_console: false,
            enable_otlp: false,
            enable_console_metrics: false,
            enable_otlp_metrics: false,
        }
    }

    pub fn with_resource(mut self, resource: OtelResource) -> Self {
        self.resource = resource;
        self
    }

    pub fn with_console_exporter(mut self) -> Self {
        self.enable_console = true;
        self
    }

    pub fn with_otlp_exporter(mut self) -> Self {
        self.enable_otlp = true;
        self
    }

    pub fn with_console_metrics(mut self) -> Self {
        self.enable_console_metrics = true;
        self
    }

    pub fn with_otlp_metrics(mut self) -> Self {
        self.enable_otlp_metrics = true;
        self
    }

    pub fn build(self) -> StrandsTelemetry {
        let mut telemetry = StrandsTelemetry {
            resource: self.resource,
            tracer_provider: None,
            console_enabled: false,
            otlp_enabled: false,
            meter_enabled: false,
        };

        telemetry.initialize_tracer();

        if self.enable_console {
            telemetry = telemetry.setup_console_exporter();
        }
        if self.enable_otlp {
            telemetry = telemetry.setup_otlp_exporter();
        }
        if self.enable_console_metrics || self.enable_otlp_metrics {
            telemetry = telemetry.setup_meter(self.enable_console_metrics, self.enable_otlp_metrics);
        }

        telemetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otel_resource_default() {
        let resource = OtelResource::default();
        assert_eq!(resource.service_name, "strands-agents");
        assert_eq!(resource.sdk_language, "rust");
    }

    #[test]
    fn test_otel_resource_attributes() {
        let resource = OtelResource::default();
        let attrs = resource.attributes();
        assert!(attrs.contains_key("service.name"));
        assert!(attrs.contains_key("telemetry.sdk.language"));
    }

    #[test]
    fn test_strands_telemetry_builder() {
        let telemetry = StrandsTelemetryBuilder::new()
            .with_console_exporter()
            .build();
        
        assert!(telemetry.tracer_provider().is_some());
    }

    #[test]
    fn test_strands_telemetry_new() {
        let telemetry = StrandsTelemetry::new();
        assert!(telemetry.tracer_provider().is_some());
        assert!(!telemetry.is_console_enabled());
        assert!(!telemetry.is_otlp_enabled());
    }

    #[test]
    fn test_strands_telemetry_chaining() {
        let telemetry = StrandsTelemetry::new()
            .setup_console_exporter();
        
        assert!(telemetry.tracer_provider().is_some());
    }

    #[tokio::test]
    #[cfg(feature = "otel-otlp")]
    async fn test_strands_telemetry_otlp() {
        let telemetry = StrandsTelemetry::new()
            .setup_otlp_exporter();
        
        assert!(telemetry.tracer_provider().is_some());
    }
}
