//! Structured output tool for validating and returning typed responses.
//!
//! Provides utilities for converting Rust types to tool specifications
//! and forcing the model to output structured JSON data.

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::types::tools::ToolSpec;

/// Converts a type implementing JsonSchema to a ToolSpec.
pub fn schema_to_tool_spec<T: JsonSchema>(name: &str, description: &str) -> ToolSpec {
    let schema = schemars::schema_for!(T);
    let mut json_schema = serde_json::to_value(schema).unwrap_or_default();

    json_schema = flatten_schema(&json_schema);

    ToolSpec::new(name, description).with_input_schema(json_schema)
}

/// Creates a structured output tool spec from a type.
pub fn structured_output_spec<T: JsonSchema>() -> ToolSpec {
    let name = std::any::type_name::<T>()
        .split("::")
        .last()
        .unwrap_or("StructuredOutput")
        .to_string();

    let description = "IMPORTANT: This StructuredOutputTool should only be invoked as the last and final tool \
         before returning the completed result to the caller.".to_string();

    schema_to_tool_spec::<T>(&name, &description)
}

/// Result of structured output parsing.
#[derive(Debug)]
pub struct StructuredOutputResult<T> {
    /// The parsed value.
    pub value: T,
    /// The raw JSON value.
    pub raw_json: Value,
}

impl<T: DeserializeOwned> StructuredOutputResult<T> {
    /// Parses a structured output from JSON.
    pub fn from_json(json: Value) -> Result<Self, serde_json::Error> {
        let value: T = serde_json::from_value(json.clone())?;
        Ok(Self { value, raw_json: json })
    }

    /// Parses a structured output from a JSON string.
    pub fn from_str(s: &str) -> Result<Self, serde_json::Error> {
        let json: Value = serde_json::from_str(s)?;
        Self::from_json(json)
    }
}

/// Flattens a JSON schema by resolving $ref references.
pub fn flatten_schema(schema: &Value) -> Value {
    let mut result = schema.clone();

    let defs_opt = result
        .as_object_mut()
        .and_then(|obj| obj.remove("$defs").or_else(|| obj.remove("definitions")));

    if let Some(defs) = defs_opt {
        resolve_refs(&mut result, &defs);
    }

    result
}

fn resolve_refs(value: &mut Value, defs: &Value) {
    match value {
        Value::Object(obj) => {
            if let Some(ref_val) = obj.remove("$ref") {
                if let Some(ref_str) = ref_val.as_str() {
                    let ref_name = ref_str.split('/').last().unwrap_or("");
                    if let Some(def) = defs.get(ref_name) {
                        let mut resolved = def.clone();
                        resolve_refs(&mut resolved, defs);
                        *value = resolved;
                        return;
                    }
                }
            }

            for (_, v) in obj.iter_mut() {
                resolve_refs(v, defs);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                resolve_refs(item, defs);
            }
        }
        _ => {}
    }
}

/// Processes a schema to handle optional fields properly.
pub fn process_schema_for_optional_fields(schema: &mut Value, required_fields: &[String]) {
    if let Some(obj) = schema.as_object_mut() {
        if let Some(Value::Object(properties)) = obj.get_mut("properties") {
            for (prop_name, prop_value) in properties.iter_mut() {
                let is_required = required_fields.contains(prop_name);
                process_property(prop_value, is_required);
            }
        }
    }
}

fn process_property(prop: &mut Value, is_required: bool) {
    if let Some(obj) = prop.as_object_mut() {
        if let Some(any_of) = obj.remove("anyOf") {
            if let Some(any_of_arr) = any_of.as_array() {
                let mut null_type = false;
                let mut non_null_type: Option<Value> = None;

                for option in any_of_arr {
                    if option.get("type") == Some(&Value::String("null".to_string())) {
                        null_type = true;
                    } else {
                        non_null_type = Some(option.clone());
                    }
                }

                if null_type && non_null_type.is_some() {
                    let non_null = non_null_type.unwrap();
                    if let Some(non_null_obj) = non_null.as_object() {
                        for (k, v) in non_null_obj {
                            obj.insert(k.clone(), v.clone());
                        }
                    }

                    if let Some(type_val) = obj.get_mut("type") {
                        if let Some(type_str) = type_val.as_str() {
                            *type_val = Value::Array(vec![
                                Value::String(type_str.to_string()),
                                Value::String("null".to_string()),
                            ]);
                        }
                    } else {
                        obj.insert(
                            "type".to_string(),
                            Value::Array(vec![
                                Value::String("object".to_string()),
                                Value::String("null".to_string()),
                            ]),
                        );
                    }
                }
            }
        } else if !is_required {
            if let Some(type_val) = obj.get_mut("type") {
                if let Some(type_str) = type_val.as_str() {
                    if type_str != "null" {
                        *type_val = Value::Array(vec![
                            Value::String(type_str.to_string()),
                            Value::String("null".to_string()),
                        ]);
                    }
                }
            }
        }

        let nested_required: Vec<String> = obj
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        if let Some(Value::Object(nested_props)) = obj.get_mut("properties") {
            for (prop_name, prop_value) in nested_props.iter_mut() {
                let is_req = nested_required.contains(prop_name);
                process_property(prop_value, is_req);
            }
        }
    }
}

/// Extracts required fields from a schema.
pub fn get_required_fields(schema: &Value) -> Vec<String> {
    schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Validates that a value conforms to a schema (basic validation).
pub fn validate_against_schema(value: &Value, schema: &Value) -> Result<(), String> {
    if let Some(schema_obj) = schema.as_object() {
        if let Some(type_val) = schema_obj.get("type") {
            let types: Vec<&str> = match type_val {
                Value::String(s) => vec![s.as_str()],
                Value::Array(arr) => arr.iter().filter_map(|v| v.as_str()).collect(),
                _ => vec![],
            };

            let value_type = match value {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            };

            let type_matches = types.iter().any(|t| {
                *t == value_type || (*t == "number" && value_type == "integer")
            });

            if !type_matches && !types.is_empty() {
                return Err(format!(
                    "Expected type {:?}, got {}",
                    types, value_type
                ));
            }
        }

        if let Some(Value::Object(properties)) = schema_obj.get("properties") {
            if let Some(value_obj) = value.as_object() {
                let required = get_required_fields(schema);

                for req_field in &required {
                    if !value_obj.contains_key(req_field) {
                        return Err(format!("Missing required field: {}", req_field));
                    }
                }

                for (prop_name, prop_schema) in properties {
                    if let Some(prop_value) = value_obj.get(prop_name) {
                        validate_against_schema(prop_value, prop_schema)?;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Creates a structured output tool that wraps a given output type.
pub struct StructuredOutputTool<T: JsonSchema + DeserializeOwned> {
    spec: ToolSpec,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: JsonSchema + DeserializeOwned> StructuredOutputTool<T> {
    /// Creates a new structured output tool.
    pub fn new() -> Self {
        let spec = structured_output_spec::<T>();
        Self {
            spec,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new structured output tool with a custom name and description.
    pub fn with_name_description(name: &str, description: &str) -> Self {
        let spec = schema_to_tool_spec::<T>(name, description);
        Self {
            spec,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the tool specification.
    pub fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    /// Parses the tool input into the output type.
    pub fn parse(&self, input: &Value) -> Result<T, serde_json::Error> {
        serde_json::from_value(input.clone())
    }
}

impl<T: JsonSchema + DeserializeOwned> Default for StructuredOutputTool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A type-erased structured output tool that can be registered with the registry.
pub struct StructuredOutputAgentTool {
    spec: ToolSpec,
}

impl StructuredOutputAgentTool {
    /// Creates a new structured output agent tool from a type.
    pub fn from_type<T: JsonSchema + DeserializeOwned>() -> Self {
        Self {
            spec: structured_output_spec::<T>(),
        }
    }

    /// Creates a new structured output agent tool from a spec.
    pub fn from_spec(spec: ToolSpec) -> Self {
        Self { spec }
    }
}

#[async_trait::async_trait]
impl super::AgentTool for StructuredOutputAgentTool {
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn description(&self) -> &str {
        &self.spec.description
    }

    fn tool_spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn tool_type(&self) -> &str {
        "structured_output"
    }

    async fn invoke(
        &self,
        input: Value,
        _context: &super::ToolContext,
    ) -> std::result::Result<super::ToolResult2, String> {

        Ok(super::ToolResult2::success_json(input))
    }
}

/// Per-invocation context for structured output execution.
#[derive(Debug, Default, Clone)]
pub struct StructuredOutputContext {
    /// Stored results by tool use ID.
    results: std::collections::HashMap<String, Value>,
    /// Expected structured output tool name.
    expected_tool_name: Option<String>,
    /// The tool specification for registration.
    tool_spec: Option<ToolSpec>,
    /// Whether structured output is enabled.
    is_enabled: bool,
    /// Whether forced mode is active.
    pub forced_mode: bool,
    /// Whether force was attempted.
    pub force_attempted: bool,
    /// Whether to stop the event loop.
    pub stop_loop: bool,
}

/// Default prompt appended when forcing structured output.
pub const DEFAULT_STRUCTURED_OUTPUT_PROMPT: &str =
    "You must format the previous response as structured output.";

impl StructuredOutputContext {
    /// Creates a new structured output context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new structured output context with a specific output type.
    pub fn with_type<T: JsonSchema + DeserializeOwned>() -> Self {
        let spec = structured_output_spec::<T>();
        let name = spec.name.clone();

        Self {
            results: std::collections::HashMap::new(),
            expected_tool_name: Some(name),
            tool_spec: Some(spec),
            is_enabled: true,
            forced_mode: false,
            force_attempted: false,
            stop_loop: false,
        }
    }

    /// Creates a new structured output context with a specific tool name and spec.
    pub fn with_tool_name(name: impl Into<String>, spec: Option<ToolSpec>) -> Self {
        Self {
            results: std::collections::HashMap::new(),
            expected_tool_name: Some(name.into()),
            tool_spec: spec,
            is_enabled: true,
            forced_mode: false,
            force_attempted: false,
            stop_loop: false,
        }
    }

    /// Returns the tool specification if available.
    pub fn get_tool_spec(&self) -> Option<&ToolSpec> {
        self.tool_spec.as_ref()
    }

    /// Registers the structured output tool with the given registry.
    ///
    /// Returns true if a tool was registered, false otherwise.
    pub fn register_tool(&self, registry: &mut super::ToolRegistry) -> bool {
        if let Some(ref spec) = self.tool_spec {
            let tool = StructuredOutputAgentTool::from_spec(spec.clone());
            if registry.register_dynamic(tool).is_ok() {
                tracing::debug!("Registered structured output tool: {}", spec.name);
                return true;
            }
        }
        false
    }

    /// Removes the structured output tool from the given registry.
    pub fn cleanup(&self, registry: &mut super::ToolRegistry) {
        if let Some(ref name) = self.expected_tool_name {
            if registry.remove_dynamic(name) {
                tracing::debug!("Cleaned up structured output tool: {}", name);
            }
        }
    }

    /// Check if structured output is enabled for this context.
    pub fn is_enabled(&self) -> bool {
        self.is_enabled
    }

    /// Get the expected tool name.
    pub fn expected_tool_name(&self) -> Option<&str> {
        self.expected_tool_name.as_deref()
    }

    /// Store a validated structured output result.
    pub fn store_result(&mut self, tool_use_id: &str, result: Value) {
        self.results.insert(tool_use_id.to_string(), result);
    }

    /// Retrieve a stored structured output result.
    pub fn get_result(&self, tool_use_id: &str) -> Option<&Value> {
        self.results.get(tool_use_id)
    }

    /// Mark this context as being in forced structured output mode.
    pub fn set_forced_mode(&mut self) {
        if !self.is_enabled {
            return;
        }
        self.forced_mode = true;
        self.force_attempted = true;
    }

    /// Check if any tool uses are for the structured output tool.
    pub fn has_structured_output_tool(&self, tool_names: &[String]) -> bool {
        if let Some(expected) = &self.expected_tool_name {
            tool_names.iter().any(|name| name == expected)
        } else {
            false
        }
    }

    /// Extract and remove structured output result from stored results.
    pub fn extract_result(&mut self, tool_use_ids: &[String]) -> Option<Value> {
        for id in tool_use_ids {
            if let Some(result) = self.results.remove(id) {
                return Some(result);
            }
        }
        None
    }

    /// Extracts a structured output from the message text as a fallback.
    ///
    /// Mirrors the Python forced-retry text validation: takes the last text
    /// block, strips code fences, parses a JSON object, and validates it
    /// against the tool spec when one is available.
    pub fn result_from_text(&self, message: &crate::types::content::Message) -> Option<Value> {
        let text = message
            .content
            .iter()
            .rev()
            .filter_map(|block| block.as_text())
            .find(|t| !t.trim().is_empty())?;

        let cleaned = strip_code_fences(text);
        let value: Value = serde_json::from_str(cleaned).ok()?;
        if !value.is_object() {
            return None;
        }

        if let Some(spec) = &self.tool_spec {
            let schema = flatten_schema(&spec.input_schema.json);
            validate_against_schema(&value, &schema).ok()?;
        }

        Some(value)
    }
}

/// Strips a wrapping markdown code fence from JSON text.
fn strip_code_fences(text: &str) -> &str {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let rest = rest.trim_start_matches(|c: char| c.is_ascii_alphanumeric());
        let rest = rest.trim_start_matches(['\r', '\n']);
        if let Some(end) = rest.rfind("```") {
            return rest[..end].trim();
        }
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct TestOutput {
        name: String,
        count: i32,
    }

    #[test]
    fn test_schema_to_tool_spec() {
        let spec = schema_to_tool_spec::<TestOutput>("test_output", "A test output type");
        assert_eq!(spec.name, "test_output");
        assert!(spec.input_schema.json.get("properties").is_some());
    }

    #[test]
    fn test_structured_output_result() {
        let json = serde_json::json!({
            "name": "test",
            "count": 42
        });

        let result: StructuredOutputResult<TestOutput> =
            StructuredOutputResult::from_json(json).unwrap();
        assert_eq!(result.value.name, "test");
        assert_eq!(result.value.count, 42);
    }

    #[test]
    fn test_flatten_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "inner": { "$ref": "#/$defs/InnerType" }
            },
            "$defs": {
                "InnerType": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    }
                }
            }
        });

        let flattened = flatten_schema(&schema);
        let inner = flattened.get("properties").unwrap().get("inner").unwrap();
        assert!(inner.get("properties").is_some());
    }

    #[test]
    fn test_validate_against_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "count": { "type": "integer" }
            },
            "required": ["name"]
        });

        let valid_value = serde_json::json!({
            "name": "test",
            "count": 42
        });

        assert!(validate_against_schema(&valid_value, &schema).is_ok());

        let invalid_value = serde_json::json!({
            "count": 42
        });

        assert!(validate_against_schema(&invalid_value, &schema).is_err());
    }

    #[test]
    fn test_structured_output_tool() {
        let tool = StructuredOutputTool::<TestOutput>::new();
        let spec = tool.spec();
        assert!(spec.name.contains("TestOutput"));

        let input = serde_json::json!({
            "name": "test",
            "count": 42
        });

        let parsed = tool.parse(&input).unwrap();
        assert_eq!(parsed.name, "test");
        assert_eq!(parsed.count, 42);
    }

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct NestedOutput {
        inner: InnerType,
        optional_field: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct InnerType {
        value: String,
    }

    #[test]
    fn test_nested_type_flattening() {
        let spec = schema_to_tool_spec::<NestedOutput>("nested", "Nested output");
        let schema = &spec.input_schema.json;

        let properties = schema.get("properties").unwrap();
        let inner_prop = properties.get("inner").unwrap();

        assert!(inner_prop.get("properties").is_some());
    }
}
