//! Tool validation and preparation utilities.
//!
//! This module provides functions for validating and preparing tools
//! before they are used in an agent's event loop.

use std::collections::HashSet;

use tracing::{debug, warn};

use crate::types::errors::StrandsError;
use crate::types::tools::{ToolSpec, ToolUse};

/// Maximum allowed length for a tool name.
pub const MAX_TOOL_NAME_LENGTH: usize = 64;

/// Minimum allowed length for a tool name.
pub const MIN_TOOL_NAME_LENGTH: usize = 1;

/// Validates a single tool specification.
///
/// This function checks that the tool spec meets requirements:
/// - Has a valid name (non-empty, proper length, valid characters)
/// - Has a description
/// - Has a valid input schema if provided
///
/// # Arguments
///
/// * `spec` - The tool specification to validate
///
/// # Returns
///
/// `Ok(())` if valid, or an error describing the validation failure.
pub fn validate_tool_spec(spec: &ToolSpec) -> Result<(), StrandsError> {

    if spec.name.is_empty() {
        return Err(StrandsError::InvalidToolName {
            name: spec.name.clone(),
            reason: "Tool name cannot be empty".to_string(),
        });
    }

    if spec.name.len() > MAX_TOOL_NAME_LENGTH {
        return Err(StrandsError::InvalidToolName {
            name: spec.name.clone(),
            reason: format!(
                "Tool name exceeds maximum length of {} characters",
                MAX_TOOL_NAME_LENGTH
            ),
        });
    }


    if !spec.name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
        return Err(StrandsError::InvalidToolName {
            name: spec.name.clone(),
            reason: "Tool name can only contain alphanumeric characters, underscores, and hyphens"
                .to_string(),
        });
    }


    if spec.description.is_empty() {
        warn!(
            tool_name = %spec.name,
            "Tool has empty description, which may reduce LLM effectiveness"
        );
    }

    Ok(())
}

/// Validates a collection of tool specifications.
///
/// Checks each tool and ensures there are no duplicate names.
///
/// # Arguments
///
/// * `specs` - A slice of tool specifications to validate
///
/// # Returns
///
/// `Ok(())` if all tools are valid with no duplicates.
pub fn validate_tool_specs(specs: &[ToolSpec]) -> Result<(), StrandsError> {
    let mut seen_names = HashSet::new();

    for spec in specs {

        validate_tool_spec(spec)?;


        if !seen_names.insert(&spec.name) {
            return Err(StrandsError::DuplicateToolName {
                name: spec.name.clone(),
            });
        }
    }

    debug!(tool_count = specs.len(), "Validated tool specifications");
    Ok(())
}

/// Validates and prepares tools for use in an agent.
///
/// This function performs comprehensive validation on tool specifications
/// and prepares them for use in the agent's event loop. It:
/// - Validates each tool specification
/// - Checks for duplicate names
/// - Optionally filters out invalid tools instead of failing
///
/// # Arguments
///
/// * `specs` - A mutable reference to the tool specifications
/// * `strict` - If true, returns error on first invalid tool; if false, filters them out
///
/// # Returns
///
/// The validated tool specifications.
pub fn validate_and_prepare_tools(
    specs: Vec<ToolSpec>,
    strict: bool,
) -> Result<Vec<ToolSpec>, StrandsError> {
    let mut validated = Vec::with_capacity(specs.len());
    let mut seen_names = HashSet::new();

    for spec in specs {
        match validate_tool_spec(&spec) {
            Ok(()) => {
                if seen_names.insert(spec.name.clone()) {
                    validated.push(spec);
                } else if strict {
                    return Err(StrandsError::DuplicateToolName { name: spec.name });
                } else {
                    warn!(
                        tool_name = %spec.name,
                        "Duplicate tool name found, skipping"
                    );
                }
            }
            Err(e) => {
                if strict {
                    return Err(e);
                }
                warn!(error = %e, "Invalid tool specification, skipping");
            }
        }
    }

    debug!(
        total = validated.len(),
        "Tools validated and prepared"
    );

    Ok(validated)
}

/// Validates a tool use request against registered tools.
///
/// Checks that the tool use references a valid, registered tool
/// and that the input parameters are appropriate.
///
/// # Arguments
///
/// * `tool_use` - The tool use request to validate
/// * `registered_tools` - Names of registered tools
///
/// # Returns
///
/// `Ok(())` if the tool use is valid.
pub fn validate_tool_use(
    tool_use: &ToolUse,
    registered_tools: &HashSet<String>,
) -> Result<(), StrandsError> {

    if !registered_tools.contains(&tool_use.name) {
        return Err(StrandsError::InvalidToolUseName {
            name: tool_use.name.clone(),
            available_tools: registered_tools.iter().cloned().collect(),
        });
    }


    if tool_use.tool_use_id.is_empty() {
        return Err(StrandsError::ToolValidationError {
            message: format!("Tool use '{}' has empty tool_use_id", tool_use.name),
        });
    }

    Ok(())
}

/// Result of tool use validation.
#[derive(Debug, Clone)]
pub struct ToolUseValidationResult {
    /// Valid tool uses that can be executed.
    pub valid: Vec<ToolUse>,
    /// Invalid tool uses with their error reasons.
    pub invalid: Vec<(ToolUse, String)>,
}

impl ToolUseValidationResult {
    /// Returns true if all tool uses were valid.
    pub fn all_valid(&self) -> bool {
        self.invalid.is_empty()
    }

    /// Returns the count of valid tool uses.
    pub fn valid_count(&self) -> usize {
        self.valid.len()
    }

    /// Returns the count of invalid tool uses.
    pub fn invalid_count(&self) -> usize {
        self.invalid.len()
    }
}

/// Validates multiple tool uses at once.
///
/// # Arguments
///
/// * `tool_uses` - The tool uses to validate
/// * `registered_tools` - Names of registered tools
///
/// # Returns
///
/// A result containing both valid and invalid tool uses.
pub fn validate_tool_uses(
    tool_uses: &[ToolUse],
    registered_tools: &HashSet<String>,
) -> ToolUseValidationResult {
    let mut valid = Vec::new();
    let mut invalid = Vec::new();

    for tool_use in tool_uses {
        match validate_tool_use(tool_use, registered_tools) {
            Ok(()) => valid.push(tool_use.clone()),
            Err(e) => invalid.push((tool_use.clone(), e.to_string())),
        }
    }

    ToolUseValidationResult { valid, invalid }
}

/// Checks if a tool name is valid according to naming rules.
///
/// # Arguments
///
/// * `name` - The tool name to check
///
/// # Returns
///
/// `true` if the name is valid.
pub fn is_valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_TOOL_NAME_LENGTH
        && name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

/// Sanitizes a tool name to make it valid.
///
/// Replaces invalid characters with underscores and truncates if needed.
///
/// # Arguments
///
/// * `name` - The tool name to sanitize
///
/// # Returns
///
/// A valid tool name.
pub fn sanitize_tool_name(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();


    if sanitized.len() > MAX_TOOL_NAME_LENGTH {
        sanitized[..MAX_TOOL_NAME_LENGTH].to_string()
    } else if sanitized.is_empty() {
        "unnamed_tool".to_string()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_tool_spec_valid() {
        let spec = ToolSpec::new("valid_tool", "A valid tool");
        assert!(validate_tool_spec(&spec).is_ok());
    }

    #[test]
    fn test_validate_tool_spec_empty_name() {
        let spec = ToolSpec::new("", "Description");
        let result = validate_tool_spec(&spec);
        assert!(result.is_err());
        assert!(matches!(result, Err(StrandsError::InvalidToolName { .. })));
    }

    #[test]
    fn test_validate_tool_spec_invalid_chars() {
        let spec = ToolSpec::new("invalid tool!", "Description");
        let result = validate_tool_spec(&spec);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tool_specs_no_duplicates() {
        let specs = vec![
            ToolSpec::new("tool1", "Tool 1"),
            ToolSpec::new("tool2", "Tool 2"),
        ];
        assert!(validate_tool_specs(&specs).is_ok());
    }

    #[test]
    fn test_validate_tool_specs_with_duplicates() {
        let specs = vec![
            ToolSpec::new("tool1", "Tool 1"),
            ToolSpec::new("tool1", "Tool 1 duplicate"),
        ];
        let result = validate_tool_specs(&specs);
        assert!(result.is_err());
        assert!(matches!(result, Err(StrandsError::DuplicateToolName { .. })));
    }

    #[test]
    fn test_validate_and_prepare_tools_strict() {
        let specs = vec![
            ToolSpec::new("valid", "Valid tool"),
            ToolSpec::new("", "Invalid tool"),
        ];
        let result = validate_and_prepare_tools(specs, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_and_prepare_tools_lenient() {
        let specs = vec![
            ToolSpec::new("valid", "Valid tool"),
            ToolSpec::new("", "Invalid tool"),
        ];
        let result = validate_and_prepare_tools(specs, false).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "valid");
    }

    #[test]
    fn test_validate_tool_use_valid() {
        let tool_use = ToolUse::new("my_tool", "123", serde_json::json!({}));
        let registered = HashSet::from(["my_tool".to_string()]);
        assert!(validate_tool_use(&tool_use, &registered).is_ok());
    }

    #[test]
    fn test_validate_tool_use_not_found() {
        let tool_use = ToolUse::new("unknown", "123", serde_json::json!({}));
        let registered = HashSet::from(["my_tool".to_string()]);
        let result = validate_tool_use(&tool_use, &registered);
        assert!(result.is_err());
        assert!(matches!(result, Err(StrandsError::InvalidToolUseName { .. })));
    }

    #[test]
    fn test_validate_tool_uses_mixed() {
        let tool_uses = vec![
            ToolUse::new("valid", "1", serde_json::json!({})),
            ToolUse::new("invalid", "2", serde_json::json!({})),
        ];
        let registered = HashSet::from(["valid".to_string()]);
        let result = validate_tool_uses(&tool_uses, &registered);
        
        assert_eq!(result.valid_count(), 1);
        assert_eq!(result.invalid_count(), 1);
        assert!(!result.all_valid());
    }

    #[test]
    fn test_is_valid_tool_name() {
        assert!(is_valid_tool_name("valid_tool"));
        assert!(is_valid_tool_name("valid-tool"));
        assert!(is_valid_tool_name("ValidTool123"));
        assert!(!is_valid_tool_name(""));
        assert!(!is_valid_tool_name("invalid tool"));
        assert!(!is_valid_tool_name("invalid!tool"));
    }

    #[test]
    fn test_sanitize_tool_name() {
        assert_eq!(sanitize_tool_name("valid_tool"), "valid_tool");
        assert_eq!(sanitize_tool_name("invalid tool"), "invalid_tool");
        assert_eq!(sanitize_tool_name("test@#$"), "test___");
        assert_eq!(sanitize_tool_name(""), "unnamed_tool");
    }
}

