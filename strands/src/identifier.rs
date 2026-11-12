//! Strands identifier utilities.

use std::path::Path;

/// Strands identifier types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Identifier {
    Agent,
    Session,
}

impl Identifier {
    /// Returns the string value of the identifier type.
    pub fn value(&self) -> &'static str {
        match self {
            Identifier::Agent => "agent",
            Identifier::Session => "session",
        }
    }
}

impl std::fmt::Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

/// Validate strands id.
///
/// # Arguments
///
/// * `id` - Id to validate.
/// * `type_` - Type of the identifier (e.g., session id, agent id, etc.)
///
/// # Returns
///
/// Validated id.
///
/// # Errors
///
/// Returns an error if the id contains path separators.
pub fn validate(id: &str, type_: Identifier) -> Result<&str, String> {

    let filename = Path::new(id)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");

    if filename != id {
        return Err(format!(
            "{}_id={} | id cannot contain path separators",
            type_.value(),
            id
        ));
    }

    Ok(id)
}

/// Generate a new monotonic UUIDv7 identifier.
///
/// Mirrors the Python `_identifier.new_uuid7` used for message tracking ids.
pub fn new_uuid7() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Returns true when the value is a parseable UUIDv7.
pub fn is_uuid7(value: &str) -> bool {
    uuid::Uuid::parse_str(value)
        .map(|u| u.get_version_num() == 7)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_id() {
        assert_eq!(validate("my-agent", Identifier::Agent).unwrap(), "my-agent");
        assert_eq!(validate("session123", Identifier::Session).unwrap(), "session123");
    }

    #[test]
    fn test_validate_invalid_id_with_path_separator() {
        assert!(validate("a/b", Identifier::Agent).is_err());
        assert!(validate("path/to/session", Identifier::Session).is_err());
    }

    #[test]
    fn test_identifier_display() {
        assert_eq!(Identifier::Agent.to_string(), "agent");
        assert_eq!(Identifier::Session.to_string(), "session");
    }
}

