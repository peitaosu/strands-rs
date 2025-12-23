//! Interrupt type definitions for human-in-the-loop workflows.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// An interrupt for pausing execution and requesting human input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interrupt {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<serde_json::Value>,
}

impl Interrupt {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            reason: None,
            response: None,
        }
    }

    pub fn with_reason(mut self, reason: serde_json::Value) -> Self {
        self.reason = Some(reason);
        self
    }

    pub fn with_response(mut self, response: serde_json::Value) -> Self {
        self.response = Some(response);
        self
    }

    pub fn has_response(&self) -> bool {
        self.response.is_some()
    }
}

/// State for managing interrupts during agent execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterruptState {
    /// Interrupts raised by the user.
    pub interrupts: HashMap<String, Interrupt>,
    /// Additional context associated with an interrupt event.
    pub context: HashMap<String, serde_json::Value>,
    /// True if agent is in an interrupt state, False otherwise.
    pub activated: bool,
}

impl InterruptState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Activate the interrupt state.
    pub fn activate(&mut self) {
        self.activated = true;
    }

    /// Deactivate the interrupt state.
    ///
    /// Interrupts and context are cleared.
    pub fn deactivate(&mut self) {
        self.interrupts.clear();
        self.context.clear();
        self.activated = false;
    }

    /// Configure the interrupt state if resuming from an interrupt event.
    pub fn resume(&mut self, responses: Vec<InterruptResponseContent>) -> Result<(), String> {
        if !self.activated {
            return Ok(());
        }

        for content in &responses {
            let interrupt_id = &content.interrupt_response.interrupt_id;
            let interrupt_response = &content.interrupt_response.response;

            if let Some(interrupt) = self.interrupts.get_mut(interrupt_id) {
                interrupt.response = Some(interrupt_response.clone());
            } else {
                return Err(format!("interrupt_id=<{}> | no interrupt found", interrupt_id));
            }
        }

        self.context.insert(
            "responses".to_string(),
            serde_json::to_value(&responses).unwrap_or_default(),
        );

        Ok(())
    }

    pub fn add(&mut self, interrupt: Interrupt) {
        self.interrupts.insert(interrupt.id.clone(), interrupt);
    }

    pub fn get(&self, id: &str) -> Option<&Interrupt> {
        self.interrupts.get(id)
    }

    pub fn get_response(&self, id: &str) -> Option<&serde_json::Value> {
        self.interrupts.get(id).and_then(|i| i.response.as_ref())
    }

    pub fn set_response(&mut self, id: &str, response: serde_json::Value) {
        if let Some(interrupt) = self.interrupts.get_mut(id) {
            interrupt.response = Some(response);
        }
    }

    pub fn pending_interrupts(&self) -> Vec<&Interrupt> {
        self.interrupts
            .values()
            .filter(|i| i.response.is_none())
            .collect()
    }

    pub fn has_pending(&self) -> bool {
        self.interrupts.values().any(|i| i.response.is_none())
    }

    pub fn to_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut dict = HashMap::new();
        dict.insert(
            "interrupts".to_string(),
            serde_json::json!(self.interrupts
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::to_value(v).unwrap_or_default()))
                .collect::<HashMap<_, _>>()),
        );
        dict.insert("context".to_string(), serde_json::json!(self.context));
        dict.insert("activated".to_string(), serde_json::json!(self.activated));
        dict
    }

    pub fn from_dict(data: HashMap<String, serde_json::Value>) -> Self {
        let interrupts = data
            .get("interrupts")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        serde_json::from_value::<Interrupt>(v.clone())
                            .ok()
                            .map(|i| (k.clone(), i))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let context = data
            .get("context")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let activated = data
            .get("activated")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            interrupts,
            context,
            activated,
        }
    }
}

/// User response to an interrupt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InterruptResponse {
    pub interrupt_id: String,
    pub response: serde_json::Value,
}

/// Content block containing a user response to an interrupt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InterruptResponseContent {
    pub interrupt_response: InterruptResponse,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interrupt_creation() {
        let interrupt = Interrupt::new("int-1", "approval")
            .with_reason(serde_json::json!({"type": "delete"}));

        assert_eq!(interrupt.id, "int-1");
        assert_eq!(interrupt.name, "approval");
        assert!(interrupt.reason.is_some());
        assert!(!interrupt.has_response());
    }

    #[test]
    fn test_interrupt_state() {
        let mut state = InterruptState::new();
        state.add(Interrupt::new("int-1", "approval"));
        state.add(Interrupt::new("int-2", "confirmation"));

        assert!(state.has_pending());
        assert_eq!(state.pending_interrupts().len(), 2);

        state.set_response("int-1", serde_json::json!("approved"));
        assert_eq!(state.pending_interrupts().len(), 1);
    }
}

