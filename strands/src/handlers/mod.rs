//! Handlers for formatting and displaying events from the agent.

use std::sync::Arc;

/// Callback handler event data.
#[derive(Debug, Clone, Default)]
pub struct CallbackEvent {
    pub reasoning_text: Option<String>,
    pub data: Option<String>,
    pub complete: bool,
    pub current_tool_use: Option<CurrentToolUse>,
}

/// Current tool use information.
#[derive(Debug, Clone, PartialEq)]
pub struct CurrentToolUse {
    pub name: String,
    pub tool_use_id: Option<String>,
    pub input: Option<serde_json::Value>,
}

/// Trait for callback handlers.
pub trait CallbackHandler: Send + Sync {
    /// Handle a callback event.
    fn call(&mut self, event: &CallbackEvent);
}

/// Handler for streaming text output and tool invocations to stdout.
pub struct PrintingCallbackHandler {
    tool_count: u32,
    previous_tool_use: Option<CurrentToolUse>,
    verbose_tool_use: bool,
}

impl PrintingCallbackHandler {
    /// Create a new printing callback handler.
    pub fn new(verbose_tool_use: bool) -> Self {
        Self {
            tool_count: 0,
            previous_tool_use: None,
            verbose_tool_use,
        }
    }
}

impl Default for PrintingCallbackHandler {
    fn default() -> Self {
        Self::new(true)
    }
}

impl CallbackHandler for PrintingCallbackHandler {
    fn call(&mut self, event: &CallbackEvent) {
        if let Some(ref reasoning_text) = event.reasoning_text {
            print!("{}", reasoning_text);
        }

        if let Some(ref data) = event.data {
            if event.complete {
                println!("{}", data);
            } else {
                print!("{}", data);
            }
        }

        if let Some(ref current_tool_use) = event.current_tool_use {
            if self.previous_tool_use.as_ref() != Some(current_tool_use) {
                self.previous_tool_use = Some(current_tool_use.clone());
                self.tool_count += 1;
                if self.verbose_tool_use {
                    println!("\nTool #{}: {}", self.tool_count, current_tool_use.name);
                }
            }
        }

        if event.complete && event.data.is_some() {
            println!();
        }
    }
}

/// Callback handler that combines multiple callback handlers.
pub struct CompositeCallbackHandler {
    handlers: Vec<Arc<std::sync::Mutex<dyn CallbackHandler>>>,
}

impl CompositeCallbackHandler {
    /// Create a new composite callback handler.
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Add a handler to the composite.
    pub fn add_handler(&mut self, handler: impl CallbackHandler + 'static) {
        self.handlers.push(Arc::new(std::sync::Mutex::new(handler)));
    }
}

impl Default for CompositeCallbackHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CallbackHandler for CompositeCallbackHandler {
    fn call(&mut self, event: &CallbackEvent) {
        for handler in &self.handlers {
            if let Ok(mut h) = handler.lock() {
                h.call(event);
            }
        }
    }
}

/// Callback handler that discards all output.
pub struct NullCallbackHandler;

impl CallbackHandler for NullCallbackHandler {
    fn call(&mut self, _event: &CallbackEvent) {
    }
}

/// Convenience function to create a null callback handler.
pub fn null_callback_handler() -> NullCallbackHandler {
    NullCallbackHandler
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_printing_callback_handler() {
        let mut handler = PrintingCallbackHandler::new(true);
        
        let event = CallbackEvent {
            data: Some("Hello".to_string()),
            complete: false,
            ..Default::default()
        };
        
        handler.call(&event);
    }

    #[test]
    fn test_null_callback_handler() {
        let mut handler = null_callback_handler();
        
        let event = CallbackEvent {
            data: Some("Should be ignored".to_string()),
            ..Default::default()
        };
        
        handler.call(&event);
    }

    #[test]
    fn test_composite_callback_handler() {
        let mut composite = CompositeCallbackHandler::new();
        composite.add_handler(NullCallbackHandler);
        
        let event = CallbackEvent::default();
        composite.call(&event);
    }
}

