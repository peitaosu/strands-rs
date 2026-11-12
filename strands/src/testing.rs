//! Test utilities: a scripted mock model.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::models::{Model, ModelConfig, StreamEventStream};
use crate::types::content::{Message, SystemContentBlock};
use crate::types::errors::StrandsError;
use crate::types::streaming::StreamEvent;
use crate::types::tools::{ToolChoice, ToolSpec};

/// One scripted model response.
#[derive(Debug, Clone)]
pub enum MockResponse {
    Events(Vec<StreamEvent>),
    Throttled { message: String },
}

impl MockResponse {
    pub fn text(text: &str) -> Self {
        Self::Events(vec![
            StreamEvent::message_start(crate::types::content::Role::Assistant),
            StreamEvent::text_delta(0, text),
            StreamEvent::content_block_stop(0),
            StreamEvent::message_stop(crate::types::streaming::StopReason::EndTurn),
        ])
    }

    pub fn tool_use(name: &str, tool_use_id: &str, input: serde_json::Value) -> Self {
        Self::Events(vec![
            StreamEvent::message_start(crate::types::content::Role::Assistant),
            StreamEvent::tool_use_start(0, name, tool_use_id),
            StreamEvent::tool_use_delta(0, serde_json::to_string(&input).unwrap_or_default()),
            StreamEvent::content_block_stop(0),
            StreamEvent::message_stop(crate::types::streaming::StopReason::ToolUse),
        ])
    }

    pub fn throttled() -> Self {
        Self::Throttled { message: "ThrottlingException".to_string() }
    }
}

/// A model that replays a scripted list of responses, one per call.
///
/// The last response repeats when the script is exhausted, so multi-turn
/// agents keep producing the same output until a stop condition hits.
#[derive(Clone, Default)]
pub struct MockModel {
    config: ModelConfig,
    responses: Arc<Mutex<Vec<MockResponse>>>,
    calls: Arc<AtomicUsize>,
}

impl MockModel {
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            config: ModelConfig::default(),
            responses: Arc::new(Mutex::new(responses)),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Number of model calls made so far.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Model for MockModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn update_config(&mut self, config: ModelConfig) {
        self.config = config;
    }

    fn stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _tool_specs: Option<&'a [ToolSpec]>,
        _system_prompt: Option<&'a str>,
        _tool_choice: Option<ToolChoice>,
        _system_prompt_content: Option<&'a [SystemContentBlock]>,
    ) -> StreamEventStream<'a> {
        self.calls.fetch_add(1, Ordering::SeqCst);

        let response = {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                MockResponse::text("")
            } else if responses.len() > 1 {
                responses.remove(0)
            } else {
                responses[0].clone()
            }
        };

        match response {
            MockResponse::Events(events) => {
                Box::pin(futures::stream::iter(events.into_iter().map(Ok)))
            }
            MockResponse::Throttled { message } => {
                Box::pin(futures::stream::iter(vec![Err(StrandsError::ModelThrottled { message })]))
            }
        }
    }
}
