//! Router, candidate, context, and strategy protocol.

use std::sync::Arc;

use async_trait::async_trait;
use futures::{Stream, StreamExt};

use crate::models::{Model, StreamEventStream};
use crate::types::content::Message;
use crate::types::errors::StrandsError;
use crate::types::streaming::StreamEvent;
use crate::types::tools::ToolSpec;

/// A model the router may route an invocation to.
#[derive(Clone)]
pub struct RoutingCandidate {
    /// Stable identifier used to correlate attempts with candidates.
    pub key: String,
    /// Human-readable name for classifier profiles.
    pub name: Option<String>,
    /// Description of what the candidate is good at, for classifier profiles.
    pub description: Option<String>,
    /// Caller-authored evidence carried to classifier strategies.
    pub metadata: serde_json::Value,
    /// The backing model.
    pub model: Arc<dyn Model>,
}

impl RoutingCandidate {
    pub fn new(key: impl Into<String>, model: Arc<dyn Model>) -> Self {
        Self {
            key: key.into(),
            name: None,
            description: None,
            metadata: serde_json::Value::Null,
            model,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// One candidate this invocation already used, and how the attempt ended.
#[derive(Debug, Clone)]
pub struct RoutingAttempt {
    /// The `RoutingCandidate::key` of the attempted candidate.
    pub candidate_key: String,
    /// `None` when the call succeeded, otherwise the model error.
    pub exception: Option<String>,
}

/// Read-only inputs a strategy sees when choosing a candidate.
pub struct RoutingContext<'a> {
    pub messages: &'a [Message],
    pub system_prompt: Option<&'a str>,
    pub tool_specs: Vec<ToolSpec>,
    pub candidates: &'a [RoutingCandidate],
    pub attempts: &'a [RoutingAttempt],
}

impl<'a> RoutingContext<'a> {
    /// The trailing run of failed attempts since the last success.
    pub fn attempts_since_success(&self) -> &[RoutingAttempt] {
        let mut index = self.attempts.len();
        while index > 0 && self.attempts[index - 1].exception.is_some() {
            index -= 1;
        }
        &self.attempts[index..]
    }
}

/// Chooses the candidate an invocation uses, including after a failure.
///
/// `select` is asked once before the first model call (empty attempts) and
/// again after each failed call. Returning `None` declines: the opening ask
/// then serves the request on the first candidate, and a later ask ends
/// routing so the model error surfaces.
#[async_trait]
pub trait RoutingStrategy: Send + Sync {
    /// Returns the index into `context.candidates` to use, or `None` to decline.
    async fn select(&self, context: &RoutingContext<'_>) -> Option<usize>;
}

/// Routes invocations among candidates through a strategy.
pub struct ModelRouter {
    candidates: Vec<RoutingCandidate>,
    strategy: Arc<dyn RoutingStrategy>,
}

impl ModelRouter {
    /// Creates a router over `candidates` with the default `FallbackStrategy`.
    pub fn new(candidates: Vec<RoutingCandidate>) -> Self {
        Self::with_strategy(candidates, Arc::new(super::fallback::FallbackStrategy))
    }

    /// Creates a router with an explicit strategy.
    pub fn with_strategy(candidates: Vec<RoutingCandidate>, strategy: Arc<dyn RoutingStrategy>) -> Self {
        Self {
            candidates,
            strategy,
        }
    }

    pub fn candidates(&self) -> &[RoutingCandidate] {
        &self.candidates
    }

    /// The first candidate serves as the router's default model.
    pub fn default_model(&self) -> Option<&Arc<dyn Model>> {
        self.candidates.first().map(|candidate| &candidate.model)
    }

    /// Asks the strategy for the candidate to use.
    pub async fn select_candidate(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        tool_specs: Vec<ToolSpec>,
        attempts: &[RoutingAttempt],
    ) -> Option<usize> {
        let context = RoutingContext {
            messages,
            system_prompt,
            tool_specs,
            candidates: &self.candidates,
            attempts,
        };
        self.strategy.select(&context).await
    }
}

#[async_trait]
impl Model for ModelRouter {
    fn config(&self) -> &crate::models::ModelConfig {
        self.candidates
            .first()
            .map(|candidate| candidate.model.config())
            .unwrap_or_else(|| {
                static EMPTY: std::sync::OnceLock<crate::models::ModelConfig> = std::sync::OnceLock::new();
                EMPTY.get_or_init(crate::models::ModelConfig::default)
            })
    }

    fn update_config(&mut self, _config: crate::models::ModelConfig) {
        tracing::warn!("ModelRouter.update_config is a no-op; configure the candidates directly");
    }

    async fn count_tokens(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
    ) -> Result<u64, StrandsError> {
        match self.default_model() {
            Some(model) => model.count_tokens(messages, system_prompt).await,
            None => Ok(crate::models::estimate_tokens(messages, system_prompt)),
        }
    }

    /// Streams through the selected candidate, failing over on stream errors
    /// before any event was produced.
    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        tool_specs: Option<&'a [ToolSpec]>,
        system_prompt: Option<&'a str>,
        tool_choice: Option<crate::types::tools::ToolChoice>,
        system_prompt_content: Option<&'a [crate::types::content::SystemContentBlock]>,
    ) -> StreamEventStream<'a> {
        let specs = tool_specs.map(|s| s.to_vec()).unwrap_or_default();
        let choice = tool_choice;

        Box::pin(async_stream::stream! {
            let mut attempts: Vec<RoutingAttempt> = Vec::new();

            loop {
                let selected = self
                    .select_candidate(messages, system_prompt, specs.clone(), &attempts)
                    .await;

                let Some(index) = selected else {
                    if attempts.is_empty() {
                        // Opening decline: serve on the default model.
                        if let Some(model) = self.default_model() {
                            for await event in model.stream(messages, tool_specs, system_prompt, choice.clone(), system_prompt_content) {
                                yield event;
                            }
                        } else {
                            yield Err(StrandsError::ConfigurationError {
                                message: "ModelRouter has no candidates".to_string(),
                            });
                        }
                    } else {
                        // Later decline: surface the pending model error.
                        if let Some(last) = attempts.last() {
                            if let Some(exception) = &last.exception {
                                yield Err(StrandsError::model_error(exception.clone()));
                            }
                        }
                    }
                    return;
                };

                let candidate = match self.candidates.get(index) {
                    Some(candidate) => candidate.clone(),
                    None => {
                        attempts.push(RoutingAttempt {
                            candidate_key: index.to_string(),
                            exception: Some(format!("no candidate at index {}", index)),
                        });
                        continue;
                    }
                };

                let mut produced_event = false;
                let mut failed: Option<String> = None;
                let mut events =
                    candidate.model.stream(messages, tool_specs, system_prompt, choice.clone(), system_prompt_content);

                while let Some(event) = events.next().await {
                    match event {
                        Ok(event) => {
                            produced_event = true;
                            yield Ok(event);
                        }
                        Err(error) => {
                            failed = Some(error.to_string());
                            break;
                        }
                    }
                }

                match failed {
                    None => return,
                    Some(exception) => {
                        attempts.push(RoutingAttempt {
                            candidate_key: candidate.key.clone(),
                            exception: Some(exception),
                        });
                        if produced_event {
                            // Mid-stream failure after output: surface it.
                            if let Some(last) = attempts.last() {
                                if let Some(exception) = &last.exception {
                                    yield Err(StrandsError::model_error(exception.clone()));
                                }
                            }
                            return;
                        }
                    }
                }
            }
        })
    }
}

/// Type alias keeping the public stream signature readable.
pub type RoutedStream<'a> = std::pin::Pin<Box<dyn Stream<Item = Result<StreamEvent, StrandsError>> + Send + 'a>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockModel;
    use crate::testing::{MockModel as Scripted, MockResponse};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn test_fallback_prefers_declaration_order() {
        let a: Arc<dyn Model> = Arc::new(MockModel::new(vec![]));
        let b: Arc<dyn Model> = Arc::new(MockModel::new(vec![]));
        let candidates = vec![
            RoutingCandidate::new("a", a),
            RoutingCandidate::new("b", b),
        ];
        let router = ModelRouter::new(candidates);

        let selected = router.select_candidate(&[], None, vec![], &[]).await;
        assert_eq!(selected, Some(0));
    }

    #[tokio::test]
    async fn test_fallback_skips_tried_candidates() {
        let a: Arc<dyn Model> = Arc::new(MockModel::new(vec![]));
        let b: Arc<dyn Model> = Arc::new(MockModel::new(vec![]));
        let candidates = vec![
            RoutingCandidate::new("a", a),
            RoutingCandidate::new("b", b),
        ];
        let router = ModelRouter::new(candidates);

        let attempts = vec![RoutingAttempt {
            candidate_key: "a".to_string(),
            exception: Some("throttled".to_string()),
        }];
        let selected = router.select_candidate(&[], None, vec![], &attempts).await;
        assert_eq!(selected, Some(1));
    }

    #[tokio::test]
    async fn test_fallback_declines_when_all_tried() {
        let a: Arc<dyn Model> = Arc::new(MockModel::new(vec![]));
        let candidates = vec![RoutingCandidate::new("a", a)];
        let router = ModelRouter::new(candidates);

        let attempts = vec![RoutingAttempt {
            candidate_key: "a".to_string(),
            exception: Some("throttled".to_string()),
        }];
        let selected = router.select_candidate(&[], None, vec![], &attempts).await;
        assert_eq!(selected, None);
    }

    #[tokio::test]
    async fn test_fallback_prefers_least_failures() {
        let a: Arc<dyn Model> = Arc::new(MockModel::new(vec![]));
        let b: Arc<dyn Model> = Arc::new(MockModel::new(vec![]));
        let c: Arc<dyn Model> = Arc::new(MockModel::new(vec![]));
        let candidates = vec![
            RoutingCandidate::new("a", a),
            RoutingCandidate::new("b", b),
            RoutingCandidate::new("c", c),
        ];
        let router = ModelRouter::new(candidates);

        // "a" failed once in a previous round (success since), "b" is clean.
        let attempts = vec![
            RoutingAttempt { candidate_key: "a".to_string(), exception: Some("boom".to_string()) },
            RoutingAttempt { candidate_key: "b".to_string(), exception: None },
        ];
        let selected = router.select_candidate(&[], None, vec![], &attempts).await;
        assert_eq!(selected, Some(1));
    }

    #[tokio::test]
    async fn test_router_streams_from_default() {
        let scripted = Scripted::new(vec![MockResponse::text("from candidate")]);
        let calls = Arc::new(AtomicUsize::new(0));

        struct CountingModel {
            inner: Scripted,
            calls: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl Model for CountingModel {
            fn config(&self) -> &crate::models::ModelConfig { self.inner.config() }
            fn update_config(&mut self, config: crate::models::ModelConfig) { self.inner.update_config(config) }
            fn stream<'a>(
                &'a self,
                messages: &'a [Message],
                tool_specs: Option<&'a [ToolSpec]>,
                system_prompt: Option<&'a str>,
                tool_choice: Option<crate::types::tools::ToolChoice>,
                system_prompt_content: Option<&'a [crate::types::content::SystemContentBlock]>,
            ) -> StreamEventStream<'a> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inner.stream(messages, tool_specs, system_prompt, tool_choice, system_prompt_content)
            }
        }

        let counting = Arc::new(CountingModel { inner: scripted, calls: calls.clone() });
        let candidates = vec![RoutingCandidate::new("primary", counting)];
        let router = ModelRouter::new(candidates);

        use futures::StreamExt;
        let mut stream = router.stream(&[], None, None, None, None);
        while let Some(event) = stream.next().await {
            assert!(event.is_ok());
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
