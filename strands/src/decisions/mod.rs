//! Decision models: asking a model structured questions.
//!
//! Skeletal port of the Python `strands.decisions` package (design 0020):
//! a `DecisionModel` answers typed questions; `LLMDecisionModel` asks a
//! language model and parses the answer as structured output.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::models::{structured_output_from, Model};
use crate::types::content::Message;
use crate::types::errors::StrandsError;

/// A decision the framework asks a model to make.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Question {
    /// Identifies the question kind, e.g. `candidate`.
    pub id: String,
    /// Full prompt presented to the decision maker.
    pub prompt: String,
    /// Structured context the question refers to.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub context: Value,
}

impl Question {
    pub fn new(id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            context: Value::Null,
        }
    }

    pub fn with_context(mut self, context: Value) -> Self {
        self.context = context;
        self
    }
}

/// The answer a decision model returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Answer {
    /// Matches the question id.
    pub id: String,
    /// The decision payload.
    pub response: Value,
}

impl Answer {
    pub fn new(id: impl Into<String>, response: Value) -> Self {
        Self {
            id: id.into(),
            response,
        }
    }
}

/// Structured response the LLM decision model parses.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionAnswer {
    /// The decision payload chosen by the model.
    pub answer: Value,
}

/// Decides questions.
#[async_trait]
pub trait DecisionModel: Send + Sync {
    /// Answers one question.
    async fn ask(&self, question: Question) -> Result<Answer, StrandsError>;
}

/// Asks a language model and parses the response as structured output.
pub struct LLMDecisionModel {
    model: std::sync::Arc<dyn Model>,
    system_prompt: Option<String>,
}

impl LLMDecisionModel {
    pub fn new(model: std::sync::Arc<dyn Model>) -> Self {
        Self {
            model,
            system_prompt: None,
        }
    }

    /// Sets a system prompt for decision questions.
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Builds the user message for a question.
    fn build_message(&self, question: &Question) -> Message {
        let payload = serde_json::json!({
            "question_id": question.id,
            "prompt": question.prompt,
            "context": question.context,
        });
        Message::user(format!(
            "Answer the following decision request.\n\n{}",
            serde_json::to_string(&payload).unwrap_or_default()
        ))
    }
}

#[async_trait]
impl DecisionModel for LLMDecisionModel {
    async fn ask(&self, question: Question) -> Result<Answer, StrandsError> {
        let message = self.build_message(&question);

        let decision = structured_output_from::<DecisionAnswer>(
            self.model.as_ref(),
            &[message],
            self.system_prompt.as_deref(),
        )
        .await?;

        Ok(Answer::new(question.id, decision.answer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockModel;
    use crate::testing::{MockModel as Scripted, MockResponse};
    use std::sync::Arc;

    fn scripted_model(answer: &str) -> Arc<dyn Model> {
        Arc::new(Scripted::new(vec![MockResponse::text(answer)]))
    }

    #[tokio::test]
    async fn test_llm_decision_model_answers() {
        let model = scripted_model(r#"{"answer": {"candidate": "c1"}}"#);
        let decision_model = LLMDecisionModel::new(model);

        let question = Question::new("candidate", "which candidate handles this?")
            .with_context(serde_json::json!({"candidates": ["c0", "c1"]}));

        let answer = decision_model.ask(question).await.unwrap();
        assert_eq!(answer.id, "candidate");
        assert_eq!(answer.response["candidate"], "c1");
    }

    #[tokio::test]
    async fn test_llm_decision_model_rejects_non_json() {
        let model = scripted_model("I prefer the second one");
        let decision_model = LLMDecisionModel::new(model);

        let error = decision_model
            .ask(Question::new("candidate", "pick one"))
            .await
            .unwrap_err();
        assert!(matches!(error, StrandsError::StructuredOutputError { .. }));
    }

    #[test]
    fn test_question_serialization() {
        let question = Question::new("goal", "what next?").with_context(serde_json::json!({"k": 1}));
        let json = serde_json::to_value(&question).unwrap();
        assert_eq!(json["id"], "goal");
        assert_eq!(json["context"]["k"], 1);
    }

    #[test]
    fn test_unscripted_mock_model_marker() {
        let _ = MockModel::new(vec![]);
    }
}
