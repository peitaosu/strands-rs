//! Default routing strategy: ordered failover preferring the least-failed candidate.

use async_trait::async_trait;

use super::{RoutingContext, RoutingStrategy};

/// Picks the healthiest candidate not yet tried since the last success.
///
/// Candidates already tried since the last success are excluded, then the
/// fewest recorded failures wins, ties going to the earlier declaration.
/// A success re-arms every candidate and clears the succeeding candidate's
/// own failure count. Returns `None` once every candidate has been tried
/// since the last success.
pub struct FallbackStrategy;

#[async_trait]
impl RoutingStrategy for FallbackStrategy {
    async fn select(&self, context: &RoutingContext<'_>) -> Option<usize> {
        let mut failures: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();

        for attempt in context.attempts {
            match attempt.exception {
                Some(_) => {
                    *failures.entry(attempt.candidate_key.as_str()).or_insert(0) += 1;
                }
                None => {
                    failures.remove(attempt.candidate_key.as_str());
                }
            }
        }

        let tried_now: Vec<&str> = context
            .attempts_since_success()
            .iter()
            .map(|attempt| attempt.candidate_key.as_str())
            .collect();

        let mut best: Option<(usize, usize)> = None;
        for (index, candidate) in context.candidates.iter().enumerate() {
            if tried_now.contains(&candidate.key.as_str()) {
                continue;
            }
            let failure_count = failures.get(candidate.key.as_str()).copied().unwrap_or(0);
            if best.map(|(_, count)| failure_count < count).unwrap_or(true) {
                best = Some((index, failure_count));
            }
        }

        best.map(|(index, _)| index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context<'a>(
        candidates: &'a [super::super::RoutingCandidate],
        attempts: &'a [super::super::RoutingAttempt],
    ) -> RoutingContext<'a> {
        RoutingContext {
            messages: &[],
            system_prompt: None,
            tool_specs: vec![],
            candidates,
            attempts,
        }
    }

    fn candidate(key: &str) -> super::super::RoutingCandidate {
        super::super::RoutingCandidate::new(key, std::sync::Arc::new(crate::testing::MockModel::new(vec![])))
    }

    fn attempt(key: &str, failed: bool) -> super::super::RoutingAttempt {
        super::super::RoutingAttempt {
            candidate_key: key.to_string(),
            exception: failed.then(|| "err".to_string()),
        }
    }

    #[tokio::test]
    async fn test_excludes_tried_since_last_success() {
        let candidates = vec![candidate("a"), candidate("b"), candidate("c")];
        let attempts = vec![attempt("a", false), attempt("a", true), attempt("b", true)];
        let strategy = FallbackStrategy;

        assert_eq!(strategy.select(&context(&candidates, &attempts)).await, Some(2));
    }

    #[tokio::test]
    async fn test_success_clears_candidate_failures() {
        let candidates = vec![candidate("a"), candidate("b")];
        let attempts = vec![attempt("a", true), attempt("a", false)];
        let strategy = FallbackStrategy;

        assert_eq!(strategy.select(&context(&candidates, &attempts)).await, Some(0));
    }

    #[tokio::test]
    async fn test_declines_when_exhausted() {
        let candidates = vec![candidate("a")];
        let attempts = vec![attempt("a", true)];
        let strategy = FallbackStrategy;

        assert_eq!(strategy.select(&context(&candidates, &attempts)).await, None);
    }
}
