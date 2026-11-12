//! Model retry strategy with exponential backoff.

use std::time::Duration;

use async_trait::async_trait;

use crate::hooks::{HookEvent, HookProvider};
use crate::types::errors::StrandsError;

/// Default retry strategy for model throttling with exponential backoff.
///
/// Retries model calls on retryable errors using exponential backoff.
/// Delay doubles after each attempt: initial_delay, initial_delay*2,
/// initial_delay*4, etc., capped at max_delay. State resets after a
/// successful call.
///
/// With defaults (initial_delay=4s, max_delay=240s, max_attempts=6), delays
/// are: 4s → 8s → 16s → 32s → 64s (5 retries before giving up on the 6th
/// attempt).
#[derive(Debug, Clone)]
pub struct ModelRetryStrategy {
    max_attempts: u32,
    initial_delay: Duration,
    max_delay: Duration,
    current_attempt: u32,
}

impl Default for ModelRetryStrategy {
    fn default() -> Self {
        Self {
            max_attempts: 6,
            initial_delay: Duration::from_secs(4),
            max_delay: Duration::from_secs(240),
            current_attempt: 0,
        }
    }
}

impl ModelRetryStrategy {
    pub fn new(max_attempts: u32, initial_delay: Duration, max_delay: Duration) -> Self {
        Self {
            max_attempts,
            initial_delay,
            max_delay,
            current_attempt: 0,
        }
    }

    /// Whether the error should be retried.
    ///
    /// Default: throttling and transient network failures.
    pub fn is_retryable(&self, error: &StrandsError) -> bool {
        matches!(
            error,
            StrandsError::ModelThrottled { .. } | StrandsError::NetworkError(_)
        )
    }

    /// Returns the delay before the next attempt, or None when attempts are exhausted.
    pub fn next_delay(&mut self) -> Option<Duration> {
        if self.current_attempt >= self.max_attempts.saturating_sub(1) {
            return None;
        }
        let delay = self
            .initial_delay
            .saturating_mul(1u32 << self.current_attempt.min(30))
            .min(self.max_delay);
        self.current_attempt += 1;
        Some(delay)
    }

    /// Resets attempt state after a successful call.
    pub fn reset(&mut self) {
        self.current_attempt = 0;
    }

    pub fn attempts(&self) -> u32 { self.current_attempt }

    pub fn max_attempts(&self) -> u32 { self.max_attempts }
}

#[async_trait]
impl HookProvider for ModelRetryStrategy {
    async fn on_event(&self, event: &mut HookEvent) {
        if let HookEvent::EventLoopThrottle(throttle) = event {
            tracing::warn!(
                "attempt=<{}> | retrying model call in {}s - {}",
                throttle.attempt,
                throttle.delay_seconds,
                throttle.reason,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn throttled() -> StrandsError {
        StrandsError::ModelThrottled { message: "throttled".to_string() }
    }

    #[test]
    fn test_default_delays() {
        let mut strategy = ModelRetryStrategy::default();
        assert_eq!(strategy.next_delay(), Some(Duration::from_secs(4)));
        assert_eq!(strategy.next_delay(), Some(Duration::from_secs(8)));
        assert_eq!(strategy.next_delay(), Some(Duration::from_secs(16)));
        assert_eq!(strategy.next_delay(), Some(Duration::from_secs(32)));
        assert_eq!(strategy.next_delay(), Some(Duration::from_secs(64)));
        assert_eq!(strategy.next_delay(), None);
    }

    #[test]
    fn test_delay_cap() {
        let mut strategy = ModelRetryStrategy::new(10, Duration::from_secs(4), Duration::from_secs(40));
        let mut last = Duration::ZERO;
        while let Some(delay) = strategy.next_delay() {
            assert!(delay <= Duration::from_secs(40));
            last = delay;
        }
        assert_eq!(last, Duration::from_secs(40));
    }

    #[test]
    fn test_reset_after_success() {
        let mut strategy = ModelRetryStrategy::default();
        assert!(strategy.next_delay().is_some());
        strategy.reset();
        assert_eq!(strategy.next_delay(), Some(Duration::from_secs(4)));
    }

    #[test]
    fn test_retryable_errors() {
        let strategy = ModelRetryStrategy::default();
        assert!(strategy.is_retryable(&throttled()));
        assert!(strategy.is_retryable(&StrandsError::NetworkError("timeout".to_string())));
        assert!(!strategy.is_retryable(&StrandsError::InvalidInput { message: "bad".to_string() }));
        assert!(!strategy.is_retryable(&StrandsError::MaxTokensReached { message: "max".to_string() }));
    }
}
