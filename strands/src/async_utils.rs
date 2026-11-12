//! Private async execution utilities.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::runtime::Handle;

/// Cooperative cancellation flag with `threading.Event` semantics.
///
/// The agent and tool executors share one signal so that `agent.cancel()`
/// can stop in-flight model streams and pending tool calls.
#[derive(Debug, Clone, Default)]
pub struct CancelSignal(Arc<AtomicBool>);

impl CancelSignal {
    pub fn new() -> Self { Self(Arc::new(AtomicBool::new(false))) }

    /// Marks the signal as cancelled.
    pub fn set(&self) { self.0.store(true, Ordering::SeqCst); }

    /// Alias for `set`, matching the Python `agent.cancel()` name.
    pub fn cancel(&self) { self.set(); }

    /// Returns true when cancellation has been requested.
    pub fn is_set(&self) -> bool { self.0.load(Ordering::SeqCst) }

    /// Resolves once cancellation has been requested.
    pub async fn wait(&self) {
        while !self.is_set() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

/// Run an async function, handling event loop conflicts.
///
/// This utility handles the common pattern of running async code from sync contexts
/// by detecting if we're already in a tokio runtime and using the appropriate method.
///
/// # Arguments
///
/// * `async_func` - A callable that returns an awaitable
///
/// # Returns
///
/// The result of the async function
pub fn run_async<F, T>(async_func: F) -> T
where
    F: Future<Output = T> + Send,
    T: Send,
{

    match Handle::try_current() {
        Ok(_handle) => {

            std::thread::scope(|s| {
                s.spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to create runtime");
                    rt.block_on(async_func)
                })
                .join()
                .expect("Thread panicked")
            })
        }
        Err(_) => {

            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create runtime");
            rt.block_on(async_func)
        }
    }
}

/// Spawn an async task and return a handle to it.
///
/// This is a convenience wrapper around tokio::spawn that works
/// from both sync and async contexts.
pub fn spawn_async<F, T>(async_func: F) -> tokio::task::JoinHandle<T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    match Handle::try_current() {
        Ok(handle) => handle.spawn(async_func),
        Err(_) => panic!("spawn_async must be called from within a tokio runtime"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_async_simple() {
        async fn add(a: i32, b: i32) -> i32 {
            a + b
        }

        let result = run_async(add(2, 3));
        assert_eq!(result, 5);
    }

    #[tokio::test]
    async fn test_run_async_from_async_context() {
        async fn multiply(a: i32, b: i32) -> i32 {
            a * b
        }


        let result = run_async(multiply(4, 5));
        assert_eq!(result, 20);
    }

    #[tokio::test]
    async fn test_spawn_async() {
        async fn compute() -> String {
            "hello".to_string()
        }

        let handle = spawn_async(compute());
        let result = handle.await.unwrap();
        assert_eq!(result, "hello");
    }
}

