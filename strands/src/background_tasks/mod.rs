//! In-process background task runtime.
//!
//! Skeletal port of the Python `strands.background_tasks` package: task
//! submission with handles, status tracking, and cancellation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use tokio::task::JoinHandle;

/// Status of a submitted background task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Handle for a submitted background task.
pub struct BackgroundTask<T> {
    pub id: String,
    join_handle: JoinHandle<std::result::Result<T, String>>,
    cancel: Arc<Notify>,
}

impl<T: Send + 'static> BackgroundTask<T> {
    /// Marks the task for cancellation; the task observes it via
    /// `cancelled().await`.
    pub fn cancel(&self) {
        self.cancel.notify_waiters();
    }

    /// Resolves when cancellation has been requested.
    pub async fn cancelled(&self) {
        self.cancel.notified().await;
    }

    /// Awaits the task outcome.
    pub async fn join(self) -> std::result::Result<T, String> {
        match self.join_handle.await {
            Ok(result) => result,
            Err(error) => Err(format!("background task panicked: {}", error)),
        }
    }
}

/// Manages background tasks with a registry of their statuses.
#[derive(Debug, Default)]
pub struct BackgroundTaskManager {
    statuses: Arc<Mutex<HashMap<String, TaskStatus>>>,
}

impl BackgroundTaskManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Submits a task body and registers it under `id`.
    pub fn submit<F, T>(&self, id: impl Into<String>, body: F) -> BackgroundTask<T>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let id = id.into();
        let statuses = self.statuses.clone();
        let cancel = Arc::new(Notify::new());
        let task_cancel = cancel.clone();
        let task_id = id.clone();

        statuses.lock().unwrap().insert(id.clone(), TaskStatus::Running);

        let join_handle = tokio::spawn(async move {
            tokio::select! {
                result = body => {
                    statuses.lock().unwrap().insert(task_id.clone(), TaskStatus::Completed);
                    Ok(result)
                }
                _ = task_cancel.notified() => {
                    statuses.lock().unwrap().insert(task_id.clone(), TaskStatus::Cancelled);
                    Err("task cancelled".to_string())
                }
            }
        });

        BackgroundTask {
            id,
            join_handle,
            cancel,
        }
    }

    /// Returns the recorded status for `id`.
    pub fn status(&self, id: &str) -> Option<TaskStatus> {
        self.statuses.lock().unwrap().get(id).copied()
    }

    /// Lists all task ids.
    pub fn ids(&self) -> Vec<String> {
        self.statuses.lock().unwrap().keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_submit_and_complete() {
        let manager = BackgroundTaskManager::new();
        let task = manager.submit("t1", async { 42 });

        let result = task.join().await.unwrap();
        assert_eq!(result, 42);
        assert_eq!(manager.status("t1"), Some(TaskStatus::Completed));
    }

    #[tokio::test]
    async fn test_cancel_running_task() {
        let manager = BackgroundTaskManager::new();
        let task = manager.submit("t2", async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            1u8
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        task.cancel();
        let error = task.join().await.unwrap_err();
        assert!(error.contains("cancelled"));
        assert_eq!(manager.status("t2"), Some(TaskStatus::Cancelled));
    }

    #[tokio::test]
    async fn test_list_ids() {
        let manager = BackgroundTaskManager::new();
        let _ = manager.submit("a", async {});
        let _ = manager.submit("b", async {});
        assert_eq!(manager.ids().len(), 2);
    }
}
