//! Bidi transports: the duplex channel between agent and model provider.

use std::collections::VecDeque;
use std::sync::Arc;

use futures::stream::BoxStream;
use tokio::sync::{mpsc, Mutex};

use crate::types::errors::StrandsError;

/// Input sent from the user toward the model.
#[derive(Debug, Clone)]
pub enum BidiInput {
    /// Raw audio bytes (provider-defined encoding).
    Audio(Vec<u8>),
    /// Text input.
    Text(String),
    /// Signals the end of the user's turn.
    EndTurn,
}

/// Events streamed from the model toward the user.
#[derive(Debug, Clone)]
pub enum BidiEvent {
    /// The connection opened.
    ConnectionStart,
    /// Audio playback started for a new response segment.
    AudioStart,
    /// Incremental audio output.
    AudioDelta(Vec<u8>),
    /// Audio playback ended for the current segment.
    AudioStop,
    /// A text response started.
    TextStart,
    /// Incremental text output.
    TextDelta(String),
    /// The text response completed.
    TextStop,
    /// The user interrupted the model.
    Interruption,
    /// The model finished its turn.
    TurnComplete,
    /// A connection-level warning that may self-heal.
    Warning(String),
    /// The connection closed.
    ConnectionStop,
}

/// A duplex transport between the bidi agent and a model provider.
#[async_trait::async_trait]
pub trait BidiTransport: Send + Sync {
    /// Sends user input toward the model.
    async fn send(&self, input: BidiInput) -> Result<(), StrandsError>;

    /// Takes the next pending user input, if any.
    async fn pop_input(&self) -> Option<BidiInput>;

    /// Streams model events toward the user.
    fn receive(&self) -> BoxStream<'static, Result<BidiEvent, StrandsError>>;

    /// Closes the transport.
    async fn close(&self) -> Result<(), StrandsError>;

    /// Returns true while the transport is open.
    async fn is_open(&self) -> bool;
}

/// Channel-backed in-memory transport for tests and local loops.
#[derive(Clone)]
pub struct InMemoryTransport {
    inbound: Arc<Mutex<VecDeque<BidiInput>>>,
    outbound: Arc<mpsc::UnboundedSender<BidiEvent>>,
    outbound_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<BidiEvent>>>,
    open: Arc<std::sync::atomic::AtomicBool>,
}

impl InMemoryTransport {
    /// Creates an open transport.
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self {
            inbound: Arc::new(Mutex::new(VecDeque::new())),
            outbound: Arc::new(sender),
            outbound_rx: Arc::new(tokio::sync::Mutex::new(receiver)),
            open: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Pushes an event to the outbound stream (model side).
    pub fn push_event(&self, event: BidiEvent) {
        let _ = self.outbound.send(event);
    }

    /// Takes the next pending inbound input (user side), if any.
    pub async fn pop_input(&self) -> Option<BidiInput> {
        self.inbound.lock().await.pop_front()
    }
}

impl Default for InMemoryTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl BidiTransport for InMemoryTransport {
    async fn send(&self, input: BidiInput) -> Result<(), StrandsError> {
        if !self.is_open().await {
            return Err(StrandsError::NetworkError("transport is closed".to_string()));
        }
        self.inbound.lock().await.push_back(input);
        Ok(())
    }

    async fn pop_input(&self) -> Option<BidiInput> {
        self.inbound.lock().await.pop_front()
    }

    fn receive(&self) -> BoxStream<'static, Result<BidiEvent, StrandsError>> {
        let receiver = self.outbound_rx.clone();
        Box::pin(async_stream::stream! {
            let mut guard = receiver.lock().await;
            while let Some(event) = guard.recv().await {
                yield Ok(event);
            }
        })
    }

    async fn close(&self) -> Result<(), StrandsError> {
        self.open
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    async fn is_open(&self) -> bool {
        self.open.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_send_and_receive_roundtrip() {
        let transport = InMemoryTransport::new();

        transport.send(BidiInput::Text("hello".to_string())).await.unwrap();
        transport.push_event(BidiEvent::ConnectionStart);
        transport.push_event(BidiEvent::TextDelta("hi".to_string()));
        transport.push_event(BidiEvent::TurnComplete);

        match transport.pop_input().await.unwrap() {
            BidiInput::Text(text) => assert_eq!(text, "hello"),
            other => panic!("unexpected input: {:?}", other),
        }

        let mut stream = transport.receive();
        assert!(matches!(stream.next().await.unwrap().unwrap(), BidiEvent::ConnectionStart));
        assert!(matches!(stream.next().await.unwrap().unwrap(), BidiEvent::TextDelta(_)));
        assert!(matches!(stream.next().await.unwrap().unwrap(), BidiEvent::TurnComplete));
    }

    #[tokio::test]
    async fn test_closed_transport_rejects_send() {
        let transport = InMemoryTransport::new();
        transport.close().await.unwrap();
        assert!(!transport.is_open().await);

        let error = transport
            .send(BidiInput::Text("late".to_string()))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("closed"));
    }

    #[tokio::test]
    async fn test_audio_input_roundtrip() {
        let transport = InMemoryTransport::new();
        transport.send(BidiInput::Audio(vec![1, 2, 3])).await.unwrap();

        match transport.pop_input().await.unwrap() {
            BidiInput::Audio(bytes) => assert_eq!(bytes, vec![1, 2, 3]),
            other => panic!("unexpected input: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_clone_shares_channel() {
        let transport = InMemoryTransport::new();
        let cloned = transport.clone();

        cloned.push_event(BidiEvent::ConnectionStart);

        let mut stream = transport.receive();
        assert!(matches!(stream.next().await.unwrap().unwrap(), BidiEvent::ConnectionStart));
    }
}
