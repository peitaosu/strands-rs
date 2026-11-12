//! The bidirectional agent loop.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};

use super::transport::{BidiEvent, BidiInput, BidiTransport};
use crate::models::Model;
use crate::types::errors::StrandsError;
use crate::types::streaming::StreamEvent;

/// Drives a duplex conversation over a transport.
///
/// User inputs arrive through the transport; each text turn runs one model
/// call whose stream is translated into `BidiEvent`s. A user interruption
/// cancels the in-flight turn.
pub struct BidiAgent<T: BidiTransport + 'static> {
    transport: Arc<T>,
    model: Arc<dyn Model>,
    system_prompt: Option<String>,
}

impl<T: BidiTransport> BidiAgent<T> {
    pub fn new(transport: Arc<T>, model: Arc<dyn Model>) -> Self {
        Self {
            transport,
            model,
            system_prompt: None,
        }
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn transport(&self) -> Arc<T> {
        self.transport.clone()
    }

    /// Runs the duplex loop until the transport closes.
    pub fn run(&self) -> BoxStream<'static, Result<BidiEvent, StrandsError>> {
        let transport = self.transport.clone();
        let model = self.model.clone();
        let system_prompt = self.system_prompt.clone();

        Box::pin(async_stream::stream! {
            yield Ok(BidiEvent::ConnectionStart);

            let mut conversation: Vec<crate::types::content::Message> = Vec::new();
            let mut pending_input: Option<BidiInput> = None;

            loop {
                if !transport.is_open().await {
                    break;
                }

                let input = match pending_input.take() {
                    Some(input) => input,
                    None => match transport.pop_input().await {
                        Some(input) => input,
                        None => {
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                            continue;
                        }
                    },
                };

                match input {
                    BidiInput::EndTurn => {
                        yield Ok(BidiEvent::TurnComplete);
                    }
                    BidiInput::Audio(_) => {
                        // Audio passthrough to the provider transport is
                        // provider-specific; the core loop surfaces a warning.
                        yield Ok(BidiEvent::Warning(
                            "audio input requires a provider-specific transport".to_string(),
                        ));
                    }
                    BidiInput::Text(text) => {
                        conversation.push(crate::types::content::Message::user(text));
                        let snapshot = conversation_snapshot(&conversation);

                        let model = model.clone();
                        let system_prompt = system_prompt.clone();
                        let mut turn = tokio::spawn(async move {
                            let mut events = Vec::new();
                            let stream = model.stream(
                                &snapshot,
                                None,
                                system_prompt.as_deref(),
                                None,
                                None,
                            );

                            futures::pin_mut!(stream);
                            while let Some(event) = stream.next().await {
                                match event {
                                    Ok(event) => events.extend(translate_stream_event(&event)),
                                    Err(e) => return Err(e),
                                }
                            }
                            Ok(events)
                        });

                        tokio::select! {
                            outcome = &mut turn => {
                                match outcome {
                                    Ok(Ok(events)) => {
                                        conversation.push(crate::types::content::Message::assistant(
                                            events.iter().filter_map(|event| match event {
                                                BidiEvent::TextDelta(text) => Some(text.clone()),
                                                _ => None,
                                            }).collect::<Vec<_>>().join(""),
                                        ));
                                        for event in events {
                                            yield Ok(event);
                                        }
                                        yield Ok(BidiEvent::TurnComplete);
                                    }
                                    Ok(Err(e)) => yield Err(e),
                                    Err(e) => {
                                        yield Err(StrandsError::InternalError {
                                            message: format!("bidi turn task failed: {}", e),
                                        });
                                    }
                                }
                            }
                            interrupted = async {
                                loop {
                                    if let Some(input) = transport.pop_input().await {
                                        return input;
                                    }
                                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                                }
                            } => {
                                turn.abort();
                                yield Ok(BidiEvent::Interruption);
                                // The interrupted user turn never got a
                                // reply; drop it so the interrupting input
                                // starts a clean turn instead of a doubled one.
                                conversation.pop();
                                pending_input = Some(interrupted);
                            }
                        }
                    }
                }
            }

            yield Ok(BidiEvent::ConnectionStop);
        })
    }
}

fn conversation_snapshot(messages: &[crate::types::content::Message]) -> Vec<crate::types::content::Message> {
    messages.to_vec()
}

/// Translates one provider stream event into bidi events.
fn translate_stream_event(event: &StreamEvent) -> Vec<BidiEvent> {
    let mut translated = Vec::new();

    if let Some(delta_event) = &event.content_block_delta {
        if let Some(delta) = &delta_event.delta {
            if let Some(text) = &delta.text {
                translated.push(BidiEvent::TextDelta(text.clone()));
            }
        }
    }
    if event.message_stop.is_some() {
        translated.push(BidiEvent::AudioStop);
    }

    translated
}

#[async_trait]
impl<T: BidiTransport> BidiTransport for BidiAgent<T> {
    async fn send(&self, input: BidiInput) -> Result<(), StrandsError> {
        self.transport.send(input).await
    }

    async fn pop_input(&self) -> Option<BidiInput> {
        self.transport.pop_input().await
    }

    fn receive(&self) -> BoxStream<'static, Result<BidiEvent, StrandsError>> {
        self.run()
    }

    async fn close(&self) -> Result<(), StrandsError> {
        self.transport.close().await
    }

    async fn is_open(&self) -> bool {
        self.transport.is_open().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidi::InMemoryTransport;
    use crate::testing::{MockModel as Scripted, MockResponse};
    use crate::types::streaming::StopReason;
    use futures::StreamExt;

    fn text_model(text: &'static str) -> Arc<dyn Model> {
        Arc::new(Scripted::new(vec![MockResponse::Events(vec![
            crate::types::streaming::StreamEvent::message_start(crate::types::content::Role::Assistant),
            crate::types::streaming::StreamEvent::text_delta(0, text),
            crate::types::streaming::StreamEvent::content_block_stop(0),
            crate::types::streaming::StreamEvent::message_stop(StopReason::EndTurn),
        ])]))
    }

    fn transport() -> Arc<InMemoryTransport> {
        Arc::new(InMemoryTransport::new())
    }

    #[tokio::test]
    async fn test_text_turn_produces_bidi_events() {
        let transport = transport();
        let agent = BidiAgent::new(transport.clone(), text_model("bidi response"));

        transport.send(BidiInput::Text("hello".to_string())).await.unwrap();

        let mut stream = agent.run();
        let mut saw_text = false;
        let mut saw_turn_complete = false;
        let mut steps = 0;
        while let Some(event) = stream.next().await {
            steps += 1;
            match event.unwrap() {
                BidiEvent::TextDelta(text) if text.contains("bidi response") => saw_text = true,
                BidiEvent::TurnComplete => {
                    saw_turn_complete = true;
                    break;
                }
                _ => {}
            }
            if steps > 20 {
                break;
            }
        }

        assert!(saw_text && saw_turn_complete);
    }

    #[tokio::test]
    async fn test_audio_input_yields_warning() {
        let transport = transport();
        let agent = BidiAgent::new(transport.clone(), text_model("unused"));

        transport.send(BidiInput::Audio(vec![0, 1])).await.unwrap();

        let mut stream = agent.run();
        let mut saw_warning = false;
        let mut steps = 0;
        while let Some(event) = stream.next().await {
            steps += 1;
            if matches!(event.unwrap(), BidiEvent::Warning(_)) {
                saw_warning = true;
                break;
            }
            if steps > 20 {
                break;
            }
        }
        assert!(saw_warning);
    }

    #[tokio::test]
    async fn test_new_input_interrupts_in_flight_turn() {
        // A model stream long enough that a second input arrives mid-turn.
        let slow_model: Arc<dyn Model> = Arc::new(Scripted::new(vec![MockResponse::Events(vec![
            crate::types::streaming::StreamEvent::message_start(crate::types::content::Role::Assistant),
            crate::types::streaming::StreamEvent::text_delta(0, "long answer"),
            crate::types::streaming::StreamEvent::content_block_stop(0),
            crate::types::streaming::StreamEvent::message_stop(StopReason::EndTurn),
        ])]));

        let transport = transport();
        let agent = BidiAgent::new(transport.clone(), slow_model);

        transport.send(BidiInput::Text("first".to_string())).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        transport.send(BidiInput::Text("interrupt!".to_string())).await.unwrap();

        let mut stream = agent.run();
        let mut saw_interruption = false;
        let mut steps = 0;
        while let Some(event) = stream.next().await {
            steps += 1;
            if matches!(event.unwrap(), BidiEvent::Interruption) {
                saw_interruption = true;
                break;
            }
            if steps > 40 {
                break;
            }
        }
        assert!(saw_interruption);
    }
}
