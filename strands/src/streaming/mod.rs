//! Streaming utilities for processing model responses.

use futures::{Stream, StreamExt};

use crate::types::errors::StrandsError;
use crate::types::streaming::StreamEvent;

/// Collects text from a stream of events.
pub async fn collect_text<S>(stream: S) -> Result<String, StrandsError>
where
    S: Stream<Item = Result<StreamEvent, StrandsError>>,
{
    let mut text = String::new();
    futures::pin_mut!(stream);

    while let Some(event) = stream.next().await {
        let event = event?;
        if let Some(delta) = event.as_text_delta() {
            text.push_str(delta);
        }
    }

    Ok(text)
}

/// Prints a stream of events to stdout.
pub async fn print_stream<S>(stream: S) -> Result<(), StrandsError>
where
    S: Stream<Item = Result<StreamEvent, StrandsError>>,
{
    use std::io::{self, Write};

    futures::pin_mut!(stream);

    while let Some(event) = stream.next().await {
        let event = event?;
        if let Some(text) = event.as_text_delta() {
            print!("{text}");
            io::stdout().flush().ok();
        }
    }
    println!();

    Ok(())
}
