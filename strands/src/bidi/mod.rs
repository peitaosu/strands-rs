//! Bidirectional streaming agent core.
//!
//! Skeletal port of the Python `strands.bidi` package: a duplex agent that
//! exchanges audio/text with a model over a provider transport, emitting
//! the typed event families (`BidiAudio*`, `BidiText*`, connection events)
//! while accepting concurrent input.

pub mod agent;
pub mod transport;

pub use agent::BidiAgent;
pub use transport::{BidiEvent, BidiInput, BidiTransport, InMemoryTransport};
