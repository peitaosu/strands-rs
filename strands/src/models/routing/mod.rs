//! Model routing primitives.
//!
//! Skeletal port of the Python `strands.models.routing` package:
//! `ModelRouter` asks its `RoutingStrategy` which candidate to use and asks
//! again after a failed call, so the strategy owns every routing decision.
//! The bundled `FallbackStrategy` prefers the candidate failing least, and
//! `ClassifierStrategy` routes by asking a classifier model one closed
//! question.

pub mod classifier;
pub mod fallback;
pub mod router;

pub use classifier::ClassifierStrategy;
pub use fallback::FallbackStrategy;
pub use router::{ModelRouter, RoutingAttempt, RoutingCandidate, RoutingContext, RoutingStrategy};
