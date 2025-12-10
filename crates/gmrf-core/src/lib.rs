//! Core utilities for building Gaussian Markov random fields in Rust.
//! The crate provides a lightweight representation for sparsely connected
//! models, along with dense linear algebra helpers built on `nalgebra`.

pub mod errors;
pub mod gmrf;
pub mod graph;

pub use errors::GmrfError;
pub use gmrf::{GaussianMarkovRandomField, LogDensity};
pub use graph::{Edge, WeightedGraph};
