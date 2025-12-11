//! Core Gaussian Markov Random Field types for the Rust port of `GaussianMarkovRandomFields.jl`.
//!
//! This crate mirrors the Julia `GMRF` constructors, precision abstractions, and solver plumbing.
//! It establishes reusable math aliases, a sparse precision representation, and sampling/variance
//! utilities that will be shared across FEM, observation, and visualization crates.

pub mod gmrf;
pub mod linear;
pub mod precision;
pub mod solver;
pub mod types;

pub use gmrf::Gmrf;
pub use linear::{ComposedOperator, LinearOperator, MatrixOperator};
pub use precision::{PrecisionOperator, PrecisionStorage};
pub use solver::{JacobiPreconditioner, Solver, SolverAlgorithm, SolverConfig};
pub use types::{GmrfError, SparseMatrix, Vector};
