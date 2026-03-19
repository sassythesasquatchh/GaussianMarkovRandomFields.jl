//! Core Gaussian Markov Random Field types for the Rust port of `GaussianMarkovRandomFields.jl`.
//!
//! This crate mirrors the Julia `GMRF` constructors, precision abstractions, and solver plumbing.
//! It establishes reusable math aliases, a sparse precision representation, and sampling/variance
//! utilities that will be shared across FEM, observation, and visualization crates.

pub mod gmrf;
pub mod linear;
pub mod observation;
pub mod precision;
pub mod solver;
pub mod types;
pub mod vtk;

pub use gmrf::{ConstrainedVarianceDecomposition, Gmrf};
pub use linear::{kronecker, ComposedOperator, LinearOperator, MatrixOperator, OperatorWithSqrt};
pub use observation::{
    add_sparse, apply_gaussian_observations, build_linear_observation_matrix, ht_weighted_h,
    ht_weighted_observations, observation_selector,
};
pub use precision::{PrecisionOperator, PrecisionStorage};
pub use solver::{JacobiPreconditioner, Solver, SolverAlgorithm, SolverConfig};
pub use types::{GmrfError, SparseCholeskyFactor, SparseMatrix, Vector};
pub use vtk::write_structured_points;
