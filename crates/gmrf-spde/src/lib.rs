//! SPDE discretization utilities for the Rust port of `GaussianMarkovRandomFields.jl`.
//!
//! This crate ports the Matérn SPDE helpers: parameter conversions, FEM-backed
//! precision assembly, and hooks for producing `gmrf-core` fields. The API is
//! intentionally lightweight so additional SPDEs (advection–diffusion, temporal
//! extensions) can follow the same patterns.

pub mod discretization;
pub mod errors;
pub mod matern;

pub use discretization::FemDiscretization2d;
pub use errors::SpdeError;
pub use matern::{range_to_kappa, smoothness_to_nu, MaternSpde2d};
