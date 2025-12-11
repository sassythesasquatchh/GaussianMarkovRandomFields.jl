//! Latent model catalog for the Rust port of `GaussianMarkovRandomFields.jl`.
//!
//! The Julia package exposes a collection of priors (AR1, RW1, IID, Besag, BYM2,
//! separable products, and block combinations). This crate provides analogous
//! constructors that emit `gmrf-core` fields and compositional utilities for
//! stacking or combining latent components.

pub mod errors;
pub mod graph;
pub mod models;

pub use errors::LatentModelError;
pub use graph::Neighborhood;
pub use models::{
    ar1_chain, besag, bym2, combine_block_diagonal, fixed_effect, iid_gaussian, rw1,
    separable_kronecker,
};
