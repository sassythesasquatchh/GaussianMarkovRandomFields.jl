//! Observation model scaffolding for the Rust port of `GaussianMarkovRandomFields.jl`.
//!
//! This crate mirrors the Julia `ObservationModel` abstractions, providing common
//! exponential-family likelihoods, linear/FEM-style transforms, and composition
//! helpers so observation sets can be combined before conditioning a `GMRF`.

#[cfg(feature = "autodiff")]
pub mod autodiff;
#[cfg(feature = "autodiff")]
pub mod gaussian_approximation;
pub mod builder;
pub mod errors;
pub mod models;
pub mod transform;

#[cfg(feature = "autodiff")]
pub use autodiff::DifferentiableObservation;
#[cfg(feature = "autodiff")]
pub use gaussian_approximation::{gaussian_approximation, GaussianApproximationError, LaplacePosterior};
pub use builder::ObservationBuilder;
pub use errors::ObservationError;
pub use models::{
    BernoulliLogitObservation, GaussianObservation, ObservationModel, PoissonLogObservation,
    StackedObservations,
};
#[cfg(feature = "autodiff")]
pub use transform::DifferentiableTransform;
pub use transform::{ComposedTransform, IdentityTransform, LinearTransform, ObservationTransform};

pub use gmrf_core as core;
