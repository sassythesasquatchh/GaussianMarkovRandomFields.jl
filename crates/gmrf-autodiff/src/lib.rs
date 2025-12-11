//! Automatic differentiation utilities shared across the Rust GMRF crates.
//!
//! The Julia package integrates ForwardDiff/Zygote for differentiating
//! observation likelihoods and latent model constructors. This crate
//! provides a lightweight forward-mode dual number implementation and
//! dense Jacobian helpers so other crates can expose optional AD-backed
//! APIs without pulling in heavy dependencies.

mod dual;
mod jacobian;

pub use dual::Dual64;
pub use jacobian::{gradient, jacobian, AutodiffError};
