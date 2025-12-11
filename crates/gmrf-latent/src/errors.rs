//! Errors for latent model construction.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LatentModelError {
    /// Inputs do not satisfy required constraints.
    #[error("invalid latent model parameters: {0}")]
    InvalidParameters(&'static str),

    /// Graph data are inconsistent with requested dimensions.
    #[error("invalid neighborhood graph: {0}")]
    InvalidGraph(&'static str),

    /// Precision matrices are required for composition but missing.
    #[error("precision matrix is required for this composition")]
    MissingPrecision,

    /// Dimensions of component models do not align for composition.
    #[error("dimension mismatch when composing latent models")]
    DimensionMismatch,
}
