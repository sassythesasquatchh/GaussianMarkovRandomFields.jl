//! Error types for observation models.
//!
//! These variants mirror the validation checks used throughout the observation
//! pipeline, ensuring inputs align with latent dimensions and probability
//! constraints.

use thiserror::Error;

/// Errors produced by observation model evaluation.
#[derive(Debug, Error)]
pub enum ObservationError {
    /// The provided inputs are inconsistent (dimension mismatch or missing data).
    #[error("inconsistent dimensions: {0}")]
    DimensionMismatch(&'static str),

    /// A probability was outside the valid open interval (0, 1).
    #[error("probability outside (0, 1) interval")]
    InvalidProbability,
}

#[cfg(feature = "autodiff")]
impl From<gmrf_autodiff::AutodiffError> for ObservationError {
    fn from(err: gmrf_autodiff::AutodiffError) -> Self {
        match err {
            gmrf_autodiff::AutodiffError::DimensionMismatch(msg) => {
                ObservationError::DimensionMismatch(msg)
            }
        }
    }
}
