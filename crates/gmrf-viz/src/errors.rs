//! Error types for visualization utilities.
//!
//! Kept minimal to propagate plotting backend failures and dimension
//! mismatches when pairing meshes with node-wise data.

use plotters::drawing::DrawingAreaErrorKind;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VizError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("plotting error: {0}")]
    Plotting(String),

    #[error("field length {actual} does not match mesh nodes {expected}")]
    DimensionMismatch { expected: usize, actual: usize },
}

impl<E> From<DrawingAreaErrorKind<E>> for VizError
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(kind: DrawingAreaErrorKind<E>) -> Self {
        VizError::Plotting(format!("{kind:?}"))
    }
}
