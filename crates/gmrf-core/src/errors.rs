use thiserror::Error;

/// Errors that can occur while constructing or manipulating a GMRF.
#[derive(Debug, Error)]
pub enum GmrfError {
    /// Occurs when mean and precision dimensions do not align.
    #[error("precision matrix shape {precision_rows}x{precision_cols} does not match mean length {mean_len}")]
    DimensionMismatch {
        mean_len: usize,
        precision_rows: usize,
        precision_cols: usize,
    },
    /// Raised when the precision matrix is not symmetric positive definite.
    #[error("precision matrix is not symmetric positive definite")]
    NotPositiveDefinite,
    /// Raised when attempting to sample with an invalid factorization.
    #[error("sampling requires a valid Cholesky factorization")]
    InvalidFactorization,
}
