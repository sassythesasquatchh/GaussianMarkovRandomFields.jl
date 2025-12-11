//! Shared math aliases and errors for Gaussian Markov Random Field primitives.
//!
//! These definitions mirror the Julia package's reliance on generic linear algebra types while
//! keeping a sparse-first mindset that will extend to FEM and observation crates.

use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use thiserror::Error;

/// Dense vector used throughout the crate.
pub type Vector = DVector<f64>;

/// Sparse matrix alias for precision and linear operator construction.
pub type SparseMatrix = CsrMatrix<f64>;

/// Error variants produced by the core GMRF routines.
#[derive(Debug, Error)]
pub enum GmrfError {
    /// The provided inputs are inconsistent (dimension mismatch or missing data).
    #[error("inconsistent dimensions: {0}")]
    DimensionMismatch(&'static str),

    /// A precision matrix was required but not available in concrete form.
    #[error("precision matrix is required for this operation")]
    MissingPrecisionMatrix,

    /// Factorization failed because the precision was not positive definite.
    #[error("precision matrix is not positive definite")]
    NonPositiveDefinite,
}
