//! Precision matrix abstractions and linear operator traits.
//!
//! The Julia package allows using explicit sparse matrices or matrix-free linear maps when
//! constructing a `GMRF`. This module mirrors that flexibility using a trait for applying the
//! precision and an enum to retain either a sparse matrix or boxed operator.

use crate::types::{GmrfError, SparseMatrix, Vector};

/// A trait representing a matrix-free precision operator.
pub trait PrecisionOperator: Send + Sync {
    /// Dimension of the operator.
    fn dimension(&self) -> usize;

    /// Apply the precision to a vector, returning the result.
    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError>;
}

/// Storage for either a concrete precision matrix or a matrix-free operator.
pub enum PrecisionStorage {
    /// Concrete sparse precision matrix (preferred for sampling and solves).
    Matrix(SparseMatrix),
    /// Matrix-free linear operator with known dimension.
    Operator {
        dimension: usize,
        operator: Box<dyn PrecisionOperator>,
    },
}

impl PrecisionStorage {
    /// Returns the number of rows/columns represented by the precision.
    pub fn dimension(&self) -> usize {
        match self {
            PrecisionStorage::Matrix(matrix) => matrix.nrows(),
            PrecisionStorage::Operator { dimension, .. } => *dimension,
        }
    }

    /// Access the underlying matrix reference when available.
    pub fn as_matrix(&self) -> Option<&SparseMatrix> {
        match self {
            PrecisionStorage::Matrix(matrix) => Some(matrix),
            PrecisionStorage::Operator { .. } => None,
        }
    }
}
