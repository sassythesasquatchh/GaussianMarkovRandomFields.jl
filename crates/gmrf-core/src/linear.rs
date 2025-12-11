//! Linear operator abstractions mirroring the Julia `LinearMaps` usage.
//!
//! These traits allow the core crate to work with both explicit sparse matrices and
//! matrix-free operators. They are intentionally lightweight so they can back
//! precision operators, preconditioners, and composed transforms without forcing
//! a particular backend.

use crate::types::{GmrfError, SparseMatrix, Vector};

/// A trait representing a generic linear operator `y = A * x`.
pub trait LinearOperator: Send + Sync {
    /// Dimension of the operator.
    fn dimension(&self) -> usize;

    /// Apply the operator to a vector, returning the result.
    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError>;
}

/// A concrete operator backed by a sparse matrix.
pub struct MatrixOperator {
    matrix: SparseMatrix,
}

impl MatrixOperator {
    /// Create a linear operator from a sparse matrix.
    pub fn new(matrix: SparseMatrix) -> Self {
        Self { matrix }
    }
}

impl LinearOperator for MatrixOperator {
    fn dimension(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError> {
        if x.len() != self.matrix.ncols() {
            return Err(GmrfError::DimensionMismatch(
                "input length must match matrix column dimension",
            ));
        }

        Ok(&self.matrix * x)
    }
}

/// Compose two operators B(A(x)) to allow lightweight chaining.
pub struct ComposedOperator<A: LinearOperator, B: LinearOperator> {
    first: A,
    second: B,
}

impl<A: LinearOperator, B: LinearOperator> ComposedOperator<A, B> {
    /// Build a composed operator `second(first(x))`.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: LinearOperator, B: LinearOperator> LinearOperator for ComposedOperator<A, B> {
    fn dimension(&self) -> usize {
        self.first.dimension()
    }

    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError> {
        let intermediate = self.first.apply(x)?;
        self.second.apply(&intermediate)
    }
}
