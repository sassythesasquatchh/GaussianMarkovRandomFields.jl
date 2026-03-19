//! Linear operator abstractions mirroring the Julia `LinearMaps` usage.
//!
//! These traits allow the core crate to work with both explicit sparse matrices and
//! matrix-free operators. They are intentionally lightweight so they can back
//! precision operators, preconditioners, and composed transforms without forcing
//! a particular backend.

use crate::types::{CooMatrix, GmrfError, SparseMatrix, Vector};

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

        Ok(sparse_matvec(&self.matrix, x))
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

/// Linear operator paired with a known square root operator, mirroring Julia's `LinearMapWithSqrt`.
pub struct OperatorWithSqrt<O: LinearOperator, S: LinearOperator> {
    operator: O,
    sqrt: S,
}

impl<O: LinearOperator, S: LinearOperator> OperatorWithSqrt<O, S> {
    /// Create a new operator-with-sqrt, verifying dimensions match.
    pub fn new(operator: O, sqrt: S) -> Result<Self, GmrfError> {
        if operator.dimension() != sqrt.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "operator and square root must share dimensions",
            ));
        }
        Ok(Self { operator, sqrt })
    }

    /// Access the stored square root operator.
    pub fn sqrt(&self) -> &S {
        &self.sqrt
    }
}

impl<O: LinearOperator, S: LinearOperator> LinearOperator for OperatorWithSqrt<O, S> {
    fn dimension(&self) -> usize {
        self.operator.dimension()
    }

    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError> {
        self.operator.apply(x)
    }
}

/// Build a Kronecker product of two matrix-backed operators.
pub fn kronecker(a: &MatrixOperator, b: &MatrixOperator) -> MatrixOperator {
    let (a_rows, a_cols) = (a.matrix.nrows(), a.matrix.ncols());
    let (b_rows, b_cols) = (b.matrix.nrows(), b.matrix.ncols());
    let mut coo = CooMatrix::new(a_rows * b_rows, a_cols * b_cols);

    for (ai, aj, av) in a.matrix.triplet_iter() {
        for (bi, bj, bv) in b.matrix.triplet_iter() {
            let row = ai * b_rows + bi;
            let col = aj * b_cols + bj;
            coo.push(row, col, *av * *bv);
        }
    }

    MatrixOperator::new(SparseMatrix::from(&coo))
}

fn sparse_matvec(matrix: &SparseMatrix, x: &Vector) -> Vector {
    let mut out = Vector::zeros(matrix.nrows());
    for (row, col, value) in matrix.triplet_iter() {
        out[row] += *value * x[col];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_matrix(size: usize) -> SparseMatrix {
        let mut coo = CooMatrix::new(size, size);
        for i in 0..size {
            coo.push(i, i, 1.0);
        }
        SparseMatrix::from(&coo)
    }

    #[test]
    fn operator_with_sqrt_applies_operator() {
        let mat = identity_matrix(3);
        let op = MatrixOperator::new(mat.clone());
        let sqrt = MatrixOperator::new(mat);
        let op_with_sqrt = OperatorWithSqrt::new(op, sqrt).unwrap();
        let x = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let y = op_with_sqrt.apply(&x).unwrap();
        assert_eq!(y, x);
        let y_sqrt = op_with_sqrt.sqrt().apply(&x).unwrap();
        assert_eq!(y_sqrt, x);
    }

    #[test]
    fn kronecker_builds_correct_dimension() {
        let a = MatrixOperator::new(identity_matrix(2));
        let b = MatrixOperator::new(identity_matrix(3));
        let kron = kronecker(&a, &b);
        assert_eq!(kron.dimension(), 6);
        let v = Vector::from_vec(vec![1.0; 6]);
        let out = kron.apply(&v).unwrap();
        assert_eq!(out.len(), 6);
    }
}
