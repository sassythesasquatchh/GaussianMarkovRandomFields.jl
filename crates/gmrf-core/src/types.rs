//! Shared math aliases and errors for Gaussian Markov Random Field primitives.
//!
//! The gmrf workspace uses faer + faer-sparse for linear algebra. This module provides
//! thin wrappers and helpers so downstream crates don't depend on faer directly.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::cholesky::supernodal::SupernodalLltRef;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LltRef, SymbolicCholesky,
    SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::linalg::solvers::Lu as FaerSparseLu;
use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
use faer::Mat;
use faer::{get_global_parallelism, Conj, Side, Unbind};
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Sub, SubAssign};
use thiserror::Error;

/// Dense matrix alias for design matrices and Jacobians.
pub type DenseMatrix = Mat<f64>;

/// Sparse matrix wrapper (CSC) used for precision and operators.
#[derive(Clone, Debug)]
pub struct SparseMatrix {
    inner: SparseColMat<usize, f64>,
}

/// Sparse triplet used for COO-style assembly.
pub type SparseTriplet = Triplet<usize, usize, f64>;

/// Simple COO builder for sparse matrices.
#[derive(Clone, Debug)]
pub struct CooMatrix {
    nrows: usize,
    ncols: usize,
    triplets: Vec<SparseTriplet>,
}

impl CooMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            triplets: Vec::new(),
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn push(&mut self, row: usize, col: usize, value: f64) {
        self.triplets.push(SparseTriplet::new(row, col, value));
    }

    pub fn triplet_iter(&self) -> impl Iterator<Item = (usize, usize, &f64)> {
        self.triplets.iter().map(|t| (t.row, t.col, &t.val))
    }
}

impl From<&CooMatrix> for SparseMatrix {
    fn from(value: &CooMatrix) -> Self {
        let mat = SparseColMat::try_new_from_triplets(value.nrows, value.ncols, &value.triplets)
            .expect("invalid COO triplets");
        Self { inner: mat }
    }
}

impl From<SparseColMat<usize, f64>> for SparseMatrix {
    fn from(inner: SparseColMat<usize, f64>) -> Self {
        Self { inner }
    }
}

impl From<&SparseMatrix> for CooMatrix {
    fn from(value: &SparseMatrix) -> Self {
        let mut coo = CooMatrix::new(value.nrows(), value.ncols());
        for (row, col, val) in value.triplet_iter() {
            coo.push(row, col, *val);
        }
        coo
    }
}

impl SparseMatrix {
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    pub fn nnz(&self) -> usize {
        self.inner.compute_nnz()
    }

    pub fn as_ref(&self) -> SparseColMatRef<'_, usize, f64> {
        self.inner.as_ref()
    }

    pub fn triplet_iter(&self) -> SparseTripletIter<'_> {
        SparseTripletIter::new(&self.inner)
    }

    pub fn mul_vec(&self, x: &Vector) -> Vector {
        let mut out = Vector::zeros(self.nrows());
        for (row, col, value) in self.triplet_iter() {
            out[row] += *value * x[col];
        }
        out
    }

    /// Compute a sparse Cholesky factorization with a fill-reducing ordering.
    ///
    /// The returned factor stores the permutation internally, so it can be reused for
    /// solves and sampling without losing the ordering information.
    pub fn cholesky_sqrt_lower(&self) -> Result<SparseCholeskyFactor, GmrfError> {
        SparseCholeskyFactor::factorize(self)
    }

    /// Alias for `cholesky_sqrt_lower` with a clearer name.
    pub fn cholesky_factor(&self) -> Result<SparseCholeskyFactor, GmrfError> {
        self.cholesky_sqrt_lower()
    }

    /// Compute a sparse LU factorization with partial row pivoting.
    pub fn lu_factor(&self) -> Result<SparseLuFactor, GmrfError> {
        SparseLuFactor::factorize(self)
    }
}

/// Sparse Cholesky factorization with permutation support.
#[derive(Debug)]
pub struct SparseCholeskyFactor {
    symbolic: SymbolicCholesky<usize>,
    values: Vec<f64>,
}

/// Sparse LU factorization with partial row pivoting.
#[derive(Debug, Clone)]
pub struct SparseLuFactor {
    dimension: usize,
    factor: FaerSparseLu<usize, f64>,
}

impl SparseCholeskyFactor {
    /// Compute a fill-reducing sparse Cholesky factorization of `precision`.
    pub fn factorize(precision: &SparseMatrix) -> Result<Self, GmrfError> {
        if precision.nrows() != precision.ncols() {
            return Err(GmrfError::DimensionMismatch(
                "precision matrix must be square",
            ));
        }

        let side = select_cholesky_side(precision);
        let symbolic = factorize_symbolic_cholesky(
            precision.as_ref().symbolic(),
            side,
            SymmetricOrdering::default(),
            CholeskySymbolicParams::default(),
        )
        .map_err(|_| GmrfError::NonPositiveDefinite)?;

        let mut values = vec![0.0; symbolic.len_val()];
        let par = get_global_parallelism();
        let mut mem =
            MemBuffer::new(symbolic.factorize_numeric_llt_scratch::<f64>(par, Default::default()));
        let mut stack = MemStack::new(&mut mem);
        symbolic
            .factorize_numeric_llt(
                &mut values,
                precision.as_ref(),
                side,
                Default::default(),
                par,
                &mut stack,
                Default::default(),
            )
            .map_err(|_| GmrfError::NonPositiveDefinite)?;

        Ok(Self { symbolic, values })
    }

    pub fn dimension(&self) -> usize {
        self.symbolic.nrows()
    }

    /// Solve `A x = rhs` in-place using the factorization.
    pub fn solve_in_place(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        if rhs.len() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "right hand side length must match precision dimension",
            ));
        }

        let par = get_global_parallelism();
        let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, par));
        let mut stack = MemStack::new(&mut mem);
        let llt = LltRef::new(&self.symbolic, &self.values);
        llt.solve_in_place_with_conj(Conj::No, rhs.as_col_mut().as_mat_mut(), par, &mut stack);
        Ok(())
    }

    /// Solve `A x = rhs`, returning the solution.
    pub fn solve(&self, rhs: &Vector) -> Result<Vector, GmrfError> {
        let mut out = rhs.clone();
        self.solve_in_place(&mut out)?;
        Ok(out)
    }

    /// Solve `Lᵀ x = rhs` in-place using the factorization.
    pub fn solve_l_transpose_in_place(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        if rhs.len() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "right hand side length must match precision dimension",
            ));
        }

        match self.symbolic.perm() {
            Some(perm) => {
                let n = rhs.len();
                let mut permuted = Vector::zeros(n);
                for (i, fwd) in perm.arrays().0.iter().enumerate() {
                    permuted[i] = rhs[*fwd];
                }
                self.solve_l_transpose_in_place_inner(&mut permuted)?;
                for (i, inv) in perm.arrays().1.iter().enumerate() {
                    rhs[i] = permuted[*inv];
                }
            }
            None => {
                self.solve_l_transpose_in_place_inner(rhs)?;
            }
        }

        Ok(())
    }

    fn solve_l_transpose_in_place_inner(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(symbolic) => {
                let l = SparseColMatRef::new(symbolic.factor(), &self.values);
                l.transpose()
                    .sp_solve_upper_triangular_in_place(rhs.as_col_mut().as_mat_mut());
            }
            SymbolicCholeskyRaw::Supernodal(symbolic) => {
                let par = get_global_parallelism();
                let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, par));
                let mut stack = MemStack::new(&mut mem);
                let llt = SupernodalLltRef::new(symbolic, &self.values);
                llt.l_transpose_solve_with_conj(
                    Conj::No,
                    rhs.as_col_mut().as_mat_mut(),
                    par,
                    &mut stack,
                );
            }
        }

        Ok(())
    }

    /// Compute `log(det(Q))` from the Cholesky factor.
    pub fn logdet_precision(&self) -> Result<f64, GmrfError> {
        let mut acc = 0.0;
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(symbolic) => {
                let factor = symbolic.factor();
                let col_ptr = factor.col_ptr();
                for col in 0..self.dimension() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    if start == end {
                        return Err(GmrfError::NonPositiveDefinite);
                    }
                    let diag = self.values[start];
                    if diag <= 0.0 {
                        return Err(GmrfError::NonPositiveDefinite);
                    }
                    acc += diag.ln();
                }
            }
            SymbolicCholeskyRaw::Supernodal(symbolic) => {
                let llt = SupernodalLltRef::new(symbolic, &self.values);
                for s in 0..symbolic.n_supernodes() {
                    let node = llt.supernode(s);
                    let matrix = node.val();
                    let size = matrix.ncols();
                    let (top, _) = matrix.split_at_row(size);
                    for i in 0..size {
                        let idx = unsafe { faer::Idx::<usize>::new_unbound(i) };
                        let diag = top[(idx, idx)];
                        if diag <= 0.0 {
                            return Err(GmrfError::NonPositiveDefinite);
                        }
                        acc += diag.ln();
                    }
                }
            }
        }
        Ok(2.0 * acc)
    }
}

impl SparseLuFactor {
    /// Compute a sparse LU factorization of `matrix`.
    pub fn factorize(matrix: &SparseMatrix) -> Result<Self, GmrfError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(GmrfError::DimensionMismatch("matrix must be square"));
        }

        let factor = matrix
            .as_ref()
            .sp_lu()
            .map_err(|_| GmrfError::SingularMatrix)?;
        Ok(Self {
            dimension: matrix.nrows(),
            factor,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Solve `A x = rhs` in-place using the factorization.
    pub fn solve_in_place(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        if rhs.len() != self.dimension {
            return Err(GmrfError::DimensionMismatch(
                "right hand side length must match matrix dimension",
            ));
        }

        self.factor.solve_in_place(rhs.as_col_mut().as_mat_mut());
        Ok(())
    }

    /// Solve `A x = rhs`, returning the solution.
    pub fn solve(&self, rhs: &Vector) -> Result<Vector, GmrfError> {
        let mut out = rhs.clone();
        self.solve_in_place(&mut out)?;
        Ok(out)
    }
}

fn select_cholesky_side(precision: &SparseMatrix) -> Side {
    let mut has_upper = false;
    let mut has_lower = false;
    for (row, col, _) in precision.triplet_iter() {
        if row < col {
            has_upper = true;
        } else if row > col {
            has_lower = true;
        }
        if has_upper {
            break;
        }
    }
    if has_upper {
        Side::Upper
    } else if has_lower {
        Side::Lower
    } else {
        Side::Lower
    }
}

/// Iterator over triplets in a CSC matrix.
pub struct SparseTripletIter<'a> {
    mat: &'a SparseColMat<usize, f64>,
    col: usize,
    idx: usize,
}

impl<'a> SparseTripletIter<'a> {
    fn new(mat: &'a SparseColMat<usize, f64>) -> Self {
        Self {
            mat,
            col: 0,
            idx: 0,
        }
    }
}

impl<'a> Iterator for SparseTripletIter<'a> {
    type Item = (usize, usize, &'a f64);

    fn next(&mut self) -> Option<Self::Item> {
        let ncols = self.mat.ncols();
        let col_ptr = self.mat.col_ptr();
        let row_idx = self.mat.row_idx();
        let vals = self.mat.val();

        while self.col < ncols {
            let end = col_ptr[self.col + 1];
            if self.idx < end {
                let row = row_idx[self.idx];
                let val = &vals[self.idx];
                let col = self.col;
                self.idx += 1;
                return Some((row, col, val));
            }
            self.col += 1;
            if self.col < ncols {
                self.idx = col_ptr[self.col];
            }
        }
        None
    }
}

/// Dense vector used throughout the crate.
#[derive(Clone, Debug, PartialEq)]
pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    pub fn from_element(len: usize, value: f64) -> Self {
        Self {
            data: vec![value; len],
        }
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn from_fn(len: usize, mut f: impl FnMut(usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push(f(i));
        }
        Self { data }
    }

    pub fn from_iterator<I: IntoIterator<Item = f64>>(len: usize, iter: I) -> Self {
        let mut data = Vec::with_capacity(len);
        data.extend(iter);
        Self { data }
    }

    pub fn from_column_slice(slice: &[f64]) -> Self {
        Self {
            data: slice.to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.data.iter()
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len());
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn component_mul(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self { data }
    }

    pub fn component_div(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a / b)
            .collect();
        Self { data }
    }

    pub fn map(&self, mut f: impl FnMut(f64) -> f64) -> Self {
        let data = self.data.iter().copied().map(|v| f(v)).collect();
        Self { data }
    }

    pub fn as_col(&self) -> faer::ColRef<'_, f64> {
        faer::ColRef::from_slice(&self.data)
    }

    pub fn as_col_mut(&mut self) -> faer::ColMut<'_, f64> {
        faer::ColMut::from_slice_mut(&mut self.data)
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<'a, 'b> Add<&'b Vector> for &'a Vector {
    type Output = Vector;

    fn add(self, rhs: &'b Vector) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Vector { data }
    }
}

impl<'a> Add<Vector> for &'a Vector {
    type Output = Vector;

    fn add(self, rhs: Vector) -> Self::Output {
        self + &rhs
    }
}

impl<'a> Add<&'a Vector> for Vector {
    type Output = Vector;

    fn add(self, rhs: &'a Vector) -> Self::Output {
        &self + rhs
    }
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, rhs: Vector) -> Self::Output {
        &self + &rhs
    }
}

impl<'a, 'b> Sub<&'b Vector> for &'a Vector {
    type Output = Vector;

    fn sub(self, rhs: &'b Vector) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Vector { data }
    }
}

impl<'a> Sub<Vector> for &'a Vector {
    type Output = Vector;

    fn sub(self, rhs: Vector) -> Self::Output {
        self - &rhs
    }
}

impl<'a> Sub<&'a Vector> for Vector {
    type Output = Vector;

    fn sub(self, rhs: &'a Vector) -> Self::Output {
        &self - rhs
    }
}

impl Sub for Vector {
    type Output = Vector;

    fn sub(self, rhs: Vector) -> Self::Output {
        &self - &rhs
    }
}

impl AddAssign<&Vector> for Vector {
    fn add_assign(&mut self, rhs: &Vector) {
        assert_eq!(self.len(), rhs.len());
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += b;
        }
    }
}

impl AddAssign<Vector> for Vector {
    fn add_assign(&mut self, rhs: Vector) {
        *self += &rhs;
    }
}

impl SubAssign<&Vector> for Vector {
    fn sub_assign(&mut self, rhs: &Vector) {
        assert_eq!(self.len(), rhs.len());
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a -= b;
        }
    }
}

impl SubAssign<Vector> for Vector {
    fn sub_assign(&mut self, rhs: Vector) {
        *self -= &rhs;
    }
}

impl Mul<f64> for &Vector {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Self::Output {
        let data = self.data.iter().map(|v| v * rhs).collect();
        Vector { data }
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Self::Output {
        &self * rhs
    }
}

impl Mul<&Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Self::Output {
        rhs * self
    }
}

impl Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        &rhs * self
    }
}

impl<'a> Mul<&'a Vector> for &'a SparseMatrix {
    type Output = Vector;

    fn mul(self, rhs: &'a Vector) -> Self::Output {
        self.mul_vec(rhs)
    }
}

impl Div<f64> for &Vector {
    type Output = Vector;

    fn div(self, rhs: f64) -> Self::Output {
        let data = self.data.iter().map(|v| v / rhs).collect();
        Vector { data }
    }
}

impl Div<f64> for Vector {
    type Output = Vector;

    fn div(self, rhs: f64) -> Self::Output {
        &self / rhs
    }
}

impl FromIterator<f64> for Vector {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        Self {
            data: iter.into_iter().collect(),
        }
    }
}

/// Error variants produced by the core GMRF routines.
#[derive(Debug, Error)]
pub enum GmrfError {
    /// The provided inputs are inconsistent (dimension mismatch or missing data).
    #[error("inconsistent dimensions: {0}")]
    DimensionMismatch(&'static str),

    /// A precision matrix was required but not available in concrete form.
    #[error("precision matrix is required for this operation")]
    MissingPrecisionMatrix,

    /// Exact marginal variance recovery requires an explicit sparse precision matrix.
    #[error("exact variance diagonal requires an explicit sparse precision matrix")]
    ExactVarianceRequiresPrecisionMatrix,

    /// A precision factorization was required but not available.
    #[error("precision factorization is required for this operation")]
    MissingPrecisionSqrt,

    /// A sparse matrix factorization failed because the matrix was singular.
    #[error("matrix is singular")]
    SingularMatrix,

    /// Factorization failed because the precision was not positive definite.
    #[error("precision matrix is not positive definite")]
    NonPositiveDefinite,

    /// Linear equality constraints were singular under the prior covariance.
    #[error("linear constraints are singular or not full row-rank")]
    SingularConstraintSystem,

    /// A covariance-derived quantity violated basic positivity constraints beyond roundoff.
    #[error("numerical instability while computing constrained covariance: {0}")]
    NumericalInstability(&'static str),
}

#[cfg(test)]
mod tests {
    use super::{CooMatrix, SparseMatrix, Vector};

    #[test]
    fn cholesky_factor_solves_linear_system() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 3.0);
        coo.push(1, 2, 1.0);
        coo.push(2, 1, 1.0);
        coo.push(2, 2, 2.0);
        let q = SparseMatrix::from(&coo);

        let factor = q.cholesky_sqrt_lower().unwrap();
        let x = Vector::from_vec(vec![0.5, -1.2, 0.7]);
        let b = q.mul_vec(&x);
        let mut solved = b.clone();
        factor.solve_in_place(&mut solved).unwrap();
        let diff = (solved - x).norm();
        assert!(diff < 1e-10);
    }

    #[test]
    fn lu_factor_solves_nonsymmetric_linear_system() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 0.0);
        coo.push(0, 1, 2.0);
        coo.push(0, 2, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 1.0);
        coo.push(2, 0, 2.0);
        coo.push(2, 2, 1.0);
        let a = SparseMatrix::from(&coo);
        let factor = a.lu_factor().unwrap();

        let x = Vector::from_vec(vec![1.0, -2.0, 0.5]);
        let b = a.mul_vec(&x);
        let solved = factor.solve(&b).unwrap();
        assert!((solved - x).norm() < 1e-10);
    }

    #[test]
    fn lu_factor_reuses_factorization_across_multiple_rhs() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, -1.0);
        coo.push(1, 0, 2.0);
        coo.push(1, 1, 3.0);
        coo.push(1, 2, 1.0);
        coo.push(2, 0, 1.0);
        coo.push(2, 2, 2.0);
        let a = SparseMatrix::from(&coo);
        let factor = a.lu_factor().unwrap();

        let x1 = Vector::from_vec(vec![0.25, -1.0, 2.0]);
        let x2 = Vector::from_vec(vec![-0.5, 1.5, 0.75]);
        let b1 = a.mul_vec(&x1);
        let b2 = a.mul_vec(&x2);

        let solved1 = factor.solve(&b1).unwrap();
        let solved2 = factor.solve(&b2).unwrap();
        assert!((solved1 - x1).norm() < 1e-10);
        assert!((solved2 - x2).norm() < 1e-10);
    }
}
