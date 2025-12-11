//! Core Gaussian Markov Random Field container and helpers.
//!
//! This module mirrors the Julia `GMRF` struct: it stores mean information, precision data,
//! and cached factorizations for repeated solves. Sampling uses the precision factor when
//! available, while RBMC-style variance estimation leverages Hutchinson probes as a fallback.

use crate::precision::PrecisionStorage;
use crate::solver::{Solver, SolverConfig};
use crate::types::{GmrfError, SparseMatrix, Vector};
use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;

/// Gaussian Markov Random Field with precision representation and solver cache.
pub struct Gmrf {
    mean: Vector,
    precision: PrecisionStorage,
    q_sqrt: Option<SparseMatrix>,
    solver: Solver,
}

impl Gmrf {
    /// Construct a GMRF from a mean vector and sparse precision matrix.
    pub fn from_mean_and_precision(
        mean: Vector,
        precision: SparseMatrix,
    ) -> Result<Self, GmrfError> {
        let dimension = precision.nrows();
        if mean.len() != dimension || precision.ncols() != dimension {
            return Err(GmrfError::DimensionMismatch(
                "mean and precision dimensions must match",
            ));
        }

        Ok(Self {
            mean,
            precision: PrecisionStorage::Matrix(precision),
            q_sqrt: None,
            solver: Solver::default(),
        })
    }

    /// Construct a GMRF from an information vector (η) and sparse precision matrix (Q).
    /// The mean is computed by solving `Q \ η` similar to Julia's constructor overload.
    pub fn from_information_and_precision(
        information: Vector,
        precision: SparseMatrix,
    ) -> Result<Self, GmrfError> {
        let dimension = precision.nrows();
        if information.len() != dimension || precision.ncols() != dimension {
            return Err(GmrfError::DimensionMismatch(
                "information vector and precision dimensions must match",
            ));
        }

        let mut solver = Solver::default();
        let mean = solver.solve_matrix(&precision, &information)?;

        Ok(Self {
            mean,
            precision: PrecisionStorage::Matrix(precision),
            q_sqrt: None,
            solver,
        })
    }

    /// Construct a GMRF with a matrix-free precision operator.
    pub fn from_operator(
        mean: Vector,
        operator: Box<dyn crate::precision::PrecisionOperator>,
    ) -> Self {
        let dimension = operator.dimension();
        assert_eq!(
            mean.len(),
            dimension,
            "mean length must match operator dimension",
        );
        Self {
            mean,
            precision: PrecisionStorage::Operator(operator),
            q_sqrt: None,
            solver: Solver::default(),
        }
    }

    /// Provide a precision square root to enable sampling without factorizing Q.
    pub fn with_precision_sqrt(mut self, q_sqrt: SparseMatrix) -> Self {
        self.q_sqrt = Some(q_sqrt);
        self
    }

    /// Configure the solver to switch between direct and iterative algorithms.
    pub fn with_solver_config(mut self, config: SolverConfig) -> Self {
        self.solver = Solver::new(config);
        self
    }

    /// Dimension of the latent field.
    pub fn dimension(&self) -> usize {
        self.precision.dimension()
    }

    /// Mean accessor.
    pub fn mean(&self) -> &Vector {
        &self.mean
    }

    /// Access the underlying precision storage (matrix or operator).
    pub fn precision(&self) -> &PrecisionStorage {
        &self.precision
    }

    /// Generate a sample by solving `Q x = z` where `z ~ N(0, I)`, yielding covariance `Q^{-1}`.
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vector, GmrfError> {
        if let Some(q_sqrt) = &self.q_sqrt {
            return self.sample_with_sqrt(q_sqrt, rng);
        }

        match &self.precision {
            PrecisionStorage::Matrix(precision) => {
                let noise = Vector::from_fn(self.dimension(), |_, _| rng.sample(StandardNormal));
                let draw = self.solver.solve_matrix(precision, &noise)?;
                Ok(&self.mean + draw)
            }
            PrecisionStorage::Operator(operator) => {
                let noise = Vector::from_fn(self.dimension(), |_, _| rng.sample(StandardNormal));
                let draw = self.solver.solve_operator(operator.as_ref(), &noise)?;
                Ok(&self.mean + draw)
            }
        }
    }

    fn sample_with_sqrt<R: Rng + ?Sized>(
        &self,
        q_sqrt: &SparseMatrix,
        rng: &mut R,
    ) -> Result<Vector, GmrfError> {
        let dimension = q_sqrt.nrows();
        if q_sqrt.ncols() != dimension || dimension != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "precision square root must be square and match mean length",
            ));
        }

        let dense: DMatrix<f64> = DMatrix::from(q_sqrt);
        let noise = Vector::from_fn(dimension, |_, _| rng.sample(StandardNormal));
        let lu = dense.transpose().lu();
        let solved = lu.solve(&noise).ok_or(GmrfError::NonPositiveDefinite)?;
        Ok(&self.mean + solved)
    }

    /// Solve `Q x = rhs` using cached factorization when possible.
    pub fn solve_precision(&mut self, rhs: &Vector) -> Result<Vector, GmrfError> {
        match &self.precision {
            PrecisionStorage::Matrix(precision) => self.solver.solve_matrix(precision, rhs),
            PrecisionStorage::Operator(operator) => {
                self.solver.solve_operator(operator.as_ref(), rhs)
            }
        }
    }

    /// Approximate marginal variances via Hutchinson-type randomized estimates (RBMC fallback).
    pub fn rbmc_variances<R: Rng + ?Sized>(
        &mut self,
        num_samples: usize,
        rng: &mut R,
    ) -> Result<Vector, GmrfError> {
        if num_samples == 0 {
            return Err(GmrfError::DimensionMismatch(
                "at least one probe is required",
            ));
        }

        let dim = self.dimension();
        let mut variances = Vector::zeros(dim);
        for _ in 0..num_samples {
            let probe = Vector::from_fn(dim, |_, _| rng.sample(StandardNormal));
            let solved = self.solve_precision(&probe)?;
            variances += solved.component_mul(&solved);
        }

        Ok(variances / (num_samples as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::CooMatrix;
    use rand::thread_rng;

    fn identity_precision(size: usize) -> SparseMatrix {
        let mut coo = CooMatrix::new(size, size);
        for i in 0..size {
            coo.push(i, i, 1.0);
        }
        SparseMatrix::from(&coo)
    }

    #[test]
    fn builds_from_information_vector() {
        let precision = identity_precision(3);
        let info = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let gmrf = Gmrf::from_information_and_precision(info.clone(), precision).unwrap();
        assert_eq!(gmrf.dimension(), 3);
        assert_eq!(gmrf.mean(), &info);
    }

    #[test]
    fn sampling_matches_mean_length() {
        let precision = identity_precision(2);
        let mean = Vector::from_vec(vec![0.5, -0.5]);
        let mut gmrf = Gmrf::from_mean_and_precision(mean.clone(), precision).unwrap();
        let mut rng = thread_rng();
        let draw = gmrf.sample(&mut rng).unwrap();
        assert_eq!(draw.len(), 2);
        assert!((draw[0] - mean[0]).abs() < 10.0); // loose check that draw is finite
    }

    #[test]
    fn rbmc_returns_reasonable_variance() {
        let precision = identity_precision(1);
        let mean = Vector::from_vec(vec![0.0]);
        let mut gmrf = Gmrf::from_mean_and_precision(mean, precision).unwrap();
        let mut rng = thread_rng();
        let variances = gmrf.rbmc_variances(32, &mut rng).unwrap();
        assert_eq!(variances.len(), 1);
        assert!(variances[0] > 0.0);
    }
}
