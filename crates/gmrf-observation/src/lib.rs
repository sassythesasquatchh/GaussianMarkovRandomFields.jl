//! Observation models for Gaussian Markov random fields.
//! This crate builds on `gmrf-core` and provides reusable conditioning
//! helpers for linear and nonlinear measurements.

use gmrf_core::{GaussianMarkovRandomField, GmrfError, LogDensity};
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Errors that can arise while evaluating observation models.
#[derive(Debug, Error)]
pub enum ObservationError {
    #[error(transparent)]
    Gmrf(#[from] GmrfError),
    #[error("measurement operator has incompatible shape")]
    DimensionMismatch,
}

/// Trait describing how observations are evaluated against latent states.
pub trait ObservationModel {
    /// Evaluate the log likelihood of an observation given a latent vector.
    fn log_likelihood(&self, x: &DVector<f64>) -> Result<LogDensity, ObservationError>;

    /// Produce a conditioned GMRF incorporating this observation.
    fn condition(
        &self,
        prior: &GaussianMarkovRandomField,
    ) -> Result<GaussianMarkovRandomField, ObservationError>;
}

/// Linear observation model: y = Hx + e, e ~ N(0, R)
pub struct LinearGaussianObservation {
    pub operator: DMatrix<f64>,
    pub observation: DVector<f64>,
    pub noise: DMatrix<f64>,
}

impl LinearGaussianObservation {
    fn validate_shapes(&self) -> Result<(), ObservationError> {
        if self.operator.nrows() != self.observation.nrows()
            || self.noise.nrows() != self.noise.ncols()
            || self.noise.nrows() != self.observation.nrows()
        {
            return Err(ObservationError::DimensionMismatch);
        }
        Ok(())
    }
}

impl ObservationModel for LinearGaussianObservation {
    fn log_likelihood(&self, x: &DVector<f64>) -> Result<LogDensity, ObservationError> {
        self.validate_shapes()?;
        let residual = &self.observation - &self.operator * x;
        let noise_inv = self
            .noise
            .clone()
            .cholesky()
            .ok_or(ObservationError::Gmrf(GmrfError::NotPositiveDefinite))?
            .solve(&DMatrix::identity(self.noise.nrows(), self.noise.ncols()));
        let quadratic = 0.5 * (residual.transpose() * noise_inv * residual.clone())[(0, 0)];
        let log_det = self.noise.determinant().ln();
        let log_norm =
            -0.5 * (self.noise.nrows() as f64 * (2.0 * std::f64::consts::PI).ln() + log_det);
        Ok(log_norm - quadratic)
    }

    fn condition(
        &self,
        prior: &GaussianMarkovRandomField,
    ) -> Result<GaussianMarkovRandomField, ObservationError> {
        self.validate_shapes()?;
        let posterior = prior.condition(&self.operator, &self.observation, &self.noise)?;
        Ok(posterior)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_model_improves_precision() {
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        let precision = DMatrix::identity(2, 2);
        let prior = GaussianMarkovRandomField::new(mean, precision).unwrap();

        let observation = LinearGaussianObservation {
            operator: DMatrix::from_row_slice(1, 2, &[1.0, -1.0]),
            observation: DVector::from_vec(vec![0.5]),
            noise: DMatrix::identity(1, 1) * 0.5,
        };

        let posterior = observation.condition(&prior).unwrap();
        assert_relative_eq!(posterior.precision()[(0, 0)], 3.0);
        assert_relative_eq!(posterior.precision()[(1, 1)], 3.0);
        assert_relative_eq!(posterior.precision()[(0, 1)], -2.0);
    }
}
