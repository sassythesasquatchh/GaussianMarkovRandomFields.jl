//! Observation model traits and concrete likelihoods.
//!
//! The Julia package exposes a family of `ObservationModel` implementations for
//! Gaussian, Bernoulli, Poisson, and stacked observation sets, often composed
//! with FEM-based point evaluations. The types below mirror that structure with
//! dense design-matrix support and reusable transforms so they can be paired
//! with latent fields from `gmrf-core` and FEM utilities from `gmrf-fem`.

use crate::errors::ObservationError;
use crate::transform::{IdentityTransform, ObservationTransform};
use gmrf_core::Vector;
use nalgebra::DMatrix;

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Common trait for observation likelihoods.
pub trait ObservationModel {
    /// Number of observed data points.
    fn num_observations(&self) -> usize;

    /// Evaluate the log-likelihood for a latent realization.
    fn log_likelihood(&self, latent: &Vector) -> Result<f64, ObservationError>;

    /// Residuals in observation space (predicted minus observed) useful for
    /// gradient-based inference.
    fn residuals(&self, latent: &Vector) -> Result<Vector, ObservationError>;
}

/// Gaussian observations with optional design matrix.
pub struct GaussianObservation<T: ObservationTransform> {
    pub(crate) data: Vector,
    pub(crate) noise_variance: f64,
    pub(crate) transform: T,
}

impl GaussianObservation<IdentityTransform> {
    /// Construct a Gaussian observation model with identity link.
    pub fn new(data: Vector, noise_variance: f64) -> Self {
        let dimension = data.len();
        Self {
            data,
            noise_variance,
            transform: IdentityTransform::new(dimension),
        }
    }
}

impl GaussianObservation<crate::transform::LinearTransform> {
    /// Construct a Gaussian model with a dense design matrix.
    pub fn with_design_matrix(data: Vector, design: DMatrix<f64>, noise_variance: f64) -> Self {
        let transform = crate::transform::LinearTransform::new(design);
        Self {
            data,
            noise_variance,
            transform,
        }
    }
}

impl<T: ObservationTransform> ObservationModel for GaussianObservation<T> {
    fn num_observations(&self) -> usize {
        self.data.len()
    }

    fn log_likelihood(&self, latent: &Vector) -> Result<f64, ObservationError> {
        let prediction = self.transform.apply(latent)?;
        if prediction.len() != self.data.len() {
            return Err(ObservationError::DimensionMismatch(
                "prediction length must match observed data",
            ));
        }

        let residual = &prediction - &self.data;
        let scale = -0.5 / self.noise_variance;
        Ok(scale * residual.dot(&residual))
    }

    fn residuals(&self, latent: &Vector) -> Result<Vector, ObservationError> {
        let prediction = self.transform.apply(latent)?;
        if prediction.len() != self.data.len() {
            return Err(ObservationError::DimensionMismatch(
                "prediction length must match observed data",
            ));
        }
        Ok(&prediction - &self.data)
    }
}

/// Bernoulli observations with logit link.
pub struct BernoulliLogitObservation<T: ObservationTransform> {
    pub(crate) data: Vector,
    pub(crate) transform: T,
}

impl BernoulliLogitObservation<IdentityTransform> {
    /// Construct a Bernoulli-logit observation with identity predictor.
    pub fn new(data: Vector) -> Self {
        let dimension = data.len();
        Self {
            data,
            transform: IdentityTransform::new(dimension),
        }
    }
}

impl BernoulliLogitObservation<crate::transform::LinearTransform> {
    /// Construct a Bernoulli-logit observation with design matrix.
    pub fn with_design_matrix(data: Vector, design: DMatrix<f64>) -> Self {
        let transform = crate::transform::LinearTransform::new(design);
        Self { data, transform }
    }
}

impl<T: ObservationTransform> ObservationModel for BernoulliLogitObservation<T> {
    fn num_observations(&self) -> usize {
        self.data.len()
    }

    fn log_likelihood(&self, latent: &Vector) -> Result<f64, ObservationError> {
        let eta = self.transform.apply(latent)?;
        if eta.len() != self.data.len() {
            return Err(ObservationError::DimensionMismatch(
                "prediction length must match observed data",
            ));
        }

        let mut logp = 0.0;
        for (y, e) in self.data.iter().zip(eta.iter()) {
            let p = logistic(*e);
            if !(0.0..1.0).contains(&p) {
                return Err(ObservationError::InvalidProbability);
            }
            logp += y * p.ln() + (1.0 - y) * (1.0 - p).ln();
        }
        Ok(logp)
    }

    fn residuals(&self, latent: &Vector) -> Result<Vector, ObservationError> {
        let eta = self.transform.apply(latent)?;
        if eta.len() != self.data.len() {
            return Err(ObservationError::DimensionMismatch(
                "prediction length must match observed data",
            ));
        }

        let residuals: Vec<f64> = eta
            .iter()
            .enumerate()
            .map(|(i, e)| logistic(*e) - self.data[i])
            .collect();
        Ok(Vector::from_iterator(self.data.len(), residuals))
    }
}

/// Poisson observations with a log link and optional offset.
pub struct PoissonLogObservation<T: ObservationTransform> {
    pub(crate) data: Vector,
    pub(crate) offset: Option<Vector>,
    pub(crate) transform: T,
}

impl PoissonLogObservation<IdentityTransform> {
    /// Construct a Poisson observation with identity transform and optional offset.
    pub fn new(data: Vector) -> Self {
        let dimension = data.len();
        Self {
            data,
            offset: None,
            transform: IdentityTransform::new(dimension),
        }
    }

    /// Attach an offset vector in the canonical log space.
    pub fn with_offset(data: Vector, offset: Vector) -> Self {
        let dimension = data.len();
        Self {
            data,
            offset: Some(offset),
            transform: IdentityTransform::new(dimension),
        }
    }
}

impl PoissonLogObservation<crate::transform::LinearTransform> {
    /// Construct a Poisson log-link observation using a design matrix.
    pub fn with_design_matrix(data: Vector, design: DMatrix<f64>, offset: Option<Vector>) -> Self {
        let transform = crate::transform::LinearTransform::new(design);
        Self {
            data,
            offset,
            transform,
        }
    }
}

impl<T: ObservationTransform> ObservationModel for PoissonLogObservation<T> {
    fn num_observations(&self) -> usize {
        self.data.len()
    }

    fn log_likelihood(&self, latent: &Vector) -> Result<f64, ObservationError> {
        let mut eta = self.transform.apply(latent)?;
        if eta.len() != self.data.len() {
            return Err(ObservationError::DimensionMismatch(
                "prediction length must match observed data",
            ));
        }

        if let Some(offset) = &self.offset {
            if offset.len() != eta.len() {
                return Err(ObservationError::DimensionMismatch(
                    "offset length must match observed data",
                ));
            }
            eta += offset;
        }

        let mut logp = 0.0;
        for (y, e) in self.data.iter().zip(eta.iter()) {
            let rate = e.exp();
            logp += y * e - rate; // drop log(y!) constant
        }
        Ok(logp)
    }

    fn residuals(&self, latent: &Vector) -> Result<Vector, ObservationError> {
        let mut eta = self.transform.apply(latent)?;
        if eta.len() != self.data.len() {
            return Err(ObservationError::DimensionMismatch(
                "prediction length must match observed data",
            ));
        }

        if let Some(offset) = &self.offset {
            if offset.len() != eta.len() {
                return Err(ObservationError::DimensionMismatch(
                    "offset length must match observed data",
                ));
            }
            eta += offset;
        }

        let residuals: Vec<f64> = eta
            .iter()
            .enumerate()
            .map(|(i, e)| e.exp() - self.data[i])
            .collect();
        Ok(Vector::from_iterator(self.data.len(), residuals))
    }
}

/// Combine multiple observation models into a single stacked likelihood.
pub struct StackedObservations {
    models: Vec<Box<dyn ObservationModel>>,
}

impl StackedObservations {
    /// Create an empty stack of observation models.
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Build a stack from an existing collection of boxed models.
    pub fn from_models(models: Vec<Box<dyn ObservationModel>>) -> Self {
        Self { models }
    }

    /// Push a new observation model onto the stack.
    pub fn push<M: ObservationModel + 'static>(&mut self, model: M) {
        self.models.push(Box::new(model));
    }

    /// Push an already boxed observation model.
    pub fn push_box(&mut self, model: Box<dyn ObservationModel>) {
        self.models.push(model);
    }
}

impl ObservationModel for StackedObservations {
    fn num_observations(&self) -> usize {
        self.models.iter().map(|m| m.num_observations()).sum()
    }

    fn log_likelihood(&self, latent: &Vector) -> Result<f64, ObservationError> {
        let mut total = 0.0;
        for model in &self.models {
            total += model.log_likelihood(latent)?;
        }
        Ok(total)
    }

    fn residuals(&self, latent: &Vector) -> Result<Vector, ObservationError> {
        let mut all = Vec::new();
        for model in &self.models {
            let r = model.residuals(latent)?;
            all.extend(r.iter());
        }
        Ok(Vector::from_column_slice(&all))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrf_core::Vector;

    #[test]
    fn gaussian_identity_matches_data() {
        let data = Vector::from_element(3, 1.0);
        let model = GaussianObservation::new(data.clone(), 0.5);
        let latent = data.clone();
        let ll = model.log_likelihood(&latent).unwrap();
        assert!(ll.abs() < 1e-9);
    }

    #[test]
    fn gaussian_design_matrix_residuals() {
        let data = Vector::from_vec(vec![1.0, 2.0]);
        let design = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let model = GaussianObservation::with_design_matrix(data.clone(), design, 1.0);
        let latent = Vector::from_vec(vec![1.0, 3.0]);
        let residuals = model.residuals(&latent).unwrap();
        assert_eq!(residuals.len(), 2);
        assert!((residuals[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bernoulli_logit_log_likelihood() {
        let data = Vector::from_vec(vec![1.0, 0.0]);
        let design = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let model = BernoulliLogitObservation::with_design_matrix(data.clone(), design);
        let latent = Vector::from_vec(vec![10.0, -10.0]);
        let ll = model.log_likelihood(&latent).unwrap();
        assert!(ll > -1e-3);
    }

    #[test]
    fn poisson_log_residuals_with_offset() {
        let data = Vector::from_vec(vec![2.0, 3.0]);
        let offset = Vector::from_element(2, 0.0);
        let model = PoissonLogObservation::with_offset(data.clone(), offset);
        let latent = Vector::from_vec(data.iter().map(|y| y.ln()).collect());
        let residuals = model.residuals(&latent).unwrap();
        assert_eq!(residuals.len(), 2);
        assert!(residuals.iter().all(|r| r.abs() < 1e-9));
    }

    #[test]
    fn stacked_observation_accumulates() {
        let data = Vector::from_element(2, 1.0);
        let gaussian = GaussianObservation::new(data.clone(), 1.0);
        let bernoulli = BernoulliLogitObservation::new(Vector::from_vec(vec![1.0, 1.0]));

        let mut stack = StackedObservations::new();
        stack.push(gaussian);
        stack.push(bernoulli);

        let latent = Vector::from_element(2, 10.0);
        let ll = stack.log_likelihood(&latent).unwrap();
        assert!(ll.is_finite());
        assert_eq!(stack.num_observations(), 4);
    }
}
