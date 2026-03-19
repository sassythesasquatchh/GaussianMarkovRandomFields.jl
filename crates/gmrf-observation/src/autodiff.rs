//! Optional automatic differentiation hooks for observation models.
//!
//! Julia integrates ForwardDiff/Zygote to differentiate log-likelihoods and
//! residuals. This module mirrors that behavior using the lightweight dual
//! numbers from `gmrf-autodiff` so callers can opt into forward-mode gradients
//! without pulling in heavyweight dependencies.

use crate::errors::ObservationError;
use crate::models::{
    BernoulliLogitObservation, GaussianObservation, ObservationModel, PoissonLogObservation,
};
use crate::transform::{DifferentiableTransform, ObservationTransform};
use gmrf_autodiff::{gradient, jacobian, Dual64};
use gmrf_core::types::DenseMatrix;
use gmrf_core::Vector;

/// Trait exposing AD-backed derivatives for observation models.
pub trait DifferentiableObservation: ObservationModel {
    /// Gradient of the log-likelihood with respect to the latent vector.
    fn log_likelihood_gradient(&self, latent: &Vector) -> Result<Vector, ObservationError>;

    /// Jacobian of residuals with respect to the latent vector.
    fn residual_jacobian(&self, latent: &Vector) -> Result<DenseMatrix, ObservationError>;
}

fn check_prediction_length(len: usize, expected: usize) -> Result<(), ObservationError> {
    if len != expected {
        return Err(ObservationError::DimensionMismatch(
            "prediction length must match observed data",
        ));
    }
    Ok(())
}

impl<T> DifferentiableObservation for GaussianObservation<T>
where
    T: ObservationTransform + DifferentiableTransform,
{
    fn log_likelihood_gradient(&self, latent: &Vector) -> Result<Vector, ObservationError> {
        gradient(latent, |dual_latent| {
            gaussian_log_likelihood_dual(self, dual_latent)
        })
    }

    fn residual_jacobian(&self, latent: &Vector) -> Result<DenseMatrix, ObservationError> {
        jacobian(latent, |dual_latent| {
            gaussian_residuals_dual(self, dual_latent)
        })
    }
}

impl<T> DifferentiableObservation for BernoulliLogitObservation<T>
where
    T: ObservationTransform + DifferentiableTransform,
{
    fn log_likelihood_gradient(&self, latent: &Vector) -> Result<Vector, ObservationError> {
        gradient(latent, |dual_latent| {
            bernoulli_log_likelihood_dual(self, dual_latent)
        })
    }

    fn residual_jacobian(&self, latent: &Vector) -> Result<DenseMatrix, ObservationError> {
        jacobian(latent, |dual_latent| {
            bernoulli_residuals_dual(self, dual_latent)
        })
    }
}

impl<T> DifferentiableObservation for PoissonLogObservation<T>
where
    T: ObservationTransform + DifferentiableTransform,
{
    fn log_likelihood_gradient(&self, latent: &Vector) -> Result<Vector, ObservationError> {
        gradient(latent, |dual_latent| {
            poisson_log_likelihood_dual(self, dual_latent)
        })
    }

    fn residual_jacobian(&self, latent: &Vector) -> Result<DenseMatrix, ObservationError> {
        jacobian(latent, |dual_latent| {
            poisson_residuals_dual(self, dual_latent)
        })
    }
}

fn gaussian_log_likelihood_dual<T>(
    model: &GaussianObservation<T>,
    latent: &[Dual64],
) -> Result<Dual64, ObservationError>
where
    T: ObservationTransform + DifferentiableTransform,
{
    let prediction = model.transform.apply_dual(latent)?;
    check_prediction_length(prediction.len(), model.data.len())?;

    let dimension = latent.len();
    let mut quad = Dual64::constant(0.0, dimension);
    for (p, y) in prediction.iter().zip(model.data.iter()) {
        let residual = p.clone() - *y;
        quad = quad + residual.clone() * residual;
    }
    Ok((-0.5 / model.noise_variance) * quad)
}

fn gaussian_residuals_dual<T>(
    model: &GaussianObservation<T>,
    latent: &[Dual64],
) -> Result<Vec<Dual64>, ObservationError>
where
    T: ObservationTransform + DifferentiableTransform,
{
    let prediction = model.transform.apply_dual(latent)?;
    check_prediction_length(prediction.len(), model.data.len())?;

    Ok(prediction
        .into_iter()
        .zip(model.data.iter())
        .map(|(p, y)| p - *y)
        .collect())
}

fn bernoulli_log_likelihood_dual<T>(
    model: &BernoulliLogitObservation<T>,
    latent: &[Dual64],
) -> Result<Dual64, ObservationError>
where
    T: ObservationTransform + DifferentiableTransform,
{
    let eta = model.transform.apply_dual(latent)?;
    check_prediction_length(eta.len(), model.data.len())?;

    let mut logp = Dual64::constant(0.0, latent.len());
    for (y, e) in model.data.iter().zip(eta.iter()) {
        let p = e.sigmoid();
        logp = logp + (*y * p.ln() + (1.0 - *y) * (Dual64::constant(1.0, latent.len()) - p).ln());
    }
    Ok(logp)
}

fn bernoulli_residuals_dual<T>(
    model: &BernoulliLogitObservation<T>,
    latent: &[Dual64],
) -> Result<Vec<Dual64>, ObservationError>
where
    T: ObservationTransform + DifferentiableTransform,
{
    let eta = model.transform.apply_dual(latent)?;
    check_prediction_length(eta.len(), model.data.len())?;

    Ok(eta
        .into_iter()
        .enumerate()
        .map(|(i, e)| e.sigmoid() - model.data[i])
        .collect())
}

fn poisson_log_likelihood_dual<T>(
    model: &PoissonLogObservation<T>,
    latent: &[Dual64],
) -> Result<Dual64, ObservationError>
where
    T: ObservationTransform + DifferentiableTransform,
{
    let mut eta = model.transform.apply_dual(latent)?;
    check_prediction_length(eta.len(), model.data.len())?;

    if let Some(offset) = &model.offset {
        if offset.len() != eta.len() {
            return Err(ObservationError::DimensionMismatch(
                "offset length must match observed data",
            ));
        }
        for (e, off) in eta.iter_mut().zip(offset.iter()) {
            *e = e.clone() + *off;
        }
    }

    let mut logp = Dual64::constant(0.0, latent.len());
    for (y, e) in model.data.iter().zip(eta.iter()) {
        logp = logp + (*y * e.clone()) - e.exp();
    }
    Ok(logp)
}

fn poisson_residuals_dual<T>(
    model: &PoissonLogObservation<T>,
    latent: &[Dual64],
) -> Result<Vec<Dual64>, ObservationError>
where
    T: ObservationTransform + DifferentiableTransform,
{
    let mut eta = model.transform.apply_dual(latent)?;
    check_prediction_length(eta.len(), model.data.len())?;

    if let Some(offset) = &model.offset {
        if offset.len() != eta.len() {
            return Err(ObservationError::DimensionMismatch(
                "offset length must match observed data",
            ));
        }
        for (e, off) in eta.iter_mut().zip(offset.iter()) {
            *e = e.clone() + *off;
        }
    }

    Ok(eta
        .into_iter()
        .enumerate()
        .map(|(i, e)| e.exp() - model.data[i])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{GaussianObservation, PoissonLogObservation};

    #[test]
    fn gaussian_gradient_matches_residuals() {
        let data = Vector::from_vec(vec![1.0, 2.0]);
        let model = GaussianObservation::new(data.clone(), 0.5);
        let latent = Vector::from_element(2, 0.0);
        let grad = model.log_likelihood_gradient(&latent).unwrap();
        // Gradient = -(1/sigma^2) * residual with residual = latent - data
        assert!((grad[0] - 2.0).abs() < 1e-9);
        assert!((grad[1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn poisson_jacobian_tracks_design_matrix() {
        let data = Vector::from_vec(vec![1.0, 1.0]);
        let design = DenseMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => 1.0,
            _ => 0.0,
        });
        let model = PoissonLogObservation::with_design_matrix(data.clone(), design, None);
        let latent = Vector::from_element(2, 0.0);
        let jac = model.residual_jacobian(&latent).unwrap();
        // exp(0) = 1 so residual derivative equals design rows
        assert_eq!(jac.nrows(), 2);
        let col0 = jac
            .as_ref()
            .col_iter()
            .next()
            .unwrap()
            .try_as_col_major()
            .unwrap();
        let col1 = jac
            .as_ref()
            .col_iter()
            .nth(1)
            .unwrap()
            .try_as_col_major()
            .unwrap();
        assert!((col0.as_slice()[0] - 1.0).abs() < 1e-9);
        assert!((col1.as_slice()[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bernoulli_gradient_handles_design_matrix() {
        let data = Vector::from_vec(vec![1.0, 0.0]);
        let design = DenseMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => 1.0,
            _ => 0.0,
        });
        let model = BernoulliLogitObservation::with_design_matrix(data.clone(), design);
        let latent = Vector::from_element(2, 0.0);
        let grad = model.log_likelihood_gradient(&latent).unwrap();
        // At zero logits, derivative equals y - 0.5 for each observation
        assert!((grad[0] - 0.5).abs() < 1e-9);
        assert!((grad[1] + 0.5).abs() < 1e-9);
    }
}
