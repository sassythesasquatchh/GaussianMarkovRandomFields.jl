//! Laplace (Gaussian) approximation of the posterior using Gauss-Newton.
//!
//! This mirrors the Julia `gaussian_approximation` helper and its AD-friendly
//! rrule: we solve for the posterior mode via Fisher scoring / Gauss-Newton and
//! return a Gaussian whose precision approximates the negative Hessian at the mode.
//! The implementation relies on the `DifferentiableObservation` trait (forward-mode
//! gradients/Jacobians) to avoid hand-written derivatives of each likelihood.

use crate::autodiff::DifferentiableObservation;
use crate::errors::ObservationError;
use gmrf_core::{Gmrf, GmrfError, SparseMatrix, Vector};
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Result of a Laplace approximation run.
pub struct LaplacePosterior {
    pub posterior: Gmrf,
    pub iterations: usize,
    pub converged: bool,
    pub final_step_norm: f64,
}

#[derive(Debug, Error)]
pub enum GaussianApproximationError {
    #[error(transparent)]
    Gmrf(#[from] GmrfError),
    #[error(transparent)]
    Observation(#[from] ObservationError),
    #[error("prior precision must be available as a matrix (operators unsupported)")]
    MissingPrecisionMatrix,
    #[error("Gauss-Newton step contained non-finite values")]
    NonFiniteStep,
}

/// Convert a dense matrix into a sparse CSR matrix (dropping exact zeros).
fn dense_to_csr(mat: &DMatrix<f64>) -> SparseMatrix {
    let mut coo = nalgebra_sparse::CooMatrix::new(mat.nrows(), mat.ncols());
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            let v = mat[(i, j)];
            if v != 0.0 {
                coo.push(i, j, v);
            }
        }
    }
    SparseMatrix::from(&coo)
}

/// Perform a Laplace (Gaussian) approximation of the posterior `p(x | y)`.
///
/// - `prior`: Gaussian prior over latent field
/// - `obs`: differentiable observation model (provides gradient & residual Jacobian)
/// - `x0`: starting point for the optimization (often prior mean)
/// - `max_iter`: maximum Gauss-Newton steps
/// - `tol`: stopping threshold on step 2-norm
pub fn gaussian_approximation<O: DifferentiableObservation>(
    prior: &Gmrf,
    obs: &O,
    mut x: Vector,
    max_iter: usize,
    tol: f64,
) -> Result<LaplacePosterior, GaussianApproximationError> {
    let q = prior
        .precision_matrix()
        .ok_or(GaussianApproximationError::MissingPrecisionMatrix)?;
    let q_dense: DMatrix<f64> = DMatrix::from(q);
    let mean = prior.mean_vector().clone();

    let mut converged = false;
    let mut step_norm = f64::INFINITY;
    let mut iters_done = 0usize;

    for iter in 0..max_iter {
        iters_done = iter + 1;
        let grad_loglik: DVector<f64> = obs.log_likelihood_gradient(&x)?.into();
        let jac_residuals = obs.residual_jacobian(&x)?;
        let h_obs = jac_residuals.transpose() * jac_residuals;

        let h_total = &q_dense + h_obs;
        let chol = h_total
            .cholesky()
            .ok_or(GaussianApproximationError::Gmrf(
                GmrfError::NonPositiveDefinite,
            ))?;

        // Gradient of negative log-posterior: Q(x-μ) - ∇loglik
        let diff = &x - &mean;
        let grad_neg = &q_dense * diff - grad_loglik;

        let step = chol.solve(&grad_neg);
        step_norm = step.norm();

        if !step.iter().all(|v| v.is_finite()) {
            return Err(GaussianApproximationError::NonFiniteStep);
        }

        x -= &step;
        if step_norm < tol {
            converged = true;
            break;
        }
    }

    let h_final = {
        let jac_residuals = obs.residual_jacobian(&x)?;
        &q_dense + jac_residuals.transpose() * jac_residuals
    };

    let posterior_precision = dense_to_csr(&h_final);
    let posterior = Gmrf::from_mean_and_precision(x.clone(), posterior_precision)?;

    Ok(LaplacePosterior {
        posterior,
        iterations: iters_done,
        converged,
        final_step_norm: step_norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GaussianObservation;

    #[test]
    fn laplace_recovers_gaussian_identity_posterior() {
        let data = Vector::from_vec(vec![1.0, 2.0]);
        let obs = GaussianObservation::new(data.clone(), 1.0);
        let prior_precision = {
            let mut coo = nalgebra_sparse::CooMatrix::new(2, 2);
            coo.push(0, 0, 1.0);
            coo.push(1, 1, 1.0);
            SparseMatrix::from(&coo)
        };
        let prior = Gmrf::from_mean_and_precision(Vector::zeros(2), prior_precision).unwrap();
        let x0 = Vector::zeros(2);

        let result = gaussian_approximation(&prior, &obs, x0, 20, 1e-10).unwrap();
        assert!(result.converged);

        // Analytic posterior for N(0, I) prior and N(y, I) likelihood is N(y/2, 2I)
        let posterior_mean = result.posterior.mean_vector();
        assert!((posterior_mean[0] - 0.5).abs() < 1e-6);
        assert!((posterior_mean[1] - 1.0).abs() < 1e-6);
    }
}
