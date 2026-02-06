//! Log-density and analytic gradients for GMRFs.
//!
//! These helpers mirror the Julia ChainRules rrules used for AD. They expose
//! value and gradients so downstream crates can plug them into AD or optimization
//! loops without re-deriving the algebra each time.

use crate::{Gmrf, GmrfError, Vector};
use nalgebra::{DMatrix, DVector};

/// Log-density of `x` under the provided GMRF (precision parameterization).
pub fn logpdf(gmrf: &Gmrf, x: &Vector) -> Result<f64, GmrfError> {
    let (q, mean) = (
        gmrf
            .precision_matrix()
            .ok_or(GmrfError::MissingPrecisionMatrix)?,
        gmrf.mean_vector(),
    );

    if x.len() != mean.len() {
        return Err(GmrfError::DimensionMismatch(
            "sample length must match mean dimension",
        ));
    }

    let q_dense: DMatrix<f64> = DMatrix::from(q);
    let diff = x - mean;
    let quad = diff.dot(&(&q_dense * &diff));

    let chol = q_dense
        .clone()
        .cholesky()
        .ok_or(GmrfError::NonPositiveDefinite)?;
    let logdet = chol
        .l()
        .diagonal()
        .iter()
        .map(|d: &f64| d.abs().ln())
        .sum::<f64>()
        * 2.0;

    let n = x.len() as f64;
    Ok(-0.5 * quad + 0.5 * logdet - 0.5 * n * (2.0 * std::f64::consts::PI).ln())
}

/// Gradients of the log-density w.r.t. latent `x`, mean `μ`, and precision `Q`.
///
/// Returns `(logpdf, ∂logp/∂x, ∂logp/∂μ, ∂logp/∂Q)` where the precision
/// gradient is dense (matching Julia's selected-inversion rrule).
pub fn logpdf_with_grads(
    gmrf: &Gmrf,
    x: &Vector,
) -> Result<(f64, Vector, Vector, DMatrix<f64>), GmrfError> {
    let (q, mean) = (
        gmrf
            .precision_matrix()
            .ok_or(GmrfError::MissingPrecisionMatrix)?,
        gmrf.mean_vector(),
    );

    if x.len() != mean.len() {
        return Err(GmrfError::DimensionMismatch(
            "sample length must match mean dimension",
        ));
    }

    let q_dense: DMatrix<f64> = DMatrix::from(q);
    let diff: DVector<f64> = x - mean;

    let chol = q_dense
        .clone()
        .cholesky()
        .ok_or(GmrfError::NonPositiveDefinite)?;
    let q_inv = chol.inverse();

    let logp = {
        let quad = diff.dot(&(&q_dense * &diff));
        let logdet = chol
            .l()
            .diagonal()
            .iter()
            .map(|d: &f64| d.abs().ln())
            .sum::<f64>()
            * 2.0;
        let n = x.len() as f64;
        -0.5 * quad + 0.5 * logdet - 0.5 * n * (2.0 * std::f64::consts::PI).ln()
    };

    // ∂logp/∂x = -Q (x - μ)
    let grad_x = -&(&q_dense * &diff);
    // ∂logp/∂μ = Q (x - μ)
    let grad_mu = &q_dense * &diff;
    // ∂logp/∂Q = 0.5 * (Q⁻¹ - (x-μ)(x-μ)ᵀ)
    let rr_t = &diff * diff.transpose();
    let grad_q = 0.5 * (q_inv - rr_t);

    Ok((logp, grad_x, grad_mu, grad_q))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SparseMatrix;
    use nalgebra_sparse::CooMatrix;

    fn identity_precision(dim: usize) -> SparseMatrix {
        let mut coo = CooMatrix::new(dim, dim);
        for i in 0..dim {
            coo.push(i, i, 1.0);
        }
        SparseMatrix::from(&coo)
    }

    #[test]
    fn logpdf_matches_standard_normal() {
        let q = identity_precision(2);
        let mean = Vector::from_vec(vec![0.0, 0.0]);
        let gmrf = Gmrf::from_mean_and_precision(mean, q).unwrap();
        let x = Vector::from_vec(vec![1.0, 0.0]);
        let logp = logpdf(&gmrf, &x).unwrap();
        // N(0, I) at x = [1, 0] has logpdf = -0.5*1 - log(2π)
        let expected = -0.5 - (2.0 * std::f64::consts::PI).ln();
        assert!((logp - expected).abs() < 1e-6);
    }

    #[test]
    fn gradients_point_opposite_signs_for_x_and_mean() {
        let q = identity_precision(1);
        let mean = Vector::from_vec(vec![0.5]);
        let gmrf = Gmrf::from_mean_and_precision(mean, q).unwrap();
        let x = Vector::from_vec(vec![1.5]);
        let (_, grad_x, grad_mu, grad_q) = logpdf_with_grads(&gmrf, &x).unwrap();

        // grad_x should be negative of grad_mu
        assert!((grad_x[0] + grad_mu[0]).abs() < 1e-9);
        // Precision gradient for 1D reduces to 0.5 * (1/Q - r^2)
        let r = x[0] - gmrf.mean_vector()[0];
        let expected_grad_q = 0.5 * (1.0 - r * r);
        assert!((grad_q[(0, 0)] - expected_grad_q).abs() < 1e-9);
    }
}
