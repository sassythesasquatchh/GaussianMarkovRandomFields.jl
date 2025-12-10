use crate::errors::GmrfError;
use nalgebra::{Cholesky, DMatrix, DVector, MatrixMN};
use rand::Rng;
use rand_distr::StandardNormal;

/// Convenience type for reporting log densities.
pub type LogDensity = f64;

/// Dense Gaussian Markov random field representation.
#[derive(Debug, Clone)]
pub struct GaussianMarkovRandomField {
    mean: DVector<f64>,
    precision: DMatrix<f64>,
    cholesky: Cholesky<f64, nalgebra::Dyn>,
    log_det: f64,
}

impl GaussianMarkovRandomField {
    /// Build a new GMRF from a mean vector and precision matrix.
    /// A Cholesky factorization is computed eagerly to validate the
    /// matrix is symmetric positive definite and to accelerate inference.
    pub fn new(mean: DVector<f64>, precision: DMatrix<f64>) -> Result<Self, GmrfError> {
        if precision.nrows() != precision.ncols() || precision.nrows() != mean.nrows() {
            return Err(GmrfError::DimensionMismatch {
                mean_len: mean.nrows(),
                precision_rows: precision.nrows(),
                precision_cols: precision.ncols(),
            });
        }

        let cholesky = precision
            .clone()
            .cholesky()
            .ok_or(GmrfError::NotPositiveDefinite)?;
        let log_det = 2.0 * cholesky.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();

        Ok(Self {
            mean,
            precision,
            cholesky,
            log_det,
        })
    }

    /// Access the mean.
    pub fn mean(&self) -> &DVector<f64> {
        &self.mean
    }

    /// Access the precision matrix.
    pub fn precision(&self) -> &DMatrix<f64> {
        &self.precision
    }

    /// Compute the log density at `x` up to a normalizing constant.
    pub fn log_density(&self, x: &DVector<f64>) -> Result<LogDensity, GmrfError> {
        if x.nrows() != self.mean.nrows() {
            return Err(GmrfError::DimensionMismatch {
                mean_len: x.nrows(),
                precision_rows: self.precision.nrows(),
                precision_cols: self.precision.ncols(),
            });
        }
        let delta = x - &self.mean;
        let quadratic = delta.transpose() * (&self.precision * delta.clone());
        let log_norm = -0.5
            * (self.precision.nrows() as f64 * (2.0 * std::f64::consts::PI).ln() - self.log_det);
        Ok(log_norm - 0.5 * quadratic[(0, 0)])
    }

    /// Draw a sample using the stored precision factorization.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<DVector<f64>, GmrfError> {
        let l = self.cholesky.l();
        let n = l.nrows();
        let mut z = DVector::from_fn(n, |_, _| rng.sample(StandardNormal));
        let mut solution = DVector::zeros(n);

        // Solve L^T x = z, then x = L^{-T} z
        upper_triangular_solve(l.transpose(), z.as_mut_slice());
        for (i, value) in z.iter().enumerate() {
            solution[i] = *value;
        }

        Ok(&self.mean + solution)
    }

    /// Perform a Gaussian conditioning step with a linear observation
    /// model y = Hx + e, e ~ N(0, R).
    pub fn condition(
        &self,
        h: &DMatrix<f64>,
        y: &DVector<f64>,
        noise_cov: &DMatrix<f64>,
    ) -> Result<Self, GmrfError> {
        let r_inv = noise_cov
            .clone()
            .cholesky()
            .ok_or(GmrfError::NotPositiveDefinite)?
            .solve(&DMatrix::identity(noise_cov.nrows(), noise_cov.ncols()));

        let updated_precision = &self.precision + h.transpose() * &r_inv * h;
        let innovation = y - h * &self.mean;
        let updated_information =
            self.precision.clone() * self.mean.clone() + h.transpose() * r_inv * innovation;
        let updated_mean = updated_precision
            .clone()
            .cholesky()
            .ok_or(GmrfError::NotPositiveDefinite)?
            .solve(&updated_information);

        Self::new(updated_mean, updated_precision)
    }
}

fn upper_triangular_solve(mut upper: MatrixMN<f64, nalgebra::Dyn, nalgebra::Dyn>, rhs: &mut [f64]) {
    let n = upper.nrows();
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= upper[(i, j)] * rhs[j];
        }
        rhs[i] = sum / upper[(i, i)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_ulps_eq};

    #[test]
    fn log_density_matches_manual_value() {
        let mean = DVector::from_vec(vec![1.0, -1.0]);
        let precision = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 3.0]);
        let gmrf = GaussianMarkovRandomField::new(mean.clone(), precision.clone()).unwrap();

        let point = DVector::from_vec(vec![0.5, -0.5]);
        let density = gmrf.log_density(&point).unwrap();

        let delta = &point - mean;
        let quadratic = 0.5 * (delta.transpose() * &precision * delta.clone())[(0, 0)];
        let log_det = precision.determinant().ln();
        let expected = -0.5 * (2.0 * (2.0 * std::f64::consts::PI).ln() - log_det) - quadratic;
        assert_relative_eq!(density, expected, epsilon = 1e-10);
    }

    #[test]
    fn conditioning_updates_precision() {
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        let precision = DMatrix::identity(2, 2);
        let gmrf = GaussianMarkovRandomField::new(mean, precision).unwrap();

        let h = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let y = DVector::from_vec(vec![1.0]);
        let noise = DMatrix::from_element(1, 1, 0.25);
        let posterior = gmrf.condition(&h, &y, &noise).unwrap();

        assert_relative_eq!(posterior.precision[(0, 0)], 5.0, epsilon = 1e-10);
        assert_relative_eq!(posterior.precision[(1, 1)], 5.0, epsilon = 1e-10);
        assert_relative_eq!(posterior.precision[(0, 1)], 4.0, epsilon = 1e-10);
        assert_relative_eq!(posterior.precision[(1, 0)], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn samples_use_precision_factor() {
        let mean = DVector::from_vec(vec![0.0]);
        let precision = DMatrix::from_element(1, 1, 4.0);
        let gmrf = GaussianMarkovRandomField::new(mean, precision).unwrap();

        let mut rng = rand::thread_rng();
        let sample = gmrf.sample(&mut rng).unwrap();
        assert_eq!(sample.len(), 1);
    }
}
