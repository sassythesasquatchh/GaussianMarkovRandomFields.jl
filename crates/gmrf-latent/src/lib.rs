use gmrf_core::GaussianMarkovRandomField;
use nalgebra::{Cholesky, DMatrix, DVector};
use thiserror::Error;

/// Errors surfaced while constructing latent GMRF components.
#[derive(Debug, Error)]
pub enum LatentModelError {
    #[error("length must be positive (got {0})")]
    InvalidLength(usize),

    #[error("|phi| must be strictly less than one (got {0})")]
    InvalidPhi(f64),

    #[error(transparent)]
    Core(#[from] gmrf_core::GmrfError),
}

fn validate_length(length: usize) -> Result<(), LatentModelError> {
    if length == 0 {
        return Err(LatentModelError::InvalidLength(length));
    }
    Ok(())
}

/// Build the precision matrix for a stationary AR(1) model with coefficient `phi`
/// and innovation precision `innovation_precision`.
///
/// The resulting matrix is symmetric positive definite for `|phi| < 1`.
pub fn ar1_precision(
    length: usize,
    phi: f64,
    innovation_precision: f64,
) -> Result<DMatrix<f64>, LatentModelError> {
    validate_length(length)?;
    if !(phi.abs() < 1.0) {
        return Err(LatentModelError::InvalidPhi(phi));
    }

    let scale = innovation_precision / (1.0 - phi * phi);
    let mut precision = DMatrix::zeros(length, length);

    for i in 0..length {
        let diagonal = if i == 0 || i == length - 1 {
            1.0
        } else {
            1.0 + phi * phi
        };
        precision[(i, i)] = scale * diagonal;
        if i + 1 < length {
            precision[(i, i + 1)] = -scale * phi;
            precision[(i + 1, i)] = -scale * phi;
        }
    }

    Ok(precision)
}

/// Build the precision matrix for a first-order random walk prior
/// with precision hyperparameter `precision_scale`.
pub fn rw1_precision(
    length: usize,
    precision_scale: f64,
) -> Result<DMatrix<f64>, LatentModelError> {
    validate_length(length)?;
    let mut precision = DMatrix::zeros(length, length);

    for i in 0..(length.saturating_sub(1)) {
        precision[(i, i)] += precision_scale;
        precision[(i + 1, i + 1)] += precision_scale;
        precision[(i, i + 1)] -= precision_scale;
        precision[(i + 1, i)] -= precision_scale;
    }

    Ok(precision)
}

/// Construct a mean-zero GMRF representing an AR(1) latent process.
pub fn ar1_field(
    length: usize,
    phi: f64,
    innovation_precision: f64,
) -> Result<GaussianMarkovRandomField, LatentModelError> {
    let precision = ar1_precision(length, phi, innovation_precision)?;
    let mean = DVector::zeros(length);
    GaussianMarkovRandomField::new(mean, precision).map_err(LatentModelError::from)
}

/// Construct a mean-zero GMRF representing an RW1 latent process.
pub fn rw1_field(
    length: usize,
    precision_scale: f64,
) -> Result<GaussianMarkovRandomField, LatentModelError> {
    let precision = rw1_precision(length, precision_scale)?;
    let mean = DVector::zeros(length);
    GaussianMarkovRandomField::new(mean, precision).map_err(LatentModelError::from)
}

/// Helper to confirm positive definiteness of a candidate precision.
fn is_spd(matrix: &DMatrix<f64>) -> bool {
    Cholesky::new(matrix.clone()).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::thread_rng;

    #[test]
    fn ar1_precision_is_spd() {
        let q = ar1_precision(4, 0.7, 2.0).unwrap();
        assert!(is_spd(&q));
        assert_relative_eq!(q[(0, 0)], 2.0 / (1.0 - 0.49));
        assert_relative_eq!(q[(1, 1)], (1.0 + 0.49) * 2.0 / (1.0 - 0.49));
    }

    #[test]
    fn rw1_precision_matches_differences() {
        let q = rw1_precision(3, 5.0).unwrap();
        assert_relative_eq!(q[(0, 0)], 5.0);
        assert_relative_eq!(q[(1, 1)], 10.0);
        assert_relative_eq!(q[(2, 2)], 5.0);
        assert_relative_eq!(q[(0, 1)], -5.0);
        assert_relative_eq!(q[(1, 2)], -5.0);
    }

    #[test]
    fn ar1_field_produces_valid_gmrf() {
        let gmrf = ar1_field(5, 0.2, 1.5).unwrap();
        let sample = gmrf.sample(&mut thread_rng()).unwrap();
        assert_eq!(sample.len(), 5);
    }

    #[test]
    fn rw1_field_handles_length_one() {
        let gmrf = rw1_field(1, 0.5).unwrap();
        let log_p = gmrf.log_density(&DVector::from_vec(vec![0.1])).unwrap();
        assert!(log_p.is_finite());
    }

    #[test]
    fn invalid_phi_rejected() {
        let err = ar1_precision(3, 1.2, 1.0).unwrap_err();
        matches!(err, LatentModelError::InvalidPhi(_));
    }
}
