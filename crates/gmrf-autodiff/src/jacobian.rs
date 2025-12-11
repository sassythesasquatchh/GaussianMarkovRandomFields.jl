use crate::Dual64;
use gmrf_core::Vector;
use nalgebra::DMatrix;
use thiserror::Error;

/// Errors surfaced by autodiff helpers when provided invalid inputs.
#[derive(Debug, Error)]
pub enum AutodiffError {
    /// Input/output dimension mismatch encountered during differentiation.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(&'static str),
}

/// Compute the gradient of a scalar function using forward-mode autodiff.
pub fn gradient<F, E>(point: &Vector, f: F) -> Result<Vector, E>
where
    F: Fn(&[Dual64]) -> Result<Dual64, E>,
{
    let dimension = point.len();
    let duals: Vec<Dual64> = (0..dimension)
        .map(|i| Dual64::variable(point[i], i, dimension))
        .collect();

    let output = f(&duals)?;
    Ok(Vector::from_column_slice(output.derivatives()))
}

/// Compute the Jacobian of a vector-valued function with respect to the input.
pub fn jacobian<F, E>(point: &Vector, f: F) -> Result<DMatrix<f64>, E>
where
    F: Fn(&[Dual64]) -> Result<Vec<Dual64>, E>,
    E: From<AutodiffError>,
{
    let dimension = point.len();
    let duals: Vec<Dual64> = (0..dimension)
        .map(|i| Dual64::variable(point[i], i, dimension))
        .collect();

    let outputs = f(&duals)?;
    if outputs.is_empty() {
        return Ok(DMatrix::from_element(0, dimension, 0.0));
    }

    let rows = outputs.len();
    let mut jac = DMatrix::zeros(rows, dimension);
    for (i, dual) in outputs.iter().enumerate() {
        if dual.derivatives().len() != dimension {
            return Err(AutodiffError::DimensionMismatch("gradient length mismatch").into());
        }
        for (j, deriv) in dual.derivatives().iter().enumerate() {
            jac[(i, j)] = *deriv;
        }
    }

    Ok(jac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_quadratic() {
        let point = Vector::from_vec(vec![1.0, 2.0]);
        let grad =
            gradient::<_, AutodiffError>(
                &point,
                |x| Ok(x[0].clone() * x[0].clone() + x[1].clone()),
            )
            .unwrap();
        assert_eq!(grad.len(), 2);
        assert!((grad[0] - 2.0).abs() < 1e-9);
        assert!((grad[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jacobian_tracks_linear_map() {
        let point = Vector::from_vec(vec![1.0, 2.0]);
        let jac = jacobian::<_, AutodiffError>(&point, |x| {
            Ok::<Vec<Dual64>, AutodiffError>(vec![
                x[0].clone() + Dual64::constant(2.0, x.len()),
                x[0].clone() * Dual64::constant(3.0, x.len()) + x[1].clone(),
            ])
        })
        .unwrap();
        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 2);
        assert!((jac[(0, 0)] - 1.0).abs() < 1e-9);
        assert!((jac[(1, 0)] - 3.0).abs() < 1e-9);
        assert!((jac[(1, 1)] - 1.0).abs() < 1e-9);
    }
}
