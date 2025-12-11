//! Latent model constructors mirrored from the Julia package.
//!
//! These helpers return `gmrf-core` fields for common priors such as AR(1),
//! random-walk, IID noise, Besag/BYM2 spatial components, and compositions
//! via block-diagonal or Kronecker products.

use gmrf_core::types::{SparseMatrix, Vector};
use gmrf_core::Gmrf;
use nalgebra_sparse::CooMatrix;

use crate::errors::LatentModelError;
use crate::graph::Neighborhood;

/// Independent Gaussian effects with common precision.
pub fn iid_gaussian(dimension: usize, precision: f64) -> Result<Gmrf, LatentModelError> {
    if dimension == 0 || !precision.is_finite() || precision <= 0.0 {
        return Err(LatentModelError::InvalidParameters(
            "IID requires positive precision and non-zero dimension",
        ));
    }

    let mut coo = CooMatrix::new(dimension, dimension);
    for i in 0..dimension {
        coo.push(i, i, precision);
    }
    build_gmrf(Vector::zeros(dimension), SparseMatrix::from(&coo))
}

/// First-order autoregressive chain with correlation `rho` and precision `tau`.
pub fn ar1_chain(length: usize, rho: f64, tau: f64) -> Result<Gmrf, LatentModelError> {
    if length == 0 {
        return Err(LatentModelError::InvalidParameters(
            "AR1 chain requires at least one node",
        ));
    }
    if !(rho.is_finite() && rho.abs() < 1.0) {
        return Err(LatentModelError::InvalidParameters(
            "|rho| must be less than one for stationarity",
        ));
    }
    if !(tau.is_finite() && tau > 0.0) {
        return Err(LatentModelError::InvalidParameters(
            "precision must be positive",
        ));
    }

    let scale = tau / (1.0 - rho * rho);
    let mut coo = CooMatrix::new(length, length);
    for i in 0..length {
        let diag = match i {
            0 | _ if i + 1 == length => 1.0,
            _ => 1.0 + rho * rho,
        };
        coo.push(i, i, scale * diag);
        if i + 1 < length {
            let val = -scale * rho;
            coo.push(i, i + 1, val);
            coo.push(i + 1, i, val);
        }
    }

    build_gmrf(Vector::zeros(length), SparseMatrix::from(&coo))
}

/// First-order random walk with precision `tau * D'D`.
pub fn rw1(length: usize, tau: f64) -> Result<Gmrf, LatentModelError> {
    if length < 2 {
        return Err(LatentModelError::InvalidParameters(
            "RW1 requires at least two nodes",
        ));
    }
    if !(tau.is_finite() && tau > 0.0) {
        return Err(LatentModelError::InvalidParameters(
            "precision must be positive",
        ));
    }

    let mut coo = CooMatrix::new(length, length);
    for i in 0..length {
        let diag = if i == 0 || i + 1 == length {
            tau
        } else {
            2.0 * tau
        };
        coo.push(i, i, diag);
        if i + 1 < length {
            let val = -tau;
            coo.push(i, i + 1, val);
            coo.push(i + 1, i, val);
        }
    }
    build_gmrf(Vector::zeros(length), SparseMatrix::from(&coo))
}

/// Intrinsic Besag model using an undirected neighbor graph.
pub fn besag(neighborhood: &Neighborhood, tau: f64) -> Result<Gmrf, LatentModelError> {
    if !(tau.is_finite() && tau > 0.0) {
        return Err(LatentModelError::InvalidParameters(
            "precision must be positive",
        ));
    }
    let n = neighborhood.node_count();
    let degrees = neighborhood.degrees();
    let mut coo = CooMatrix::new(n, n);
    for (i, deg) in degrees.iter().enumerate() {
        coo.push(i, i, tau * (*deg as f64));
    }
    for (u, v) in neighborhood.edges() {
        coo.push(u, v, -tau);
        coo.push(v, u, -tau);
    }

    build_gmrf(Vector::zeros(n), SparseMatrix::from(&coo))
}

/// BYM2-style structured + unstructured spatial model stacked in a single field.
pub fn bym2(neighborhood: &Neighborhood, rho: f64, tau: f64) -> Result<Gmrf, LatentModelError> {
    if !(0.0..=1.0).contains(&rho) {
        return Err(LatentModelError::InvalidParameters("rho must be in [0, 1]"));
    }
    if !(tau.is_finite() && tau > 0.0) {
        return Err(LatentModelError::InvalidParameters(
            "precision must be positive",
        ));
    }

    let structured = besag(neighborhood, tau)?;
    let scaled_structured = rescale_to_unit_variance(&structured)?;
    let iid = iid_gaussian(neighborhood.node_count(), tau)?;

    let combined = combine_block_diagonal(&[
        scale_precision(&scaled_structured, rho)?,
        scale_precision(&iid, 1.0 - rho)?,
    ])?;

    Ok(combined)
}

/// Block-diagonal combination of independent latent fields.
pub fn combine_block_diagonal(models: &[Gmrf]) -> Result<Gmrf, LatentModelError> {
    if models.is_empty() {
        return Err(LatentModelError::InvalidParameters(
            "at least one model is required",
        ));
    }

    let total_dim: usize = models.iter().map(|m| m.dimension()).sum();
    let mut coo = CooMatrix::new(total_dim, total_dim);
    let mut mean = Vector::zeros(total_dim);
    let mut offset = 0;

    for model in models {
        let precision = extract_matrix(model)?;
        for (i, j, v) in precision.triplet_iter() {
            coo.push(i + offset, j + offset, *v);
        }
        for (i, val) in model.mean().iter().enumerate() {
            mean[offset + i] = *val;
        }
        offset += model.dimension();
    }

    build_gmrf(mean, SparseMatrix::from(&coo))
}

/// Separable model with Kronecker precision `Q_space ⊗ Q_time`.
pub fn separable_kronecker(a: &Gmrf, b: &Gmrf) -> Result<Gmrf, LatentModelError> {
    let qa = extract_matrix(a)?;
    let qb = extract_matrix(b)?;
    let (na, nb) = (qa.nrows(), qb.nrows());
    let mut coo = CooMatrix::new(na * nb, na * nb);

    for (ia, ja, va) in qa.triplet_iter() {
        for (ib, jb, vb) in qb.triplet_iter() {
            let row = ia * nb + ib;
            let col = ja * nb + jb;
            coo.push(row, col, va * vb);
        }
    }

    build_gmrf(Vector::zeros(na * nb), SparseMatrix::from(&coo))
}

/// Fixed-effect helper that pins coefficients tightly around provided values.
pub fn fixed_effect(values: Vector) -> Result<Gmrf, LatentModelError> {
    if values.len() == 0 {
        return Err(LatentModelError::InvalidParameters(
            "fixed effect requires at least one coefficient",
        ));
    }
    let tau = 1e12; // large precision to approximate determinism
    let mut coo = CooMatrix::new(values.len(), values.len());
    for i in 0..values.len() {
        coo.push(i, i, tau);
    }
    build_gmrf(values, SparseMatrix::from(&coo))
}

fn extract_matrix(model: &Gmrf) -> Result<&SparseMatrix, LatentModelError> {
    model
        .precision()
        .as_matrix()
        .ok_or(LatentModelError::MissingPrecision)
}

fn scale_precision(model: &Gmrf, factor: f64) -> Result<Gmrf, LatentModelError> {
    let q = extract_matrix(model)?;
    let mut coo = CooMatrix::new(q.nrows(), q.ncols());
    for (i, j, v) in q.triplet_iter() {
        coo.push(i, j, factor * v);
    }
    build_gmrf(model.mean().clone(), SparseMatrix::from(&coo))
}

fn rescale_to_unit_variance(model: &Gmrf) -> Result<Gmrf, LatentModelError> {
    let q = extract_matrix(model)?;
    let mut diag_sum = 0.0;
    for (i, j, v) in q.triplet_iter() {
        if i == j {
            diag_sum += v;
        }
    }
    if diag_sum <= 0.0 {
        return Err(LatentModelError::InvalidParameters(
            "precision diagonal must be positive",
        ));
    }
    let scale = (model.dimension() as f64) / diag_sum;
    scale_precision(model, scale)
}

fn build_gmrf(mean: Vector, precision: SparseMatrix) -> Result<Gmrf, LatentModelError> {
    Gmrf::from_mean_and_precision(mean, precision)
        .map_err(|_| LatentModelError::InvalidParameters("failed to build GMRF"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iid_builds_identity_precision() {
        let gmrf = iid_gaussian(3, 2.0).unwrap();
        let q = gmrf.precision().as_matrix().unwrap();
        assert_eq!(gmrf.dimension(), 3);
        for (i, j, v) in q.triplet_iter() {
            if i == j {
                assert!((*v - 2.0).abs() < 1e-12);
            } else {
                assert!(v.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn ar1_enforces_stationarity() {
        assert!(ar1_chain(4, 1.2, 1.0).is_err());
        let gmrf = ar1_chain(4, 0.5, 2.0).unwrap();
        assert_eq!(gmrf.dimension(), 4);
    }

    #[test]
    fn rw1_precision_matches_pattern() {
        let gmrf = rw1(3, 1.0).unwrap();
        let q = gmrf.precision().as_matrix().unwrap();
        let mut diag = vec![0.0; 3];
        for (i, j, v) in q.triplet_iter() {
            if i == j {
                diag[i] = *v;
            }
        }
        assert_eq!(diag, vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn besag_uses_graph_degrees() {
        let neighborhood = Neighborhood::from_edges(3, &[(0, 1), (1, 2)]).unwrap();
        let gmrf = besag(&neighborhood, 1.0).unwrap();
        let q = gmrf.precision().as_matrix().unwrap();
        let mut diag = vec![0.0; q.nrows()];
        for (i, j, v) in q.triplet_iter() {
            if i == j {
                diag[i] = *v;
            }
        }
        assert_eq!(diag, vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn bym2_stacks_components() {
        let neighborhood = Neighborhood::from_edges(2, &[(0, 1)]).unwrap();
        let gmrf = bym2(&neighborhood, 0.7, 1.0).unwrap();
        assert_eq!(gmrf.dimension(), 4);
    }

    #[test]
    fn block_diagonal_combination_concatenates_means() {
        let a = iid_gaussian(1, 1.0).unwrap();
        let b = iid_gaussian(2, 1.0).unwrap();
        let combined = combine_block_diagonal(&[a, b]).unwrap();
        assert_eq!(combined.dimension(), 3);
        let q = combined.precision().as_matrix().unwrap();
        assert_eq!(q.nnz(), 3);
    }

    #[test]
    fn separable_builds_kronecker_precision() {
        let a = ar1_chain(2, 0.2, 1.0).unwrap();
        let b = rw1(2, 1.0).unwrap();
        let sep = separable_kronecker(&a, &b).unwrap();
        assert_eq!(sep.dimension(), 4);
    }

    #[test]
    fn fixed_effect_keeps_mean() {
        let values = Vector::from_vec(vec![1.0, -1.0]);
        let gmrf = fixed_effect(values.clone()).unwrap();
        assert_eq!(gmrf.mean(), &values);
    }
}
