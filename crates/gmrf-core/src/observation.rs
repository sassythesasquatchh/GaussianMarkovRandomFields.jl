//! Observation and conditioning helpers for Gaussian observation models.
//!
//! These utilities mirror the Julia workflow of forming H^T R^-1 H and H^T R^-1 y
//! updates when conditioning a Gaussian prior on linear observations.

use crate::types::{SparseMatrix, Vector};
use nalgebra_sparse::CooMatrix;

/// Build an observation selector matrix that picks `indices` from a latent vector.
pub fn observation_selector(dimension: usize, indices: &[usize]) -> SparseMatrix {
    let mut coo = CooMatrix::new(indices.len(), dimension);
    for (row, idx) in indices.iter().copied().enumerate() {
        coo.push(row, idx, 1.0);
    }
    SparseMatrix::from(&coo)
}

/// Build a sparse linear observation matrix with two-point stencils per row.
///
/// Each row is defined by `(left, right, weight_right)` so that the row evaluates
/// `weight_left * x[left] + weight_right * x[right]`, where `weight_left = 1 - weight_right`.
pub fn build_linear_observation_matrix(
    dimension: usize,
    rows: &[(usize, usize, f64)],
) -> SparseMatrix {
    let mut coo = CooMatrix::new(rows.len(), dimension);
    for (row, (left, right, weight_right)) in rows.iter().enumerate() {
        let weight_right = *weight_right;
        let weight_left = 1.0 - weight_right;
        coo.push(row, *left, weight_left);
        coo.push(row, *right, weight_right);
    }
    SparseMatrix::from(&coo)
}

/// Compute H^T R^-1 y for scalar observation variance.
pub fn ht_weighted_observations(h: &SparseMatrix, y: &Vector, inv_var: f64) -> Vector {
    let mut out = Vector::zeros(h.ncols());
    for row in 0..h.nrows() {
        let weight = inv_var * y[row];
        let row_view = h.row(row);
        for (col, value) in row_view.col_indices().iter().zip(row_view.values().iter()) {
            out[*col] += value * weight;
        }
    }
    out
}

/// Compute H^T R^-1 H for scalar observation variance.
pub fn ht_weighted_h(h: &SparseMatrix, inv_var: f64) -> SparseMatrix {
    let mut coo = CooMatrix::new(h.ncols(), h.ncols());
    for row in 0..h.nrows() {
        let row_view = h.row(row);
        let cols = row_view.col_indices();
        let vals = row_view.values();
        for i in 0..cols.len() {
            for j in 0..cols.len() {
                coo.push(cols[i], cols[j], inv_var * vals[i] * vals[j]);
            }
        }
    }
    SparseMatrix::from(&coo)
}

/// Add two sparse matrices, preserving duplicate entries as additive contributions.
pub fn add_sparse(a: &SparseMatrix, b: &SparseMatrix) -> SparseMatrix {
    let mut coo = CooMatrix::from(a);
    for (i, j, v) in b.triplet_iter() {
        coo.push(i, j, *v);
    }
    SparseMatrix::from(&coo)
}

/// Apply Gaussian observation conditioning in one step.
///
/// Returns `(posterior_precision, information)` where
/// `posterior_precision = prior_precision + H^T R^-1 H` and `information = H^T R^-1 y`.
pub fn apply_gaussian_observations(
    prior_precision: &SparseMatrix,
    observation_matrix: &SparseMatrix,
    observations: &Vector,
    noise_variance: f64,
) -> (SparseMatrix, Vector) {
    let inv_var = 1.0 / noise_variance;
    let info = ht_weighted_observations(observation_matrix, observations, inv_var);
    let htwh = ht_weighted_h(observation_matrix, inv_var);
    let posterior_precision = add_sparse(prior_precision, &htwh);
    (posterior_precision, info)
}
