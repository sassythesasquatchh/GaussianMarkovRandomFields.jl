//! Observation and conditioning helpers for Gaussian observation models.
//!
//! These utilities mirror the Julia workflow of forming H^T R^-1 H and H^T R^-1 y
//! updates when conditioning a Gaussian prior on linear observations.

use crate::types::{CooMatrix, SparseMatrix, Vector};
use faer::get_global_parallelism;
use faer::sparse::linalg::matmul::sparse_sparse_matmul;
use faer::sparse::ops::binary_op;

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
    for (row, col, value) in h.triplet_iter() {
        let weight = inv_var * y[row];
        out[col] += *value * weight;
    }
    out
}

/// Compute H^T R^-1 H for scalar observation variance.
pub fn ht_weighted_h(h: &SparseMatrix, inv_var: f64) -> SparseMatrix {
    let h_ref = h.as_ref();
    let h_transpose = h_ref
        .transpose()
        .to_col_major()
        .expect("failed to build H^T in column-major form");
    let htwh = sparse_sparse_matmul(
        h_transpose.as_ref(),
        h_ref,
        inv_var,
        get_global_parallelism(),
    )
    .expect("sparse-sparse matmul failed for H^T H");
    SparseMatrix::from(htwh)
}

fn sparse_matvec(matrix: &SparseMatrix, vector: &Vector) -> Vector {
    assert_eq!(
        matrix.ncols(),
        vector.len(),
        "sparse matvec dimension mismatch"
    );
    let mut out = Vector::zeros(matrix.nrows());
    for (row, col, value) in matrix.triplet_iter() {
        out[row] += *value * vector[col];
    }
    out
}

/// Compute H^T Q y for sparse observation precision `Q`.
pub fn ht_precision_weighted_observations(
    h: &SparseMatrix,
    y: &Vector,
    precision: &SparseMatrix,
) -> Vector {
    assert_eq!(
        precision.nrows(),
        precision.ncols(),
        "observation precision must be square"
    );
    assert_eq!(
        precision.nrows(),
        h.nrows(),
        "observation precision row count must match observation operator rows"
    );
    assert_eq!(
        y.len(),
        h.nrows(),
        "observation vector length must match observation operator rows"
    );
    let weighted = sparse_matvec(precision, y);
    let mut out = Vector::zeros(h.ncols());
    for (row, col, value) in h.triplet_iter() {
        out[col] += *value * weighted[row];
    }
    out
}

/// Compute H^T Q H for sparse observation precision `Q`.
pub fn ht_precision_weighted_h(h: &SparseMatrix, precision: &SparseMatrix) -> SparseMatrix {
    assert_eq!(
        precision.nrows(),
        precision.ncols(),
        "observation precision must be square"
    );
    assert_eq!(
        precision.nrows(),
        h.nrows(),
        "observation precision row count must match observation operator rows"
    );
    let qh = sparse_sparse_matmul(
        precision.as_ref(),
        h.as_ref(),
        1.0,
        get_global_parallelism(),
    )
    .expect("sparse-sparse matmul failed for QH");
    let h_transpose = h
        .as_ref()
        .transpose()
        .to_col_major()
        .expect("failed to build H^T in column-major form");
    let htqh = sparse_sparse_matmul(
        h_transpose.as_ref(),
        qh.as_ref(),
        1.0,
        get_global_parallelism(),
    )
    .expect("sparse-sparse matmul failed for H^T Q H");
    SparseMatrix::from(htqh)
}

/// Add two sparse matrices, preserving duplicate entries as additive contributions.
pub fn add_sparse(a: &SparseMatrix, b: &SparseMatrix) -> SparseMatrix {
    let sum = binary_op(a.as_ref(), b.as_ref(), |lhs, rhs| {
        lhs.copied().unwrap_or(0.0) + rhs.copied().unwrap_or(0.0)
    })
    .expect("sparse add failed");
    SparseMatrix::from(sum)
}

/// Apply Gaussian observation conditioning in one step.
///
/// Observations follow `y = H x + b + noise`, where `b` is optional. Returns
/// `(posterior_precision, information)` with
/// `posterior_precision = prior_precision + H^T R^-1 H` and
/// `information = H^T R^-1 (y - b)`.
pub fn apply_gaussian_observations(
    prior_precision: &SparseMatrix,
    observation_matrix: &SparseMatrix,
    observations: &Vector,
    observation_bias: Option<&Vector>,
    noise_variance: f64,
) -> (SparseMatrix, Vector) {
    let inv_var = 1.0 / noise_variance;
    let info = match observation_bias {
        Some(bias) => {
            assert_eq!(
                bias.len(),
                observations.len(),
                "observation bias length must match observations length"
            );
            let centered = observations - bias;
            ht_weighted_observations(observation_matrix, &centered, inv_var)
        }
        None => ht_weighted_observations(observation_matrix, observations, inv_var),
    };
    let htwh = ht_weighted_h(observation_matrix, inv_var);
    let posterior_precision = add_sparse(prior_precision, &htwh);
    (posterior_precision, info)
}

/// Apply Gaussian observation conditioning with an explicit sparse observation precision.
///
/// Observations follow `y = H x + b + noise`, where the noise precision is `Q_eps`.
/// Returns `(posterior_precision, information)` with
/// `posterior_precision = prior_precision + H^T Q_eps H` and
/// `information = H^T Q_eps (y - b)`.
pub fn apply_gaussian_observations_with_precision(
    prior_precision: &SparseMatrix,
    observation_matrix: &SparseMatrix,
    observations: &Vector,
    observation_bias: Option<&Vector>,
    observation_precision: &SparseMatrix,
) -> (SparseMatrix, Vector) {
    let info = match observation_bias {
        Some(bias) => {
            assert_eq!(
                bias.len(),
                observations.len(),
                "observation bias length must match observations length"
            );
            let centered = observations - bias;
            ht_precision_weighted_observations(observation_matrix, &centered, observation_precision)
        }
        None => ht_precision_weighted_observations(
            observation_matrix,
            observations,
            observation_precision,
        ),
    };
    let htqh = ht_precision_weighted_h(observation_matrix, observation_precision);
    let posterior_precision = add_sparse(prior_precision, &htqh);
    (posterior_precision, info)
}
