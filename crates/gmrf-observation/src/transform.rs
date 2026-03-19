//! Transformation helpers for mapping latent fields to observation space.
//!
//! The Julia `ObservationModel` API allows stacking linear maps or FEM
//! point evaluations before applying likelihoods. Here we provide a small
//! set of reusable transforms that can be composed with observation models
//! to mirror that flexibility.

use crate::errors::ObservationError;
#[cfg(feature = "autodiff")]
use gmrf_autodiff::Dual64;
use gmrf_core::types::DenseMatrix;
use gmrf_core::Vector;

/// A transformation applied to the latent field before evaluating a likelihood.
pub trait ObservationTransform {
    /// Number of outputs produced by the transform.
    fn output_dim(&self) -> usize;

    /// Apply the transform to a latent vector.
    fn apply(&self, latent: &Vector) -> Result<Vector, ObservationError>;
}

/// Optional trait for AD-enabled transforms that can propagate dual numbers.
#[cfg(feature = "autodiff")]
pub trait DifferentiableTransform {
    /// Number of outputs produced by the transform.
    fn output_dim(&self) -> usize;

    /// Apply the transform to a dual-valued latent vector.
    fn apply_dual(&self, latent: &[Dual64]) -> Result<Vec<Dual64>, ObservationError>;
}

/// Identity transform that passes the latent values through unchanged.
pub struct IdentityTransform {
    dimension: usize,
}

impl IdentityTransform {
    /// Create an identity transform that expects a fixed latent dimension.
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl ObservationTransform for IdentityTransform {
    fn output_dim(&self) -> usize {
        self.dimension
    }

    fn apply(&self, latent: &Vector) -> Result<Vector, ObservationError> {
        if latent.len() != self.dimension {
            return Err(ObservationError::DimensionMismatch(
                "latent dimension does not match transform output",
            ));
        }
        Ok(latent.clone())
    }
}

#[cfg(feature = "autodiff")]
impl DifferentiableTransform for IdentityTransform {
    fn output_dim(&self) -> usize {
        self.dimension
    }

    fn apply_dual(&self, latent: &[Dual64]) -> Result<Vec<Dual64>, ObservationError> {
        if latent.len() != self.dimension {
            return Err(ObservationError::DimensionMismatch(
                "latent dimension does not match transform output",
            ));
        }
        Ok(latent.to_vec())
    }
}

/// Dense linear transform `y = A * x` used for design matrices or FEM evaluation.
pub struct LinearTransform {
    matrix: DenseMatrix,
}

impl LinearTransform {
    /// Create a new transform from a dense design matrix.
    pub fn new(matrix: DenseMatrix) -> Self {
        Self { matrix }
    }
}

impl ObservationTransform for LinearTransform {
    fn output_dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply(&self, latent: &Vector) -> Result<Vector, ObservationError> {
        if latent.len() != self.matrix.ncols() {
            return Err(ObservationError::DimensionMismatch(
                "latent dimension does not match transform columns",
            ));
        }
        Ok(dense_matvec(&self.matrix, latent))
    }
}

#[cfg(feature = "autodiff")]
impl DifferentiableTransform for LinearTransform {
    fn output_dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply_dual(&self, latent: &[Dual64]) -> Result<Vec<Dual64>, ObservationError> {
        if latent.len() != self.matrix.ncols() {
            return Err(ObservationError::DimensionMismatch(
                "latent dimension does not match transform columns",
            ));
        }

        let mut output = vec![Dual64::constant(0.0, latent.len()); self.matrix.nrows()];
        for (j, col) in self.matrix.as_ref().col_iter().enumerate() {
            let weight_col = col.try_as_col_major().unwrap();
            let latent_j = latent[j].clone();
            for (i, weight) in weight_col.as_slice().iter().enumerate() {
                if *weight != 0.0 {
                    output[i] = output[i].clone() + latent_j.clone() * *weight;
                }
            }
        }
        Ok(output)
    }
}

fn dense_matvec(matrix: &DenseMatrix, vector: &Vector) -> Vector {
    let mut out = Vector::zeros(matrix.nrows());
    for (j, col) in matrix.as_ref().col_iter().enumerate() {
        let xj = vector[j];
        if xj == 0.0 {
            continue;
        }
        let col = col.try_as_col_major().unwrap();
        for (i, val) in col.as_slice().iter().enumerate() {
            out[i] += *val * xj;
        }
    }
    out
}

/// Chain two transforms to mimic Julia's composite observation maps.
pub struct ComposedTransform<T1, T2>
where
    T1: ObservationTransform,
    T2: ObservationTransform,
{
    first: T1,
    second: T2,
}

impl<T1, T2> ComposedTransform<T1, T2>
where
    T1: ObservationTransform,
    T2: ObservationTransform,
{
    /// Create a composed transform that applies `first` then `second`.
    pub fn new(first: T1, second: T2) -> Self {
        Self { first, second }
    }
}

impl<T1, T2> ObservationTransform for ComposedTransform<T1, T2>
where
    T1: ObservationTransform,
    T2: ObservationTransform,
{
    fn output_dim(&self) -> usize {
        self.second.output_dim()
    }

    fn apply(&self, latent: &Vector) -> Result<Vector, ObservationError> {
        let intermediate = self.first.apply(latent)?;
        self.second.apply(&intermediate)
    }
}

#[cfg(feature = "autodiff")]
impl<T1, T2> DifferentiableTransform for ComposedTransform<T1, T2>
where
    T1: DifferentiableTransform + ObservationTransform,
    T2: DifferentiableTransform + ObservationTransform,
{
    fn output_dim(&self) -> usize {
        ObservationTransform::output_dim(&self.second)
    }

    fn apply_dual(&self, latent: &[Dual64]) -> Result<Vec<Dual64>, ObservationError> {
        let intermediate = self.first.apply_dual(latent)?;
        self.second.apply_dual(&intermediate)
    }
}
