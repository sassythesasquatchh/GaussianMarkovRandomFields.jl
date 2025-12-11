//! Matérn SPDE definitions and discretization utilities.
//!
//! The routines follow the Lindgren et al. formulation used in the Julia package:
//! build mass and stiffness matrices, combine them into a precision operator for
//! α = 1 or 2, and scale to achieve the desired marginal variance.

use std::f64::consts::PI;

use crate::discretization::FemDiscretization2d;
use crate::errors::SpdeError;
use gmrf_core::types::{SparseMatrix, Vector};
use gmrf_core::Gmrf;
use nalgebra::{DMatrix, Matrix2};
use nalgebra_sparse::CooMatrix;
use statrs::function::gamma::gamma;

/// Whittle–Matérn SPDE parameters for 2D domains.
pub struct MaternSpde2d {
    pub kappa: f64,
    pub nu: f64,
    pub sigma2: f64,
    pub diffusion_factor: Matrix2<f64>,
}

impl MaternSpde2d {
    /// Construct a Matérn SPDE with explicit κ and ν.
    pub fn from_kappa_nu(
        kappa: f64,
        nu: f64,
        sigma2: f64,
        diffusion_factor: Option<Matrix2<f64>>,
    ) -> Result<Self, SpdeError> {
        if !(kappa.is_finite() && kappa > 0.0) {
            return Err(SpdeError::InvalidParameters("κ must be positive"));
        }
        if !(nu.is_finite() && nu >= 0.0) {
            return Err(SpdeError::InvalidParameters("ν must be non-negative"));
        }
        if !(sigma2.is_finite() && sigma2 > 0.0) {
            return Err(SpdeError::InvalidParameters("σ² must be positive"));
        }

        Ok(Self {
            kappa,
            nu,
            sigma2,
            diffusion_factor: diffusion_factor.unwrap_or_else(Matrix2::identity),
        })
    }

    /// Construct a Matérn SPDE using a practical range and smoothness (integer) parameter.
    pub fn from_range_and_smoothness(
        range: f64,
        smoothness: u32,
        sigma2: f64,
        diffusion_factor: Option<Matrix2<f64>>,
    ) -> Result<Self, SpdeError> {
        let nu = smoothness_to_nu(smoothness, 2)?;
        let kappa = range_to_kappa(range, nu)?;
        Self::from_kappa_nu(kappa, nu, sigma2, diffusion_factor)
    }

    /// α recursion depth (α = ν + d/2 with d = 2). Only integer α ≥ 1 is supported.
    pub fn alpha(&self) -> Result<u32, SpdeError> {
        let alpha = self.nu + 1.0;
        let rounded = alpha.round();
        if (alpha - rounded).abs() > 1e-9 || rounded < 1.0 {
            return Err(SpdeError::InvalidParameters(
                "only integer α ≥ 1 is supported for Matérn discretization",
            ));
        }
        Ok(rounded as u32)
    }

    /// Build the precision matrix for the discretized SPDE.
    pub fn precision_matrix(&self, fem: &FemDiscretization2d) -> Result<SparseMatrix, SpdeError> {
        let alpha = self.alpha()?;
        let scale = variance_scaling(self.nu, self.kappa, self.sigma2);
        let precision = match alpha {
            1 => alpha_one_precision(self.kappa, &fem.mass, &fem.stiffness, scale),
            2 => alpha_two_precision(
                self.kappa,
                &fem.mass,
                &fem.stiffness,
                &fem.lumped_mass,
                scale,
            ),
            _ => return Err(SpdeError::UnsupportedAlpha(alpha)),
        }?;
        Ok(precision)
    }

    /// Discretize the SPDE using a precomputed FEM discretization.
    pub fn discretize(&self, fem: &FemDiscretization2d) -> Result<Gmrf, SpdeError> {
        let precision = self.precision_matrix(fem)?;
        let mean = Vector::zeros(fem.dimension());
        Ok(Gmrf::from_mean_and_precision(mean, precision)?)
    }
}

/// Convert a practical range to κ (κ = √(8ν) / range).
pub fn range_to_kappa(range: f64, nu: f64) -> Result<f64, SpdeError> {
    if !(range.is_finite() && range > 0.0) {
        return Err(SpdeError::InvalidParameters("range must be positive"));
    }
    if !(nu.is_finite() && nu >= 0.0) {
        return Err(SpdeError::InvalidParameters("ν must be non-negative"));
    }
    Ok((8.0 * nu).sqrt() / range)
}

/// Convert an integer smoothness parameter to ν for a given dimension.
pub fn smoothness_to_nu(smoothness: u32, dimension: u32) -> Result<f64, SpdeError> {
    if dimension % 2 == 0 {
        Ok((smoothness + 1) as f64)
    } else {
        Ok((smoothness as f64) + 0.5)
    }
}

pub(crate) fn variance_scaling(nu: f64, kappa: f64, sigma2_goal: f64) -> f64 {
    if nu <= 0.0 {
        return 1.0;
    }
    // For 2D, σ²_natural = Γ(ν) / (Γ(ν + 1) (4π) * κ^{2ν}).
    let sigma_natural = gamma(nu) / (gamma(nu + 1.0) * (4.0 * PI) * kappa.powf(2.0 * nu));
    sigma_natural / sigma2_goal
}

fn alpha_one_precision(
    kappa: f64,
    mass: &SparseMatrix,
    stiffness: &SparseMatrix,
    scaling: f64,
) -> Result<SparseMatrix, SpdeError> {
    let mass_dense: DMatrix<f64> = DMatrix::from(mass);
    let stiffness_dense: DMatrix<f64> = DMatrix::from(stiffness);
    let precision_dense = scaling * (kappa * kappa * mass_dense + stiffness_dense);
    Ok(dense_to_sparse(&precision_dense))
}

fn alpha_two_precision(
    kappa: f64,
    mass: &SparseMatrix,
    stiffness: &SparseMatrix,
    lumped_mass: &Vector,
    scaling: f64,
) -> Result<SparseMatrix, SpdeError> {
    let kappa2 = kappa * kappa;
    let mass_dense: DMatrix<f64> = DMatrix::from(mass);
    let stiffness_dense: DMatrix<f64> = DMatrix::from(stiffness);
    let mass_inv_diag = lumped_mass.map(|v| 1.0 / v);
    let mass_inv = DMatrix::from_diagonal(&mass_inv_diag);

    let q = kappa2 * kappa2 * &mass_dense
        + 2.0 * kappa2 * &stiffness_dense
        + &stiffness_dense * &mass_inv * &stiffness_dense;
    Ok(dense_to_sparse(&(scaling * q)))
}

fn dense_to_sparse(mat: &DMatrix<f64>) -> SparseMatrix {
    let mut coo = CooMatrix::new(mat.nrows(), mat.ncols());
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            let val = mat[(i, j)];
            if val.abs() > 1e-14 {
                coo.push(i, j, val);
            }
        }
    }
    SparseMatrix::from(&coo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::FemDiscretization2d;
    use gmrf_fem::{ElementConnectivity, Mesh2d, Point2};

    fn unit_square_mesh() -> Mesh2d {
        Mesh2d::new(
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(1.0, 1.0),
                Point2::new(0.0, 1.0),
            ],
            vec![
                ElementConnectivity::Triangle([0, 1, 2]),
                ElementConnectivity::Triangle([0, 2, 3]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn range_and_smoothness_match_nu_conversion() {
        let spde =
            MaternSpde2d::from_range_and_smoothness(0.5, 1, 1.0, None).expect("valid params");
        assert!((spde.nu - 2.0).abs() < 1e-12);
        assert!(spde.kappa.is_finite());
    }

    #[test]
    fn alpha_checks_integer_values() {
        let spde = MaternSpde2d::from_kappa_nu(1.0, 0.0, 1.0, None).unwrap();
        assert_eq!(spde.alpha().unwrap(), 1);
        let invalid = MaternSpde2d::from_kappa_nu(1.0, 0.3, 1.0, None).unwrap();
        assert!(invalid.alpha().is_err());
    }

    #[test]
    fn discretize_builds_precision() {
        let mesh = unit_square_mesh();
        let fem = FemDiscretization2d::new(mesh, None, None, None).unwrap();
        let spde = MaternSpde2d::from_kappa_nu(1.0, 1.0, 1.0, None).unwrap();
        let gmrf = spde.discretize(&fem).unwrap();
        assert_eq!(gmrf.dimension(), fem.dimension());
    }

    #[test]
    fn diffusion_factor_changes_precision() {
        let mesh = unit_square_mesh();
        let fem_iso = FemDiscretization2d::new(mesh.clone(), None, None, None).unwrap();
        let fem_scaled =
            FemDiscretization2d::new(mesh, None, Some(Matrix2::new(3.0, 0.0, 0.0, 3.0)), None)
                .unwrap();
        let spde = MaternSpde2d::from_kappa_nu(1.0, 1.0, 1.0, None).unwrap();
        let p_iso = spde.precision_matrix(&fem_iso).unwrap();
        let p_scaled = spde.precision_matrix(&fem_scaled).unwrap();

        let mut ratio_sum = 0.0;
        for ((_, _, base), (_, _, scaled)) in p_iso.triplet_iter().zip(p_scaled.triplet_iter()) {
            ratio_sum += scaled / base;
        }
        let avg_ratio = ratio_sum / (p_iso.nnz() as f64);
        assert!(avg_ratio > 1.0);
    }
}
