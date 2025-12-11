//! Advection–diffusion and spatiotemporal SPDE helpers.
//!
//! This module extends the Matérn discretization with a first-order advection term
//! and optional streamline diffusion stabilization. Constraint handling is mirrored
//! via diagonal penalties supplied through the FEM discretization.

use crate::discretization::FemDiscretization2d;
use crate::errors::SpdeError;
use crate::matern::variance_scaling;
use gmrf_core::types::{SparseMatrix, Vector};
use gmrf_core::Gmrf;
use nalgebra::{DMatrix, Matrix2, Vector2};
use nalgebra_sparse::CooMatrix;

/// Advection–diffusion SPDE parameters for 2D domains.
pub struct AdvectionDiffusionSpde2d {
    pub kappa: f64,
    pub alpha: u32,
    pub sigma2: f64,
    pub diffusion_factor: Matrix2<f64>,
    pub advection_velocity: Vector2<f64>,
    pub streamline_scale: f64,
}

impl AdvectionDiffusionSpde2d {
    /// Construct an advection–diffusion SPDE with validation of parameters.
    pub fn new(
        kappa: f64,
        alpha: u32,
        sigma2: f64,
        diffusion_factor: Option<Matrix2<f64>>,
        advection_velocity: Vector2<f64>,
        streamline_scale: f64,
    ) -> Result<Self, SpdeError> {
        if !(kappa.is_finite() && kappa > 0.0) {
            return Err(SpdeError::InvalidParameters("κ must be positive"));
        }
        if alpha == 0 || alpha > 2 {
            return Err(SpdeError::InvalidParameters(
                "only α = 1 or α = 2 are supported for advection–diffusion",
            ));
        }
        if !(sigma2.is_finite() && sigma2 > 0.0) {
            return Err(SpdeError::InvalidParameters("σ² must be positive"));
        }
        if !streamline_scale.is_finite() || streamline_scale < 0.0 {
            return Err(SpdeError::InvalidParameters(
                "streamline stabilization must be non-negative",
            ));
        }

        Ok(Self {
            kappa,
            alpha,
            sigma2,
            diffusion_factor: diffusion_factor.unwrap_or_else(Matrix2::identity),
            advection_velocity,
            streamline_scale,
        })
    }

    /// Build the precision matrix for the discretized advection–diffusion SPDE.
    pub fn precision_matrix(&self, fem: &FemDiscretization2d) -> Result<SparseMatrix, SpdeError> {
        let scale = variance_scaling((self.alpha as f64) - 1.0, self.kappa, self.sigma2);
        let advection = fem.advection_matrix(self.advection_velocity)?;
        let advection_sym = symmetrize(&advection);
        let stiffness = combine_symmetric(&fem.stiffness, &advection_sym);

        let stabilization = if self.streamline_scale > 0.0 {
            fem.streamline_diffusion(self.advection_velocity, self.streamline_scale)?
        } else {
            SparseMatrix::from(&CooMatrix::new(fem.dimension(), fem.dimension()))
        };
        let stiffness = combine_symmetric(&stiffness, &stabilization);

        let precision = match self.alpha {
            1 => alpha_one_precision(self.kappa, &fem.mass, &stiffness, scale),
            2 => alpha_two_precision(self.kappa, &fem.mass, &stiffness, &fem.lumped_mass, scale),
            _ => unreachable!(),
        }?;

        Ok(fem.apply_constraints(&precision))
    }

    /// Discretize the SPDE using a precomputed FEM discretization.
    pub fn discretize(&self, fem: &FemDiscretization2d) -> Result<Gmrf, SpdeError> {
        let precision = self.precision_matrix(fem)?;
        let mean = Vector::zeros(fem.dimension());
        Ok(Gmrf::from_mean_and_precision(mean, precision)?)
    }
}

fn combine_symmetric(a: &SparseMatrix, b: &SparseMatrix) -> SparseMatrix {
    let mut coo = CooMatrix::from(a);
    for (i, j, v) in b.triplet_iter() {
        coo.push(i, j, *v);
    }
    SparseMatrix::from(&coo)
}

fn symmetrize(mat: &SparseMatrix) -> SparseMatrix {
    let mut coo = CooMatrix::new(mat.nrows(), mat.ncols());
    for (i, j, v) in mat.triplet_iter() {
        coo.push(i, j, v * 0.5);
        coo.push(j, i, v * 0.5);
    }
    SparseMatrix::from(&coo)
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
    fn advection_changes_precision_entries() {
        let mesh = unit_square_mesh();
        let fem = FemDiscretization2d::new(mesh, None, None, None).unwrap();
        let base =
            AdvectionDiffusionSpde2d::new(1.0, 1, 1.0, None, Vector2::new(0.0, 0.0), 0.0).unwrap();
        let adv =
            AdvectionDiffusionSpde2d::new(1.0, 1, 1.0, None, Vector2::new(1.0, 0.5), 0.0).unwrap();

        let q_base = base.precision_matrix(&fem).unwrap();
        let q_adv = adv.precision_matrix(&fem).unwrap();
        assert!(q_base.nnz() == q_adv.nnz());

        let mut diff_norm = 0.0;
        for ((_, _, a), (_, _, b)) in q_base.triplet_iter().zip(q_adv.triplet_iter()) {
            diff_norm += (a - b).abs();
        }
        assert!(diff_norm > 0.0);
    }

    #[test]
    fn constraint_noise_inflates_diagonal() {
        let mesh = unit_square_mesh();
        let constraint_noise = Vector::from_vec(vec![0.1; mesh.num_nodes()]);
        let fem = FemDiscretization2d::new(mesh, None, None, Some(constraint_noise)).unwrap();
        let spde =
            AdvectionDiffusionSpde2d::new(1.0, 1, 1.0, None, Vector2::new(0.0, 0.0), 0.0).unwrap();
        let q = spde.precision_matrix(&fem).unwrap();
        let mut diag_min = f64::INFINITY;
        for (i, j, v) in q.triplet_iter() {
            if i == j {
                diag_min = diag_min.min(*v);
            }
        }
        assert!(diag_min > 0.0);
    }
}
