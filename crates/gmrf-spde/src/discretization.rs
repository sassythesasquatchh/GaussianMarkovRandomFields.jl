//! Lightweight FEM discretization wrapper for SPDE assembly.
//!
//! This mirrors the Julia `FEMDiscretization` helper by caching the assembled
//! mass, stiffness, and lumped mass terms needed by Matérn recursions.

use crate::errors::SpdeError;
use gmrf_core::types::{CooMatrix, SparseMatrix, Vector};
use gmrf_fem::quadrature::QuadratureRule;
use gmrf_fem::{
    assemble_advection_matrix, assemble_mass_matrix, assemble_stiffness_matrix_with_diffusion,
    assemble_streamline_diffusion_matrix, lumped_mass_vector, Mesh2d,
};
use gmrf_fem::{Matrix2, Vector2};

/// Precomputed FEM data required for SPDE discretization.
pub struct FemDiscretization2d {
    pub mesh: Mesh2d,
    pub mass: SparseMatrix,
    pub stiffness: SparseMatrix,
    pub lumped_mass: Vector,
    pub quadrature: Option<QuadratureRule>,
    pub constraint_noise: Option<Vector>,
}

impl FemDiscretization2d {
    /// Assemble FEM matrices for a 2D mesh with an optional quadrature override
    /// and diffusion factor.
    pub fn new(
        mesh: Mesh2d,
        quadrature: Option<QuadratureRule>,
        diffusion_factor: Option<Matrix2>,
        constraint_noise: Option<Vector>,
    ) -> Result<Self, SpdeError> {
        let quad_ref = quadrature.as_ref();
        let mass = assemble_mass_matrix(&mesh, quad_ref)?;
        let stiffness =
            assemble_stiffness_matrix_with_diffusion(&mesh, quad_ref, diffusion_factor)?;
        let lumped_mass = lumped_mass_vector(&mesh)?;

        Ok(Self {
            mesh,
            mass,
            stiffness,
            lumped_mass,
            quadrature,
            constraint_noise,
        })
    }

    /// Dimension of the discretized field (number of mesh nodes).
    pub fn dimension(&self) -> usize {
        self.mesh.num_nodes()
    }

    /// Assemble an advection matrix using the discretization's quadrature settings.
    pub fn advection_matrix(&self, velocity: Vector2) -> Result<SparseMatrix, SpdeError> {
        let quad_ref = self.quadrature.as_ref();
        assemble_advection_matrix(&self.mesh, quad_ref, velocity).map_err(SpdeError::from)
    }

    /// Assemble a streamline diffusion stabilization matrix using the stored quadrature.
    pub fn streamline_diffusion(
        &self,
        velocity: Vector2,
        stabilization_scale: f64,
    ) -> Result<SparseMatrix, SpdeError> {
        let quad_ref = self.quadrature.as_ref();
        assemble_streamline_diffusion_matrix(&self.mesh, quad_ref, velocity, stabilization_scale)
            .map_err(SpdeError::from)
    }

    /// Apply soft constraint noise by adding diagonal precision penalties.
    pub fn apply_constraints(&self, precision: &SparseMatrix) -> SparseMatrix {
        if let Some(noise) = &self.constraint_noise {
            let mut coo = CooMatrix::from(precision);
            for (idx, noise_val) in noise.iter().enumerate() {
                if *noise_val > 0.0 && noise_val.is_finite() {
                    let penalty = 1.0 / (noise_val * noise_val);
                    coo.push(idx, idx, penalty);
                }
            }
            SparseMatrix::from(&coo)
        } else {
            precision.clone()
        }
    }
}
