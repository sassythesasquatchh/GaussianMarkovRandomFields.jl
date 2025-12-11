//! Lightweight FEM discretization wrapper for SPDE assembly.
//!
//! This mirrors the Julia `FEMDiscretization` helper by caching the assembled
//! mass, stiffness, and lumped mass terms needed by Matérn recursions.

use crate::errors::SpdeError;
use gmrf_core::types::{SparseMatrix, Vector};
use gmrf_fem::quadrature::QuadratureRule;
use gmrf_fem::{
    assemble_mass_matrix, assemble_stiffness_matrix_with_diffusion, lumped_mass_vector, Mesh2d,
};
use nalgebra::Matrix2;

/// Precomputed FEM data required for SPDE discretization.
pub struct FemDiscretization2d {
    pub mesh: Mesh2d,
    pub mass: SparseMatrix,
    pub stiffness: SparseMatrix,
    pub lumped_mass: Vector,
}

impl FemDiscretization2d {
    /// Assemble FEM matrices for a 2D mesh with an optional quadrature override
    /// and diffusion factor.
    pub fn new(
        mesh: Mesh2d,
        quadrature: Option<QuadratureRule>,
        diffusion_factor: Option<Matrix2<f64>>,
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
        })
    }

    /// Dimension of the discretized field (number of mesh nodes).
    pub fn dimension(&self) -> usize {
        self.mesh.num_nodes()
    }
}
