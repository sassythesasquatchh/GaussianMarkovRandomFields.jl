//! Finite-element scaffolding for the Rust port of `GaussianMarkovRandomFields.jl`.
//!
//! This crate mirrors the Julia Ferrite mesh handling helpers. It defines lightweight mesh types
//! (triangles and quadrilaterals), quadrature/interpolation utilities, basic mass and stiffness
//! assembly routines, and a minimal Gmsh importer for 2D meshes. The APIs are intentionally
//! composable so later SPDE discretizations can plug into `gmrf-core` precision construction.

pub mod assembly;
pub mod errors;
pub mod gmsh;
pub mod interpolation;
pub mod mesh;
pub mod quadrature;

pub use assembly::{
    assemble_advection_matrix, assemble_mass_matrix, assemble_stiffness_matrix,
    assemble_stiffness_matrix_with_diffusion, assemble_streamline_diffusion_matrix,
    lumped_mass_vector,
};
pub use errors::FemError;
pub use gmrf_core as core;
pub use gmsh::GmshMesh;
pub use interpolation::{
    quad_bilinear_gradients, quad_bilinear_shapes, triangle_linear_gradients,
    triangle_linear_shapes,
};
pub use mesh::{CellType, ElementConnectivity, Mesh2d, Point2};
pub use quadrature::{quad_gauss_2x2, triangle_degree_two, QuadraturePoint, QuadratureRule};
