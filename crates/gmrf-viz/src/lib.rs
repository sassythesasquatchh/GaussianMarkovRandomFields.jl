//! Optional visualization helpers mirroring the Julia Makie recipes.
//!
//! This crate keeps plotting lightweight by exporting simple SVG writers
//! backed by `plotters`, focusing on mesh outlines and node-wise field
//! coloring needed for quick sanity checks in examples.

pub mod errors;
pub mod mesh;

pub use errors::VizError;
pub use mesh::{plot_field_svg, plot_mesh_svg};
pub use mesh::{FieldPlotOptions, MeshPlotOptions};

pub use gmrf_core as core;
pub use gmrf_fem as fem;
