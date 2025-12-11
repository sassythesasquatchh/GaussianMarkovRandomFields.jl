//! Error types for FEM scaffolding.
//!
//! These mirror the shape of `GmrfError` in the core crate but capture mesh/assembly specific
//! validation and parsing failures.

use std::io;

use thiserror::Error;

/// Errors produced by mesh handling and FEM assembly routines.
#[derive(Debug, Error)]
pub enum FemError {
    /// Inputs are inconsistent for the requested operation (e.g., wrong element order).
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),

    /// A mesh references nodes that are not present.
    #[error("mesh connectivity references missing node indices")]
    MissingNode,

    /// Gmsh parsing failed.
    #[error("failed to parse gmsh mesh: {0}")]
    GmshParse(String),

    /// IO errors when loading mesh files.
    #[error(transparent)]
    Io(#[from] io::Error),
}
