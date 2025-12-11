//! Lightweight 2D mesh types for simplex and quadrilateral elements.
//!
//! The structures here mirror Ferrite's mesh representation: node coordinates plus element
//! connectivity. They are intentionally minimal but sufficient for mass/stiffness assembly and
//! SPDE discretizations.

use nalgebra::SVector;

use crate::errors::FemError;

/// 2D point used for node coordinates.
pub type Point2 = SVector<f64, 2>;

/// Supported cell types for 2D meshes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellType {
    Triangle,
    Quadrilateral,
}

/// Element connectivity for a 2D mesh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElementConnectivity {
    Triangle([usize; 3]),
    Quadrilateral([usize; 4]),
}

impl ElementConnectivity {
    /// Returns node indices for iteration regardless of the concrete shape.
    pub fn node_indices(&self) -> Vec<usize> {
        match self {
            ElementConnectivity::Triangle(nodes) => nodes.to_vec(),
            ElementConnectivity::Quadrilateral(nodes) => nodes.to_vec(),
        }
    }

    /// Returns the cell type for downstream dispatch.
    pub fn cell_type(&self) -> CellType {
        match self {
            ElementConnectivity::Triangle(_) => CellType::Triangle,
            ElementConnectivity::Quadrilateral(_) => CellType::Quadrilateral,
        }
    }
}

/// A 2D mesh storing node coordinates and element connectivity.
#[derive(Debug, Clone)]
pub struct Mesh2d {
    nodes: Vec<Point2>,
    elements: Vec<ElementConnectivity>,
}

impl Mesh2d {
    /// Construct a mesh from node coordinates and element connectivity, validating that all
    /// indices are within bounds.
    pub fn new(nodes: Vec<Point2>, elements: Vec<ElementConnectivity>) -> Result<Self, FemError> {
        let mesh = Self { nodes, elements };
        mesh.validate()?;
        Ok(mesh)
    }

    /// Number of nodes in the mesh.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Returns node coordinates.
    pub fn nodes(&self) -> &[Point2] {
        &self.nodes
    }

    /// Returns element connectivity.
    pub fn elements(&self) -> &[ElementConnectivity] {
        &self.elements
    }

    /// Ensure all node indices referenced by elements exist.
    fn validate(&self) -> Result<(), FemError> {
        let max_index = self.nodes.len();
        for elem in &self.elements {
            if elem.node_indices().iter().any(|&idx| idx >= max_index) {
                return Err(FemError::MissingNode);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_validation_catches_missing_nodes() {
        let nodes = vec![Point2::new(0.0, 0.0)];
        let elements = vec![ElementConnectivity::Triangle([0, 1, 2])];
        let mesh = Mesh2d::new(nodes, elements);
        assert!(matches!(mesh, Err(FemError::MissingNode)));
    }
}
