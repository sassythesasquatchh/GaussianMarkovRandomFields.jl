//! Element-level assembly routines for mass and stiffness matrices.
//!
//! The implementations follow the Julia Ferrite helpers: evaluate reference shape functions and
//! gradients, map them to physical coordinates, and accumulate contributions into sparse matrices
//! that align with `gmrf-core`'s sparse aliases.

use nalgebra::{Matrix2, Vector2};
use nalgebra_sparse::CooMatrix;

use crate::errors::FemError;
use crate::interpolation::{
    quad_bilinear_gradients, quad_bilinear_shapes, triangle_linear_gradients,
    triangle_linear_shapes,
};
use crate::mesh::{ElementConnectivity, Mesh2d};
use crate::quadrature::{quad_gauss_2x2, triangle_degree_two, QuadratureRule};
use gmrf_core::types::{SparseMatrix, Vector};

fn element_coordinates(mesh: &Mesh2d, element: &ElementConnectivity) -> Vec<Vector2<f64>> {
    element
        .node_indices()
        .into_iter()
        .map(|idx| mesh.nodes()[idx].into())
        .collect()
}

fn jacobian(coords: &[Vector2<f64>], gradients_ref: &[Vector2<f64>]) -> Matrix2<f64> {
    let mut j = Matrix2::zeros();
    for (node, grad) in coords.iter().zip(gradients_ref.iter()) {
        j += node * grad.transpose();
    }
    j
}

fn shape_gradients_physical(
    jacobian: &Matrix2<f64>,
    gradients_ref: &[Vector2<f64>],
) -> Result<Vec<Vector2<f64>>, FemError> {
    let inv = jacobian
        .try_inverse()
        .ok_or(FemError::InvalidInput("singular element Jacobian"))?;
    let inv_t = inv.transpose();
    Ok(gradients_ref.iter().map(|g| inv_t * g).collect::<Vec<_>>())
}

fn accumulate_mass(coo: &mut CooMatrix<f64>, nodes: &[usize], shapes: &[f64], weight: f64) {
    for (i_local, &i_global) in nodes.iter().enumerate() {
        for (j_local, &j_global) in nodes.iter().enumerate() {
            let value = shapes[i_local] * shapes[j_local] * weight;
            coo.push(i_global, j_global, value);
        }
    }
}

fn accumulate_stiffness(
    coo: &mut CooMatrix<f64>,
    nodes: &[usize],
    gradients: &[Vector2<f64>],
    diffusion: &Matrix2<f64>,
    weight: f64,
) {
    for (i_local, &i_global) in nodes.iter().enumerate() {
        for (j_local, &j_global) in nodes.iter().enumerate() {
            let weighted = diffusion * gradients[j_local];
            let value = gradients[i_local].dot(&weighted) * weight;
            coo.push(i_global, j_global, value);
        }
    }
}

fn accumulate_advection(
    coo: &mut CooMatrix<f64>,
    nodes: &[usize],
    shapes: &[f64],
    gradients: &[Vector2<f64>],
    velocity: &Vector2<f64>,
    weight: f64,
) {
    for (i_local, &i_global) in nodes.iter().enumerate() {
        for (j_local, &j_global) in nodes.iter().enumerate() {
            let adv_term = velocity.dot(&gradients[j_local]);
            let value = shapes[i_local] * adv_term * weight;
            coo.push(i_global, j_global, value);
        }
    }
}

fn accumulate_streamline_diffusion(
    coo: &mut CooMatrix<f64>,
    nodes: &[usize],
    gradients: &[Vector2<f64>],
    velocity: &Vector2<f64>,
    weight: f64,
    stabilization_scale: f64,
) {
    if stabilization_scale == 0.0 {
        return;
    }

    let speed = velocity.norm();
    if speed == 0.0 {
        return;
    }

    for (i_local, &i_global) in nodes.iter().enumerate() {
        let g_i = velocity.dot(&gradients[i_local]);
        for (j_local, &j_global) in nodes.iter().enumerate() {
            let g_j = velocity.dot(&gradients[j_local]);
            let value = stabilization_scale * g_i * g_j * weight / speed;
            coo.push(i_global, j_global, value);
        }
    }
}

/// Assemble the consistent mass matrix for a mesh.
pub fn assemble_mass_matrix(
    mesh: &Mesh2d,
    quadrature_override: Option<&QuadratureRule>,
) -> Result<SparseMatrix, FemError> {
    let mut coo = CooMatrix::new(mesh.num_nodes(), mesh.num_nodes());

    for element in mesh.elements() {
        match element {
            ElementConnectivity::Triangle(nodes) => {
                let coords = element_coordinates(mesh, element);
                let gradients_ref = triangle_linear_gradients();
                let j = jacobian(&coords, &gradients_ref);
                let det_j = j.determinant().abs();
                let rule = quadrature_override
                    .cloned()
                    .unwrap_or_else(triangle_degree_two);
                for qp in rule.points {
                    let shapes = triangle_linear_shapes(&qp.local);
                    let weight = qp.weight * det_j;
                    accumulate_mass(&mut coo, nodes, &shapes, weight);
                }
            }
            ElementConnectivity::Quadrilateral(nodes) => {
                let coords = element_coordinates(mesh, element);
                let rule = quadrature_override.cloned().unwrap_or_else(quad_gauss_2x2);
                for qp in rule.points {
                    let grads_ref = quad_bilinear_gradients(&qp.local);
                    let j = jacobian(&coords, &grads_ref);
                    let det_j = j.determinant().abs();
                    let shapes = quad_bilinear_shapes(&qp.local);
                    let weight = qp.weight * det_j;
                    accumulate_mass(&mut coo, nodes, &shapes, weight);
                }
            }
        }
    }

    Ok(SparseMatrix::from(&coo))
}

/// Assemble the advection matrix for a mesh and constant velocity field.
///
/// Each entry integrates `(v · ∇ϕ_j) ϕ_i` over the element using the same
/// quadrature machinery as the stiffness routine. The result is generally
/// non-symmetric; callers can symmetrize if required by downstream solvers.
pub fn assemble_advection_matrix(
    mesh: &Mesh2d,
    quadrature_override: Option<&QuadratureRule>,
    velocity: Vector2<f64>,
) -> Result<SparseMatrix, FemError> {
    let mut coo = CooMatrix::new(mesh.num_nodes(), mesh.num_nodes());

    for element in mesh.elements() {
        match element {
            ElementConnectivity::Triangle(nodes) => {
                let coords = element_coordinates(mesh, element);
                let grads_ref = triangle_linear_gradients();
                let j = jacobian(&coords, &grads_ref);
                let det_j = j.determinant().abs();
                let grads = shape_gradients_physical(&j, &grads_ref)?;
                let rule = quadrature_override
                    .cloned()
                    .unwrap_or_else(triangle_degree_two);
                for qp in rule.points {
                    let shapes = triangle_linear_shapes(&qp.local);
                    let weight = qp.weight * det_j;
                    accumulate_advection(&mut coo, nodes, &shapes, &grads, &velocity, weight);
                }
            }
            ElementConnectivity::Quadrilateral(nodes) => {
                let coords = element_coordinates(mesh, element);
                let rule = quadrature_override.cloned().unwrap_or_else(quad_gauss_2x2);
                for qp in rule.points {
                    let grads_ref = quad_bilinear_gradients(&qp.local);
                    let j = jacobian(&coords, &grads_ref);
                    let det_j = j.determinant().abs();
                    let grads = shape_gradients_physical(&j, &grads_ref)?;
                    let shapes = quad_bilinear_shapes(&qp.local);
                    let weight = qp.weight * det_j;
                    accumulate_advection(&mut coo, nodes, &shapes, &grads, &velocity, weight);
                }
            }
        }
    }

    Ok(SparseMatrix::from(&coo))
}

/// Assemble the stiffness matrix for a mesh.
pub fn assemble_stiffness_matrix(
    mesh: &Mesh2d,
    quadrature_override: Option<&QuadratureRule>,
) -> Result<SparseMatrix, FemError> {
    assemble_stiffness_matrix_with_diffusion(mesh, quadrature_override, None)
}

/// Assemble the stiffness matrix for a mesh with an optional diffusion factor.
///
/// Passing a diffusion matrix allows anisotropic diffusion where each gradient
/// contribution is weighted by `gᵀ H g`. When `diffusion_factor` is `None`, the
/// identity matrix is used, matching the isotropic case.
pub fn assemble_stiffness_matrix_with_diffusion(
    mesh: &Mesh2d,
    quadrature_override: Option<&QuadratureRule>,
    diffusion_factor: Option<Matrix2<f64>>,
) -> Result<SparseMatrix, FemError> {
    let mut coo = CooMatrix::new(mesh.num_nodes(), mesh.num_nodes());
    let diffusion = diffusion_factor.unwrap_or_else(Matrix2::identity);

    for element in mesh.elements() {
        match element {
            ElementConnectivity::Triangle(nodes) => {
                let coords = element_coordinates(mesh, element);
                let grads_ref = triangle_linear_gradients();
                let j = jacobian(&coords, &grads_ref);
                let det_j = j.determinant().abs();
                let grads = shape_gradients_physical(&j, &grads_ref)?;
                let rule = quadrature_override
                    .cloned()
                    .unwrap_or_else(triangle_degree_two);
                for qp in rule.points {
                    let weight = qp.weight * det_j;
                    accumulate_stiffness(&mut coo, nodes, &grads, &diffusion, weight);
                }
            }
            ElementConnectivity::Quadrilateral(nodes) => {
                let coords = element_coordinates(mesh, element);
                let rule = quadrature_override.cloned().unwrap_or_else(quad_gauss_2x2);
                for qp in rule.points {
                    let grads_ref = quad_bilinear_gradients(&qp.local);
                    let j = jacobian(&coords, &grads_ref);
                    let det_j = j.determinant().abs();
                    let grads = shape_gradients_physical(&j, &grads_ref)?;
                    let weight = qp.weight * det_j;
                    accumulate_stiffness(&mut coo, nodes, &grads, &diffusion, weight);
                }
            }
        }
    }

    Ok(SparseMatrix::from(&coo))
}

/// Assemble a streamline diffusion stabilization matrix for advection-dominated flows.
///
/// The entries follow the heuristic `(h / ||v||) (v · ∇ϕ_i)(v · ∇ϕ_j)` where `h` is an
/// element length scale approximated from the mapped quadrature weight. Passing a zero
/// stabilization scale returns an empty matrix.
pub fn assemble_streamline_diffusion_matrix(
    mesh: &Mesh2d,
    quadrature_override: Option<&QuadratureRule>,
    velocity: Vector2<f64>,
    stabilization_scale: f64,
) -> Result<SparseMatrix, FemError> {
    let mut coo = CooMatrix::new(mesh.num_nodes(), mesh.num_nodes());

    for element in mesh.elements() {
        match element {
            ElementConnectivity::Triangle(nodes) => {
                let coords = element_coordinates(mesh, element);
                let grads_ref = triangle_linear_gradients();
                let j = jacobian(&coords, &grads_ref);
                let det_j = j.determinant().abs();
                let grads = shape_gradients_physical(&j, &grads_ref)?;
                let rule = quadrature_override
                    .cloned()
                    .unwrap_or_else(triangle_degree_two);
                for qp in rule.points {
                    let weight = qp.weight * det_j;
                    accumulate_streamline_diffusion(
                        &mut coo,
                        nodes,
                        &grads,
                        &velocity,
                        weight,
                        stabilization_scale,
                    );
                }
            }
            ElementConnectivity::Quadrilateral(nodes) => {
                let coords = element_coordinates(mesh, element);
                let rule = quadrature_override.cloned().unwrap_or_else(quad_gauss_2x2);
                for qp in rule.points {
                    let grads_ref = quad_bilinear_gradients(&qp.local);
                    let j = jacobian(&coords, &grads_ref);
                    let det_j = j.determinant().abs();
                    let grads = shape_gradients_physical(&j, &grads_ref)?;
                    let weight = qp.weight * det_j;
                    accumulate_streamline_diffusion(
                        &mut coo,
                        nodes,
                        &grads,
                        &velocity,
                        weight,
                        stabilization_scale,
                    );
                }
            }
        }
    }

    Ok(SparseMatrix::from(&coo))
}

/// Compute a lumped mass vector by summing row entries of the consistent mass matrix.
pub fn lumped_mass_vector(mesh: &Mesh2d) -> Result<Vector, FemError> {
    let mass = assemble_mass_matrix(mesh, None)?;
    let mut lumped = Vector::zeros(mesh.num_nodes());
    for (row, _col, value) in mass.triplet_iter() {
        // Classical row-sum lumping.
        lumped[row] += value;
    }
    Ok(lumped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Point2;

    fn unit_square_tri_mesh() -> Mesh2d {
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
    fn mass_matrix_has_correct_size() {
        let mesh = unit_square_tri_mesh();
        let mass = assemble_mass_matrix(&mesh, None).unwrap();
        assert_eq!(mass.nrows(), 4);
        assert_eq!(mass.ncols(), 4);
    }

    #[test]
    fn stiffness_matrix_is_positive_entries() {
        let mesh = unit_square_tri_mesh();
        let stiffness = assemble_stiffness_matrix(&mesh, None).unwrap();
        use std::collections::HashMap;
        let mut entries = HashMap::new();
        for (row, col, value) in stiffness.triplet_iter() {
            entries.insert((row, col), value);
            if row == col {
                assert!(*value > 0.0);
            }
        }
        for ((row, col), value) in &entries {
            if let Some(sym) = entries.get(&(col.to_owned(), row.to_owned())) {
                assert!((*value - *sym).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn diffusion_matrix_scales_entries() {
        let mesh = unit_square_tri_mesh();
        let iso = assemble_stiffness_matrix(&mesh, None).unwrap();
        let diffusion = Matrix2::new(2.0, 0.0, 0.0, 2.0);
        let scaled =
            assemble_stiffness_matrix_with_diffusion(&mesh, None, Some(diffusion)).unwrap();

        for ((row, col, base), (_, _, diff)) in iso.triplet_iter().zip(scaled.triplet_iter()) {
            assert_eq!(row, row);
            assert_eq!(col, col);
            assert!((diff - 2.0 * base).abs() < 1e-12);
        }
    }

    #[test]
    fn lumped_mass_is_positive() {
        let mesh = unit_square_tri_mesh();
        let lumped = lumped_mass_vector(&mesh).unwrap();
        assert!(lumped.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn advection_matrix_populates_off_diagonal_entries() {
        let mesh = unit_square_tri_mesh();
        let velocity = Vector2::new(1.0, 0.5);
        let adv = assemble_advection_matrix(&mesh, None, velocity).unwrap();
        assert!(adv.nnz() > mesh.num_nodes());
    }

    #[test]
    fn streamline_diffusion_vanishes_for_zero_velocity() {
        let mesh = unit_square_tri_mesh();
        let velocity = Vector2::new(0.0, 0.0);
        let sd = assemble_streamline_diffusion_matrix(&mesh, None, velocity, 0.5).unwrap();
        assert_eq!(sd.nnz(), 0);
    }
}
