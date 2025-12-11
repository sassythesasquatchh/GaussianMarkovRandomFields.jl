//! Reference shape functions and gradients for supported elements.
//!
//! These correspond to the lowest-order shapes used by Ferrite. The gradients are expressed in
//! reference coordinates and converted to physical space during assembly.

use crate::mesh::Point2;

/// Linear triangle shape values at a local coordinate (ξ, η) with ξ, η ≥ 0 and ξ + η ≤ 1.
pub fn triangle_linear_shapes(local: &Point2) -> [f64; 3] {
    let xi = local[0];
    let eta = local[1];
    [1.0 - xi - eta, xi, eta]
}

/// Reference gradients for linear triangle shape functions.
pub fn triangle_linear_gradients() -> [Point2; 3] {
    [
        Point2::new(-1.0, -1.0),
        Point2::new(1.0, 0.0),
        Point2::new(0.0, 1.0),
    ]
}

/// Bilinear quadrilateral shape values at a local coordinate on [-1, 1]².
pub fn quad_bilinear_shapes(local: &Point2) -> [f64; 4] {
    let (xi, eta) = (local[0], local[1]);
    [
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta),
    ]
}

/// Reference gradients for bilinear quadrilateral shape functions.
pub fn quad_bilinear_gradients(local: &Point2) -> [Point2; 4] {
    let (xi, eta) = (local[0], local[1]);
    [
        Point2::new(-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)),
        Point2::new(0.25 * (1.0 - eta), -0.25 * (1.0 + xi)),
        Point2::new(0.25 * (1.0 + eta), 0.25 * (1.0 + xi)),
        Point2::new(-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_shape_sum_to_one() {
        let shapes = triangle_linear_shapes(&Point2::new(0.2, 0.3));
        let sum: f64 = shapes.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn quad_shapes_match_corners() {
        let shapes = quad_bilinear_shapes(&Point2::new(-1.0, -1.0));
        assert!((shapes[0] - 1.0).abs() < 1e-12);
        assert!(shapes[1..].iter().all(|v| v.abs() < 1e-12));
    }
}
