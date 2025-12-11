//! Quadrature rules for common 2D reference elements.
//!
//! These mirror Ferrite's built-in rules and are sufficient for first-order mass/stiffness
//! assembly. Additional rules can be added when higher-order elements are introduced.

use crate::mesh::Point2;

/// A single quadrature point and weight on the reference element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraturePoint {
    /// Local coordinates (ξ, η) on the reference element.
    pub local: Point2,
    /// Quadrature weight.
    pub weight: f64,
}

/// A quadrature rule consisting of points and weights.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadratureRule {
    pub points: Vec<QuadraturePoint>,
}

impl QuadratureRule {
    /// Create a new rule from weights and local coordinates.
    pub fn new(points: Vec<QuadraturePoint>) -> Self {
        Self { points }
    }
}

/// One-point degree-two quadrature on the reference triangle (barycenter rule).
pub fn triangle_degree_two() -> QuadratureRule {
    QuadratureRule {
        points: vec![QuadraturePoint {
            local: Point2::new(1.0 / 3.0, 1.0 / 3.0),
            weight: 0.5,
        }],
    }
}

/// 2x2 Gauss-Legendre quadrature for the reference square [-1, 1]².
pub fn quad_gauss_2x2() -> QuadratureRule {
    let a = 1.0 / f64::sqrt(3.0);
    let points = vec![
        QuadraturePoint {
            local: Point2::new(-a, -a),
            weight: 1.0,
        },
        QuadraturePoint {
            local: Point2::new(a, -a),
            weight: 1.0,
        },
        QuadraturePoint {
            local: Point2::new(-a, a),
            weight: 1.0,
        },
        QuadraturePoint {
            local: Point2::new(a, a),
            weight: 1.0,
        },
    ];
    QuadratureRule { points }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_rule_has_expected_weight() {
        let rule = triangle_degree_two();
        assert_eq!(rule.points.len(), 1);
        assert!((rule.points[0].weight - 0.5).abs() < 1e-12);
    }

    #[test]
    fn quad_rule_has_four_points() {
        let rule = quad_gauss_2x2();
        assert_eq!(rule.points.len(), 4);
    }
}
