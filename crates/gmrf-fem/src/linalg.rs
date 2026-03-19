//! Minimal 2D linear algebra helpers to avoid pulling in a full dense backend.

use std::ops::{Add, AddAssign, Mul, Sub};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vector2 {
    pub x: f64,
    pub y: f64,
}

impl Vector2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn outer(&self, other: &Self) -> Matrix2 {
        Matrix2::new(
            self.x * other.x,
            self.x * other.y,
            self.y * other.x,
            self.y * other.y,
        )
    }
}

impl Add for Vector2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign for Vector2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vector2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<f64> for Vector2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Vector2> for f64 {
    type Output = Vector2;

    fn mul(self, rhs: Vector2) -> Self::Output {
        rhs * self
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Matrix2 {
    pub a11: f64,
    pub a12: f64,
    pub a21: f64,
    pub a22: f64,
}

impl Matrix2 {
    pub const fn new(a11: f64, a12: f64, a21: f64, a22: f64) -> Self {
        Self { a11, a12, a21, a22 }
    }

    pub const fn zeros() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0)
    }

    pub fn determinant(&self) -> f64 {
        self.a11 * self.a22 - self.a12 * self.a21
    }

    pub fn transpose(&self) -> Self {
        Self::new(self.a11, self.a21, self.a12, self.a22)
    }

    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self::new(
            self.a22 * inv_det,
            -self.a12 * inv_det,
            -self.a21 * inv_det,
            self.a11 * inv_det,
        ))
    }
}

impl Add for Matrix2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.a11 + rhs.a11,
            self.a12 + rhs.a12,
            self.a21 + rhs.a21,
            self.a22 + rhs.a22,
        )
    }
}

impl AddAssign for Matrix2 {
    fn add_assign(&mut self, rhs: Self) {
        self.a11 += rhs.a11;
        self.a12 += rhs.a12;
        self.a21 += rhs.a21;
        self.a22 += rhs.a22;
    }
}

impl Mul<Vector2> for Matrix2 {
    type Output = Vector2;

    fn mul(self, rhs: Vector2) -> Self::Output {
        Vector2::new(
            self.a11 * rhs.x + self.a12 * rhs.y,
            self.a21 * rhs.x + self.a22 * rhs.y,
        )
    }
}

impl Mul<Vector2> for &Matrix2 {
    type Output = Vector2;

    fn mul(self, rhs: Vector2) -> Self::Output {
        (*self).mul(rhs)
    }
}
