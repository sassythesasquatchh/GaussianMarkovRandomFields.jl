use std::ops::{Add, Div, Mul, Neg, Sub};

/// Forward-mode dual number with a dense derivative vector.
#[derive(Clone, Debug)]
pub struct Dual64 {
    value: f64,
    derivatives: Vec<f64>,
}

impl Dual64 {
    /// Create a dual variable with derivative 1.0 at the provided index.
    pub fn variable(value: f64, index: usize, dimension: usize) -> Self {
        let mut derivatives = vec![0.0; dimension];
        derivatives[index] = 1.0;
        Self { value, derivatives }
    }

    /// Create a constant dual number with zero derivatives.
    pub fn constant(value: f64, dimension: usize) -> Self {
        Self {
            value,
            derivatives: vec![0.0; dimension],
        }
    }

    /// Extract the primal value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Access the gradient components.
    pub fn derivatives(&self) -> &[f64] {
        &self.derivatives
    }

    fn map_derivatives<F>(&self, rhs: &Dual64, value: f64, f: F) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        let derivatives = self
            .derivatives
            .iter()
            .zip(rhs.derivatives.iter())
            .map(|(a, b)| f(*a, *b))
            .collect();
        Self { value, derivatives }
    }

    /// Exponential of a dual number.
    pub fn exp(&self) -> Self {
        let value = self.value.exp();
        let derivatives = self.derivatives.iter().map(|d| d * value).collect();
        Self { value, derivatives }
    }

    /// Natural logarithm of a dual number.
    pub fn ln(&self) -> Self {
        let value = self.value.ln();
        let derivatives = self.derivatives.iter().map(|d| d / self.value).collect();
        Self { value, derivatives }
    }

    /// Sigmoid/logistic function.
    pub fn sigmoid(&self) -> Self {
        let exp_neg = (-self.clone()).exp();
        let denom = Dual64::constant(1.0, self.derivatives.len()) + exp_neg;
        Dual64::constant(1.0, self.derivatives.len()) / denom
    }
}

impl Add for Dual64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.map_derivatives(&rhs, self.value + rhs.value, |a, b| a + b)
    }
}

impl<'a> Add<&'a Dual64> for Dual64 {
    type Output = Self;

    fn add(self, rhs: &'a Dual64) -> Self::Output {
        self.map_derivatives(rhs, self.value + rhs.value, |a, b| a + b)
    }
}

impl Add<f64> for Dual64 {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        let mut derivatives = self.derivatives.clone();
        Self {
            value: self.value + rhs,
            derivatives: derivatives.drain(..).collect(),
        }
    }
}

impl Sub for Dual64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.map_derivatives(&rhs, self.value - rhs.value, |a, b| a - b)
    }
}

impl<'a> Sub<&'a Dual64> for Dual64 {
    type Output = Self;

    fn sub(self, rhs: &'a Dual64) -> Self::Output {
        self.map_derivatives(rhs, self.value - rhs.value, |a, b| a - b)
    }
}

impl Sub<f64> for Dual64 {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        let mut derivatives = self.derivatives.clone();
        Self {
            value: self.value - rhs,
            derivatives: derivatives.drain(..).collect(),
        }
    }
}

impl Neg for Dual64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let derivatives = self.derivatives.iter().map(|d| -d).collect();
        Self {
            value: -self.value,
            derivatives,
        }
    }
}

impl Mul for Dual64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let value = self.value * rhs.value;
        let derivatives = self
            .derivatives
            .iter()
            .zip(rhs.derivatives.iter())
            .map(|(a, b)| self.value * b + rhs.value * a)
            .collect();
        Self { value, derivatives }
    }
}

impl<'a> Mul<&'a Dual64> for Dual64 {
    type Output = Self;

    fn mul(self, rhs: &'a Dual64) -> Self::Output {
        let value = self.value * rhs.value;
        let derivatives = self
            .derivatives
            .iter()
            .zip(rhs.derivatives.iter())
            .map(|(a, b)| self.value * b + rhs.value * a)
            .collect();
        Self { value, derivatives }
    }
}

impl Mul<f64> for Dual64 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        let derivatives = self.derivatives.iter().map(|d| d * rhs).collect();
        Self {
            value: self.value * rhs,
            derivatives,
        }
    }
}

impl Mul<Dual64> for f64 {
    type Output = Dual64;

    fn mul(self, rhs: Dual64) -> Self::Output {
        rhs * self
    }
}

impl Div for Dual64 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let value = self.value / rhs.value;
        let derivatives = self
            .derivatives
            .iter()
            .zip(rhs.derivatives.iter())
            .map(|(a, b)| (rhs.value * a - self.value * b) / (rhs.value * rhs.value))
            .collect();
        Self { value, derivatives }
    }
}

impl<'a> Div<&'a Dual64> for Dual64 {
    type Output = Self;

    fn div(self, rhs: &'a Dual64) -> Self::Output {
        let value = self.value / rhs.value;
        let derivatives = self
            .derivatives
            .iter()
            .zip(rhs.derivatives.iter())
            .map(|(a, b)| (rhs.value * a - self.value * b) / (rhs.value * rhs.value))
            .collect();
        Self { value, derivatives }
    }
}

impl Div<f64> for Dual64 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        let derivatives = self.derivatives.iter().map(|d| d / rhs).collect();
        Self {
            value: self.value / rhs,
            derivatives,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_derivative_matches_definition() {
        let x = Dual64::variable(0.2, 0, 1);
        let s = x.sigmoid();
        let expected = s.value() * (1.0 - s.value());
        assert!((s.derivatives()[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn multiplication_applies_product_rule() {
        let x = Dual64::variable(2.0, 0, 1);
        let y = Dual64::variable(3.0, 0, 1);
        let prod = x * y;
        assert_eq!(prod.value(), 6.0);
        assert_eq!(prod.derivatives()[0], 5.0);
    }
}
