//! Ergonomic helpers for assembling observation stacks.
//!
//! The Julia package exposes formula-style constructors for building
//! composite observation sets. This module mirrors that convenience
//! with a builder that chains likelihood constructors before emitting
//! a `StackedObservations` instance.

use gmrf_core::types::DenseMatrix;
use gmrf_core::Vector;

use crate::models::{
    BernoulliLogitObservation, GaussianObservation, ObservationModel, PoissonLogObservation,
    StackedObservations,
};

/// Builder for assembling stacked observation models.
#[derive(Default)]
pub struct ObservationBuilder {
    models: Vec<Box<dyn ObservationModel>>,
}

impl ObservationBuilder {
    /// Start a new, empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a Gaussian observation with identity transform.
    pub fn gaussian(mut self, data: Vector, noise_variance: f64) -> Self {
        self.models
            .push(Box::new(GaussianObservation::new(data, noise_variance)));
        self
    }

    /// Add a Gaussian observation backed by a dense design matrix.
    pub fn gaussian_with_design(
        mut self,
        data: Vector,
        design: DenseMatrix,
        noise_variance: f64,
    ) -> Self {
        self.models
            .push(Box::new(GaussianObservation::with_design_matrix(
                data,
                design,
                noise_variance,
            )));
        self
    }

    /// Add a Bernoulli-logit observation with identity transform.
    pub fn bernoulli_logit(mut self, data: Vector) -> Self {
        self.models
            .push(Box::new(BernoulliLogitObservation::new(data)));
        self
    }

    /// Add a Bernoulli-logit observation with design matrix.
    pub fn bernoulli_logit_with_design(mut self, data: Vector, design: DenseMatrix) -> Self {
        self.models
            .push(Box::new(BernoulliLogitObservation::with_design_matrix(
                data, design,
            )));
        self
    }

    /// Add a Poisson-log observation with identity transform.
    pub fn poisson_log(mut self, data: Vector, offset: Option<Vector>) -> Self {
        let model = match offset {
            Some(o) => PoissonLogObservation::with_offset(data, o),
            None => PoissonLogObservation::new(data),
        };
        self.models.push(Box::new(model));
        self
    }

    /// Add a Poisson-log observation with design matrix.
    pub fn poisson_log_with_design(
        mut self,
        data: Vector,
        design: DenseMatrix,
        offset: Option<Vector>,
    ) -> Self {
        self.models
            .push(Box::new(PoissonLogObservation::with_design_matrix(
                data, design, offset,
            )));
        self
    }

    /// Finalize the builder into a stacked observation set.
    pub fn build(self) -> StackedObservations {
        StackedObservations::from_models(self.models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_multiple_models() {
        let data = Vector::from_element(2, 1.0);
        let design = DenseMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => 1.0,
            _ => 0.0,
        });

        let stack = ObservationBuilder::new()
            .gaussian(data.clone(), 0.5)
            .bernoulli_logit(data.clone())
            .poisson_log_with_design(data.clone(), design, None)
            .build();

        assert_eq!(stack.num_observations(), 6);
    }
}
