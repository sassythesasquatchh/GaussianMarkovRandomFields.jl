//! Integration tests that mirror Julia workflows for the Rust port.
//! These tests validate SPDE discretization and stacked observation flows
//! to guard against regressions as features are added.

use gmrf_core::types::DenseMatrix;
use gmrf_core::Vector;
use gmrf_latent::ar1_chain;
use gmrf_observation::{
    BernoulliLogitObservation, GaussianObservation, ObservationBuilder, ObservationModel,
    PoissonLogObservation,
};
use gmrf_spde::{FemDiscretization2d, MaternSpde2d};
use rand::{rngs::StdRng, SeedableRng};

use gmrf_fem::{ElementConnectivity, Mesh2d, Point2};

fn unit_square_mesh() -> Mesh2d {
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
    .expect("valid unit square mesh")
}

#[test]
fn matern_spde_pipeline_produces_valid_precision_and_samples() {
    let mesh = unit_square_mesh();
    let fem = FemDiscretization2d::new(mesh, None, None, None).expect("assemble FEM data");
    let spde = MaternSpde2d::from_range_and_smoothness(0.8, 1, 1.0, None)
        .expect("valid Matérn parameters");

    let precision = spde.precision_matrix(&fem).expect("precision assembly");
    assert_eq!(precision.nrows(), fem.dimension());
    let diagonal_positive = precision
        .triplet_iter()
        .filter(|(i, j, _)| i == j)
        .all(|(_, _, v)| *v > 0.0);
    assert!(diagonal_positive, "precision diagonal should be positive");

    let mut gmrf = spde.discretize(&fem).expect("discretize Matérn");
    let mut rng = StdRng::seed_from_u64(7);
    let draw = gmrf.sample(&mut rng).expect("sample from SPDE field");
    assert_eq!(draw.len(), fem.dimension());

    let variances = gmrf
        .rbmc_variances(8, &mut rng)
        .expect("rbmc variances available");
    assert_eq!(variances.len(), fem.dimension());
    assert!(variances.iter().all(|v| v.is_finite() && *v > 0.0));
}

#[test]
fn stacked_observation_matches_component_loglikelihoods() {
    let latent = ar1_chain(3, 0.2, 1.5).expect("ar1 latent");
    let state = latent.mean().clone();

    let gaussian = GaussianObservation::new(Vector::from_vec(vec![0.1, -0.2, 0.05]), 0.25);
    let design = DenseMatrix::from_fn(2, 3, |i, j| {
        match (i, j) {
            (0, 0) => 1.0,
            (0, 2) => 0.5,
            (1, 1) => 1.0,
            (1, 2) => -0.25,
            _ => 0.0,
        }
    });
    let bernoulli = BernoulliLogitObservation::with_design_matrix(
        Vector::from_vec(vec![1.0, 0.0]),
        design.clone(),
    );
    let poisson = PoissonLogObservation::with_offset(
        Vector::from_vec(vec![1.0]),
        Vector::from_vec(vec![0.1]),
    );

    let stack = ObservationBuilder::new()
        .gaussian(Vector::from_vec(vec![0.1, -0.2, 0.05]), 0.25)
        .bernoulli_logit_with_design(Vector::from_vec(vec![1.0, 0.0]), design)
        .poisson_log(Vector::from_vec(vec![1.0]), Some(Vector::from_vec(vec![0.1])))
        .build();

    let component_ll = gaussian.log_likelihood(&state).expect("gaussian ll")
        + bernoulli.log_likelihood(&state).expect("bernoulli ll")
        + poisson.log_likelihood(&state).expect("poisson ll");
    let stacked_ll = stack.log_likelihood(&state).expect("stacked ll");
    assert!(
        (component_ll - stacked_ll).abs() < 1e-12,
        "stacked log-likelihood should match component sum"
    );

    let residuals = stack.residuals(&state).expect("stacked residuals");
    assert_eq!(residuals.len(), stack.num_observations());
}
