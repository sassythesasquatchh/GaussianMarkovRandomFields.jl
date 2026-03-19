//! Example: Matérn/SPDE-style prior with linear observations, using sparse-only operations.
//!
//! This mirrors the Julia workflow of building an SPDE precision matrix, conditioning on
//! linear observations (H x + ε), and drawing samples from the posterior. The SPDE
//! discretization here is intentionally a placeholder (sparse 1D Laplacian + κ² I) so it
//! can be swapped with a FEM-based assembly later without changing the conditioning steps.

use gmrf_core::observation::{apply_gaussian_observations, build_linear_observation_matrix};
use gmrf_core::solver::{DirectBackend, Solver, SolverAlgorithm, SolverConfig};
use gmrf_core::types::{CooMatrix, SparseMatrix, Vector};
use gmrf_core::Gmrf;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dimension = 30;
    let kappa = 1.2;
    let noise_variance: f64 = 0.05;
    let solver_config = SolverConfig {
        algorithm: SolverAlgorithm::Direct(DirectBackend::SparseCholesky),
        ..Default::default()
    };

    // Placeholder SPDE precision: Q = κ² I + L, where L is a 1D Laplacian.
    let prior_precision = placeholder_matern_precision_1d(dimension, kappa);

    // Sparse linear observation matrix H (each row is a short stencil).
    let observation_rows = vec![
        (2, 3, 0.2),
        (8, 9, 0.7),
        (13, 14, 0.5),
        (19, 20, 0.3),
        (25, 26, 0.8),
    ];
    let observation_matrix = build_linear_observation_matrix(dimension, &observation_rows);

    // Draw a latent sample from the prior to synthesize observations.
    let mut rng = thread_rng();
    let mut prior =
        Gmrf::from_mean_and_precision(Vector::zeros(dimension), prior_precision.clone())?
            .with_solver_config(solver_config);
    let latent_true = prior.sample(&mut rng)?;

    let noise = Normal::new(0.0, noise_variance.sqrt())?;
    let mut observations = &observation_matrix * &latent_true;
    for i in 0..observations.len() {
        observations[i] += noise.sample(&mut rng);
    }

    // Condition the prior on linear observations: Q_post = Q + Hᵀ R⁻¹ H, η = Hᵀ R⁻¹ y.
    let (posterior_precision, info) = apply_gaussian_observations(
        &prior_precision,
        &observation_matrix,
        &observations,
        None,
        noise_variance,
    );

    // Compute posterior mean with the same sparse iterative solver.
    let mut solver = Solver::new(solver_config);
    let posterior_mean = solver.solve_matrix(&posterior_precision, &info)?;
    let mut posterior = Gmrf::from_mean_and_precision(posterior_mean, posterior_precision)?
        .with_solver_config(solver_config);

    let posterior_sample = posterior.sample(&mut rng)?;

    let mean_preview: Vec<f64> = posterior.mean().iter().take(6).cloned().collect();
    let sample_preview: Vec<f64> = posterior_sample.iter().take(6).cloned().collect();
    println!("Posterior mean (first 6): {:?}", mean_preview);
    println!("Posterior sample (first 6): {:?}", sample_preview);
    Ok(())
}

fn placeholder_matern_precision_1d(dimension: usize, kappa: f64) -> SparseMatrix {
    let mut coo = CooMatrix::new(dimension, dimension);
    let kappa2 = kappa * kappa;
    for i in 0..dimension {
        let diag = 2.0 + kappa2;
        coo.push(i, i, diag);
        if i > 0 {
            coo.push(i, i - 1, -1.0);
        }
        if i + 1 < dimension {
            coo.push(i, i + 1, -1.0);
        }
    }
    SparseMatrix::from(&coo)
}
