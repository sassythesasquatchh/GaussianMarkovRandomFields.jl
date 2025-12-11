# Rust Port Feature Parity Checklist

This checklist tracks progress toward matching the Julia `GaussianMarkovRandomFields.jl` capabilities in the Rust workspace.
Each section summarizes what has landed and highlights remaining validation or ergonomics gaps.

## Core & Solvers
- [x] GMRF container with mean/information constructors, precision storage, sampling, and RBMC variance fallback.
- [x] Linear operator abstraction, solver configuration, Jacobi preconditioner, and conjugate-gradient path.
- [ ] Optional direct solver backends (e.g., Pardiso bindings) and selected-inversion variants for marginal variances.
- [x] Criterion benchmarks for core solver paths (solve and sampling) to monitor performance regressions.

## FEM & SPDE
- [x] Mesh primitives, quadrature rules, interpolation helpers, and mass/stiffness assembly with diffusion support.
- [x] Matérn SPDE discretization for α ∈ {1, 2} with variance scaling and FEM caching.
- [x] Advection–diffusion extensions with streamline stabilization, soft node constraints, and spatiotemporal-ready operators.

## Latent Models
- [x] AR1, RW1, IID, Besag, BYM2, separable Kronecker, block-diagonal combinations, and fixed effects.
- [x] Separable spatiotemporal kernels plus structured priors like cyclic RW1 to mirror the Julia catalogue.

## Observation Models
- [x] Gaussian, Bernoulli-logit, and Poisson log-likelihoods with dense design matrices and stacking helpers.
- [x] Observation builder API to mirror Julia's formula-style ergonomics.
- [x] Optional autodiff gradients/Jacobians for likelihoods and transforms.

## Visualization
- [x] Plotters-backed mesh and field rendering in the optional `gmrf-viz` crate.
- [ ] Gallery of example renders matching the Julia tutorials.

## Validation & Testing
- [x] Unit tests across crates for assembly, SPDE parameters, latent models, and observation likelihoods.
- [x] Integration tests that follow Julia-style workflows (SPDE discretization and stacked observations).
- [ ] Cross-language parity checks against saved Julia outputs and expanded tutorial coverage.

## Benchmarks
- [x] Criterion benchmarks for core solver solves and sampling.
- [ ] Broader solver/Sparse algebra benchmarks to compare iterative vs. direct backends once available.

