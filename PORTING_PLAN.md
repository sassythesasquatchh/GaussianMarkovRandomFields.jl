# Rust Porting Plan for GaussianMarkovRandomFields

This document summarizes the feature set from the original Julia package (now
maintained externally) and outlines an incremental strategy to recreate the full
feature set in Rust. Each phase is designed to land in a working state, building
confidence and test coverage as functionality accumulates.

## Repository Overview

- **Core distributions**: `GMRF` implements sparse multivariate Gaussian fields with mean, precision maps/matrices, sampling helpers, and RBMC-based variance estimation, backed by `LinearSolve` caches for efficient factorizations and solves. Constructors accept mean vectors or information vectors and handle algorithm configuration internally.
- **SPDE discretizations**: The Matérn SPDE pipeline builds FEM mass and stiffness matrices with Ferrite, applies boundary constraints, and assembles precision matrices/square roots (including α-recursions and constraint handling). Discretization produces `GMRF` instances from finite-element meshes.
- **Meshes and FEM utilities**: Mesh creation and manipulation rely on Ferrite/FerriteGmsh/Gmsh, with helpers for quadrature rules, interpolations, lumping mass matrices, and assembling diffusion/mass matrices.
- **Latent model catalog**: Provides AR(1), RW1, IID, Besag/BYM2, Matérn, separable, combined, and fixed-effects latent components that emit `GMRF`s or combine multiple fields.
- **Observation models**: ObservationModel abstractions support exponential-family likelihoods, composite/stacked observations, FEM-based point evaluations, linear transformations, and automatic differentiation of likelihoods (SparseDiffTools/Symbolics or Zygote/ForwardDiff backends).
- **Linear maps and solvers**: Thin wrappers around `LinearMaps`, `LinearSolve`, `SelectedInversion`, and optional Pardiso to expose precision operators, preconditioners, and solver configuration helpers; sampling can fall back to provided precision square roots.
- **Autodiff hooks**: Extension modules integrate with ForwardDiff/Zygote/Enzyme and sparse Jacobian utilities for differentiating likelihoods and latent model constructors.
- **Plotting**: Optional Makie recipes for visualizing fields and meshes.

## Rust Dependency Candidates

- **Linear algebra and sparse matrices**: `nalgebra` + `nalgebra_sparse` or `sprs` for CSC/CSR matrices; `linfa` or `ndarray` for dense vectors; `sprs` + `pardiso`/`intel-mkl-sys` bindings for direct solvers; `parry` or `faer` (if preferred) for factorization kernels.
- **Iterative/direct solvers and preconditioning**: `argmin` or `minres`-style crates for Krylov methods; `sprsolve` or `pardiso` bindings for sparse Cholesky/LDLᵀ; `lin-solve` abstraction trait to mirror `LinearSolve`.
- **Finite elements and meshes**: `fenris`, `meshx`, or `gmsh-parser` for reading/meshing; `geo`/`geo-types` + `kiddo` for spatial indexing akin to NearestNeighbors.
- **Probability/distributions**: `statrs` for Gaussian utilities; custom implementations for precision-based multivariate normals with sampling via `rand` + factorization.
- **Automatic differentiation**: `autodiff` or `enzyme-rust` (when mature) for reverse-mode; `nalgebra-autodiff` or `forward_autodiff` for forward-mode; consider `pyo3` bridges if leveraging Python-based `jax` for prototyping sparse Jacobians.
- **Plotting/visualization**: `plotters`, `egui`/`eframe`, or `vtkio` for mesh/field visualization as optional feature flags.

## Incremental Porting Roadmap

1. **Foundations & workspace**
   - Set up a Cargo workspace with core crate (e.g., `gmrf-core`) plus optional feature crates (`gmrf-fem`, `gmrf-observation`, `gmrf-viz`).
   - Establish shared math types (vector/matrix aliases, sparse matrix wrappers) using `nalgebra_sparse`/`sprs`; define error handling and traits for precision operators and solver backends.
   - Port the `GMRF` data structure with mean/information-vector constructors, precision map abstraction, and solver cache plumbing; implement sampling (using factorization or provided `Q_sqrt`) and RBMC-like variance fallback.

2. **Linear operators & solvers**
   - Introduce linear map traits mirroring `LinearMaps` to allow matrix-free precision operators and compositions.
   - Implement solver configuration analogous to `configure_algorithm`/`prepare_for_linsolve`, supporting both direct (sparse Cholesky/LDLᵀ via `nalgebra-sparse` or external backends like `sprs`/`pardiso`) and iterative methods; cache factorizations for repeated solves.
   - Provide preconditioners (diagonal/Jacobi, incomplete factorizations where available) and selected-inversion alternatives or approximations for marginal variance extraction.

3. **Mesh handling & FEM scaffolding**
   - Add mesh types and converters (simplex/quadrilateral cells), Gmsh importers, and utility functions for quadrature/interpolation to match Ferrite helpers.
   - Implement mass and stiffness assembly, lumped mass computation, and diffusion matrix routines; encode boundary condition and constraint handling analogous to `apply_soft_constraints!` and `ConstraintHandler` workflows.

4. **SPDE discretizations (Matérn and variants)**
   - Port Matérn SPDE definitions (κ/ν/range/smoothness conversions) and α-recursive precision assembly to Rust, producing `GMRF`s from FEM discretizations; include diffusion factor support and constraint noise injection.
   - Add advection–diffusion SPDE support and spatiotemporal extensions (separable kernels, anisotropy) following the Julia structure.

5. **Latent model library**
   - Recreate AR1/RW1/IID/Besag/BYM2/combined/separable/fixed-effect models as constructors returning `GMRF` instances or compositions; ensure graph-based neighborhood handling uses `petgraph` equivalents.
   - Implement meta-models for combining/stacking latent fields, ensuring precision block assembly and marginalization utilities mirror Julia behavior.

6. **Observation models & likelihoods**
   - Define observation model traits and exponential-family implementations (Gaussian, Bernoulli, Poisson, etc.) with ability to attach to latent fields.
   - Port FEM-based point evaluation helpers, linear transformations, composite/stacked observations, and nonlinear least-squares wrappers; ensure compatibility with autodiff traits.

7. **Automatic differentiation integration**
   - Introduce optional AD features (forward/reverse) gated by Cargo features; provide sparse Jacobian/Hessian utilities and compatibility shims for observation likelihoods and latent constructors.

8. **Visualization & ergonomics**
   - Add optional plotting utilities (e.g., `plotters` or `vtkio`) for meshes and field summaries; provide examples mirroring Julia demos.
   - Offer builder-style APIs or macros to mimic Julia formula constructors where sensible.

9. **Validation & parity**
   - Create Rust integration tests mirroring Julia unit tests and tutorial examples (Matérn discretization, AR1/Besag priors, observation likelihood conditioning).
   - Document feature parity checklist; add benchmarks for solver performance and sampling.

## Migration Notes

- Keep feature flags aligned with optional Julia extensions (Pardiso, Makie, AD backends) to allow lean builds.
- Preserve numerical behavior (precision scaling, constraint handling, α-recursion) via reference tests against Julia outputs where feasible.
- Prefer trait-based abstractions to retain composability across latent models, observation models, and solvers.
- Provide interop guides (e.g., exporting meshes or precision matrices) to ease cross-language validation during the port.
