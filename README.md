# GaussianMarkovRandomFields (Rust)

Gaussian Markov Random Fields (GMRFs) are Gaussian distributions with sparse
precision (inverse covariance) matrices. This workspace provides Rust crates
for constructing, conditioning, and sampling GMRFs, with FEM/SPDE tooling and
observation models as optional components.

## Workspace crates

- `gmrf-core`: core GMRF type, linear solves, conditioning utilities.
- `gmrf-latent`: latent model constructors and combinators.
- `gmrf-observation`: observation models and likelihoods.
- `gmrf-spde`: SPDE-based discretizations and helpers.
- `gmrf-fem`: FEM mesh/discretization utilities.
- `gmrf-autodiff`: optional AD integrations.
- `gmrf-viz`: optional visualization helpers.

## Build and test

```sh
cargo test
cargo test --examples
```

## Run examples

```sh
cargo run -p gmrf-core --example isotropic_grid_posterior
cargo run -p gmrf-spde --example matern_linear_observation
```

## Contributing

See `CONTRIBUTING.md`.
