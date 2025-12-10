# Gaussian Markov Random Fields (Rust)

This repository now hosts a Rust implementation of Gaussian Markov random fields (GMRFs).
The code is organized as a Cargo workspace with three crates:

- `gmrf-core`: Fundamental types for representing weighted graphs, constructing Laplacian-based precision matrices, and working
with dense Gaussian Markov random fields.
- `gmrf-observation`: Reusable observation models built on top of the core crate, including linear Gaussian observations and conditioning helpers.
- `gmrf-latent`: Latent model building blocks such as AR(1) and random walk precision constructors that create ready-to-use GMRF priors.

You will need Rust (edition 2021) installed via [`rustup`](https://rustup.rs/).
Run the full test suite with:

```bash
cargo test
```

Add the crates to your own project by pointing at the workspace or by publishing them to crates.io.

## Examples

Construct a simple precision matrix from a graph and evaluate a log density:

```rust
use gmrf_core::{GaussianMarkovRandomField, WeightedGraph};
use nalgebra::DVector;

let mut graph = WeightedGraph::new(3);
graph.add_edge(0, 1, 1.0);
graph.add_edge(1, 2, 2.0);
let (mean, precision) = graph.as_gmrf(1e-3);
let gmrf = GaussianMarkovRandomField::new(mean, precision).unwrap();

let x = DVector::from_vec(vec![0.1, -0.2, 0.3]);
let log_p = gmrf.log_density(&x).unwrap();
println!("log p(x) = {}", log_p);
```

Apply a linear observation to condition the prior:

```rust
use gmrf_observation::{LinearGaussianObservation, ObservationModel};
use nalgebra::{DMatrix, DVector};

let observation = LinearGaussianObservation {
    operator: DMatrix::from_row_slice(1, 3, &[1.0, 0.0, 1.0]),
    observation: DVector::from_vec(vec![0.25]),
    noise: DMatrix::identity(1, 1) * 0.1,
};
let posterior = observation.condition(&gmrf).unwrap();
println!("posterior precision: {}", posterior.precision());
```

Build a simple AR(1) latent field prior with the latent crate:

```rust
use gmrf_latent::ar1_field;

let prior = ar1_field(8, 0.6, 2.0).unwrap();
println!("latent precision shape: {}x{}", prior.precision().nrows(), prior.precision().ncols());
```

## License

The project retains the original MIT license.
