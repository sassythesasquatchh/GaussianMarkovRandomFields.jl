# Observation Models: Mathematics and Interfaces

This page summarizes the mathematical foundations behind every component in `src/observation_models/`. It is organized by the abstractions exposed in Julia and mirrors how the implementation structures log-likelihoods, derivatives, and composition utilities.

## Core abstractions

- **ObservationModel**: A factory that, when called with data `y` and hyperparameters `θ`, returns a materialized **ObservationLikelihood** closing over `y, θ`. Everything that follows operates on the latent field `x` only.
- **ObservationLikelihood**: Provides `loglik(x)`, optional `loggrad(x)`, `loghessian(x)`, and `pointwise_loglik(x)` when observations are conditionally independent.
- **Conditional independence trait**: `ConditionallyIndependent` allows per-observation log-likelihoods; `ConditionallyDependent` disallows them.

### Log-likelihood and AD fallbacks

Given `ℓ(x) = log p(y | x, θ)`, gradients and Hessians are computed via DifferentiationInterface (DI) when model-specific methods are absent:

```math
\nabla_x \ell(x) = \text{DI.gradient}(\ell, x), \qquad
\nabla_x^2 \ell(x) = \text{DI.hessian}(\ell, x)
```

Models can override `autodiff_gradient_backend`, `autodiff_hessian_backend`, and precomputation hooks to choose back-ends and reuse trace state.

## Linear transformations of the latent field

`LinearlyTransformedObservationModel` wraps a base model with design matrix `A`, mapping full latent field `x_full` to predictors `η = A x_full`.

- **Log-likelihood**: `ℓ(x_full) = ℓ_base(η)`
- **Gradient (chain rule)**: `∇ℓ(x_full) = Aᵀ ∇ℓ_base(η)`
- **Hessian (chain rule)**: `∇²ℓ(x_full) = Aᵀ (∇²ℓ_base(η)) A`

Pointwise log-likelihood delegates after computing `η`.

## FEM-derived transforms

`PointEvaluationObsModel`, `PointDerivativeObsModel`, and `PointSecondDerivativeObsModel` build linear transforms from FEM evaluation / derivative matrices, then delegate to an exponential-family base model. Mathematically identical to the linear case above; the matrices encode basis evaluations or derivatives at requested points.

## Exponential family subdirectory

### Distributions and links

Supported families: Normal, Poisson, Bernoulli, Binomial. Link functions `g` map mean parameter `μ` to linear predictor `η = g(μ)` with inverse `μ = g⁻¹(η)`:

- Identity: `μ = η`
- Log: `μ = exp(η)`
- Logit: `μ = (1 + e^{-η})^{-1}`

The derivatives used for chain rules:

```math
\frac{d μ}{d η} = g^{-1'}(η), \qquad
\frac{d^2 μ}{d η^2} = g^{-1''}(η)
```

### Likelihoods (canonical links)

Let `y_i` be observations, `μ_i = g^{-1}(η_i)`.

- **Normal (σ known)**  
  `ℓ = -\tfrac{n}{2}\log(2π) - n\log σ - \tfrac{1}{2σ^2} ∑ (y_i - μ_i)^2`  
  `∂ℓ/∂η_i = (y_i - μ_i) / σ^2` (identity link ⇒ `μ_i = η_i`)  
  `∂²ℓ/∂η_i² = -1/σ^2`

- **Poisson with log link (offset optional)**  
  `ℓ = ∑ (y_i η_i - exp(η_i + offset_i))`  
  `∂ℓ/∂η_i = y_i - μ_i` with `μ_i = exp(η_i + offset_i)`  
  `∂²ℓ/∂η_i² = -μ_i`

- **Bernoulli with logit link**  
  `ℓ = ∑ [ y_i log μ_i + (1 - y_i) log(1 - μ_i) ]`, `μ_i = logistic(η_i)`  
  `∂ℓ/∂η_i = y_i - μ_i`  
  `∂²ℓ/∂η_i² = -μ_i (1 - μ_i)`

- **Binomial with logit link (n_i trials)**  
  Replace Bernoulli `y_i` with successes `y_i`, scale by `n_i`:  
  `∂ℓ/∂η_i = y_i - n_i μ_i`, `∂²ℓ/∂η_i² = -n_i μ_i (1 - μ_i)`

### Non-canonical links

Fallback implementations apply chain rule using `dμ/dη` and `d²μ/dη²`:

```math
\frac{∂ℓ}{∂η} = \frac{∂ℓ}{∂μ} \circ \frac{d μ}{d η}, \qquad
\frac{∂²ℓ}{∂η²} = \frac{∂²ℓ}{∂μ²} \circ \left(\frac{d μ}{d η}\right)^2 + \frac{∂ℓ}{∂μ} \circ \frac{d^2 μ}{d η^2}
```

### Indexing and embedding

Models can observe a subset of latent components via `indices`; gradients/Hessians are embedded back into the full latent space by zero-filling outside the observed indices.

### Pointwise log-likelihoods

For `ConditionallyIndependent` likelihoods, per-observation terms are `logpdf` evaluations of the family distribution (Normal, Poisson, Bernoulli, Binomial). Offsets (Poisson) are applied on the `η` scale before inversion.

## Composite likelihoods

`CompositeObservationModel` materializes a tuple of component likelihoods sharing the same latent field.

- **Log-likelihood**: `ℓ_total(x) = Σ_k ℓ_k(x)`
- **Gradient**: `∇ℓ_total = Σ_k ∇ℓ_k`
- **Hessian**: `∇²ℓ_total = Σ_k ∇²ℓ_k`
- **Pointwise**: concatenates each component’s pointwise vector when all are conditionally independent; errors otherwise.

Component observations are stored as `CompositeObservations`, a logical concatenation preserving component boundaries.

## Automatic differentiation conveniences

### AutoDiffObservationModel / AutoDiffLikelihood

User-provided log-likelihood `f(x; θ, y)` is wrapped so that:

- AD backends (Enzyme, Mooncake, Zygote, ForwardDiff) are prepared once on a prototype `x`.
- `loggrad` / `loghessian` call DI with stored backends and preparation state.
- Optional user-supplied `pointwise_loglik_func` enables per-observation metrics.

### Built-in AD for primitive models (feature-gated)

`autodiff_gradient_backend` defaults to model-specific backends; when absent, DI falls back to numerical AD. Hessian backends default to gradient backends unless a sparse-aware backend is available.

## Nonlinear least squares model

For `y | x ~ Normal(f(x), σ)`, with differentiable `f`:

- **Log-likelihood**: `ℓ = -\tfrac{1}{2} (y - f(x))ᵀ Σ^{-1} (y - f(x)) + const`
- **Gradient**: `J(x)ᵀ Σ^{-1} (y - f(x))`, where `J` is Jacobian of `f`
- **Hessian (Gauss–Newton approximation)**: `- Jᵀ Σ^{-1} J`
- Uses DI sparse Jacobian backend when available; pointwise logpdf comes from elementwise Normal terms.

## Conditional sampling

`conditional_distribution` constructs `p(y | x, θ)` for supported models:

- Exponential family → product distributions with mean `μ = g^{-1}(η)` (and optional Poisson offsets).
- Nonlinear least squares → product of Normals with mean `f(x)` and provided `σ`.
- Linearly transformed models first map `x_full` to `η = A x_full`, then delegate.

## Summary of mathematical guarantees

- All gradients/Hessians respect chain rule under linear transforms and composite summation.
- Pointwise log-likelihoods are only valid under `ConditionallyIndependent`.
- AD backends are pluggable; analytical closed forms are used where available (canonical links, Gauss–Newton).

