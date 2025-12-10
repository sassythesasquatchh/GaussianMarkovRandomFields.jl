# Agent Guidance for Rust Port

This repository currently documents the Julia implementation of GaussianMarkovRandomFields.jl and a roadmap for porting it to Rust. Use the following expectations when extending the plan or implementing code:

- **Scope**: This guidance applies to the entire repository and any new Rust-port planning artifacts. If you add nested instructions later, they take precedence inside their directories.
- **Planning first**: Align changes with `PORTING_PLAN.md`. Update the plan before making architectural changes that diverge from it. Avoid speculative code without a documented step.
- **Parity mindset**: Strive for feature-for-feature parity with the Julia package (latent models, observation models, SPDE discretizations, solver configuration, and optional extensions). Keep optional features behind cargo-style flags.
- **Dependencies**: Prefer Rust-native crates listed in the plan. If alternatives are needed, justify them in the plan and keep ergonomics/composability similar to the Julia APIs.
- **Documentation quality**: When adding new files, include concise module-level comments explaining how they map to Julia components and how they interact with the wider system. Update examples/tests alongside new functionality.
- **Testing discipline**: Maintain runnable examples or integration tests mirroring Julia tutorials when implementing features. Ensure solvers and discretizations are validated with deterministic seeds where possible.

If you modify this guidance, highlight rationale in your commit message to help future agents understand the change.
