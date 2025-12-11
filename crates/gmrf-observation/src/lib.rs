//! Placeholder crate for observation model abstractions.
//!
//! Observation likelihoods and transforms will follow the Julia `ObservationModel` APIs. For now we
//! keep this crate minimal so the Cargo workspace mirrors the porting roadmap while core pieces are
//! built out in `gmrf-core`.

pub use gmrf_core as core;
