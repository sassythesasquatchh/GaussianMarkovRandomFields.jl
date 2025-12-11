use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpdeError {
    #[error("invalid parameters: {0}")]
    InvalidParameters(&'static str),

    #[error("unsupported α value: {0}")]
    UnsupportedAlpha(u32),

    #[error(transparent)]
    Fem(#[from] gmrf_fem::FemError),

    #[error(transparent)]
    Core(#[from] gmrf_core::types::GmrfError),
}
