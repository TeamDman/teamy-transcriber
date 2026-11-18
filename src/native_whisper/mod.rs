//! Source-defined CUDA Whisper and CPU Silero, with external tensor weights.
#![expect(
    clippy::missing_errors_doc,
    reason = "Numerical implementation boundaries document runtime errors."
)]
pub mod cuda;
pub mod model;
pub mod prepare;
pub mod safetensors_manifest;
pub mod selection;
pub mod speech;
pub mod whisper;
