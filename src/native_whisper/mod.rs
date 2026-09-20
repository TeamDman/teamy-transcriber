//! Source-defined CUDA Whisper and CPU Silero, with external tensor weights.
#![expect(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    clippy::missing_errors_doc,
    reason = "Model metadata uses serde for canonical external formats; numerical implementation boundaries document runtime errors."
)]
pub mod cuda;
pub mod model;
pub mod prepare;
pub mod safetensors_manifest;
pub mod speech;
pub mod whisper;
