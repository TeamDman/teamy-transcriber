use crate::cli::output::CliOutput;
use crate::native_whisper::speech::prepare as prepare_speech_model;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::Path;
use std::time::Duration;

/// Validate and add local data-only Silero weights to a native Whisper package.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelPrepareVadArgs {
    /// Existing Whisper model package. Its weights are retained.
    #[facet(args::named)]
    pub model_dir: String,
    /// Local 16 kHz Silero FP32 safetensors (see the development exporter).
    #[facet(args::named)]
    pub source_weights: String,
}

impl ModelPrepareVadArgs {
    /// # Errors
    /// Returns an error for incompatible assets or an existing VAD package.
    #[expect(
        clippy::unused_async,
        reason = "CLI dispatch uses async invoke methods"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let report =
            prepare_speech_model(Path::new(&self.source_weights), Path::new(&self.model_dir))?;
        Ok(CliOutput::facet(report, Some(Duration::from_secs(10))))
    }
}
