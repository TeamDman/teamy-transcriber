use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;

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
        let report = crate::native_whisper::speech::prepare(
            std::path::Path::new(&self.source_weights),
            std::path::Path::new(&self.model_dir),
        )?;
        Ok(CliOutput::facet(report))
    }
}
