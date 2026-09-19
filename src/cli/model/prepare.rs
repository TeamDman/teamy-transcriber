use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;

#[derive(Facet, Debug)]
struct ModelPrepareReport {
    source_dir: String,
    output_dir: String,
    layout: String,
    dimensions: Vec<String>,
    acquisition_policy: String,
}

/// Prepare a local canonical Hugging Face Whisper model for direct Rust/tch use.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelPrepareArgs {
    /// Directory containing model.safetensors, config.json, and tokenizer.json.
    #[facet(args::named)]
    pub source_dir: String,
    /// New directory to create for the native application package.
    #[facet(args::named)]
    pub output_dir: String,
}

impl ModelPrepareArgs {
    /// # Errors
    ///
    /// Returns an error when the source package is incomplete, incompatible, or
    /// the output directory cannot be created.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        #[cfg(any(feature = "tch-native", feature = "cuda-native"))]
        {
            let artifacts = crate::native_whisper::prepare::prepare_safetensors_model(
                std::path::Path::new(&self.source_dir),
                std::path::Path::new(&self.output_dir),
            )?;
            let dimensions = artifacts.dims.as_ref().map_or_else(
                Vec::new,
                crate::native_whisper::whisper::WhisperDims::render_lines,
            );
            Ok(CliOutput::facet(ModelPrepareReport {
                source_dir: self.source_dir,
                output_dir: artifacts.root.display().to_string(),
                layout: artifacts.layout.as_str().to_string(),
                dimensions,
                acquisition_policy: "local files only; no download or CDN mutation".to_string(),
            }))
        }
        #[cfg(not(any(feature = "tch-native", feature = "cuda-native")))]
        {
            let _ = self;
            Err(eyre::eyre!(
                "model preparation requires the cuda-native or tch-native feature"
            ))
        }
    }
}
