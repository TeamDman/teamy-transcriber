use crate::cli::output::CliOutput;
use crate::native_whisper::selection;
use crate::paths::AppHome;
use crate::paths::CacheHome;
use crate::paths::ModelHome;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::PathBuf;

#[derive(Facet, Debug)]
struct ModelPrepareReport {
    source_dir: String,
    output_dir: String,
    layout: String,
    dimensions: Vec<String>,
    acquisition_policy: String,
}

/// Prepare a local canonical Hugging Face Whisper model for source-defined CUDA inference.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelPrepareArgs {
    /// Prepared model folder, or canonical safetensors source. Omitted discovers the local HF cache.
    #[facet(args::named)]
    pub source_dir: Option<String>,
    /// Optional new package directory. Omit to select prepared weights in place.
    #[facet(args::named)]
    pub output_dir: Option<String>,
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
        let source = if let Some(source) = self.source_dir {
            PathBuf::from(source)
        } else {
            let configured = ModelHome::resolve()?.0;
            if configured.is_dir() {
                configured
            } else {
                selection::find_cached_model()?
            }
        };
        let prepared = if let Some(output) = self.output_dir {
            crate::native_whisper::prepare::prepare_safetensors_model(
                &source,
                std::path::Path::new(&output),
            )?
            .root
        } else if source.join("dims.json").is_file() {
            source.clone()
        } else {
            let output = CacheHome::resolve()?.0.join("models");
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            crate::native_whisper::prepare::prepare_safetensors_model(&source, &output)?.root
        };
        let artifacts = selection::select_model(&AppHome::resolve()?.0, &prepared)?;
        let dimensions = artifacts.dims.as_ref().map_or_else(
            Vec::new,
            crate::native_whisper::whisper::WhisperDims::render_lines,
        );
        Ok(CliOutput::facet(ModelPrepareReport {
            source_dir: source.display().to_string(),
            output_dir: artifacts.root.display().to_string(),
            layout: artifacts.layout.as_str().to_string(),
            dimensions,
            acquisition_policy: "validated local package selected; no download or weight re-encoding; TEAMY_TRANSCRIBER_MODEL_DIR overrides this selection".to_string(),
        }))
    }
}
