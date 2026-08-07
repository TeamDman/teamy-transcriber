use crate::cli::output::CliOutput;
use crate::media::FfmpegMediaAdapter;
use crate::transcription::LocalModelInventory;
use crate::transcription::NativeWhisperBackend;
use crate::transcription::NativeWhisperConfig;
use crate::transcription::RuntimeAssetStatus;
use crate::transcription::TranscriptionBackend;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;

#[derive(Facet, Debug)]
struct DoctorReport {
    app_home: String,
    app_home_exists: bool,
    cache_home: String,
    cache_home_exists: bool,
    model_home: String,
    model_home_exists: bool,
    model_file_count: usize,
    native_backend: String,
    model_status: RuntimeAssetStatus,
    weights_status: RuntimeAssetStatus,
    dims_status: RuntimeAssetStatus,
    tokenizer_status: RuntimeAssetStatus,
    ffmpeg_executable: String,
    ffprobe_executable: String,
}

/// Report local application paths and native Rust Whisper readiness.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct DoctorArgs;

impl DoctorArgs {
    /// # Errors
    ///
    /// This function returns an error if the platform application paths cannot be resolved.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let app_home = crate::paths::AppHome::resolve()?;
        let cache_home = crate::paths::CacheHome::resolve()?;
        let model_home = crate::paths::ModelHome::resolve()?;
        let model_inventory = LocalModelInventory::inspect(model_home.0.clone())?;
        let backend = NativeWhisperBackend::new(NativeWhisperConfig {
            model_dir: model_home.0.clone(),
            max_decode_tokens: crate::native_whisper::whisper::DEFAULT_MAX_DECODE_TOKENS,
        });
        let readiness = backend.readiness();
        let media_adapter = FfmpegMediaAdapter::from_environment();

        Ok(CliOutput::facet(DoctorReport {
            app_home: app_home.display().to_string(),
            app_home_exists: app_home.exists(),
            cache_home: cache_home.display().to_string(),
            cache_home_exists: cache_home.exists(),
            model_home: model_home.display().to_string(),
            model_home_exists: model_inventory.exists,
            model_file_count: model_inventory.file_count,
            native_backend: backend.capabilities().backend_id,
            model_status: readiness.model_dir,
            weights_status: readiness.weights,
            dims_status: readiness.dims,
            tokenizer_status: readiness.tokenizer,
            ffmpeg_executable: media_adapter.ffmpeg_executable.display().to_string(),
            ffprobe_executable: media_adapter.ffprobe_executable.display().to_string(),
        }))
    }
}
