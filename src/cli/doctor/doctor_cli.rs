use crate::cli::output::CliOutput;
use crate::media::FfmpegMediaAdapter;
use crate::transcription::LocalModelInventory;
use crate::transcription::LocalWhisperXBackend;
use crate::transcription::LocalWhisperXConfig;
use crate::transcription::RuntimeAssetStatus;
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
    whisperx_worker: String,
    whisperx_worker_status: RuntimeAssetStatus,
    python_executable: String,
    python_status: RuntimeAssetStatus,
    model_status: RuntimeAssetStatus,
    ffmpeg_executable: String,
    ffprobe_executable: String,
}

/// Report local application paths and local `WhisperX` runtime readiness.
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
        let worker_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("runtime")
            .join("whisperx_worker.py");
        let python_executable =
            std::env::var("TEAMY_TRANSCRIBER_PYTHON").unwrap_or_else(|_| "python".to_string());
        let backend = LocalWhisperXBackend::new(LocalWhisperXConfig {
            python_executable: std::path::PathBuf::from(&python_executable),
            worker_script: worker_path.clone(),
            model_dir: model_home.0.clone(),
            model_name: "small".to_string(),
            device: "cpu".to_string(),
            compute_type: "int8".to_string(),
            batch_size: 1,
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
            whisperx_worker: format!(
                "{} ({})",
                worker_path.display(),
                if worker_path.is_file() {
                    "present"
                } else {
                    "missing"
                }
            ),
            whisperx_worker_status: readiness.worker_script,
            python_executable,
            python_status: readiness.python,
            model_status: readiness.model_dir,
            ffmpeg_executable: media_adapter.ffmpeg_executable.display().to_string(),
            ffprobe_executable: media_adapter.ffprobe_executable.display().to_string(),
        }))
    }
}
