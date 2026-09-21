use crate::cli::output::CliOutput;
use crate::paths::AppHome;
use crate::paths::ModelHome;
use arbitrary::Arbitrary;
use eyre::Result;
use eyre::ensure;
use facet::Facet;
use figue as args;
use std::path::PathBuf;
use std::time::Duration;
use teamy_cancellation::CancellationToken;

/// Print microphone transcription live, then drain and exit on Ctrl+C or duration.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct MicrophoneTranscribeArgs {
    /// Capture endpoint ID; defaults to the Windows default microphone.
    #[facet(args::named)]
    pub device_id: Option<String>,
    /// Stop capture after this many milliseconds; omitted records until Ctrl+C.
    #[facet(args::named)]
    pub duration_ms: Option<u64>,
    /// Maximum audio window in milliseconds; VAD submits pauses sooner (500..30000; default 5000).
    #[facet(args::named)]
    pub chunk_duration_ms: Option<u64>,
    /// Override the selected native model package.
    #[facet(args::named)]
    pub model_dir: Option<String>,
}

impl MicrophoneTranscribeArgs {
    /// # Errors
    /// Returns capture, model, persistence or output errors with recovery information.
    pub fn invoke(self, token: &CancellationToken) -> Result<CliOutput> {
        let chunk_ms = self.chunk_duration_ms.unwrap_or(5000);
        ensure!(
            (500..=30000).contains(&chunk_ms),
            "--chunk-duration-ms must be between 500 and 30000"
        );
        ensure!(
            self.duration_ms != Some(0),
            "--duration-ms must be greater than zero"
        );
        let model = self.model_dir.map_or_else(
            || ModelHome::resolve().map(|home| home.0),
            |path| Ok(PathBuf::from(path)),
        )?;
        crate::native_whisper::selection::validate_prepared(&model)?;
        let store = crate::storage::RecordingStore::new(AppHome::resolve()?.0);
        eprintln!(
            "Transcribing at speech pauses (320 ms quiet), with a {chunk_ms} ms maximum window. Captured recordings are retained; use recording list to find them."
        );
        crate::live_microphone::transcribe(
            &store,
            &model,
            chunk_ms,
            |abort, sink| {
                crate::capture::live::capture(
                    self.device_id.as_deref(),
                    self.duration_ms.map(Duration::from_millis),
                    token,
                    abort,
                    sink,
                )
            },
            &mut std::io::stdout().lock(),
        )?;
        eprintln!("Capture stopped; all queued transcription drained.");
        Ok(CliOutput::none())
    }
}
