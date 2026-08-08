use crate::cli::output::CliOutput;
use crate::domain::RecordingId;
use crate::paths::ModelHome;
use crate::storage::RecordingStore;
use crate::transcription::NativeWhisperConfig;
use crate::workflow::transcribe_recording;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::PathBuf;

#[derive(Facet, Debug)]
struct RecordingTranscribeReport {
    recording_id: String,
    backend_id: String,
    chunk_count: usize,
    chunks: Vec<TranscribedChunkReport>,
}

#[derive(Facet, Debug)]
struct TranscribedChunkReport {
    clip_id: String,
    transcript_id: String,
    start_us: u64,
    end_us: u64,
    audio_path: String,
    text: String,
}

/// Run the local pure-Rust Whisper backend against a prepared recording clip.
#[derive(Default, Facet, Arbitrary, Debug, PartialEq)]
pub struct RecordingTranscribeArgs {
    /// Recording UUID returned by recording create.
    #[facet(args::positional)]
    pub recording_id: String,
    /// Local native Whisper model directory; defaults to the resolved model home.
    #[facet(args::named)]
    pub model_dir: Option<String>,
    /// Maximum number of decoder tokens generated for each clip.
    #[facet(args::named)]
    pub max_decode_tokens: Option<usize>,
    /// Maximum source chunk duration in milliseconds; omitted uses one full-recording clip.
    #[facet(args::named)]
    pub chunk_duration_ms: Option<u64>,
}

impl RecordingTranscribeArgs {
    /// # Errors
    ///
    /// Returns an error when the recording is not prepared, the local
    /// native Whisper configuration is unavailable, or the transcript cannot be
    /// committed to the event store.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let recording_id =
            RecordingId::parse(&self.recording_id).wrap_err("recording ID must be a UUID")?;
        let app_home = crate::paths::AppHome::resolve()?;
        let store = RecordingStore::new(app_home.0);
        let chunk_duration_us = self
            .chunk_duration_ms
            .map(|duration_ms| {
                duration_ms
                    .checked_mul(1_000)
                    .ok_or_else(|| eyre::eyre!("--chunk-duration-ms is too large"))
            })
            .transpose()?;
        let config = self.native_whisper_config()?;
        let report = transcribe_recording(
            &store,
            recording_id,
            config.model_dir,
            config.max_decode_tokens,
            chunk_duration_us,
        )?;
        let chunks = report
            .chunks
            .into_iter()
            .map(|chunk| TranscribedChunkReport {
                clip_id: chunk.clip_id.to_string(),
                transcript_id: chunk.transcript_id.to_string(),
                start_us: chunk.source_range.start_us,
                end_us: chunk.source_range.end_us,
                audio_path: chunk.audio_path.display().to_string(),
                text: chunk.text,
            })
            .collect::<Vec<_>>();

        Ok(CliOutput::facet(RecordingTranscribeReport {
            recording_id: recording_id.to_string(),
            backend_id: report.backend_id,
            chunk_count: chunks.len(),
            chunks,
        }))
    }

    fn native_whisper_config(&self) -> Result<NativeWhisperConfig> {
        let model_home = ModelHome::resolve()?;
        Ok(NativeWhisperConfig {
            model_dir: self
                .model_dir
                .as_deref()
                .map_or(model_home.0, PathBuf::from),
            max_decode_tokens: self
                .max_decode_tokens
                .unwrap_or(crate::native_whisper::whisper::DEFAULT_MAX_DECODE_TOKENS),
        })
    }
}
