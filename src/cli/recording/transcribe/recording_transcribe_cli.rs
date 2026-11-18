use crate::cli::cuda_whisper_options::resolve_cuda_whisper_options;
use crate::cli::decode_token_limit::DecodeTokenLimit;
use crate::cli::output::CliOutput;
use crate::cli::recording_chunk_milliseconds::RecordingChunkMilliseconds;
use crate::domain::RecordingId;
use crate::media::AudioProfile;
use crate::native_whisper::whisper::DEFAULT_MAX_DECODE_TOKENS;
use crate::paths::AppHome;
use crate::paths::CacheHome;
use crate::paths::WhisperModelDir;
use crate::storage::RecordingStore;
use crate::workflow::TranscriptionOptions;
use crate::workflow::TranscriptionSession;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::PathBuf;

#[derive(Facet, Debug)]
struct RecordingTranscribeReport {
    recording_id: String,
    backend_id: String,
    chunk_count: usize,
    cancelled: bool,
    no_speech: bool,
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
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct RecordingTranscribeArgs {
    /// Recording UUID returned by recording create.
    #[facet(args::positional)]
    pub recording_id: RecordingId,
    /// Local native Whisper model directory; defaults to the resolved model home.
    #[facet(args::named)]
    pub model_dir: Option<String>,
    /// Maximum number of decoder tokens generated for each clip.
    #[facet(args::named)]
    pub max_decode_tokens: Option<DecodeTokenLimit>,
    /// Fixed chunk duration in milliseconds (at most 30000); omitted preserves
    /// existing clips or uses prepared speech detection, then 30-second chunks.
    #[facet(args::named)]
    pub chunk_duration_ms: Option<RecordingChunkMilliseconds>,
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
    pub async fn invoke(self, app_home: &AppHome, cache_home: &CacheHome) -> Result<CliOutput> {
        let store = RecordingStore::new(app_home.as_ref());
        let chunk_duration_us = self
            .chunk_duration_ms
            .map(RecordingChunkMilliseconds::microseconds);
        let options = self.transcription_options(app_home, cache_home, chunk_duration_us)?;
        let report = TranscriptionSession::default().transcribe(
            &store,
            self.recording_id,
            options,
            None,
            None,
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

        Ok(CliOutput::facet(
            RecordingTranscribeReport {
                recording_id: self.recording_id.to_string(),
                backend_id: report.backend_id,
                chunk_count: chunks.len(),
                cancelled: report.cancelled,
                no_speech: report.no_speech,
                chunks,
            },
            None, /* Media duration and inference work are unbounded. */
        ))
    }

    fn transcription_options(
        &self,
        app_home: &AppHome,
        cache_home: &CacheHome,
        chunk_duration_us: Option<u64>,
    ) -> Result<TranscriptionOptions> {
        let model_home = WhisperModelDir::resolve(app_home, cache_home)?;
        Ok(TranscriptionOptions {
            chunk_duration_us,
            profile: AudioProfile::Original,
            cuda: resolve_cuda_whisper_options()?,
            model_dir: self
                .model_dir
                .as_deref()
                .map_or_else(|| model_home.into(), PathBuf::from),
            max_decode_tokens: self
                .max_decode_tokens
                .map_or(DEFAULT_MAX_DECODE_TOKENS, DecodeTokenLimit::get),
        })
    }
}
