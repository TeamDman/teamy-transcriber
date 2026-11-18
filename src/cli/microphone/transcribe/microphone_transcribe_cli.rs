use crate::capture::live::capture as capture_live;
use crate::cli::cuda_whisper_options::resolve_cuda_whisper_options;
use crate::cli::output::CliOutput;
use crate::cli::positive_milliseconds::PositiveMilliseconds;
use crate::live_microphone::LiveTranscriptionOptions;
use crate::live_microphone::Recognition;
use crate::microphone_chunk_milliseconds::MicrophoneChunkMilliseconds;
use crate::native_whisper::selection::validate_prepared;
use crate::paths::AppHome;
use crate::paths::CacheHome;
use crate::paths::PhoneModelDir;
use crate::paths::PhoneModelDirSelection;
use crate::paths::WhisperModelDir;
use crate::storage::RecordingStore;
use arbitrary::Arbitrary;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;
use std::path::PathBuf;
use std::time::Duration;
use teamy_cancellation::CancellationScope;

#[derive(Clone, Copy, Default, Facet, Arbitrary, Debug, PartialEq)]
#[facet(rename_all = "kebab-case")]
#[repr(u8)]
pub enum MicrophoneTranscribeMode {
    #[default]
    Words,
    Phonemes,
    WordsAndPhonemes,
}

/// Print microphone transcription live, then drain and exit on Ctrl+C or duration.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct MicrophoneTranscribeArgs {
    /// Capture endpoint ID; defaults to the Windows default microphone.
    #[facet(args::named)]
    pub device_id: Option<String>,
    /// Stop capture after this many milliseconds; omitted records until Ctrl+C.
    #[facet(args::named)]
    pub duration_ms: Option<PositiveMilliseconds>,
    /// Maximum audio window in milliseconds; VAD submits pauses sooner (500..30000; default 5000).
    #[facet(args::named)]
    pub chunk_duration_ms: Option<MicrophoneChunkMilliseconds>,
    /// Whisper package; in phone mode this supplies only the VAD weights.
    #[facet(args::named)]
    pub model_dir: Option<String>,
    /// Output mode: words (default), phonemes, or words-and-phonemes.
    #[facet(args::positional, default)]
    #[arbitrary(default)]
    pub mode: MicrophoneTranscribeMode,
    /// Phone model folder for phonemes or words-and-phonemes.
    #[facet(args::named)]
    pub phone_model_dir: Option<String>,
    /// Milliseconds of output inactivity before printing ... (default 10000 in combined mode).
    #[facet(args::named)]
    pub inactivity_ms: Option<PositiveMilliseconds>,
    /// Keep each microphone recording and intermediate result after successful output.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub keep_recording: bool,
}

impl MicrophoneTranscribeArgs {
    /// # Errors
    /// Returns capture, model, persistence or output errors with recovery information.
    pub fn invoke(
        self,
        app_home: &AppHome,
        cache_home: &CacheHome,
        cancellation: &CancellationScope,
    ) -> Result<CliOutput> {
        let phone_model_dir = self.phone_model_dir.as_deref();
        let recognition = match self.mode {
            MicrophoneTranscribeMode::Words => Recognition::Words,
            MicrophoneTranscribeMode::Phonemes => {
                Recognition::Phonemes(PhoneModelDir::resolve(app_home, phone_model_dir)?)
            }
            MicrophoneTranscribeMode::WordsAndPhonemes => {
                Recognition::WordsAndPhonemes(PhoneModelDir::resolve(app_home, phone_model_dir)?)
            }
        };
        if phone_model_dir.is_some() && !recognition.includes_phonemes() {
            bail!("--phone-model-dir requires phonemes or words-and-phonemes");
        }
        let inactivity = self
            .inactivity_ms
            .map(PositiveMilliseconds::duration)
            .or_else(|| {
                (self.mode == MicrophoneTranscribeMode::WordsAndPhonemes)
                    .then_some(Duration::from_secs(10))
            });
        // Fixed captures include a minute for startup/draining; open captures are user-interactive.
        let elapsed_duration_warn_threshold = self
            .duration_ms
            .map(|duration| duration.duration().saturating_add(Duration::from_mins(1)));
        let chunk_ms = self.chunk_duration_ms.unwrap_or_default();
        let model = self.model_dir.map_or_else(
            || WhisperModelDir::resolve(app_home, cache_home),
            |path| Ok(WhisperModelDir::from(PathBuf::from(path))),
        )?;
        if recognition.includes_words() {
            validate_prepared(&model)?;
        }
        let store = RecordingStore::new(app_home.as_ref());
        tracing::info!(
            "Transcribing at speech pauses (320 ms quiet), with a {} ms maximum window.",
            chunk_ms.get()
        );
        if self.keep_recording {
            tracing::info!(
                "Keeping completed microphone recordings; use recording list to find them."
            );
        } else {
            tracing::info!(
                "Completed microphone recordings are removed after output; failed or unfinished recordings are retained for recovery."
            );
        }
        crate::live_microphone::transcribe(
            &store,
            &model,
            &recognition,
            LiveTranscriptionOptions {
                cuda: if recognition.includes_words() {
                    resolve_cuda_whisper_options()?
                } else {
                    Default::default()
                },
                chunk_ms,
                keep_recording: self.keep_recording,
                inactivity,
            },
            cancellation,
            |cancellation, sink| {
                capture_live(
                    self.device_id.as_deref(),
                    self.duration_ms.map(PositiveMilliseconds::duration),
                    cancellation,
                    sink,
                )
            },
            &mut std::io::stdout(),
        )?;
        tracing::info!("Capture stopped; all queued transcription drained.");
        Ok(CliOutput::none(elapsed_duration_warn_threshold))
    }
}
