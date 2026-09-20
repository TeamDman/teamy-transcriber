//! Shared application workflows used by the GUI and diagnostic CLI.
//!
//! The GUI owns presentation and interaction; this module owns the deeper
//! orchestration of domain commands, storage, media preparation, capture, local
//! transcription, and explicit export. Keeping these operations here prevents
//! a second GUI-only implementation from drifting away from the CLI surface.

use crate::capture::AudioCaptureReport;
use crate::capture::record_audio_input;
use crate::capture::record_audio_input_until_stopped;
use crate::domain::AppState;
use crate::domain::AssetKind;
use crate::domain::Clip;
use crate::domain::ClipId;
use crate::domain::ClipStatus;
use crate::domain::Command;
use crate::domain::RecordingId;
use crate::domain::SourceAsset;
use crate::domain::TimeRange;
use crate::domain::TranscriptId;
use crate::domain::TranscriptProvenance;
use crate::media::AudioProfile;
use crate::media::FfmpegMediaAdapter;
use crate::media::MediaAdapter;
use crate::media::MediaMetadata;
use crate::media::PreparedAudio;
use crate::media::WavMediaAdapter;
use crate::media::apply_audio_profile;
use crate::media::plan_time_chunks;
use crate::storage::RecordingStore;
use crate::transcription::NativeWhisperBackend;
use crate::transcription::NativeWhisperConfig;
use crate::transcription::TranscriptionBackend;
use crate::transcription::TranscriptionError;
use crate::transcription::TranscriptionRequest;
use eyre::Context;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;
use thiserror::Error;

#[cfg(test)]
#[path = "workflow_batch_tests.rs"]
mod batch_tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrepareReport {
    pub normalized_path: PathBuf,
    pub metadata: MediaMetadata,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipMoveReport {
    pub clip_id: ClipId,
    pub target_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipDeleteReport {
    pub clip_id: ClipId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipSplitReport {
    pub original_clip_id: ClipId,
    pub left_clip_id: ClipId,
    pub right_clip_id: ClipId,
    pub split_at_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipAppendReport {
    pub first_clip_id: ClipId,
    pub second_clip_id: ClipId,
    pub appended_clip_id: ClipId,
    pub source_range: TimeRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaToolConfig {
    pub ffmpeg_executable: PathBuf,
    pub ffprobe_executable: PathBuf,
}

impl MediaToolConfig {
    #[must_use]
    pub fn from_environment() -> Self {
        let adapter = FfmpegMediaAdapter::from_environment();
        Self {
            ffmpeg_executable: adapter.ffmpeg_executable,
            ffprobe_executable: adapter.ffprobe_executable,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptionOptions {
    pub model_dir: PathBuf,
    pub max_decode_tokens: usize,
    pub chunk_duration_us: Option<u64>,
    pub profile: AudioProfile,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscribedChunk {
    pub clip_id: ClipId,
    pub transcript_id: TranscriptId,
    pub source_range: TimeRange,
    pub audio_path: PathBuf,
    pub text: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptionReport {
    pub backend_id: String,
    pub chunks: Vec<TranscribedChunk>,
    pub cancelled: bool,
}

/// One application's resident model. Recordings share weights and workspaces,
/// while each request retains its own audio, cancellation and persisted state.
/// A different model/configuration replaces the sole cached backend. Dropping
/// the session releases it, including joining the native CUDA inference thread.
#[derive(Debug, Default)]
pub struct TranscriptionSession {
    backend: Option<NativeWhisperBackend>,
}

impl TranscriptionSession {
    /// Transcribe a recording with a model retained across successful requests.
    ///
    /// # Errors
    /// Returns an error for invalid recording/model assets, inference failures
    /// or persistence failures. Failed requests evict cached initialization
    /// errors so repairing local assets permits a fresh attempt.
    pub fn transcribe(
        &mut self,
        store: &RecordingStore,
        recording_id: RecordingId,
        options: TranscriptionOptions,
        stop_requested: Option<&AtomicBool>,
        progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<TranscriptionReport> {
        let config = NativeWhisperConfig {
            model_dir: options.model_dir,
            max_decode_tokens: options.max_decode_tokens,
        };
        if self
            .backend
            .as_ref()
            .is_none_or(|backend| backend.config() != &config)
        {
            self.backend = Some(NativeWhisperBackend::new(config));
        }
        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| eyre::eyre!("transcription session has no backend"))?;
        let result = transcribe_recording_inner(
            store,
            recording_id,
            options.chunk_duration_us,
            options.profile,
            stop_requested,
            progress,
            backend,
        );
        if result.is_err() {
            self.backend = None;
        }
        result
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExportReport {
    pub output_path: PathBuf,
    pub transcript_count: usize,
    pub byte_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MicrophoneReport {
    pub recording_id: RecordingId,
    pub capture: AudioCaptureReport,
}

#[derive(Debug, Error)]
#[error("microphone recording {recording_id} failed: {reason}")]
pub struct MicrophoneFailure {
    pub recording_id: RecordingId,
    pub reason: String,
}

/// Persist a new recording source and return its stable ID.
///
/// # Errors
///
/// Returns an error when the source is empty or the manifest cannot be saved.
pub fn create_recording(
    store: &RecordingStore,
    kind: AssetKind,
    source: impl AsRef<Path>,
) -> Result<RecordingId> {
    let recording_id = RecordingId::new();
    let source = SourceAsset::new(kind, source)?;
    let mut state = AppState::new();
    store
        .apply_command(
            &mut state,
            Command::CreateRecording {
                recording_id,
                source,
            },
        )
        .wrap_err("failed to persist recording manifest")?;
    Ok(recording_id)
}

/// Move an existing clip in the persisted recording order.
///
/// # Errors
///
/// Returns an error when the recording or clip is missing, the target index is
/// invalid, or the replayable move event cannot be persisted.
pub fn move_clip(
    store: &RecordingStore,
    recording_id: RecordingId,
    clip_id: ClipId,
    target_index: usize,
) -> Result<ClipMoveReport> {
    let mut state = store
        .load_state(recording_id)
        .wrap_err("failed to load recording event state")?;
    store
        .apply_command(
            &mut state,
            Command::MoveClip {
                recording_id,
                clip_id,
                target_index,
            },
        )
        .wrap_err("failed to persist clip movement")?;
    Ok(ClipMoveReport {
        clip_id,
        target_index,
    })
}

/// Mark an existing clip deleted in the replayable recording history.
///
/// The source and derived audio artifacts are retained so deletion is a
/// reversible-history operation rather than an irreversible filesystem erase.
///
/// # Errors
///
/// Returns an error when the recording or clip is missing, or the delete event
/// cannot be persisted.
pub fn delete_clip(
    store: &RecordingStore,
    recording_id: RecordingId,
    clip_id: ClipId,
) -> Result<ClipDeleteReport> {
    let mut state = store
        .load_state(recording_id)
        .wrap_err("failed to load recording event state")?;
    store
        .apply_command(
            &mut state,
            Command::DeleteClip {
                recording_id,
                clip_id,
            },
        )
        .wrap_err("failed to persist clip deletion")?;
    Ok(ClipDeleteReport { clip_id })
}

/// Replace one active clip with two adjacent source-time clips.
///
/// The original clip is soft-deleted and its source and derived artifacts are
/// retained. The new clips intentionally start without a transcript so the
/// caller can transcribe the revised boundaries explicitly.
///
/// # Errors
///
/// Returns an error when the recording or clip is missing, the split point is
/// outside the clip, or one of the replayable events cannot be persisted.
pub fn split_clip_at(
    store: &RecordingStore,
    recording_id: RecordingId,
    clip_id: ClipId,
    split_at_us: u64,
) -> Result<ClipSplitReport> {
    let mut state = store
        .load_state(recording_id)
        .wrap_err("failed to load recording event state")?;
    let original = state
        .recording(recording_id)
        .and_then(|recording| {
            recording
                .clips
                .iter()
                .find(|clip| clip.id == clip_id && clip.status != ClipStatus::Deleted)
        })
        .cloned()
        .ok_or_else(|| eyre::eyre!("active clip {clip_id} was not found"))?;
    if split_at_us <= original.source_range.start_us || split_at_us >= original.source_range.end_us
    {
        bail!(
            "split point {split_at_us} is outside clip range {}..{}",
            original.source_range.start_us,
            original.source_range.end_us
        );
    }
    let left_range = TimeRange::new(original.source_range.start_us, split_at_us)?;
    let right_range = TimeRange::new(split_at_us, original.source_range.end_us)?;
    let left_clip_id = ClipId::new();
    let right_clip_id = ClipId::new();
    store
        .apply_command(
            &mut state,
            Command::DeleteClip {
                recording_id,
                clip_id,
            },
        )
        .wrap_err("failed to persist the original clip deletion for split")?;
    store
        .apply_command(
            &mut state,
            Command::AddClip {
                recording_id,
                clip_id: left_clip_id,
                source_range: left_range,
            },
        )
        .wrap_err("failed to persist the left split clip")?;
    store
        .apply_command(
            &mut state,
            Command::AddClip {
                recording_id,
                clip_id: right_clip_id,
                source_range: right_range,
            },
        )
        .wrap_err("failed to persist the right split clip")?;
    Ok(ClipSplitReport {
        original_clip_id: clip_id,
        left_clip_id,
        right_clip_id,
        split_at_us,
    })
}

/// Replace two source-time-adjacent active clips with one combined clip.
///
/// The two inputs may be supplied in either order; the resulting range is
/// always ordered by source time. The original clips and their derived audio
/// remain in replayable history, while the combined clip starts pending and
/// can be transcribed as one unit.
///
/// # Errors
///
/// Returns an error when either clip is missing/deleted, the clips are not
/// adjacent in source time, or one of the replayable events cannot be
/// persisted.
pub fn append_adjacent_clips(
    store: &RecordingStore,
    recording_id: RecordingId,
    first_clip_id: ClipId,
    second_clip_id: ClipId,
) -> Result<ClipAppendReport> {
    if first_clip_id == second_clip_id {
        bail!("cannot append a clip to itself");
    }
    let mut state = store
        .load_state(recording_id)
        .wrap_err("failed to load recording event state")?;
    let recording = state
        .recording(recording_id)
        .ok_or_else(|| eyre::eyre!("recording {recording_id} was not found"))?;
    let first = recording
        .clips
        .iter()
        .find(|clip| clip.id == first_clip_id && clip.status != ClipStatus::Deleted)
        .cloned()
        .ok_or_else(|| eyre::eyre!("active clip {first_clip_id} was not found"))?;
    let second = recording
        .clips
        .iter()
        .find(|clip| clip.id == second_clip_id && clip.status != ClipStatus::Deleted)
        .cloned()
        .ok_or_else(|| eyre::eyre!("active clip {second_clip_id} was not found"))?;
    let (source_first, source_second) = if first.source_range.end_us == second.source_range.start_us
    {
        (first, second)
    } else if second.source_range.end_us == first.source_range.start_us {
        (second, first)
    } else {
        bail!("clips {first_clip_id} and {second_clip_id} are not adjacent in source time");
    };
    let source_range = TimeRange::new(
        source_first.source_range.start_us,
        source_second.source_range.end_us,
    )?;
    let appended_clip_id = ClipId::new();
    for clip_id in [source_first.id, source_second.id] {
        store
            .apply_command(
                &mut state,
                Command::DeleteClip {
                    recording_id,
                    clip_id,
                },
            )
            .wrap_err("failed to persist an original clip deletion for append")?;
    }
    store
        .apply_command(
            &mut state,
            Command::AddClip {
                recording_id,
                clip_id: appended_clip_id,
                source_range,
            },
        )
        .wrap_err("failed to persist the appended clip")?;
    Ok(ClipAppendReport {
        first_clip_id: source_first.id,
        second_clip_id: source_second.id,
        appended_clip_id,
        source_range,
    })
}

/// Normalize an imported or captured source into the Whisper audio format.
///
/// # Errors
///
/// Returns an error when the recording or source cannot be loaded or the media
/// adapter cannot produce normalized audio.
pub fn prepare_recording(
    store: &RecordingStore,
    recording_id: RecordingId,
) -> Result<PrepareReport> {
    prepare_recording_with_tools_and_profile(
        store,
        recording_id,
        &MediaToolConfig::from_environment(),
        AudioProfile::Original,
    )
}

/// Normalize a recording using explicitly selected local media tools.
///
/// # Errors
///
/// Returns an error when the recording or source cannot be loaded, or the
/// selected media adapter cannot produce normalized audio.
pub fn prepare_recording_with_tools(
    store: &RecordingStore,
    recording_id: RecordingId,
    tools: &MediaToolConfig,
) -> Result<PrepareReport> {
    prepare_recording_with_tools_and_profile(store, recording_id, tools, AudioProfile::Original)
}

/// Normalize a recording and apply one explicit derived audio profile.
///
/// # Errors
///
/// Returns an error when the recording or source cannot be loaded, the
/// selected media adapter cannot produce normalized audio, or the profile
/// receipt cannot be written.
pub fn prepare_recording_with_tools_and_profile(
    store: &RecordingStore,
    recording_id: RecordingId,
    tools: &MediaToolConfig,
    profile: AudioProfile,
) -> Result<PrepareReport> {
    let recording = store
        .load_recording(recording_id)
        .wrap_err("failed to load recording manifest")?;
    let source = Path::new(&recording.source.path);
    let output_dir = store.recording_dir(recording_id).join("audio");
    let normalized = match recording.source.kind {
        AssetKind::AudioFile | AssetKind::MicrophoneRecording
            if source.extension().is_some_and(|extension| {
                extension.to_string_lossy().eq_ignore_ascii_case("wav")
            }) =>
        {
            WavMediaAdapter
                .prepare_audio(source, &output_dir)
                .wrap_err("failed to normalize WAV source")?
        }
        AssetKind::AudioFile | AssetKind::VideoFile | AssetKind::MicrophoneRecording => {
            FfmpegMediaAdapter {
                ffmpeg_executable: tools.ffmpeg_executable.clone(),
                ffprobe_executable: tools.ffprobe_executable.clone(),
            }
            .prepare_audio(source, &output_dir)
            .wrap_err("failed to normalize source through ffmpeg")?
        }
    };
    let prepared = apply_audio_profile(&normalized.path, &output_dir.join("profiles"), profile)
        .wrap_err("failed to apply audio profile")?;
    if profile != AudioProfile::Original {
        write_audio_profile_receipt(&output_dir, &normalized, &prepared, profile)?;
    }
    Ok(PrepareReport {
        normalized_path: prepared.path,
        metadata: prepared.metadata,
    })
}

/// Return the persisted derived audio path for a profile selection.
#[must_use]
pub fn audio_path_for_profile(
    store: &RecordingStore,
    recording_id: RecordingId,
    profile: AudioProfile,
) -> PathBuf {
    let audio_dir = store.recording_dir(recording_id).join("audio");
    profile.file_stem().map_or_else(
        || audio_dir.join("normalized-16khz-mono.wav"),
        |stem| audio_dir.join("profiles").join(format!("{stem}.wav")),
    )
}

#[derive(Facet)]
struct AudioProfileReceipt {
    profile: AudioProfile,
    source_path: String,
    output_path: String,
    duration_us: u64,
    frame_count: u64,
}

fn write_audio_profile_receipt(
    output_dir: &Path,
    normalized: &PreparedAudio,
    prepared: &PreparedAudio,
    profile: AudioProfile,
) -> Result<()> {
    let profiles_dir = output_dir.join("profiles");
    std::fs::create_dir_all(&profiles_dir)?;
    let stem = profile
        .file_stem()
        .ok_or_else(|| eyre::eyre!("original audio has no profile receipt"))?;
    let receipt = AudioProfileReceipt {
        profile,
        source_path: normalized.path.to_string_lossy().into_owned(),
        output_path: prepared.path.to_string_lossy().into_owned(),
        duration_us: prepared.metadata.duration_us,
        frame_count: prepared.metadata.frame_count,
    };
    let contents = facet_json::to_string_pretty(&receipt)?;
    write_atomic_text(&profiles_dir.join(format!("{stem}.json")), &contents)
}

/// Capture a microphone recording until the returned cancellation flag is set.
///
/// The workflow persists the created/recording/saved or failed lifecycle, so a
/// GUI shutdown or capture error remains recoverable in the same event receipt.
///
/// # Errors
///
/// Returns an error when capture or lifecycle persistence fails.
pub fn record_microphone(
    store: &RecordingStore,
    endpoint_id: Option<&str>,
    stop_requested: Arc<AtomicBool>,
) -> std::result::Result<MicrophoneReport, MicrophoneFailure> {
    capture_microphone(
        store,
        endpoint_id,
        None,
        CaptureRequest::UntilStopped(stop_requested),
    )
}

/// Capture a bounded microphone interval through the same recording lifecycle
/// used by the GUI's stop-controlled capture.
///
/// # Errors
///
/// Returns an error when the duration is zero, capture fails, or lifecycle
/// persistence fails.
pub fn record_microphone_for_duration(
    store: &RecordingStore,
    endpoint_id: Option<&str>,
    output_path: Option<PathBuf>,
    duration: Duration,
) -> std::result::Result<MicrophoneReport, MicrophoneFailure> {
    if duration.is_zero() {
        return Err(MicrophoneFailure {
            recording_id: RecordingId::new(),
            reason: "capture duration must be greater than zero".to_string(),
        });
    }
    capture_microphone(
        store,
        endpoint_id,
        output_path,
        CaptureRequest::Bounded(duration),
    )
}

enum CaptureRequest {
    Bounded(Duration),
    UntilStopped(Arc<AtomicBool>),
}

fn capture_microphone(
    store: &RecordingStore,
    endpoint_id: Option<&str>,
    requested_output_path: Option<PathBuf>,
    request: CaptureRequest,
) -> std::result::Result<MicrophoneReport, MicrophoneFailure> {
    let recording_id = RecordingId::new();
    let output_path = requested_output_path.unwrap_or_else(|| {
        store
            .recording_dir(recording_id)
            .join("source")
            .join("microphone.wav")
    });
    let source =
        SourceAsset::new(AssetKind::MicrophoneRecording, &output_path).map_err(|error| {
            MicrophoneFailure {
                recording_id,
                reason: error.to_string(),
            }
        })?;
    let mut state = AppState::new();
    store
        .apply_command(
            &mut state,
            Command::CreateRecording {
                recording_id,
                source,
            },
        )
        .map_err(|error| MicrophoneFailure {
            recording_id,
            reason: format!("failed to persist microphone recording manifest: {error}"),
        })?;
    store
        .apply_command(&mut state, Command::StartRecording { recording_id })
        .map_err(|error| MicrophoneFailure {
            recording_id,
            reason: format!("failed to persist microphone start state: {error}"),
        })?;

    let capture_result = match request {
        CaptureRequest::Bounded(duration) => {
            record_audio_input(endpoint_id, &output_path, duration)
        }
        CaptureRequest::UntilStopped(stop_requested) => {
            record_audio_input_until_stopped(endpoint_id, &output_path, &stop_requested)
        }
    };
    let capture = match capture_result {
        Ok(capture) => capture,
        Err(error) => {
            let reason = error.to_string();
            store
                .apply_command(
                    &mut state,
                    Command::FailRecording {
                        recording_id,
                        reason,
                    },
                )
                .map_err(|persist_error| MicrophoneFailure {
                    recording_id,
                    reason: format!(
                        "microphone capture failed: {error}; failed to persist failure state: {persist_error}"
                    ),
                })?;
            return Err(MicrophoneFailure {
                recording_id,
                reason: format!("microphone capture failed: {error}"),
            });
        }
    };
    store
        .apply_command(&mut state, Command::CompleteRecording { recording_id })
        .map_err(|error| MicrophoneFailure {
            recording_id,
            reason: format!("failed to persist microphone saved state: {error}"),
        })?;
    Ok(MicrophoneReport {
        recording_id,
        capture,
    })
}

/// Transcribe every active clip in one prepared recording with local native Whisper.
///
/// # Errors
///
/// Returns an error when the recording is not prepared, model assets are not
/// ready, inference fails, or a lifecycle/transcript event cannot be saved.
pub fn transcribe_recording(
    store: &RecordingStore,
    recording_id: RecordingId,
    model_dir: PathBuf,
    max_decode_tokens: usize,
    chunk_duration_us: Option<u64>,
) -> Result<TranscriptionReport> {
    TranscriptionSession::default().transcribe(
        store,
        recording_id,
        TranscriptionOptions {
            model_dir,
            max_decode_tokens,
            chunk_duration_us,
            profile: AudioProfile::Original,
        },
        None,
        None,
    )
}

/// Transcribe a recording with cooperative cancellation.
///
/// CUDA observes cancellation between decoder steps and encoder invocations;
/// other backends stop between clips. Committed clips remain saved. Incomplete
/// clips return to pending or their latest committed transcript state.
///
/// # Errors
///
/// Returns an error when the recording is not prepared, model assets are not
/// ready, inference fails, or a lifecycle/transcript event cannot be saved.
pub fn transcribe_recording_with_cancellation(
    store: &RecordingStore,
    recording_id: RecordingId,
    model_dir: PathBuf,
    max_decode_tokens: usize,
    chunk_duration_us: Option<u64>,
    stop_requested: &Arc<AtomicBool>,
) -> Result<TranscriptionReport> {
    TranscriptionSession::default().transcribe(
        store,
        recording_id,
        TranscriptionOptions {
            model_dir,
            max_decode_tokens,
            chunk_duration_us,
            profile: AudioProfile::Original,
        },
        Some(stop_requested.as_ref()),
        None,
    )
}

/// Transcribe a prepared derived audio profile while allowing cooperative
/// cancellation between clips.
///
/// # Errors
///
/// Returns an error when the recording is not prepared, the selected profile
/// artifact or model is unavailable, inference fails, or persistence fails.
pub fn transcribe_recording_with_profile_and_cancellation(
    store: &RecordingStore,
    recording_id: RecordingId,
    model_dir: PathBuf,
    max_decode_tokens: usize,
    chunk_duration_us: Option<u64>,
    profile: AudioProfile,
    stop_requested: &Arc<AtomicBool>,
) -> Result<TranscriptionReport> {
    let mut no_progress = |_: usize, _: usize| {};
    transcribe_recording_with_profile_and_cancellation_and_progress(
        store,
        recording_id,
        TranscriptionOptions {
            model_dir,
            max_decode_tokens,
            chunk_duration_us,
            profile,
        },
        stop_requested,
        &mut no_progress,
    )
}

/// Transcribe a prepared derived audio profile with cooperative cancellation
/// and bounded per-clip progress callbacks.
///
/// The callback receives `(completed_clips, total_clips)` after each committed
/// clip and once before work begins with `(0, total_clips)`. It runs on the
/// calling thread, so callers should keep it non-blocking.
///
/// # Errors
///
/// Returns an error when the recording is not prepared, the selected profile
/// artifact or model is unavailable, or persistence/inference fails.
pub fn transcribe_recording_with_profile_and_cancellation_and_progress(
    store: &RecordingStore,
    recording_id: RecordingId,
    options: TranscriptionOptions,
    stop_requested: &Arc<AtomicBool>,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<TranscriptionReport> {
    TranscriptionSession::default().transcribe(
        store,
        recording_id,
        options,
        Some(stop_requested.as_ref()),
        Some(progress),
    )
}

fn transcribe_recording_inner(
    store: &RecordingStore,
    recording_id: RecordingId,
    chunk_duration_us: Option<u64>,
    profile: AudioProfile,
    stop_requested: Option<&AtomicBool>,
    mut progress: Option<&mut dyn FnMut(usize, usize)>,
    backend: &dyn TranscriptionBackend,
) -> Result<TranscriptionReport> {
    let mut state = store
        .load_state(recording_id)
        .wrap_err("failed to load recording event state")?;
    let normalized_path = audio_path_for_profile(store, recording_id, profile);
    let metadata = WavMediaAdapter
        .inspect(&normalized_path)
        .wrap_err("recording is not prepared; prepare it from the GUI first")?;
    let full_range = TimeRange::new(0, metadata.duration_us)
        .wrap_err("prepared recording has no transcribable duration")?;
    // The native Whisper frontend has a fixed 30-second context window. Keep
    // the CLI/GUI safe for long recordings even when the caller omits the
    // option; callers can still choose a shorter explicit duration.
    let chunk_duration_us = chunk_duration_us.or_else(|| {
        Some(
            crate::native_whisper::frontend::N_SAMPLES as u64 * 1_000_000
                / u64::from(crate::media::WHISPER_SAMPLE_RATE_HZ),
        )
    });
    let clips = if let Some(chunk_duration_us) = chunk_duration_us {
        let ranges = plan_time_chunks(metadata.duration_us, chunk_duration_us)?;
        ensure_recording_chunks(store, &mut state, recording_id, &ranges)?
    } else {
        vec![ensure_recording_clip(
            store,
            &mut state,
            recording_id,
            full_range,
        )?]
    };
    let backend_id = backend.capabilities().backend_id;
    let total_clips = clips.len();
    if let Some(progress) = progress.as_deref_mut() {
        progress(0, total_clips);
    }
    let mut chunks = Vec::with_capacity(clips.len());
    let mut cancelled = false;
    let mut should_stop = || {
        stop_requested.is_some_and(|requested| requested.load(std::sync::atomic::Ordering::Relaxed))
    };
    let batch = BatchWorkflow {
        store,
        recording_id,
        full_range,
        normalized_path: &normalized_path,
        backend,
    };
    for group in clips.chunks(backend.batch_capacity().clamp(1, 8)) {
        if should_stop() {
            cancelled = true;
            break;
        }
        cancelled = batch.run(&mut state, group, &mut should_stop, &mut |chunk| {
            chunks.push(chunk);
            if let Some(progress) = progress.as_deref_mut() {
                progress(chunks.len(), total_clips);
            }
        })?;
        if cancelled {
            break;
        }
    }
    Ok(TranscriptionReport {
        backend_id,
        chunks,
        cancelled,
    })
}

/// Commit a user-edited transcript as a new provenance-preserving version.
///
/// # Errors
///
/// Returns an error when the clip cannot accept an edit or the event cannot be
/// persisted.
pub fn commit_transcript_edit(
    store: &RecordingStore,
    recording_id: RecordingId,
    clip_id: ClipId,
    text: String,
) -> Result<TranscriptId> {
    let mut state = store.load_state(recording_id)?;
    let transcript_id = TranscriptId::new();
    store
        .apply_command(
            &mut state,
            Command::CommitTranscript {
                recording_id,
                clip_id,
                transcript_id,
                provenance: TranscriptProvenance::UserEdit,
                text,
            },
        )
        .wrap_err("failed to persist transcript edit")?;
    Ok(transcript_id)
}

/// Export the latest transcript for each active clip to an explicit file.
///
/// # Errors
///
/// Returns an error when no committed transcript exists or the output cannot be written.
pub fn export_recording(
    store: &RecordingStore,
    recording_id: RecordingId,
    requested_path: Option<PathBuf>,
) -> Result<ExportReport> {
    export_recording_inner(store, recording_id, requested_path, false)
}

/// Export the latest transcript for each active clip with source-time ranges.
///
/// The ranges identify the persisted transcription segments; they are not
/// word-level alignment timestamps.
///
/// # Errors
///
/// Returns an error when no committed transcript exists or the output cannot be written.
pub fn export_recording_with_timestamps(
    store: &RecordingStore,
    recording_id: RecordingId,
    requested_path: Option<PathBuf>,
) -> Result<ExportReport> {
    export_recording_inner(store, recording_id, requested_path, true)
}

fn export_recording_inner(
    store: &RecordingStore,
    recording_id: RecordingId,
    requested_path: Option<PathBuf>,
    include_timestamps: bool,
) -> Result<ExportReport> {
    let recording = store
        .load_recording(recording_id)
        .wrap_err("failed to load recording manifest")?;
    let mut text = String::new();
    let mut transcript_count = 0;
    for clip in recording
        .clips
        .iter()
        .filter(|clip| clip.status != ClipStatus::Deleted)
    {
        let Some(transcript) = recording
            .transcripts
            .iter()
            .rev()
            .find(|transcript| transcript.clip_id == clip.id)
        else {
            continue;
        };
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        if include_timestamps {
            let _ = writeln!(
                text,
                "[{} - {}] [clip {} | {}]",
                format_timestamp(clip.source_range.start_us),
                format_timestamp(clip.source_range.end_us),
                clip.id,
                provenance_label(transcript.provenance),
            );
            text.push_str(transcript.text.trim());
        } else {
            let _ = write!(
                text,
                "[clip {} | {}]\n{}",
                clip.id,
                provenance_label(transcript.provenance),
                transcript.text.trim()
            );
        }
        transcript_count += 1;
    }
    if transcript_count == 0 {
        bail!("recording has no committed transcripts for active clips");
    }
    let output_path = requested_path.unwrap_or_else(|| {
        store
            .recording_dir(recording_id)
            .join("transcripts")
            .join("transcript.txt")
    });
    write_atomic_text(&output_path, &text)?;
    Ok(ExportReport {
        output_path,
        transcript_count,
        byte_count: text.len(),
    })
}

fn format_timestamp(microseconds: u64) -> String {
    let total_milliseconds = microseconds / 1_000;
    let milliseconds = total_milliseconds % 1_000;
    let total_seconds = total_milliseconds / 1_000;
    let seconds = total_seconds % 60;
    let total_minutes = total_seconds / 60;
    let minutes = total_minutes % 60;
    let hours = total_minutes / 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}")
}

struct BatchWorkflow<'a> {
    store: &'a RecordingStore,
    recording_id: RecordingId,
    full_range: TimeRange,
    normalized_path: &'a Path,
    backend: &'a dyn TranscriptionBackend,
}

impl BatchWorkflow<'_> {
    fn run(
        &self,
        state: &mut AppState,
        clips: &[Clip],
        should_stop: &mut dyn FnMut() -> bool,
        on_complete: &mut dyn FnMut(TranscribedChunk),
    ) -> Result<bool> {
        let outcome = self.transcribe(state, clips, should_stop, on_complete);
        if outcome.is_err() {
            // An append can succeed before a manifest write fails. Replay the
            // durable log before deciding which clips still need cleanup.
            *state = self
                .store
                .load_state(self.recording_id)
                .wrap_err("failed to reload transcription state after an error")?;
        }
        let mut cleanup_errors = Vec::new();
        for clip in clips {
            let processing = state.recording(self.recording_id).is_some_and(|r| {
                r.clips
                    .iter()
                    .any(|c| c.id == clip.id && c.status == ClipStatus::Processing)
            });
            if !processing {
                continue;
            }
            let command = match &outcome {
                Ok(_) => Command::CancelTranscription {
                    recording_id: self.recording_id,
                    clip_id: clip.id,
                },
                Err(error) => Command::FailTranscription {
                    recording_id: self.recording_id,
                    clip_id: clip.id,
                    reason: error.to_string(),
                },
            };
            if let Err(error) = self.store.apply_command(state, command) {
                cleanup_errors.push(error.to_string());
                // Cleanup itself may append successfully before a manifest
                // failure. Never append the next clip with a stale sequence.
                match self.store.load_state(self.recording_id) {
                    Ok(replayed) => *state = replayed,
                    Err(error) => {
                        cleanup_errors.push(format!("could not replay cleanup: {error}"));
                        break;
                    }
                }
            }
        }
        if !cleanup_errors.is_empty() {
            bail!(
                "transcription outcome: {outcome:?}; failed to save clip lifecycle: {}",
                cleanup_errors.join("; ")
            );
        }
        outcome
    }

    fn transcribe(
        &self,
        state: &mut AppState,
        clips: &[Clip],
        should_stop: &mut dyn FnMut() -> bool,
        on_complete: &mut dyn FnMut(TranscribedChunk),
    ) -> Result<bool> {
        let mut requests = Vec::with_capacity(clips.len());
        for clip in clips {
            if should_stop() {
                return Ok(true);
            }
            self.store
                .apply_command(
                    state,
                    Command::BeginTranscription {
                        recording_id: self.recording_id,
                        clip_id: clip.id,
                    },
                )
                .wrap_err("failed to persist transcription start state")?;
            let audio_path = if clip.source_range == self.full_range {
                self.normalized_path.to_path_buf()
            } else {
                WavMediaAdapter
                    .prepare_clip(
                        self.normalized_path,
                        &self
                            .store
                            .recording_dir(self.recording_id)
                            .join("audio")
                            .join("clips"),
                        clip.source_range,
                        clip.id,
                    )?
                    .path
            };
            requests.push(TranscriptionRequest {
                recording_id: self.recording_id,
                clip_id: clip.id,
                audio_path,
            });
        }
        let mut completed = 0;
        let mut persist_error = None;
        let outcome =
            self.backend
                .transcribe_batch(&requests, should_stop, &mut |index, result| {
                    if index != completed || index >= clips.len() {
                        return Err(TranscriptionError::Inference(
                            "backend returned an out-of-order completion".into(),
                        ));
                    }
                    let clip = &clips[index];
                    let transcript_id = TranscriptId::new();
                    self.store
                        .apply_command(
                            state,
                            Command::CommitTranscript {
                                recording_id: self.recording_id,
                                clip_id: clip.id,
                                transcript_id,
                                provenance: result.provenance,
                                text: result.text.clone(),
                            },
                        )
                        .map_err(|error| {
                            let message = error.to_string();
                            persist_error = Some(
                                eyre::eyre!(error).wrap_err("failed to persist the transcript"),
                            );
                            TranscriptionError::Inference(message)
                        })?;
                    completed += 1;
                    on_complete(TranscribedChunk {
                        clip_id: clip.id,
                        transcript_id,
                        source_range: clip.source_range,
                        audio_path: requests[index].audio_path.clone(),
                        text: result.text,
                    });
                    Ok(())
                });
        if let Some(error) = persist_error {
            return Err(error);
        }
        let cancelled = outcome.wrap_err("native Whisper transcription failed")?;
        if !cancelled && completed != clips.len() {
            bail!("backend omitted a transcript without cancelling");
        }
        Ok(cancelled)
    }
}

fn ensure_recording_clip(
    store: &RecordingStore,
    state: &mut AppState,
    recording_id: RecordingId,
    full_range: TimeRange,
) -> Result<Clip> {
    let existing_clip = state
        .recording(recording_id)
        .ok_or_else(|| eyre::eyre!("recording was not found in loaded state"))?
        .clips
        .iter()
        .find(|clip| !matches!(clip.status, ClipStatus::Deleted))
        .cloned();
    if let Some(clip) = existing_clip {
        return Ok(clip);
    }
    let clip_id = ClipId::new();
    store
        .apply_command(
            state,
            Command::AddClip {
                recording_id,
                clip_id,
                source_range: full_range,
            },
        )
        .wrap_err("failed to persist the full-recording clip")?;
    state
        .recording(recording_id)
        .and_then(|recording| recording.clips.iter().find(|clip| clip.id == clip_id))
        .cloned()
        .ok_or_else(|| eyre::eyre!("new clip was not found after persistence"))
}

fn ensure_recording_chunks(
    store: &RecordingStore,
    state: &mut AppState,
    recording_id: RecordingId,
    ranges: &[TimeRange],
) -> Result<Vec<Clip>> {
    let mut active_clips = state
        .recording(recording_id)
        .ok_or_else(|| eyre::eyre!("recording was not found in loaded state"))?
        .clips
        .iter()
        .filter(|clip| !matches!(clip.status, ClipStatus::Deleted))
        .cloned()
        .collect::<Vec<_>>();
    active_clips.sort_by_key(|clip| clip.source_range.start_us);
    let active_ranges = active_clips
        .iter()
        .map(|clip| clip.source_range)
        .collect::<Vec<_>>();
    if !active_clips.is_empty() {
        if active_ranges != ranges {
            bail!("recording already has active clips that do not match the selected chunk size");
        }
        return Ok(active_clips);
    }
    let mut clip_ids = Vec::with_capacity(ranges.len());
    for &source_range in ranges {
        let clip_id = ClipId::new();
        store
            .apply_command(
                state,
                Command::AddClip {
                    recording_id,
                    clip_id,
                    source_range,
                },
            )
            .wrap_err("failed to persist a planned transcription chunk")?;
        clip_ids.push(clip_id);
    }
    clip_ids
        .into_iter()
        .map(|clip_id| {
            state
                .recording(recording_id)
                .and_then(|recording| recording.clips.iter().find(|clip| clip.id == clip_id))
                .cloned()
                .ok_or_else(|| eyre::eyre!("planned transcription chunk was not found"))
        })
        .collect()
}

fn provenance_label(provenance: TranscriptProvenance) -> &'static str {
    match provenance {
        TranscriptProvenance::RawAsr => "raw_asr",
        TranscriptProvenance::UserEdit => "user_edit",
        TranscriptProvenance::LocalLlm => "local_llm",
        TranscriptProvenance::Imported => "imported",
    }
}

fn write_atomic_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary_path = path.with_extension("txt.tmp");
    let mut file = File::create(&temporary_path)?;
    file.write_all(text.as_bytes())?;
    file.write_all(b"\n")?;
    file.flush()?;
    drop(file);
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    std::fs::rename(temporary_path, path)?;
    Ok(())
}
