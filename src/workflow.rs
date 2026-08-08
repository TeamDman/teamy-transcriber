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
    transcribe_recording_inner(
        store,
        recording_id,
        model_dir,
        max_decode_tokens,
        chunk_duration_us,
        AudioProfile::Original,
        None,
    )
}

/// Transcribe a recording while allowing the caller to stop between clips.
///
/// The current native backend is a synchronous per-clip operation, so a stop
/// request is observed at the next safe clip boundary. Any clips completed
/// before that boundary remain committed and are included in the report.
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
    transcribe_recording_inner(
        store,
        recording_id,
        model_dir,
        max_decode_tokens,
        chunk_duration_us,
        AudioProfile::Original,
        Some(stop_requested.as_ref()),
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
    transcribe_recording_inner(
        store,
        recording_id,
        model_dir,
        max_decode_tokens,
        chunk_duration_us,
        profile,
        Some(stop_requested.as_ref()),
    )
}

fn transcribe_recording_inner(
    store: &RecordingStore,
    recording_id: RecordingId,
    model_dir: PathBuf,
    max_decode_tokens: usize,
    chunk_duration_us: Option<u64>,
    profile: AudioProfile,
    stop_requested: Option<&AtomicBool>,
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
    let backend = NativeWhisperBackend::new(NativeWhisperConfig {
        model_dir,
        max_decode_tokens,
    });
    let backend_id = backend.capabilities().backend_id;
    let mut chunks = Vec::with_capacity(clips.len());
    let mut cancelled = false;
    for clip in clips {
        if stop_requested
            .as_ref()
            .is_some_and(|requested| requested.load(std::sync::atomic::Ordering::Relaxed))
        {
            cancelled = true;
            break;
        }
        chunks.push(transcribe_clip(
            store,
            &mut state,
            recording_id,
            &clip,
            full_range,
            &normalized_path,
            &backend,
        )?);
        if stop_requested
            .as_ref()
            .is_some_and(|requested| requested.load(std::sync::atomic::Ordering::Relaxed))
        {
            cancelled = true;
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
        let _ = write!(
            text,
            "[clip {} | {}]\n{}",
            clip.id,
            provenance_label(transcript.provenance),
            transcript.text.trim()
        );
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

fn transcribe_clip(
    store: &RecordingStore,
    state: &mut AppState,
    recording_id: RecordingId,
    clip: &Clip,
    full_range: TimeRange,
    normalized_path: &Path,
    backend: &NativeWhisperBackend,
) -> Result<TranscribedChunk> {
    store
        .apply_command(
            state,
            Command::BeginTranscription {
                recording_id,
                clip_id: clip.id,
            },
        )
        .wrap_err("failed to persist transcription start state")?;
    let clip_audio_path = if clip.source_range == full_range {
        normalized_path.to_path_buf()
    } else {
        WavMediaAdapter
            .prepare_clip(
                normalized_path,
                &store
                    .recording_dir(recording_id)
                    .join("audio")
                    .join("clips"),
                clip.source_range,
                clip.id,
            )?
            .path
    };
    let result = match backend.transcribe(&TranscriptionRequest {
        recording_id,
        clip_id: clip.id,
        audio_path: clip_audio_path.clone(),
    }) {
        Ok(result) => result,
        Err(error) => {
            let reason = error.to_string();
            store
                .apply_command(
                    state,
                    Command::FailTranscription {
                        recording_id,
                        clip_id: clip.id,
                        reason,
                    },
                )
                .wrap_err("failed to persist transcription failure state")?;
            return Err(eyre::eyre!("{error}")).wrap_err("native Whisper transcription failed");
        }
    };
    let transcript_id = TranscriptId::new();
    store
        .apply_command(
            state,
            Command::CommitTranscript {
                recording_id,
                clip_id: clip.id,
                transcript_id,
                provenance: result.provenance,
                text: result.text.clone(),
            },
        )
        .wrap_err("failed to persist the transcript")?;
    Ok(TranscribedChunk {
        clip_id: clip.id,
        transcript_id,
        source_range: clip.source_range,
        audio_path: clip_audio_path,
        text: result.text,
    })
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
