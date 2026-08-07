use crate::cli::output::CliOutput;
use crate::domain::AppState;
use crate::domain::Clip;
use crate::domain::ClipId;
use crate::domain::ClipStatus;
use crate::domain::Command;
use crate::domain::RecordingId;
use crate::domain::TimeRange;
use crate::domain::TranscriptId;
use crate::media::MediaAdapter;
use crate::media::WavMediaAdapter;
use crate::paths::ModelHome;
use crate::storage::RecordingStore;
use crate::transcription::LocalWhisperXBackend;
use crate::transcription::LocalWhisperXConfig;
use crate::transcription::TranscriptionBackend;
use crate::transcription::TranscriptionRequest;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::PathBuf;

#[derive(Facet, Debug)]
struct RecordingTranscribeReport {
    recording_id: String,
    clip_id: String,
    transcript_id: String,
    backend_id: String,
    audio_path: String,
    text: String,
}

/// Run the local `WhisperX` worker against a prepared recording clip.
#[derive(Default, Facet, Arbitrary, Debug, PartialEq)]
pub struct RecordingTranscribeArgs {
    /// Recording UUID returned by recording create.
    #[facet(args::positional)]
    pub recording_id: String,
    /// Python executable; defaults to `TEAMY_TRANSCRIBER_PYTHON` or python.
    #[facet(args::named)]
    pub python: Option<String>,
    /// `WhisperX` worker script; defaults to the repository runtime script.
    #[facet(args::named)]
    pub worker_script: Option<String>,
    /// Local model directory; defaults to the resolved model home.
    #[facet(args::named)]
    pub model_dir: Option<String>,
    /// `WhisperX` model identifier.
    #[facet(args::named)]
    pub model_name: Option<String>,
    /// `WhisperX` device, such as cpu or cuda.
    #[facet(args::named)]
    pub device: Option<String>,
    /// `WhisperX` compute type, such as int8 or float16.
    #[facet(args::named)]
    pub compute_type: Option<String>,
    /// `WhisperX` batch size.
    #[facet(args::named)]
    pub batch_size: Option<u32>,
}

impl RecordingTranscribeArgs {
    /// # Errors
    ///
    /// Returns an error when the recording is not prepared, the local
    /// `WhisperX` configuration is unavailable, or the transcript cannot be
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
        let mut state = store
            .load_state(recording_id)
            .wrap_err("failed to load recording event state")?;
        let normalized_path = store
            .recording_dir(recording_id)
            .join("audio")
            .join("normalized-16khz-mono.wav");
        let metadata = WavMediaAdapter
            .inspect(&normalized_path)
            .wrap_err("recording is not prepared; run recording prepare first")?;
        let full_range = TimeRange::new(0, metadata.duration_us)
            .wrap_err("prepared recording has no transcribable duration")?;
        let clip = ensure_recording_clip(&store, &mut state, recording_id, full_range)?;
        let clip_audio_path = if clip.source_range == full_range {
            normalized_path.clone()
        } else {
            WavMediaAdapter
                .prepare_clip(
                    &normalized_path,
                    &store
                        .recording_dir(recording_id)
                        .join("audio")
                        .join("clips"),
                    clip.source_range,
                    clip.id,
                )?
                .path
        };
        store
            .apply_command(
                &mut state,
                Command::BeginTranscription {
                    recording_id,
                    clip_id: clip.id,
                },
            )
            .wrap_err("failed to persist transcription start state")?;
        let backend = LocalWhisperXBackend::new(self.local_whisperx_config()?);
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
                        &mut state,
                        Command::FailTranscription {
                            recording_id,
                            clip_id: clip.id,
                            reason,
                        },
                    )
                    .wrap_err("failed to persist transcription failure state")?;
                return Err(eyre::eyre!("{error}")).wrap_err("local WhisperX transcription failed");
            }
        };
        let transcript_id = TranscriptId::new();
        store
            .apply_command(
                &mut state,
                Command::CommitTranscript {
                    recording_id,
                    clip_id: clip.id,
                    transcript_id,
                    provenance: result.provenance,
                    text: result.text.clone(),
                },
            )
            .wrap_err("failed to persist the transcript")?;

        Ok(CliOutput::facet(RecordingTranscribeReport {
            recording_id: recording_id.to_string(),
            clip_id: clip.id.to_string(),
            transcript_id: transcript_id.to_string(),
            backend_id: backend.capabilities().backend_id,
            audio_path: clip_audio_path.display().to_string(),
            text: result.text,
        }))
    }

    fn local_whisperx_config(self) -> Result<LocalWhisperXConfig> {
        let model_home = ModelHome::resolve()?;
        Ok(LocalWhisperXConfig {
            python_executable: self
                .python
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var("TEAMY_TRANSCRIBER_PYTHON")
                        .ok()
                        .map(PathBuf::from)
                })
                .unwrap_or_else(|| PathBuf::from("python")),
            worker_script: self.worker_script.map_or_else(
                || {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("runtime")
                        .join("whisperx_worker.py")
                },
                PathBuf::from,
            ),
            model_dir: self.model_dir.map_or(model_home.0, PathBuf::from),
            model_name: self.model_name.unwrap_or_else(|| "small".to_string()),
            device: self.device.unwrap_or_else(|| "cpu".to_string()),
            compute_type: self.compute_type.unwrap_or_else(|| "int8".to_string()),
            batch_size: self.batch_size.unwrap_or(1),
        })
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
