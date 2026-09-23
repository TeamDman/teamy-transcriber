//! Remove only completed microphone windows owned by the recording store.

use crate::domain::AssetKind;
use crate::domain::ClipStatus;
use crate::domain::Recording;
use crate::domain::RecordingId;
use crate::domain::RecordingStatus;
use crate::domain::TranscriptProvenance;
use crate::storage::RecordingStore;
use eyre::Context;
use eyre::Result;
use eyre::ensure;
use std::path::Path;

pub(crate) fn is_completed_microphone(store: &RecordingStore, recording: &Recording) -> bool {
    if recording.source.kind != AssetKind::MicrophoneRecording
        || recording.status != RecordingStatus::Saved
        || recording.failure.is_some()
        || recording
            .clips
            .iter()
            .any(|clip| clip.status != ClipStatus::Transcribed)
        || recording
            .transcripts
            .iter()
            .any(|transcript| transcript.provenance != TranscriptProvenance::RawAsr)
    {
        return false;
    }

    // A zero-clip VAD plan means a successful no-speech result. Older phone
    // runs saved their output in phones.json instead of a transcript event.
    !recording.transcripts.is_empty()
        || (recording.clip_plan.is_some() && recording.clips.is_empty())
        || store
            .recording_dir(recording.id)
            .join("phones.json")
            .is_file()
}

pub(crate) fn remove_completed_microphone(store: &RecordingStore, id: RecordingId) -> Result<()> {
    let recording = store
        .load_recording(id)
        .wrap_err_with(|| format!("failed to inspect recording {id} before cleanup"))?;
    ensure!(
        is_completed_microphone(store, &recording),
        "recording {id} is not a completed microphone transcription; it was retained"
    );

    // Refuse redirected UUID directories. The caller owns only children of
    // this store's recordings folder, even if a junction or symlink appears.
    let root = store.root().canonicalize()?;
    let expected = root.join("recordings").join(id.to_string());
    let actual = store.recording_dir(id).canonicalize()?;
    ensure!(
        actual == expected,
        "refusing cleanup of a redirected recording directory"
    );
    let source = Path::new(&recording.source.path);
    if let Ok(source) = source.canonicalize() {
        ensure!(
            source.starts_with(&actual),
            "refusing cleanup of a microphone recording whose source is outside its directory"
        );
    }
    std::fs::remove_dir_all(&actual)
        .wrap_err_with(|| format!("failed to remove recording {id} at {}", actual.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ClipPlan;
    use crate::domain::ClipPlanKind;
    use crate::domain::SourceAsset;

    #[test]
    fn empty_vad_plan_is_complete_but_unprocessed_audio_is_not() {
        let store = RecordingStore::new(std::env::temp_dir());
        let id = RecordingId::new();
        let mut recording = Recording {
            id,
            source: SourceAsset::new(
                AssetKind::MicrophoneRecording,
                store.recording_dir(id).join("source/microphone.wav"),
            )
            .unwrap(),
            status: RecordingStatus::Saved,
            failure: None,
            clips: Vec::new(),
            transcripts: Vec::new(),
            clip_plan: None,
        };
        assert!(!is_completed_microphone(&store, &recording));
        recording.clip_plan = Some(ClipPlan {
            kind: ClipPlanKind::VoiceActivity {
                weights_sha256: "0".repeat(64),
            },
            duration_us: 1_000_000,
            clips: Vec::new(),
        });
        assert!(is_completed_microphone(&store, &recording));
        recording.status = RecordingStatus::Created;
        assert!(!is_completed_microphone(&store, &recording));
    }
}
