use std::path::PathBuf;
use teamy_transcriber::domain::AppState;
use teamy_transcriber::domain::AssetKind;
use teamy_transcriber::domain::ClipId;
use teamy_transcriber::domain::ClipStatus;
use teamy_transcriber::domain::Command;
use teamy_transcriber::domain::RecordingId;
use teamy_transcriber::domain::SourceAsset;
use teamy_transcriber::domain::TimeRange;
use teamy_transcriber::domain::TranscriptId;
use teamy_transcriber::domain::TranscriptProvenance;
use teamy_transcriber::presentation::ActionId;
use teamy_transcriber::presentation::FocusContext;
use teamy_transcriber::presentation::KeyChord;
use teamy_transcriber::presentation::PresentationState;
use teamy_transcriber::presentation::UiKey;
use teamy_transcriber::presentation::action_for_key;

#[test]
fn contextual_key_resolution_has_stable_precedence() {
    assert_eq!(
        action_for_key(FocusContext::Transcript, KeyChord::plain(UiKey::Escape)),
        Some(ActionId::CancelOperation)
    );
    assert_eq!(
        action_for_key(FocusContext::Transcript, KeyChord::control(UiKey::Enter)),
        Some(ActionId::CommitTranscriptEdit)
    );
    assert_eq!(
        action_for_key(FocusContext::ClipTimeline, KeyChord::control(UiKey::Enter)),
        Some(ActionId::TranscribeSelectedClip)
    );
    assert_eq!(
        action_for_key(
            FocusContext::RecordingControl,
            KeyChord::control(UiKey::Enter)
        ),
        None
    );
}

#[test]
fn projection_exposes_selected_transcript_and_clip_diagnostics() {
    let recording_id = RecordingId::new();
    let clip_id = ClipId::new();
    let source = SourceAsset::new(AssetKind::AudioFile, PathBuf::from("fixture.wav"))
        .expect("audio source should be accepted");
    let mut state = AppState::new();
    state
        .execute(Command::CreateRecording {
            recording_id,
            source,
        })
        .expect("recording should be created");
    state
        .execute(Command::AddClip {
            recording_id,
            clip_id,
            source_range: TimeRange::new(0, 1_000_000).expect("range should be valid"),
        })
        .expect("clip should be created");
    state
        .execute(Command::BeginTranscription {
            recording_id,
            clip_id,
        })
        .expect("transcription should start");
    state
        .execute(Command::CommitTranscript {
            recording_id,
            clip_id,
            transcript_id: TranscriptId::new(),
            provenance: TranscriptProvenance::RawAsr,
            text: "locally projected".to_string(),
        })
        .expect("transcript should commit");

    let recording = state
        .recording(recording_id)
        .expect("recording should exist");
    let projection = PresentationState::from_recording(recording, Some(clip_id));
    assert_eq!(projection.recording_id, Some(recording_id));
    assert_eq!(projection.selected_clip_id, Some(clip_id));
    assert_eq!(projection.clips[0].status, ClipStatus::Transcribed);
    assert_eq!(
        projection
            .transcript
            .as_ref()
            .map(|transcript| transcript.text.as_str()),
        Some("locally projected")
    );
    assert!(projection.diagnostics.is_empty());
}
