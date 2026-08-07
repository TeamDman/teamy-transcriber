use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use teamy_transcriber::domain::AppState;
use teamy_transcriber::domain::AssetKind;
use teamy_transcriber::domain::ClipId;
use teamy_transcriber::domain::Command;
use teamy_transcriber::domain::DomainError;
use teamy_transcriber::domain::RecordingId;
use teamy_transcriber::domain::RecordingStatus;
use teamy_transcriber::domain::SourceAsset;
use teamy_transcriber::domain::TimeRange;
use teamy_transcriber::domain::TranscriptId;
use teamy_transcriber::domain::TranscriptProvenance;
use teamy_transcriber::storage::RecordingStore;
use teamy_transcriber::transcription::FakeTranscriptionBackend;
use teamy_transcriber::transcription::LocalModelInventory;
use teamy_transcriber::transcription::TranscriptionBackend;
use teamy_transcriber::transcription::TranscriptionRequest;

#[test]
fn domain_events_replay_to_the_same_state() {
    let recording_id = RecordingId::new();
    let first_clip = ClipId::new();
    let second_clip = ClipId::new();
    let source = SourceAsset::new(AssetKind::AudioFile, PathBuf::from("fixture.wav"))
        .expect("fixture path should be accepted");
    let mut state = AppState::new();
    let mut records = Vec::new();

    records.push(
        state
            .execute(Command::CreateRecording {
                recording_id,
                source,
            })
            .expect("recording creation should succeed"),
    );
    records.push(
        state
            .execute(Command::AddClip {
                recording_id,
                clip_id: first_clip,
                source_range: TimeRange::new(0, 1_000_000).expect("range should be valid"),
            })
            .expect("first clip should succeed"),
    );
    records.push(
        state
            .execute(Command::AddClip {
                recording_id,
                clip_id: second_clip,
                source_range: TimeRange::new(1_000_000, 2_000_000).expect("range should be valid"),
            })
            .expect("second clip should succeed"),
    );
    records.push(
        state
            .execute(Command::MoveClip {
                recording_id,
                clip_id: second_clip,
                target_index: 0,
            })
            .expect("clip movement should succeed"),
    );
    records.push(
        state
            .execute(Command::CommitTranscript {
                recording_id,
                clip_id: second_clip,
                transcript_id: TranscriptId::new(),
                provenance: TranscriptProvenance::RawAsr,
                text: "hello locally".to_string(),
            })
            .expect("transcript commit should succeed"),
    );

    let replayed = AppState::replay(records).expect("events should replay");
    assert_eq!(replayed, state);
    assert_eq!(
        state
            .recording(recording_id)
            .expect("recording should exist")
            .clips[0]
            .id,
        second_clip
    );
}

#[test]
fn storage_writes_manifest_and_ndjson_receipt() {
    let root = unique_temp_dir("teamy-transcriber-storage");
    let store = RecordingStore::new(&root);
    let recording_id = RecordingId::new();
    let source = SourceAsset::new(AssetKind::MicrophoneRecording, PathBuf::from("mic.wav"))
        .expect("microphone path should be accepted");
    let mut state = AppState::new();

    store
        .apply_command(
            &mut state,
            Command::CreateRecording {
                recording_id,
                source,
            },
        )
        .expect("recording should persist");
    store
        .apply_command(
            &mut state,
            Command::AddClip {
                recording_id,
                clip_id: ClipId::new(),
                source_range: TimeRange::new(0, 500_000).expect("range should be valid"),
            },
        )
        .expect("clip should persist");

    let loaded = store
        .load_recording(recording_id)
        .expect("recording should load from receipt");
    assert_eq!(
        Some(&loaded),
        state.recording(recording_id),
        "replayed state should match the in-memory state"
    );
    assert!(store.events_path(recording_id).is_file());
    assert!(store.manifest_path(recording_id).is_file());

    std::fs::remove_dir_all(root).expect("test directory should be removable");
}

#[test]
fn fake_backend_is_explicitly_local_and_deterministic() {
    let backend = FakeTranscriptionBackend::with_text("fixture transcript");
    let capabilities = backend.capabilities();
    assert!(capabilities.local_only);
    assert_eq!(capabilities.backend_id, "fake");

    let result = backend
        .transcribe(&TranscriptionRequest {
            recording_id: RecordingId::new(),
            clip_id: ClipId::new(),
            audio_path: PathBuf::from("fixture.wav"),
        })
        .expect("fake backend should return its configured result");
    assert_eq!(result.text, "fixture transcript");
}

#[test]
fn local_model_inventory_does_not_download_or_modify() {
    let root = unique_temp_dir("teamy-transcriber-model");
    std::fs::create_dir_all(root.join("nested")).expect("model fixture directory should exist");
    std::fs::write(root.join("model.bin"), b"fixture").expect("model fixture should be writable");
    std::fs::write(root.join("nested").join("config.json"), b"{}")
        .expect("model metadata fixture should be writable");

    let inventory =
        LocalModelInventory::inspect(&root).expect("model inventory should be inspectable");
    assert!(inventory.exists);
    assert_eq!(inventory.file_count, 2);

    std::fs::remove_dir_all(root).expect("test directory should be removable");
}

#[test]
fn invalid_time_ranges_are_rejected_before_events_exist() {
    assert!(TimeRange::new(10, 10).is_err());
    assert!(TimeRange::new(20, 10).is_err());
}

#[test]
fn active_clip_ranges_must_not_overlap() {
    let recording_id = RecordingId::new();
    let source = SourceAsset::new(AssetKind::AudioFile, PathBuf::from("fixture.wav"))
        .expect("fixture path should be accepted");
    let mut state = AppState::new();
    state
        .execute(Command::CreateRecording {
            recording_id,
            source,
        })
        .expect("recording creation should succeed");
    state
        .execute(Command::AddClip {
            recording_id,
            clip_id: ClipId::new(),
            source_range: TimeRange::new(0, 1_000_000).expect("range should be valid"),
        })
        .expect("first clip should succeed");

    let result = state.execute(Command::AddClip {
        recording_id,
        clip_id: ClipId::new(),
        source_range: TimeRange::new(999_999, 2_000_000).expect("range should be valid"),
    });
    assert!(matches!(result, Err(DomainError::ClipOverlaps { .. })));
}

#[test]
fn recording_capture_lifecycle_is_typed_and_replayable() {
    let recording_id = RecordingId::new();
    let source = SourceAsset::new(AssetKind::MicrophoneRecording, PathBuf::from("mic.wav"))
        .expect("microphone path should be accepted");
    let mut state = AppState::new();
    state
        .execute(Command::CreateRecording {
            recording_id,
            source,
        })
        .expect("recording creation should succeed");

    let invalid_save = state.execute(Command::CompleteRecording { recording_id });
    assert!(matches!(
        invalid_save,
        Err(DomainError::InvalidRecordingTransition {
            action: "save",
            actual: RecordingStatus::Created,
            ..
        })
    ));

    state
        .execute(Command::StartRecording { recording_id })
        .expect("recording should start");
    state
        .execute(Command::CompleteRecording { recording_id })
        .expect("recording should save");
    let saved = state
        .recording(recording_id)
        .expect("recording should exist");
    assert_eq!(saved.status, RecordingStatus::Saved);
    assert_eq!(saved.failure, None);

    let failed_id = RecordingId::new();
    let mut failed_state = AppState::new();
    let mut records = Vec::new();
    records.push(
        failed_state
            .execute(Command::CreateRecording {
                recording_id: failed_id,
                source: SourceAsset::new(
                    AssetKind::MicrophoneRecording,
                    PathBuf::from("other-mic.wav"),
                )
                .expect("microphone path should be accepted"),
            })
            .expect("failed recording creation should succeed"),
    );
    records.push(
        failed_state
            .execute(Command::StartRecording {
                recording_id: failed_id,
            })
            .expect("failed recording should start"),
    );
    records.push(
        failed_state
            .execute(Command::FailRecording {
                recording_id: failed_id,
                reason: "simulated device disconnect".to_string(),
            })
            .expect("failure should be recorded"),
    );
    let replayed = AppState::replay(records).expect("capture lifecycle should replay");
    let failed = replayed
        .recording(failed_id)
        .expect("failed recording should exist");
    assert_eq!(failed.status, RecordingStatus::Failed);
    assert_eq!(
        failed.failure.as_deref(),
        Some("simulated device disconnect")
    );
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{timestamp}", std::process::id()))
}
