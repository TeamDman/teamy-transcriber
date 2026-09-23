use std::path::Path;
use std::process::Command;
use std::process::Output;
use teamy_transcriber::domain::AssetKind;
use teamy_transcriber::storage::RecordingStore;

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_teamy-transcriber"))
        .args(args)
        .env("TEAMY_TRANSCRIBER_HOME_DIR", root.join("app"))
        .env("TEAMY_TRANSCRIBER_CACHE_DIR", root.join("cache"))
        .env("TEAMY_TRANSCRIBER_MODEL_DIR", root.join("missing-model"))
        .output()
        .expect("CLI should start")
}

#[test]
fn missing_model_retains_one_recording_and_resume_reuses_it() {
    let root = std::env::temp_dir().join(format!("transcribe-cli-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("speaker's [example].WEBM");
    std::fs::write(&source, b"source is not decoded before model validation").unwrap();
    let result = run(&root, &["transcribe", source.to_str().unwrap()]);
    assert!(!result.status.success());
    let store = RecordingStore::new(root.join("app"));
    let recordings = store.list_recordings().unwrap();
    assert_eq!(recordings.len(), 1);
    assert_eq!(recordings[0].source.kind, AssetKind::VideoFile);
    let id = recordings[0].id.to_string();
    let receipt_path = store.events_path(recordings[0].id);
    let saved_receipt = std::fs::read(&receipt_path).unwrap();
    for format in ["text", "json", "csv"] {
        let listed = run(&root, &["--output-format", format, "recording", "list"]);
        assert!(
            listed.status.success(),
            "{}",
            String::from_utf8_lossy(&listed.stderr)
        );
        let listing = String::from_utf8_lossy(&listed.stdout);
        assert!(listing.contains(&id), "{listing}");
        assert!(listing.contains("speaker's [example].WEBM"), "{listing}");
        assert!(listing.contains("clip_count"), "{listing}");
        assert!(listing.contains("transcript_count"), "{listing}");
    }
    assert_eq!(std::fs::read(receipt_path).unwrap(), saved_receipt);
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("NOT cleaned up"), "{error}");
    assert!(
        error.contains(&format!("transcribe --resume {id}")),
        "{error}"
    );
    let retry = run(&root, &["transcribe", "--resume", &id]);
    assert!(!retry.status.success());
    assert_eq!(store.list_recordings().unwrap().len(), 1);
    assert!(source.is_file());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn recording_list_on_a_fresh_home_is_empty_and_does_not_create_storage() {
    let root = std::env::temp_dir().join(format!("recording-list-{}", uuid::Uuid::new_v4()));
    let listed = run(&root, &["--output-format", "json", "recording", "list"]);
    assert!(
        listed.status.success(),
        "{}",
        String::from_utf8_lossy(&listed.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&listed.stdout).trim(), "[]");
    assert!(!root.exists());
}

#[test]
fn recording_create_infers_video_and_accepts_an_override() {
    let root = std::env::temp_dir().join(format!("recording-kind-{}", uuid::Uuid::new_v4()));
    let source = root.join("sample.WEBM");
    assert!(
        run(&root, &["recording", "create", source.to_str().unwrap()])
            .status
            .success()
    );
    let store = RecordingStore::new(root.join("app"));
    assert_eq!(
        store.list_recordings().unwrap()[0].source.kind,
        AssetKind::VideoFile
    );
    assert!(
        run(
            &root,
            &[
                "recording",
                "create",
                source.to_str().unwrap(),
                "--kind",
                "audio"
            ]
        )
        .status
        .success()
    );
    assert!(
        store
            .list_recordings()
            .unwrap()
            .iter()
            .any(|r| r.source.kind == AssetKind::AudioFile)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn live_microphone_rejects_invalid_options_before_capture() {
    let root = std::env::temp_dir().join(format!("microphone-cli-{}", uuid::Uuid::new_v4()));
    for (args, expected) in [
        (
            vec!["microphone", "transcribe", "--duration-ms", "0"],
            "greater than zero",
        ),
        (
            vec!["microphone", "transcribe", "--chunk-duration-ms", "499"],
            "between 500 and 30000",
        ),
        (
            vec!["--output-format", "json", "microphone", "transcribe"],
            "emits plain text",
        ),
    ] {
        let output = run(&root, &args);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(expected), "{error}");
    }
    assert!(!root.exists());
}

#[test]
fn phone_model_option_requires_phone_mode_without_opening_microphone() {
    let root = std::env::temp_dir().join(format!("phone-options-{}", uuid::Uuid::new_v4()));
    let output = run(
        &root,
        &["microphone", "transcribe", "--phone-model-dir", "missing"],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires --phones"));
    assert!(output.stdout.is_empty());
    assert!(!root.exists());
}

#[test]
fn file_transcribe_phone_flags_select_phone_model_and_preserve_retry_mode() {
    let root = std::env::temp_dir().join(format!("file-phones-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("sample.wav");
    std::fs::write(&source, b"RIFF\x00\x00\x00\x00WAVE").unwrap();
    for flag in ["--phonemes", "--phones"] {
        let result = run(
            &root,
            &[
                "transcribe",
                source.to_str().unwrap(),
                flag,
                "--phone-model-dir",
                root.join("missing-phone-model").to_str().unwrap(),
            ],
        );
        assert!(!result.status.success());
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(error.contains("phone model validation"), "{error}");
        assert!(
            error.contains("--resume") && error.contains("--phonemes"),
            "{error}"
        );
    }
    assert_eq!(
        RecordingStore::new(root.join("app"))
            .list_recordings()
            .unwrap()
            .len(),
        2
    );
    std::fs::remove_dir_all(root).unwrap();
}
