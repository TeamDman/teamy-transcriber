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
