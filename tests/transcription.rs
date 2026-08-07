use std::path::PathBuf;
use teamy_transcriber::domain::ClipId;
use teamy_transcriber::domain::RecordingId;
use teamy_transcriber::transcription::LocalWhisperXBackend;
use teamy_transcriber::transcription::LocalWhisperXConfig;
use teamy_transcriber::transcription::RuntimeAssetStatus;
use teamy_transcriber::transcription::TranscriptionBackend;
use teamy_transcriber::transcription::TranscriptionError;
use teamy_transcriber::transcription::TranscriptionRequest;

#[test]
fn local_whisperx_rejects_missing_configuration_before_launching_worker() {
    let backend = LocalWhisperXBackend::new(LocalWhisperXConfig {
        python_executable: PathBuf::from("missing-python"),
        worker_script: PathBuf::from("missing-worker.py"),
        model_dir: PathBuf::from("missing-model-directory"),
        model_name: "small".to_string(),
        device: "cpu".to_string(),
        compute_type: "int8".to_string(),
        batch_size: 1,
    });

    let capabilities = backend.capabilities();
    assert_eq!(capabilities.backend_id, "whisperx-local");
    assert!(capabilities.local_only);
    assert!(capabilities.accepts_normalized_audio);

    let result = backend.transcribe(&TranscriptionRequest {
        recording_id: RecordingId::new(),
        clip_id: ClipId::new(),
        audio_path: PathBuf::from("missing-audio.wav"),
    });
    assert!(matches!(result, Err(TranscriptionError::Configuration(_))));
}

#[test]
fn local_whisperx_readiness_resolves_an_executable_path() {
    let python_executable = std::env::current_exe().expect("test executable should resolve");
    let backend = LocalWhisperXBackend::new(LocalWhisperXConfig {
        python_executable,
        worker_script: PathBuf::from("missing-worker.py"),
        model_dir: PathBuf::from("missing-model-directory"),
        model_name: "small".to_string(),
        device: "cpu".to_string(),
        compute_type: "int8".to_string(),
        batch_size: 1,
    });

    let readiness = backend.readiness();
    assert_eq!(readiness.python, RuntimeAssetStatus::Present);
    assert_eq!(readiness.worker_script, RuntimeAssetStatus::Missing);
    assert_eq!(readiness.model_dir, RuntimeAssetStatus::Missing);
}
