use std::path::PathBuf;
use teamy_transcriber::domain::ClipId;
use teamy_transcriber::domain::RecordingId;
use teamy_transcriber::transcription::NativeWhisperBackend;
use teamy_transcriber::transcription::NativeWhisperConfig;
use teamy_transcriber::transcription::RuntimeAssetStatus;
use teamy_transcriber::transcription::TranscriptionBackend;
use teamy_transcriber::transcription::TranscriptionError;
use teamy_transcriber::transcription::TranscriptionRequest;

#[test]
fn native_whisper_rejects_missing_configuration_before_model_load() {
    let backend = NativeWhisperBackend::new(NativeWhisperConfig {
        model_dir: PathBuf::from("missing-model-directory"),
        max_decode_tokens: 64,
    });

    let capabilities = backend.capabilities();
    assert_eq!(capabilities.backend_id, "whisper-burn-native-cpu");
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
fn native_whisper_readiness_reports_missing_artifacts() {
    let backend = NativeWhisperBackend::new(NativeWhisperConfig {
        model_dir: PathBuf::from("missing-model-directory"),
        max_decode_tokens: 64,
    });

    let readiness = backend.readiness();
    assert_eq!(readiness.model_dir, RuntimeAssetStatus::Missing);
    assert_eq!(readiness.weights, RuntimeAssetStatus::Missing);
    assert_eq!(readiness.dims, RuntimeAssetStatus::Missing);
    assert_eq!(readiness.tokenizer, RuntimeAssetStatus::Missing);
}
