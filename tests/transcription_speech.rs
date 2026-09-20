#![cfg(feature = "cuda-native")]

use eyre::Result;
use eyre::ensure;
use std::path::Path;
use std::path::PathBuf;
use teamy_transcriber::domain::AssetKind;
use teamy_transcriber::domain::ClipPlanKind;
use teamy_transcriber::media::AudioProfile;
use teamy_transcriber::native_whisper::speech;
use teamy_transcriber::storage::RecordingStore;
use teamy_transcriber::transcription::SpeechPlan;
use teamy_transcriber::workflow::TranscriptionOptions;
use teamy_transcriber::workflow::TranscriptionSession;
use teamy_transcriber::workflow::create_recording;
use teamy_transcriber::workflow::prepare_recording;

#[test]
#[ignore = "requires SILERO_TEST_WEIGHTS and TEAMY_TRANSCRIBER_TEST_MODEL; CPU-only inference"]
fn prepared_vad_checks_assets_streams_and_persists_silence() -> Result<()> {
    let model = PathBuf::from(std::env::var("TEAMY_TRANSCRIBER_TEST_MODEL")?);
    let weights = PathBuf::from(std::env::var("SILERO_TEST_WEIGHTS")?);
    let root = std::env::temp_dir().join(format!("transcriber-speech-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&root)?;
    let result = exercise(&root, &model, &weights);
    // This test owns this newly-created UUID directory and all copied files.
    std::fs::remove_dir_all(&root)?;
    result
}

fn exercise(root: &Path, model: &Path, weights: &Path) -> Result<()> {
    let staged = root.join("model");
    std::fs::create_dir(&staged)?;
    for name in [
        "model.safetensors",
        "tokenizer.json",
        "dims.json",
        "config.json",
    ] {
        std::fs::copy(model.join(name), staged.join(name))?;
    }
    let policy = r#"{"suppress_tokens":[1],"begin_suppress_tokens":[2]}"#;
    std::fs::write(staged.join("generation_config.json"), policy)?;
    let prepared = root.join("prepared");
    teamy_transcriber::native_whisper::prepare::prepare_safetensors_model(&staged, &prepared)?;
    ensure!(std::fs::read_to_string(prepared.join("generation_config.json"))? == policy);
    let staged = prepared;
    let bad = root.join("bad.safetensors");
    std::fs::write(&bad, b"invalid tensor data")?;
    let _ = speech::prepare(&bad, &staged).unwrap_err();
    ensure!(
        !staged.join("vad").exists(),
        "invalid package was published"
    );
    let manifest = speech::prepare(weights, &staged)?;
    ensure!(
        manifest.bytes == std::fs::metadata(weights)?.len(),
        "wrong asset length"
    );
    let _ = speech::prepare(weights, &staged).unwrap_err();
    let saved = std::fs::read(staged.join("vad").join("silero.safetensors"))?;
    ensure!(
        saved == std::fs::read(weights)?,
        "preparation modified weights"
    );
    let wav = root.join("silence.wav");
    write_silence(&wav, 32017)?;
    let mut checks = 0;
    ensure!(
        matches!(
            speech::detect(&staged, &wav, &mut || {
                checks += 1;
                checks >= 4
            })?,
            SpeechPlan::Cancelled
        ),
        "streaming detector ignored cancellation"
    );
    for _ in 0..2 {
        let SpeechPlan::Detected {
            ranges,
            weights_sha256,
        } = speech::detect(&staged, &wav, &mut || false)?
        else {
            eyre::bail!("expected complete detection");
        };
        ensure!(
            ranges.is_empty() && weights_sha256 == manifest.sha256,
            "silence/reset/hash mismatch"
        );
    }
    // A valid shape is insufficient: reject any byte change against the manifest.
    let destination = staged.join("vad").join("silero.safetensors");
    let mut corrupted = saved.clone();
    *corrupted.last_mut().unwrap() ^= 1;
    std::fs::write(&destination, corrupted)?;
    let error = speech::detect(&staged, &wav, &mut || false).unwrap_err();
    ensure!(
        error.to_string().contains("checksum"),
        "corrupt weights were not rejected"
    );
    std::fs::write(&destination, saved)?;
    exercise_silent_workflow(root, &staged, &wav)?;
    Ok(())
}

fn exercise_silent_workflow(root: &Path, model: &Path, wav: &Path) -> Result<()> {
    let store = RecordingStore::new(root.join("home"));
    let id = create_recording(&store, AssetKind::AudioFile, wav)?;
    prepare_recording(&store, id)?;
    let options = TranscriptionOptions {
        model_dir: model.into(),
        max_decode_tokens: 448,
        chunk_duration_us: None,
        profile: AudioProfile::Original,
    };
    let mut session = TranscriptionSession::default();
    let mut invalid = options.clone();
    invalid.max_decode_tokens = 0;
    ensure!(
        session.transcribe(&store, id, invalid, None, None).is_err(),
        "silence bypassed configuration validation"
    );
    ensure!(store.load_recording(id)?.clip_plan.is_none());
    for _ in 0..2 {
        let report = session.transcribe(&store, id, options.clone(), None, None)?;
        ensure!(
            report.no_speech && !report.cancelled && report.chunks.is_empty(),
            "silent workflow attempted ASR"
        );
        let saved = store.load_recording(id)?;
        ensure!(
            saved.clips.is_empty() && saved.transcripts.is_empty(),
            "silence produced transcript artifacts"
        );
        ensure!(
            matches!(
                saved.clip_plan.map(|plan| plan.kind),
                Some(ClipPlanKind::VoiceActivity { .. })
            ),
            "empty speech plan was not persisted"
        );
    }
    Ok(())
}

fn write_silence(path: &Path, samples: usize) -> Result<()> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for _ in 0..samples {
        writer.write_sample(0f32)?;
    }
    writer.finalize()?;
    Ok(())
}
