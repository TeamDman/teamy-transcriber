#![cfg(feature = "cuda-native")]

use eyre::{Result, ensure};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use teamy_transcriber::domain::AssetKind;
use teamy_transcriber::media::AudioProfile;
use teamy_transcriber::storage::RecordingStore;
use teamy_transcriber::workflow::{
    TranscriptionOptions, TranscriptionSession, create_recording, export_recording_with_timestamps,
    prepare_recording,
};

/// Real model lifecycle regression, opt-in because it needs local CUDA/assets.
/// Supply a small prepared single-file model and a short speech WAV. Files are
/// copied into a unique test directory; original model/audio are never changed.
#[test]
#[ignore = "requires CUDA and TEAMY_TRANSCRIBER_TEST_MODEL / TEAMY_TRANSCRIBER_TEST_WAV"]
fn session_recovers_from_model_repair_and_cancellation() -> Result<()> {
    let model = PathBuf::from(std::env::var("TEAMY_TRANSCRIBER_TEST_MODEL")?);
    let wav = PathBuf::from(std::env::var("TEAMY_TRANSCRIBER_TEST_WAV")?);
    let root = std::env::temp_dir().join(format!("transcriber-session-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&root)?;
    let result = exercise_session(&model, &wav, &root);
    // `root` is the unique directory this test just created, never a model or
    // user-supplied output directory. Drop the session before removing its files.
    std::fs::remove_dir_all(&root)?;
    result
}

fn exercise_session(
    model: &std::path::Path,
    wav: &std::path::Path,
    root: &std::path::Path,
) -> Result<()> {
    let staged = root.join("model");
    std::fs::create_dir(&staged)?;
    for file in ["dims.json", "tokenizer.json", "model.safetensors"] {
        std::fs::copy(model.join(file), staged.join(file))?;
    }
    let store = RecordingStore::new(root.join("home"));
    let id = create_recording(&store, AssetKind::AudioFile, wav)?;
    let prepared = prepare_recording(&store, id)?;
    ensure!(
        (2_000_000..=30_000_000).contains(&prepared.metadata.duration_us),
        "supply a speech WAV between two and thirty seconds long"
    );
    let options = TranscriptionOptions {
        model_dir: staged.clone(),
        max_decode_tokens: 448,
        chunk_duration_us: Some(prepared.metadata.duration_us.div_ceil(2)),
        profile: AudioProfile::Original,
    };
    let mut session = TranscriptionSession::default();
    // All required files exist, but inspection fails inside the cached loader.
    // Repairing the same path must work without replacing the session manually.
    std::fs::write(staged.join("dims.json"), "invalid dimensions")?;
    ensure!(
        session
            .transcribe(&store, id, options.clone(), None, None)
            .is_err(),
        "malformed model unexpectedly loaded"
    );
    std::fs::copy(model.join("dims.json"), staged.join("dims.json"))?;
    let cancelled = AtomicBool::new(false);
    let mut cancel_after_first = |completed, _| {
        if completed == 1 {
            cancelled.store(true, Ordering::Relaxed);
        }
    };
    let partial = session.transcribe(
        &store,
        id,
        options.clone(),
        Some(&cancelled),
        Some(&mut cancel_after_first),
    )?;
    ensure!(
        partial.cancelled && partial.chunks.len() == 1,
        "cancellation did not stop after one committed clip"
    );
    ensure!(
        export_recording_with_timestamps(&store, id, None)?.transcript_count == 1,
        "cancelled work was not persisted"
    );
    let resumed = session.transcribe(&store, id, options.clone(), None, None)?;
    ensure!(
        !resumed.cancelled && resumed.chunks.len() >= 2,
        "session did not recover after cancellation"
    );
    ensure!(
        resumed.chunks[0].clip_id == partial.chunks[0].clip_id
            && resumed.chunks[0].text == partial.chunks[0].text,
        "resumed clip identity/text changed"
    );
    ensure!(
        export_recording_with_timestamps(&store, id, None)?.transcript_count
            == resumed.chunks.len(),
        "resumed transcripts did not persist"
    );
    let mut different_model = options.clone();
    different_model.model_dir = root.join("missing-model");
    ensure!(
        session
            .transcribe(&store, id, different_model, None, None)
            .is_err(),
        "model selection incorrectly reused the previous model"
    );
    let fresh_id = create_recording(&store, AssetKind::AudioFile, wav)?;
    prepare_recording(&store, fresh_id)?;
    let fresh = session.transcribe(&store, fresh_id, options, None, None)?;
    ensure!(
        fresh
            .chunks
            .iter()
            .map(|chunk| &chunk.text)
            .eq(resumed.chunks.iter().map(|chunk| &chunk.text)),
        "a later recording inherited stale decoder state"
    );
    Ok(())
}
