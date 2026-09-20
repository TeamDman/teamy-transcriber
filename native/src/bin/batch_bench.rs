//! Batched ASR regression harness, with resident reuse and a final partial batch.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_macros,
    reason = "Development receipts use serde for reference interoperability."
)]
use anyhow::{Result, ensure};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
use teamy_whisper_native::{
    Engine, MAX_BATCH_SIZE, decoding::GreedySuppression, frontend::Frontend,
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 6,
        "usage: batch_bench MODEL WAV_LIST REPEATS BATCH_SIZE fp32|tf32 GENERATION_CONFIG"
    );
    let paths: Vec<PathBuf> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let repeats: usize = args[2].parse()?;
    let batch_size: usize = args[3].parse()?;
    ensure!(
        !paths.is_empty() && repeats > 0 && (1..=MAX_BATCH_SIZE).contains(&batch_size),
        "invalid benchmark dimensions"
    );
    ensure!(args[4] == "fp32" || args[4] == "tf32", "invalid precision");
    let policy: GreedySuppression = serde_json::from_slice(&std::fs::read(&args[5])?)?;
    let started = Instant::now();
    let mut engine = Engine::load(Path::new(&args[0]), 0, args[4] == "tf32")?;
    engine.configure_greedy(&policy)?;
    let mut frontend = Frontend::new(engine.dims().audio.n_mels)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let mut first_result_ms = None;
    let mut runs = Vec::new();
    let mut batches = Vec::new();
    for index in 0..repeats {
        for group in paths.chunks(batch_size) {
            let begin = Instant::now();
            let mut features = Vec::new();
            for path in group {
                let mut wav = hound::WavReader::open(path)?;
                let spec = wav.spec();
                ensure!(
                    spec.sample_rate == 16000 && spec.channels == 1,
                    "expected mono 16 kHz WAV"
                );
                let audio: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
                    (hound::SampleFormat::Float, 32) => {
                        wav.samples::<f32>().collect::<Result<_, _>>()?
                    }
                    (hound::SampleFormat::Int, 16) => wav
                        .samples::<i16>()
                        .map(|s| s.map(|s| f32::from(s) / 32768.))
                        .collect::<Result<_, _>>()?,
                    _ => anyhow::bail!("expected PCM16 or float32 WAV"),
                };
                features.push(frontend.compute(&audio)?);
            }
            let frontend_ms = begin.elapsed().as_secs_f64() * 1000.;
            let mels: Vec<_> = features.iter().map(Vec::as_slice).collect();
            let result = engine.transcribe_mel_batch(&mels, 448)?;
            let total_ms = begin.elapsed().as_secs_f64() * 1000.;
            first_result_ms.get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.);
            for (path, result) in group.iter().zip(result.transcripts) {
                runs.push(serde_json::json!({"file":path,"index":index,"batch_index":batches.len(),"result":result}));
            }
            batches.push(
                serde_json::json!({"index":index,"size":group.len(),"frontend_ms":frontend_ms,
                "prepare_ms":result.prepare_ms,"decode_ms":result.decode_ms,"total_ms":total_ms}),
            );
        }
    }
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({"backend":"native-batched-asr",
        "scope":"WAV reads, frontend and batched greedy English ASR; no VAD/alignment/diarization",
        "batch_size":batch_size,"precision":args[4],"suppression":policy,
        "load_ms":load_ms,"first_result_ms":first_result_ms,"batch_workspace_bytes":engine.batch_workspace_bytes(),
        "runs":runs,"batches":batches}))?
    );
    Ok(())
}
