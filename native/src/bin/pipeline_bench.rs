//! Full-recording development harness: WAV, native VAD, merge and native ASR.
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
    Engine,
    decoding::GreedySuppression,
    frontend::Frontend,
    vad::{Silero, merge_segments, speech_segments},
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 5,
        "usage: pipeline_bench WHISPER_MODEL SILERO_WEIGHTS WAV_LIST REPEATS GENERATION_CONFIG"
    );
    let paths: Vec<PathBuf> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let repeats: usize = args[3].parse()?;
    ensure!(!paths.is_empty() && repeats > 0, "empty benchmark");
    let suppression: GreedySuppression = serde_json::from_slice(&std::fs::read(&args[4])?)?;
    let begin = Instant::now();
    let mut vad = Silero::load(Path::new(&args[1]))?;
    let mut engine = Engine::load(Path::new(&args[0]), 0, true)?;
    engine.configure_greedy(&suppression)?;
    let mut frontend = Frontend::new(engine.dims().audio.n_mels)?;
    let load_ms = begin.elapsed().as_secs_f64() * 1000.;
    let mut first_result_ms = None;
    let mut runs = Vec::new();
    for path in paths {
        for index in 0..repeats {
            let started = Instant::now();
            let mut wav = hound::WavReader::open(&path)?;
            let spec = wav.spec();
            ensure!(
                spec.channels == 1
                    && spec.sample_rate == 16000
                    && spec.sample_format == hound::SampleFormat::Int
                    && spec.bits_per_sample == 16,
                "expected mono 16 kHz PCM16 WAV"
            );
            let audio: Vec<f32> = wav
                .samples::<i16>()
                .map(|x| x.map(|x| f32::from(x) / 32768.))
                .collect::<Result<_, _>>()?;
            let audio_ms = started.elapsed().as_secs_f64() * 1000.;
            let vad_start = Instant::now();
            let probabilities = vad.probabilities(&audio)?;
            let speech = speech_segments(&probabilities, audio.len(), 0.5, 30.)?;
            let merged = merge_segments(&speech, 480000)?;
            let vad_ms = vad_start.elapsed().as_secs_f64() * 1000.;
            let mut segments = Vec::new();
            for span in &merged {
                let start = span.start as f64 / 16000.;
                let end = span.end as f64 / 16000.;
                // Match WhisperX's seconds-to-samples truncation exactly, including
                // potential one-sample round-trip loss at some decimal boundaries.
                let start_sample = (start * 16000.) as usize;
                let end_sample = (end * 16000.) as usize;
                let chunk_start = Instant::now();
                let mel = frontend.compute(&audio[start_sample..end_sample])?;
                let result = engine.transcribe_mel(&mel, 448)?;
                segments.push(serde_json::json!({"start":start,"end":end,
                    "start_sample":start_sample,"end_sample":end_sample,
                    "total_ms":chunk_start.elapsed().as_secs_f64()*1000.,
                    "text":result.text,"result":result}));
            }
            let total_ms = started.elapsed().as_secs_f64() * 1000.;
            first_result_ms.get_or_insert_with(|| begin.elapsed().as_secs_f64() * 1000.);
            eprintln!(
                "{} request {}: {:.1} ms, {} segments",
                path.display(),
                index,
                total_ms,
                segments.len()
            );
            runs.push(
                serde_json::json!({"source":path,"index":index,"batch_size":1,
                "audio_seconds":audio.len() as f64/16000.,"audio_ms":audio_ms,"vad_ms":vad_ms,
                "total_ms":total_ms,"result":{"segments":segments,"language":"en"}}),
            );
        }
    }
    println!(
        "{}",
        serde_json::to_string(
            &serde_json::json!({"backend":"source-defined-native-pipeline",
        "scope":"PCM16 WAV loading, CPU Silero VAD, merging, CUDA greedy English ASR and assembly; batch one. Excludes alignment, diarization and file export. Development harness, not application integration.",
        "precision":"tf32","suppression":suppression,"load_ms":load_ms,
        "first_result_ms":first_result_ms,"runs":runs,"full_goal_acceptance":false})
        )?
    );
    Ok(())
}
