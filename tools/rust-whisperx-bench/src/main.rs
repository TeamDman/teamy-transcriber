//! Pinned public ASR engine used by native-whisperx; no patched dependencies.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_macros,
    reason = "This isolated reference adapter preserves upstream serde-only configuration and transcript contracts in its receipts."
)]
use anyhow::{Result, ensure};
use audio_analysis_transcription::{
    AsrRequest, CandleWhisperComputeType, CandleWhisperOptions, CandleWhisperTimingMode,
    CandleWhisperTranscriptionRequestConfig, CandleWhisperWindowControls, LoadedAudio,
    NativeDevicePreference, ReusableCandleWhisperTranscriber, SpeechActivitySegment,
    TranscriptionPipelineEvent, TranscriptionPipelineObserver, TranscriptionTask,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::PathBuf, time::Instant};

#[derive(Default)]
struct Observer {
    load_ms: f64,
    reuses: usize,
}
impl TranscriptionPipelineObserver for Observer {
    fn observe(&mut self, event: TranscriptionPipelineEvent) {
        match event {
            TranscriptionPipelineEvent::ModelLoadEnd {
                duration_seconds, ..
            } => {
                self.load_ms += duration_seconds * 1000.;
            }
            TranscriptionPipelineEvent::ModelReuse { .. } => self.reuses += 1,
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let process_start = Instant::now();
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 5,
        "usage: rust-whisperx-bench CANONICAL_MODEL WAV_OR_JSON_LIST REPEATS fp32|fp16 MODEL_ID"
    );
    let root = PathBuf::from(&args[0]);
    ensure!(root.is_dir(), "canonical model directory missing");
    let generation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("generation_config.json"))?)?;
    ensure!(
        generation["max_length"] == 448,
        "benchmark requires a 448-token context bound"
    );
    let suppression = json!({"suppress_tokens":generation["suppress_tokens"],"begin_suppress_tokens":generation["begin_suppress_tokens"]});
    let compute_type = match args[3].as_str() {
        "fp32" => CandleWhisperComputeType::Fp32,
        "fp16" => CandleWhisperComputeType::Fp16,
        _ => anyhow::bail!("precision must be fp32 or fp16"),
    };
    let paths: Vec<PathBuf> = if args[1].ends_with(".json") {
        serde_json::from_slice(&std::fs::read(&args[1])?)?
    } else {
        vec![PathBuf::from(&args[1])]
    };
    let repeats: usize = args[2].parse()?;
    ensure!(!paths.is_empty() && repeats > 0, "empty benchmark");
    let mut provider = ReusableCandleWhisperTranscriber::new(CandleWhisperOptions {
        model_id: args[4].clone(),
        language: Some("en".into()),
        device: NativeDevicePreference::Cuda,
        compute_type,
        model_bundle: Some(root.clone()),
        model_cache_only: true,
        batch_chunks: false,
        max_batch_size: Some(1),
        ..Default::default()
    });
    let config = CandleWhisperTranscriptionRequestConfig {
        window: CandleWhisperWindowControls {
            timing_mode: CandleWhisperTimingMode::NoTimestamps,
            leading_context_seconds: 0.,
            trailing_context_seconds: 0.,
        },
        ..Default::default()
    };
    let mut observer = Observer::default();
    let mut first_result_ms = None;
    let mut runs = Vec::new();
    for path in paths {
        for index in 0..repeats {
            let started = Instant::now();
            let mut wav = hound::WavReader::open(&path)?;
            let spec = wav.spec();
            ensure!(
                spec.sample_rate == 16000 && spec.channels == 1,
                "expected mono 16 kHz WAV"
            );
            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => wav.samples::<f32>().collect::<Result<_, _>>()?,
                hound::SampleFormat::Int => {
                    ensure!(spec.bits_per_sample == 16, "expected FP32 or PCM16 WAV");
                    wav.samples::<i16>()
                        .map(|s| s.map(|n| n as f32 / 32768.))
                        .collect::<Result<_, _>>()?
                }
            };
            ensure!(
                !samples.is_empty()
                    && samples.len() <= 480000
                    && samples.iter().all(|n| n.is_finite()),
                "expected finite audio up to 30 seconds"
            );
            let audio_seconds = samples.len() as f64 / 16000.;
            let response = provider.transcribe_with_request_config_and_observer(
                AsrRequest {
                    audio: LoadedAudio {
                        samples,
                        sample_rate: 16000,
                        channels: 1,
                        source: None,
                    },
                    chunks: vec![SpeechActivitySegment::new(0., audio_seconds, 1.)?],
                    task: TranscriptionTask::Transcribe,
                    language: Some("en".into()),
                    model_id: args[4].clone(),
                },
                config.clone(),
                &mut observer,
            )?;
            let text = response
                .transcript
                .segments
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let total_ms = started.elapsed().as_secs_f64() * 1000.;
            first_result_ms.get_or_insert_with(|| process_start.elapsed().as_secs_f64() * 1000.);
            ensure!(
                response.diagnostics.iter().any(|d| d == "cuda=true"),
                "reference did not use CUDA"
            );
            ensure!(
                response
                    .diagnostics
                    .iter()
                    .any(|d| d == "timingMode=noTimestamps"),
                "reference timing mode mismatch"
            );
            runs.push(json!({"id":path.file_stem().unwrap().to_string_lossy(),"index":index,"audio_seconds":audio_seconds,"total_ms":total_ms,"text":text,"tokens":null,"diagnostics":response.diagnostics,"transcript":response.transcript}));
        }
    }
    let mut digest = Sha256::new();
    let mut file = File::open(root.join("model.safetensors"))?;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":1,"backend":"rust-whisperx-asr","version":"moenarch-audio-analysis-transcription=0.1.15",
            "scope":"WAV-to-text ASR component only; greedy English, batch one, canonical generation_config suppression; no VAD/alignment/diarization",
            "device":"cuda","compute_type":args[3],"model_sha256":format!("{:x}",digest.finalize()),
            "generation_config_sha256":format!("{:x}",Sha256::digest(std::fs::read(root.join("generation_config.json"))?)),
        "config":config,"suppression":suppression,"load_ms":observer.load_ms,"model_reuses":observer.reuses,
            "first_result_ms":first_result_ms,"runs":runs
        }))?
    );
    Ok(())
}
