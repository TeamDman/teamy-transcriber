//! Complete upstream VAD/ASR pipeline; this adapter only selects public controls.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_macros,
    reason = "This isolated reference adapter preserves upstream serde-only configuration and transcript contracts in its receipts."
)]
use anyhow::Result;
use anyhow::ensure;
use audio_analysis_transcription::AsrRequest;
use audio_analysis_transcription::AsrResponse;
use audio_analysis_transcription::AudioTranscriptionProvider;
use audio_analysis_transcription::CandleWhisperComputeType;
use audio_analysis_transcription::CandleWhisperDecodeRuntime;
use audio_analysis_transcription::CandleWhisperOptions;
use audio_analysis_transcription::CandleWhisperTimingMode;
use audio_analysis_transcription::CandleWhisperTranscriptionRequestConfig;
use audio_analysis_transcription::CandleWhisperWindowControls;
use audio_analysis_transcription::NativeDevicePreference;
use audio_analysis_transcription::ReusableCandleWhisperTranscriber;
use audio_analysis_transcription::SileroVadOptions;
use audio_analysis_transcription::SileroVadTranscriptionProvider;
use audio_analysis_transcription::TranscriptionPipelineEvent;
use audio_analysis_transcription::TranscriptionPipelineObserver;
use audio_analysis_transcription::TranscriptionPipelineRequest;
use audio_analysis_transcription::TranscriptionProviderSelection;
use audio_analysis_transcription::TranscriptionSource;
use audio_analysis_transcription::run_transcription_pipeline_with_observer;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

struct ConfiguredAsr {
    model: ReusableCandleWhisperTranscriber,
    config: CandleWhisperTranscriptionRequestConfig,
}
impl AudioTranscriptionProvider for ConfiguredAsr {
    fn provider_id(&self) -> &str {
        "candle-whisper"
    }
    fn transcribe(&mut self, request: AsrRequest) -> video_analysis_core::Result<AsrResponse> {
        self.transcribe_with_observer(request, &mut Observer::default())
    }
    fn transcribe_with_observer(
        &mut self,
        request: AsrRequest,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> video_analysis_core::Result<AsrResponse> {
        self.model.transcribe_with_request_config_and_observer(
            request,
            self.config.clone(),
            observer,
        )
    }
}

#[derive(Default)]
struct Observer {
    vad_started: Option<Instant>,
    asr_started: Option<Instant>,
    phases: BTreeMap<&'static str, f64>,
    model_reuses: usize,
    vad_segments: Option<usize>,
}
impl TranscriptionPipelineObserver for Observer {
    fn observe(&mut self, event: TranscriptionPipelineEvent) {
        match event {
            TranscriptionPipelineEvent::DecodeEnd {
                duration_seconds, ..
            } => {
                self.phases.insert("audio_ms", duration_seconds * 1000.);
            }
            TranscriptionPipelineEvent::VadStart { .. } => self.vad_started = Some(Instant::now()),
            TranscriptionPipelineEvent::VadEnd { segments, .. } => {
                self.vad_segments = Some(segments);
                self.phases.insert(
                    "vad_ms",
                    self.vad_started.unwrap().elapsed().as_secs_f64() * 1000.,
                );
            }
            TranscriptionPipelineEvent::AsrStart { .. } => self.asr_started = Some(Instant::now()),
            TranscriptionPipelineEvent::AsrEnd { .. } => {
                self.phases.insert(
                    "asr_ms",
                    self.asr_started.unwrap().elapsed().as_secs_f64() * 1000.,
                );
            }
            TranscriptionPipelineEvent::ModelLoadEnd {
                duration_seconds, ..
            } => {
                *self.phases.entry("model_load_ms").or_default() += duration_seconds * 1000.;
            }
            TranscriptionPipelineEvent::ModelReuse { .. } => self.model_reuses += 1,
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let process_start = Instant::now();
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 7,
        "usage: rust-whisperx-pipeline-bench CANONICAL_MODEL SILERO_ONNX WAV_OR_LIST REPEATS BATCH fp32|fp16 MODEL_ID"
    );
    let root = PathBuf::from(&args[0]);
    let onnx = PathBuf::from(&args[1]);
    ensure!(
        root.is_dir() && onnx.is_file(),
        "supply existing local models"
    );
    let generation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("generation_config.json"))?)?;
    ensure!(
        generation["max_length"] == 448,
        "expected 448-token context limit"
    );
    let paths: Vec<PathBuf> = if args[2].ends_with(".json") {
        serde_json::from_slice(&std::fs::read(&args[2])?)?
    } else {
        vec![PathBuf::from(&args[2])]
    };
    let repeats: usize = args[3].parse()?;
    let batch: usize = args[4].parse()?;
    ensure!(
        !paths.is_empty() && repeats > 0 && (1..=16).contains(&batch),
        "invalid trial size"
    );
    let compute_type = match args[5].as_str() {
        "fp32" => CandleWhisperComputeType::Fp32,
        "fp16" => CandleWhisperComputeType::Fp16,
        _ => anyhow::bail!("precision must be fp32 or fp16"),
    };
    let options = CandleWhisperOptions {
        model_id: args[6].clone(),
        language: Some("en".into()),
        device: NativeDevicePreference::Cuda,
        compute_type,
        model_bundle: Some(root.clone()),
        model_cache_only: true,
        batch_chunks: batch > 1,
        max_batch_size: Some(batch),
        decode_runtime: if batch > 1 {
            CandleWhisperDecodeRuntime::ActiveRowTensorBatch
        } else {
            CandleWhisperDecodeRuntime::AutoregressiveKvCache
        },
        ..Default::default()
    };
    let config = CandleWhisperTranscriptionRequestConfig {
        window: CandleWhisperWindowControls {
            timing_mode: CandleWhisperTimingMode::NoTimestamps,
            leading_context_seconds: 0.,
            trailing_context_seconds: 0.,
        },
        ..Default::default()
    };
    let mut asr = ConfiguredAsr {
        model: ReusableCandleWhisperTranscriber::new(options.clone()),
        config: config.clone(),
    };
    let vad_load = Instant::now();
    let mut vad = SileroVadTranscriptionProvider::from_options(
        SileroVadOptions {
            model_path: onnx.clone(),
            input_name: None,
            output_name: None,
            threshold: 0.5,
            max_speech_duration_seconds: 30.,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
        },
        vec![],
    )?;
    let vad_load_ms = vad_load.elapsed().as_secs_f64() * 1000.;
    let mut first_result_ms = None;
    let mut runs = Vec::new();
    let mut failed_requests = 0;
    for path in paths {
        for index in 0..repeats {
            let started = Instant::now();
            let mut observer = Observer::default();
            let response = run_transcription_pipeline_with_observer(
                TranscriptionPipelineRequest {
                    source: TranscriptionSource::Path { path: path.clone() },
                    provider: TranscriptionProviderSelection::CandleWhisper(options.clone()),
                    vad: Default::default(),
                    alignment: Default::default(),
                    diarization: Default::default(),
                    output: Default::default(),
                },
                &mut vad,
                &mut asr,
                None,
                None,
                &mut observer,
            );
            let total_ms = started.elapsed().as_secs_f64() * 1000.;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    // Retain failures and later corpus entries, without treating
                    // the elapsed time of a failed request as a transcription.
                    failed_requests += 1;
                    runs.push(json!({"source":path,"index":index,
                        "failed_elapsed_ms":total_ms,"error":error.to_string(),
                        "vad_segments":observer.vad_segments,"phases":observer.phases}));
                    continue;
                }
            };
            first_result_ms.get_or_insert_with(|| process_start.elapsed().as_secs_f64() * 1000.);
            ensure!(
                response.diagnostics.iter().any(|d| d == "cuda=true"),
                "ASR did not use CUDA"
            );
            ensure!(
                response
                    .diagnostics
                    .iter()
                    .any(|d| d == "timingMode=noTimestamps"),
                "decoding mode mismatch"
            );
            ensure!(
                response
                    .diagnostics
                    .iter()
                    .any(|d| d == "native Silero VAD completed"),
                "missing upstream VAD"
            );
            ensure!(
                response.alignment.is_none() && response.diarization.is_none(),
                "feature mismatch"
            );
            if batch > 1 && response.vad_segments.len() > 1 {
                ensure!(
                    response
                        .diagnostics
                        .iter()
                        .any(|d| d == "batchExecution=candle-whisper-active-row-tensor-batch"),
                    "reference did not actually batch its decoder rows"
                );
            }
            runs.push(json!({"source":path,"index":index,"total_ms":total_ms,
                "phases":observer.phases,"model_reuses":observer.model_reuses,"result":response}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "backend":"rust-whisperx-library-pipeline", "version":"moenarch-audio-analysis-transcription=0.1.15",
            "scope":"Unmodified upstream WAV loader, single-threaded ONNX Silero VAD/merge, resident batched CUDA ASR and assembly. Public request controls select greedy English without token timestamps. Excludes alignment, diarization and file export; not stock CLI defaults.",
            "compute_type":args[5],"batch_size":batch,"request_config":config,
            "suppression":{"suppress_tokens":generation["suppress_tokens"],"begin_suppress_tokens":generation["begin_suppress_tokens"]},
            "model_sha256":hash(&root.join("model.safetensors"))?,"generation_config_sha256":hash(&root.join("generation_config.json"))?,
            "silero_onnx_sha256":hash(&onnx)?,"vad_load_ms":vad_load_ms,"first_result_ms":first_result_ms,"failed_requests":failed_requests,"runs":runs,
        }))?
    );
    ensure!(
        failed_requests == 0,
        "{failed_requests} upstream requests failed; retained in receipt"
    );
    Ok(())
}

fn hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = vec![0; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut bytes)?;
        if count == 0 {
            break;
        }
        hash.update(&bytes[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
