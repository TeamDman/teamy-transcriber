//! Full-recording development harness: WAV, native VAD, merge and native ASR.
use anyhow::Result;
use anyhow::ensure;
use facet::Facet;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;
use teamy_whisper_native::BatchTranscript;
use teamy_whisper_native::CudaDeviceIndex;
use teamy_whisper_native::Engine;
use teamy_whisper_native::MAX_BATCH_SIZE;
use teamy_whisper_native::decoding::GreedySuppression;
use teamy_whisper_native::frontend::Frontend;
use teamy_whisper_native::vad::Silero;
use teamy_whisper_native::vad::merge_segments;
use teamy_whisper_native::vad::speech_segments;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 5 || args.len() == 6,
        "usage: pipeline_bench WHISPER_MODEL SILERO_WEIGHTS WAV_LIST REPEATS GENERATION_CONFIG [BATCH_SIZE]"
    );
    let paths: Vec<PathBuf> = facet_json::from_slice(&std::fs::read(&args[2])?)?;
    let repeats: usize = args[3].parse()?;
    ensure!(!paths.is_empty() && repeats > 0, "empty benchmark");
    let batch_size: usize = args.get(5).map_or(Ok(1), |s| s.parse())?;
    ensure!(
        (1..=MAX_BATCH_SIZE).contains(&batch_size),
        "unsupported batch size"
    );
    let suppression: GreedySuppression = facet_json::from_slice(&std::fs::read(&args[4])?)?;
    let begin = Instant::now();
    let mut vad = Silero::load(Path::new(&args[1]))?;
    let mut engine = Engine::load(Path::new(&args[0]), CudaDeviceIndex::FIRST, true)?;
    engine.configure_greedy(&suppression)?;
    let mut frontend = Frontend::new(engine.dims().audio.n_mels)?;
    let load_ms = begin.elapsed().as_secs_f64() * 1000.;
    let mut first_result_ms = None;
    #[derive(Facet)]
    struct Segment {
        start: f64,
        end: f64,
        start_sample: usize,
        end_sample: usize,
        batch_index: usize,
        text: String,
        result: BatchTranscript,
    }

    #[derive(Facet)]
    struct Batch {
        size: usize,
        frontend_ms: f64,
        prepare_ms: f64,
        decode_ms: f64,
        total_ms: f64,
    }

    #[derive(Facet)]
    struct RunResult {
        segments: Vec<Segment>,
        language: &'static str,
    }

    #[derive(Facet)]
    struct Run<'a> {
        source: &'a Path,
        index: usize,
        batch_size: usize,
        batches: Vec<Batch>,
        audio_seconds: f64,
        audio_ms: f64,
        vad_ms: f64,
        total_ms: f64,
        result: RunResult,
    }

    let mut runs: Vec<Run<'_>> = Vec::new();
    for path in &paths {
        for index in 0..repeats {
            let started = Instant::now();
            let mut wav = hound::WavReader::open(path)?;
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
            let mut batches = Vec::new();
            for spans in merged.chunks(batch_size) {
                let batch_start = Instant::now();
                let mut features = Vec::new();
                let mut bounds = Vec::new();
                for span in spans {
                    let start = span.start as f64 / 16000.;
                    let end = span.end as f64 / 16000.;
                    // Match WhisperX's seconds-to-samples truncation exactly, including
                    // potential one-sample round-trip loss at some decimal boundaries.
                    let start_sample = (start * 16000.) as usize;
                    let end_sample = (end * 16000.) as usize;
                    let mel = frontend.compute(&audio[start_sample..end_sample])?;
                    features.push(mel);
                    bounds.push((start, end, start_sample, end_sample));
                }
                let frontend_ms = batch_start.elapsed().as_secs_f64() * 1000.;
                let (texts, prepare_ms, decode_ms) = if batch_size == 1 {
                    let r = engine.transcribe_mel(&features[0], 448)?;
                    (
                        vec![BatchTranscript {
                            text: r.text,
                            tokens: r.tokens,
                            ended: r.ended,
                        }],
                        r.encoder_ms,
                        r.decoder_ms,
                    )
                } else {
                    let mels: Vec<_> = features.iter().map(Vec::as_slice).collect();
                    let r = engine.transcribe_mel_batch(&mels, 448)?;
                    (r.transcripts, r.prepare_ms, r.decode_ms)
                };
                for ((start, end, start_sample, end_sample), result) in
                    bounds.into_iter().zip(texts)
                {
                    segments.push(Segment {
                        start,
                        end,
                        start_sample,
                        end_sample,
                        batch_index: batches.len(),
                        text: result.text.clone(),
                        result,
                    });
                }
                batches.push(Batch {
                    size: spans.len(),
                    frontend_ms,
                    prepare_ms,
                    decode_ms,
                    total_ms: batch_start.elapsed().as_secs_f64() * 1000.,
                });
            }
            let total_ms = started.elapsed().as_secs_f64() * 1000.;
            first_result_ms.get_or_insert_with(|| begin.elapsed().as_secs_f64() * 1000.);
            writeln!(
                std::io::stderr().lock(),
                "{} request {}: {:.1} ms, {} segments",
                path.display(),
                index,
                total_ms,
                segments.len()
            )?;
            runs.push(Run {
                source: path,
                index,
                batch_size,
                batches,
                audio_seconds: audio.len() as f64 / 16000.,
                audio_ms,
                vad_ms,
                total_ms,
                result: RunResult {
                    segments,
                    language: "en",
                },
            });
        }
    }
    #[derive(Facet)]
    struct Receipt<'a> {
        backend: &'static str,
        scope: &'static str,
        batch_size: usize,
        batch_workspace_bytes: usize,
        precision: &'static str,
        suppression: GreedySuppression,
        load_ms: f64,
        first_result_ms: Option<f64>,
        runs: Vec<Run<'a>>,
        full_goal_acceptance: bool,
    }

    writeln!(
        std::io::stdout().lock(),
        "{}",
        facet_json::to_string(&Receipt {
            backend: "source-defined-native-pipeline",
            scope: "PCM16 WAV loading, CPU Silero VAD, merging, CUDA greedy English ASR and assembly. Excludes alignment, diarization and file export. Development harness, not application integration.",
            batch_size,
            batch_workspace_bytes: engine.batch_workspace_bytes(),
            precision: "tf32",
            suppression,
            load_ms,
            first_result_ms,
            runs,
            full_goal_acceptance: false
        })?
    )?;
    Ok(())
}
