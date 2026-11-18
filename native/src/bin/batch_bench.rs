//! Batched ASR regression harness, with resident reuse and a final partial batch.
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

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 6,
        "usage: batch_bench MODEL WAV_LIST REPEATS BATCH_SIZE fp32|tf32 GENERATION_CONFIG"
    );
    let paths: Vec<PathBuf> = facet_json::from_slice(&std::fs::read(&args[1])?)?;
    let repeats: usize = args[2].parse()?;
    let batch_size: usize = args[3].parse()?;
    ensure!(
        !paths.is_empty() && repeats > 0 && (1..=MAX_BATCH_SIZE).contains(&batch_size),
        "invalid benchmark dimensions"
    );
    ensure!(args[4] == "fp32" || args[4] == "tf32", "invalid precision");
    let policy: GreedySuppression = facet_json::from_slice(&std::fs::read(&args[5])?)?;
    let started = Instant::now();
    let mut engine = Engine::load(
        Path::new(&args[0]),
        CudaDeviceIndex::FIRST,
        args[4] == "tf32",
    )?;
    engine.configure_greedy(&policy)?;
    let mut frontend = Frontend::new(engine.dims().audio.n_mels)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let mut first_result_ms = None;
    #[derive(Facet)]
    struct Run<'a> {
        file: &'a Path,
        index: usize,
        batch_index: usize,
        result: BatchTranscript,
    }

    let mut runs: Vec<Run<'_>> = Vec::new();
    #[derive(Facet)]
    struct Batch {
        index: usize,
        size: usize,
        frontend_ms: f64,
        prepare_ms: f64,
        decode_ms: f64,
        total_ms: f64,
    }

    let mut batches: Vec<Batch> = Vec::new();
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
                runs.push(Run {
                    file: path,
                    index,
                    batch_index: batches.len(),
                    result,
                });
            }
            batches.push(Batch {
                index,
                size: group.len(),
                frontend_ms,
                prepare_ms: result.prepare_ms,
                decode_ms: result.decode_ms,
                total_ms,
            });
        }
    }
    #[derive(Facet)]
    struct Receipt<'a> {
        backend: &'static str,
        scope: &'static str,
        batch_size: usize,
        precision: &'a str,
        suppression: GreedySuppression,
        load_ms: f64,
        first_result_ms: Option<f64>,
        batch_workspace_bytes: usize,
        runs: Vec<Run<'a>>,
        batches: Vec<Batch>,
    }

    writeln!(
        std::io::stdout().lock(),
        "{}",
        facet_json::to_string(&Receipt {
            backend: "native-batched-asr",
            scope: "WAV reads, frontend and batched greedy English ASR; no VAD/alignment/diarization",
            batch_size,
            precision: &args[4],
            suppression: policy,
            load_ms,
            first_result_ms,
            batch_workspace_bytes: engine.batch_workspace_bytes(),
            runs,
            batches
        })?
    )?;
    Ok(())
}
