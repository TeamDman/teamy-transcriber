//! Explicit-input development benchmark. Timings include real decoding to EOT
//! or the model context limit. This measures ASR, not alignment/diarization.
use eyre::Context;
use eyre::Result;
use eyre::ensure;
use facet::Facet;
use sha2::Digest;
use sha2::Sha256;
use std::path::Path;
use std::time::Instant;
use teamy_transcriber::native_whisper::frontend;
use teamy_transcriber::native_whisper::model;

#[derive(Facet)]
struct Run {
    index: usize,
    frontend_ms: f64,
    inference_ms: f64,
    total_ms: f64,
    text: String,
}

#[derive(Facet)]
struct ModelHash {
    name: Option<String>,
    sha256: String,
}

#[derive(Facet)]
struct Receipt {
    schema: u32,
    backend: String,
    scope: String,
    device_setting: String,
    cuda_available: bool,
    cuda_device_count: i64,
    revision: String,
    profile: String,
    audio_sha256: String,
    audio_seconds: f64,
    model_hashes: Vec<ModelHash>,
    inspect_ms: f64,
    load_ms: f64,
    measured_elapsed_ms: f64,
    runs: Vec<Run>,
}

fn main() -> Result<()> {
    let started = Instant::now();
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 2,
        "usage: whisper_bench MODEL_DIR MONO_16KHZ_WAV [REPEATS] [MEL_OUTPUT]"
    );
    let repeats: usize = args.get(2).map_or(Ok(5), |n| n.parse())?;
    ensure!(repeats > 0, "REPEATS must be positive");
    let audio_path = Path::new(&args[1]);
    let mut wav = hound::WavReader::open(audio_path)?;
    let spec = wav.spec();
    ensure!(
        spec.channels == 1 && spec.sample_rate == 16_000,
        "expected 16 kHz mono WAV"
    );
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => wav.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            ensure!(spec.bits_per_sample <= 16, "expected at most 16-bit PCM");
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
            wav.samples::<i16>()
                .map(|n| n.map(|n| f32::from(n) / scale))
                .collect::<Result<_, _>>()?
        }
    };
    ensure!(
        samples.len() <= frontend::N_SAMPLES,
        "benchmark accepts one complete <=30-second window; split longer audio explicitly"
    );
    let inspect_start = Instant::now();
    let artifacts = model::inspect_model_dir(Path::new(&args[0]))?;
    let inspect_ms = inspect_start.elapsed().as_secs_f64() * 1_000.0;
    let load_start = Instant::now();
    #[cfg(feature = "tch-native")]
    let runtime = teamy_transcriber::native_whisper::tch_safetensors::TchSafetensorsWhisperRuntime::from_artifacts(&artifacts)?;
    #[cfg(not(feature = "tch-native"))]
    compile_error!("whisper_bench currently requires the tch-native feature");
    let load_ms = load_start.elapsed().as_secs_f64() * 1_000.0;
    let mut runs = Vec::new();
    for index in 0..repeats {
        let request_start = Instant::now();
        let features = frontend::whisper_log_mel_spectrogram(&samples);
        let frontend_ms = request_start.elapsed().as_secs_f64() * 1_000.0;
        if index == 0
            && let Some(path) = args.get(3)
        {
            let bytes: Vec<u8> = features
                .values
                .iter()
                .flat_map(|n| n.to_le_bytes())
                .collect();
            std::fs::write(path, bytes)?;
        }
        let decode_start = Instant::now();
        let text = runtime
            .greedy_decode(&artifacts, &features, 448)
            .wrap_err("real Whisper inference failed")?;
        let inference_ms = decode_start.elapsed().as_secs_f64() * 1_000.0;
        let total_ms = frontend_ms + inference_ms;
        runs.push(Run {
            index,
            frontend_ms,
            inference_ms,
            total_ms,
            text,
        });
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    // Hash after timing: verification IO must not masquerade as inference work.
    let model_hashes = hash_model(&artifacts)?;
    println!(
        "{}",
        facet_json::to_string_pretty(&Receipt {
            schema: 1,
            backend: "rust-tch".into(),
            scope: "ASR only; fixed English greedy prompt, no word alignment or diarization".into(),
            device_setting: std::env::var("TEAMY_TRANSCRIBER_TORCH_DEVICE")
                .unwrap_or_else(|_| "0".to_string()),
            cuda_available: tch::Cuda::is_available(),
            cuda_device_count: tch::Cuda::device_count(),
            revision: env!("GIT_REVISION").into(),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .into(),
            audio_sha256: format!("{:x}", Sha256::digest(std::fs::read(audio_path)?)),
            audio_seconds: f64::from(u32::try_from(samples.len())?) / 16_000.0,
            model_hashes,
            inspect_ms,
            load_ms,
            measured_elapsed_ms: elapsed_ms,
            runs,
        })?
    );
    Ok(())
}

fn hash_model(artifacts: &model::WhisperModelArtifacts) -> Result<Vec<ModelHash>> {
    artifacts
        .safetensors_paths
        .iter()
        .map(|path| {
            Ok(ModelHash {
                name: path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned()),
                sha256: format!("{:x}", Sha256::digest(std::fs::read(path)?)),
            })
        })
        .collect()
}
