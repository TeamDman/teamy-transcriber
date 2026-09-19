//! Explicit-input development benchmark. Timings include real decoding to EOT
//! or the model context limit. This measures ASR, not alignment/diarization.
use eyre::Context;
use eyre::Result;
use eyre::ensure;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use std::path::Path;
use std::time::Instant;
use teamy_transcriber::native_whisper::frontend;
use teamy_transcriber::native_whisper::model;

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
        runs.push(json!({"index":index,"frontend_ms":frontend_ms,"inference_ms":inference_ms,"total_ms":total_ms,"text":text}));
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    // Hash after timing: verification IO must not masquerade as inference work.
    let mut model_hashes = Vec::new();
    for path in &artifacts.safetensors_paths {
        let bytes = std::fs::read(path)?;
        model_hashes
            .push(json!({"name":path.file_name().map(|name| name.to_string_lossy()),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":1,"backend":"rust-tch","scope":"ASR only; fixed English greedy prompt, no word alignment or diarization",
            "device_setting":std::env::var("TEAMY_TRANSCRIBER_TORCH_DEVICE").unwrap_or_else(|_| "0".to_string()),
            "cuda_available":tch::Cuda::is_available(),"cuda_device_count":tch::Cuda::device_count(),
            "revision":env!("GIT_REVISION"),"profile":if cfg!(debug_assertions) {"debug"} else {"release"},
            "audio_sha256":format!("{:x}",Sha256::digest(std::fs::read(audio_path)?)),
            "audio_seconds":samples.len() as f64 / 16_000.0,"model_hashes":model_hashes,
            "inspect_ms":inspect_ms,"load_ms":load_ms,"measured_elapsed_ms":elapsed_ms,"runs":runs
        }))?
    );
    Ok(())
}
