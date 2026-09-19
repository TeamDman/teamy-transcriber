//! Complete WAV-to-text timing. A JSON array of WAV paths exercises resident reuse.
use anyhow::Result;
use anyhow::ensure;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 2,
        "usage: wav_bench MODEL WAV_OR_JSON_LIST [REPEATS] [DUMP_PREFIX_OR_DASH] [tf32]"
    );
    let paths: Vec<PathBuf> = if args[1].ends_with(".json") {
        serde_json::from_slice(&std::fs::read(&args[1])?)?
    } else {
        vec![PathBuf::from(&args[1])]
    };
    let started = Instant::now();
    let tf32 = args.get(4).is_some_and(|s| s == "tf32");
    let mut engine = teamy_whisper_native::Engine::load(Path::new(&args[0]), 0, tf32)?;
    let mut frontend = teamy_whisper_native::frontend::Frontend::new(engine.dims().audio.n_mels)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let repeats: usize = args.get(2).map_or(Ok(3), |s| s.parse())?;
    ensure!(repeats > 0 && !paths.is_empty(), "empty benchmark");
    let mut runs = Vec::new();
    let mut first_result_ms = None;
    for path in &paths {
        for index in 0..repeats {
            let request_start = Instant::now();
            let mut wav = hound::WavReader::open(path)?;
            let spec = wav.spec();
            ensure!(
                spec.channels == 1 && spec.sample_rate == 16000,
                "expected 16 kHz mono WAV"
            );
            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => wav.samples::<f32>().collect::<Result<_, _>>()?,
                hound::SampleFormat::Int => {
                    ensure!(spec.bits_per_sample <= 16, "expected <=16-bit PCM");
                    let scale = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
                    wav.samples::<i16>()
                        .map(|n| n.map(|n| f32::from(n) / scale))
                        .collect::<Result<_, _>>()?
                }
            };
            let mel = frontend.compute(&samples)?;
            let frontend_ms = request_start.elapsed().as_secs_f64() * 1000.;
            let result = engine.transcribe_mel(&mel, 448)?;
            let total_ms = request_start.elapsed().as_secs_f64() * 1000.;
            first_result_ms.get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.);
            if index == 0
                && let Some(prefix) = args.get(3).filter(|s| s.as_str() != "-")
            {
                let prefix = if paths.len() > 1 {
                    format!("{prefix}-{}", path.file_stem().unwrap().to_string_lossy())
                } else {
                    prefix.clone()
                };
                std::fs::write(
                    format!("{prefix}.mel.f32"),
                    mel.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>(),
                )?;
                let logits = engine.prompt_logits(&mel)?;
                std::fs::write(
                    format!("{prefix}.logits.f32"),
                    logits
                        .iter()
                        .flat_map(|n| n.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            }
            runs.push(serde_json::json!({"file":path.file_name().map(|s|s.to_string_lossy()),"index":index,"audio_seconds":samples.len() as f64/16000.,"frontend_ms":frontend_ms,"total_ms":total_ms,"result":result}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"backend":"native-cuda","precision":if tf32 {"tf32"} else {"fp32"},"scope":"WAV-to-text ASR; greedy English, batch one; excludes VAD/alignment/diarization","load_ms":load_ms,"first_result_ms":first_result_ms,"runs":runs})
        )?
    );
    Ok(())
}
