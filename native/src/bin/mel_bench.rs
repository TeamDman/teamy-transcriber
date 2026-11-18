//! Low-level parity/timing harness. Input is an explicit little-endian f32 mel matrix.
use anyhow::Result;
use anyhow::ensure;
use facet::Facet;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use teamy_whisper_native::CudaDeviceIndex;
use teamy_whisper_native::DecodeResult;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 2,
        "usage: mel_bench MODEL MEL_F32 [REPEATS] [TF32:0|1]"
    );
    let started = Instant::now();
    let tf32 = args.get(3).is_some_and(|s| s == "1");
    let mut engine =
        teamy_whisper_native::Engine::load(Path::new(&args[0]), CudaDeviceIndex::FIRST, tf32)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let data = std::fs::read(&args[1])?;
    ensure!(data.len() % 4 == 0, "truncated mel input");
    let mel: Vec<_> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let repeats: usize = args.get(2).map_or(Ok(5), |s| s.parse())?;
    let mut runs = Vec::new();
    for _ in 0..repeats {
        runs.push(engine.transcribe_mel(&mel, 448)?);
    }
    #[derive(Facet)]
    struct Receipt {
        backend: &'static str,
        precision: &'static str,
        scope: &'static str,
        load_ms: f64,
        runs: Vec<DecodeResult>,
    }

    writeln!(
        std::io::stdout().lock(),
        "{}",
        facet_json::to_string_pretty(&Receipt {
            backend: "native-cuda",
            precision: if tf32 { "tf32" } else { "fp32" },
            scope: "mel-to-text only; excludes frontend, alignment, diarization",
            load_ms,
            runs
        })?
    )?;
    Ok(())
}
