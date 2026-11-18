//! Development-only VAD parity/timing receipt; no transcription or downloads.
use anyhow::Result;
use anyhow::ensure;
use facet::Facet;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;
use teamy_whisper_native::vad::Silero;
use teamy_whisper_native::vad::SpeechSegment;
use teamy_whisper_native::vad::merge_segments;
use teamy_whisper_native::vad::speech_segments;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: vad_bench WEIGHTS WAV_OR_JSON_LIST REPEATS"
    );
    let paths: Vec<PathBuf> = if args[1].ends_with(".json") {
        facet_json::from_slice(&std::fs::read(&args[1])?)?
    } else {
        vec![PathBuf::from(&args[1])]
    };
    let repeats: usize = args[2].parse()?;
    ensure!(!paths.is_empty() && repeats > 0, "empty benchmark");
    let started = Instant::now();
    let mut model = Silero::load(Path::new(&args[0]))?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    #[derive(Facet)]
    struct Run<'a> {
        file: &'a Path,
        index: usize,
        samples: usize,
        vad_ms: f64,
        probabilities: Vec<f32>,
        segments: Vec<SpeechSegment>,
        merged: Vec<SpeechSegment>,
    }

    let mut runs: Vec<Run<'_>> = Vec::new();
    for path in &paths {
        let mut wav = hound::WavReader::open(path)?;
        let spec = wav.spec();
        ensure!(
            spec.channels == 1 && spec.sample_rate == 16000,
            "expected mono 16 kHz WAV"
        );
        let audio: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Int, 16) => wav
                .samples::<i16>()
                .map(|x| x.map(|x| f32::from(x) / 32768.))
                .collect::<Result<_, _>>()?,
            (hound::SampleFormat::Float, 32) => wav.samples::<f32>().collect::<Result<_, _>>()?,
            _ => anyhow::bail!("expected PCM16 or float32 WAV"),
        };
        for index in 0..repeats {
            let begin = Instant::now();
            let probabilities = model.probabilities(&audio)?;
            let segments = speech_segments(&probabilities, audio.len(), 0.5, 30.)?;
            let merged = merge_segments(&segments, 480000)?;
            let vad_ms = begin.elapsed().as_secs_f64() * 1000.;
            runs.push(Run {
                file: path,
                index,
                samples: audio.len(),
                vad_ms,
                probabilities,
                segments,
                merged,
            });
        }
    }
    #[derive(Facet)]
    struct Receipt<'a> {
        backend: &'static str,
        scope: &'static str,
        load_ms: f64,
        runs: Vec<Run<'a>>,
    }

    writeln!(
        std::io::stdout().lock(),
        "{}",
        facet_json::to_string(&Receipt {
            backend: "source-defined-silero-cpu",
            scope: "VAD only, no audio I/O in request timing; no ASR/alignment/diarization",
            load_ms,
            runs
        })?
    )?;
    Ok(())
}
