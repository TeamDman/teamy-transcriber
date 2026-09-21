//! Local, validated weights and streaming source-defined speech detection.
use crate::domain::TimeRange;
use crate::transcription::SpeechPlan;
use eyre::Context;
use eyre::Result;
use eyre::ensure;
use facet::Facet;
use sha2::Digest;
use sha2::Sha256;
use std::io::Read;
use std::path::Path;
use teamy_whisper_native::vad::Silero;
use teamy_whisper_native::vad::WINDOW;
use teamy_whisper_native::vad::merge_segments;
use teamy_whisper_native::vad::speech_segments;

pub const DIRECTORY: &str = "vad";
const WEIGHTS: &str = "silero.safetensors";
const MANIFEST: &str = "manifest.json";
const ARCHITECTURE: &str = "silero-16k-context64-v1";
const MAX_WEIGHT_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Debug, Facet)]
pub struct SpeechModelManifest {
    pub schema_version: u16,
    pub architecture: String,
    pub bytes: u64,
    pub sha256: String,
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .wrap_err_with(|| format!("opening {}", path.display()))?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "model asset exceeds its size limit"
    );
    Ok(bytes)
}

/// Validate and stage local data weights beneath an existing Whisper package.
/// Publication is one directory rename; existing assets are never replaced.
pub fn prepare(source: &Path, model_dir: &Path) -> Result<SpeechModelManifest> {
    super::model::inspect_model_dir(model_dir)?;
    let destination = model_dir.join(DIRECTORY);
    ensure!(
        !destination.exists(),
        "speech model directory already exists: {}",
        destination.display()
    );
    let bytes = read_bounded(source, MAX_WEIGHT_BYTES)?;
    Silero::from_bytes(&bytes).map_err(|e| eyre::eyre!(e.to_string()))?;
    let manifest = SpeechModelManifest {
        schema_version: 1,
        architecture: ARCHITECTURE.into(),
        bytes: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    };
    let staging = model_dir.join(format!(".vad-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&staging)?;
    let mut partial = super::prepare::PartialModelDirectory::new(&staging);
    std::fs::write(staging.join(WEIGHTS), bytes)?;
    std::fs::write(
        staging.join(MANIFEST),
        facet_json::to_string_pretty(&manifest)?,
    )?;
    std::fs::write(
        staging.join("LICENSE.txt"),
        include_str!("../../native/licenses/silero.txt"),
    )?;
    let _ = load(&staging)?;
    std::fs::rename(&staging, &destination)
        .wrap_err("publishing the validated local speech model")?;
    partial.commit();
    Ok(manifest)
}

/// Validate a prepared speech detector without initializing CUDA.
pub fn validate(directory: &Path) -> Result<()> {
    load(directory).map(|_| ())
}

pub(crate) fn load(directory: &Path) -> Result<(Silero, String)> {
    let manifest: SpeechModelManifest =
        facet_json::from_slice(&read_bounded(&directory.join(MANIFEST), 8192)?)?;
    ensure!(
        manifest.schema_version == 1 && manifest.architecture == ARCHITECTURE,
        "unsupported speech model manifest"
    );
    let bytes = read_bounded(&directory.join(WEIGHTS), MAX_WEIGHT_BYTES)?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    ensure!(
        manifest.bytes == bytes.len() as u64 && manifest.sha256 == hash,
        "speech model checksum/length mismatch"
    );
    let model = Silero::from_bytes(&bytes).map_err(|e| eyre::eyre!(e.to_string()))?;
    Ok((model, hash))
}

/// Scan normalized audio with one 512-sample window and one probability per
/// window in memory. No CUDA model is created for silence-only recordings.
pub fn detect(
    model_dir: &Path,
    audio: &Path,
    should_stop: &mut dyn FnMut() -> bool,
) -> Result<SpeechPlan> {
    if should_stop() {
        return Ok(SpeechPlan::Cancelled);
    }
    let (mut model, hash) = load(&model_dir.join(DIRECTORY))?;
    let mut wav = hound::WavReader::open(audio)?;
    let spec = wav.spec();
    ensure!(
        spec.channels == 1 && spec.sample_rate == 16000,
        "speech detection expects normalized 16 kHz mono audio"
    );
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => analyze(&mut model, wav.samples::<f32>(), should_stop)?,
        hound::SampleFormat::Int => {
            ensure!(
                (1..=16).contains(&spec.bits_per_sample),
                "speech detection expects at most 16-bit PCM"
            );
            let divisor = f32::from(1u16 << (spec.bits_per_sample - 1));
            analyze(
                &mut model,
                wav.samples::<i16>()
                    .map(|s| s.map(|v| f32::from(v) / divisor)),
                should_stop,
            )?
        }
    };
    let Some((probabilities, count)) = samples else {
        return Ok(SpeechPlan::Cancelled);
    };
    let spans = merge_segments(
        &speech_segments(&probabilities, count, 0.5, 30.)
            .map_err(|e| eyre::eyre!(e.to_string()))?,
        480_000,
    )
    .map_err(|e| eyre::eyre!(e.to_string()))?;
    let ranges = spans
        .iter()
        .map(|span| {
            TimeRange::new(
                span.start as u64 * 1_000_000 / 16000,
                span.end as u64 * 1_000_000 / 16000,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SpeechPlan::Detected {
        ranges,
        weights_sha256: hash,
    })
}

fn analyze(
    model: &mut Silero,
    samples: impl Iterator<Item = Result<f32, hound::Error>>,
    should_stop: &mut dyn FnMut() -> bool,
) -> Result<Option<(Vec<f32>, usize)>> {
    model.reset();
    let mut samples = samples.peekable();
    let mut probabilities = Vec::new();
    let mut count = 0;
    while samples.peek().is_some() {
        if should_stop() {
            return Ok(None);
        }
        let mut window = [0.; WINDOW];
        for (slot, sample) in window.iter_mut().zip(samples.by_ref()) {
            let sample = sample?;
            ensure!(
                sample.is_finite(),
                "speech detection received non-finite PCM"
            );
            // Float decoders/resamplers can overshoot full scale. Saturate only
            // the detector input; preserve the saved waveform and ASR input.
            *slot = sample.clamp(-1., 1.);
            count += 1;
        }
        probabilities.push(
            model
                .step(&window)
                .map_err(|e| eyre::eyre!(e.to_string()))?,
        );
    }
    if should_stop() {
        return Ok(None);
    }
    Ok(Some((probabilities, count)))
}
