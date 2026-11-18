//! Normalized audio input, and durable IPA output.
use eyre::Result;
use eyre::ensure;
use facet::Facet;
use std::path::Path;
use teamy_cancellation::CancellationScope;
use teamy_whisper_native::phones::PhoneModel;

#[derive(Debug, Facet)]
pub(crate) struct PhoneChunk {
    pub start_ms: u64,
    pub duration_ms: u64,
    pub phones: Vec<String>,
    pub ipa: String,
}
pub(crate) fn recognize_wav(
    model: &PhoneModel,
    path: &Path,
    cancellation: Option<&CancellationScope>,
) -> Result<Vec<PhoneChunk>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1 && spec.sample_rate == 16000,
        "phone input must be normalized mono16kHz"
    );
    let mut samples: Box<dyn Iterator<Item = Result<f32, hound::Error>> + '_> =
        match spec.sample_format {
            hound::SampleFormat::Float => Box::new(reader.samples::<f32>()),
            hound::SampleFormat::Int => {
                ensure!(spec.bits_per_sample == 16, "phone input requires16-bit PCM");
                Box::new(
                    reader
                        .samples::<i16>()
                        .map(|v| v.map(|v| f32::from(v) / 32768.)),
                )
            }
        };
    let mut chunks = Vec::new();
    let mut start = 0u64;
    loop {
        if let Some(cancellation) = cancellation {
            cancellation.bail_if_cancelled()?;
        }
        let mut audio: Vec<f32> = samples.by_ref().take(480_000).collect::<Result<_, _>>()?;
        if audio.is_empty() {
            break;
        }
        let count = audio.len();
        if count < 400 {
            audio.resize(400, 0.);
        }
        let result = model.recognize(&audio).map_err(|e| eyre::eyre!("{e:#}"))?;
        chunks.push(PhoneChunk {
            start_ms: start / 16,
            duration_ms: count as u64 / 16,
            phones: result.phones,
            ipa: result.ipa,
        });
        start += count as u64;
    }
    Ok(chunks)
}
