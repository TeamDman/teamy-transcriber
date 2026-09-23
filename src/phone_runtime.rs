//! Phone model discovery, normalized audio input, and durable IPA output.
use eyre::Context;
use eyre::Result;
use eyre::ensure;
use facet::Facet;
use std::path::Path;
use std::path::PathBuf;
use teamy_cancellation::CancellationToken;
use teamy_whisper_native::phones::PhoneModel;

pub const REPOSITORY: &str = "changelinglab/PhoneticXeus";
pub const REVISION: &str = "3a8d860fa68f8936ceb4196651221215bab9dae4";
#[derive(Facet)]
struct Selection {
    model_dir: String,
}
pub(crate) fn resolve(explicit: Option<&str>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("TEAMY_TRANSCRIBER_PHONE_MODEL_DIR") {
        return Ok(PathBuf::from(path));
    }
    let selection = crate::paths::AppHome::resolve()?
        .0
        .join("phone-model-selection.json");
    if selection.is_file() {
        let s: Selection = facet_json::from_slice(&std::fs::read(selection)?)?;
        return Ok(PathBuf::from(s.model_dir));
    }
    crate::native_whisper::selection::find_cached_repository(REPOSITORY,REVISION)
        .wrap_err("Phone model missing. Use hf download changelinglab/PhoneticXeus model.safetensors config.json ipa_vocab.json --revision 3a8d860fa68f8936ceb4196651221215bab9dae4; or pass --model-dir to phones / --phone-model-dir to transcribe or microphone transcribe")
}
pub(crate) fn select(path: &Path) -> Result<PathBuf> {
    let path = path.canonicalize()?;
    let _ = PhoneModel::load(&path, 0).map_err(|e| eyre::eyre!("{e:#}"))?;
    let home = crate::paths::AppHome::resolve()?.0;
    std::fs::create_dir_all(&home)?;
    let temporary = home.join(format!(".phone-model-{}.tmp", uuid::Uuid::new_v4()));
    std::fs::write(
        &temporary,
        facet_json::to_string(&Selection {
            model_dir: path
                .to_str()
                .ok_or_else(|| eyre::eyre!("model path is not UTF-8"))?
                .to_owned(),
        })?,
    )?;
    std::fs::rename(temporary, home.join("phone-model-selection.json"))?;
    Ok(path)
}
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
    token: Option<&CancellationToken>,
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
        if let Some(token) = token {
            token.bail_if_cancelled()?;
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
