use crate::cli::output::CliOutput;
use crate::cli::output::CliOutputValue;
use crate::cli::output::OutputFormat;
use crate::domain::AppState;
use crate::domain::Command;
use crate::domain::RecordingId;
use crate::domain::SourceAsset;
use crate::phone_runtime::PhoneChunk;
use crate::storage::RecordingStore;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::Path;
use teamy_cancellation::CancellationToken;

/// Recognize IPA phones directly from audio/video using native CUDA `PhoneticXeus`.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct PhonesArgs {
    /// Audio or video file to recognize.
    #[facet(args::positional)]
    pub source: String,
    /// Canonical `PhoneticXeus` folder; defaults to selection, environment, or hf cache.
    #[facet(args::named)]
    pub model_dir: Option<String>,
}
#[derive(Facet)]
struct Report {
    recording_id: String,
    chunks: Vec<PhoneChunk>,
}
impl PhonesArgs {
    /// # Errors
    /// Returns source/model/audio/output errors, retaining created recordings.
    pub fn invoke(self, token: &CancellationToken) -> Result<CliOutput> {
        let source = Path::new(&self.source)
            .canonicalize()
            .wrap_err("phone source is unavailable")?;
        let model_dir = crate::phone_runtime::resolve(self.model_dir.as_deref())?;
        let model = teamy_whisper_native::phones::PhoneModel::load(&model_dir, 0)
            .map_err(|e| eyre::eyre!("{e:#}"))?;
        token.bail_if_cancelled()?;
        let store = RecordingStore::new(crate::paths::AppHome::resolve()?.0);
        let id = RecordingId::new();
        let run = (|| -> Result<CliOutput> {
            store.apply_command(
                &mut AppState::new(),
                Command::CreateRecording {
                    recording_id: id,
                    source: SourceAsset::new(
                        crate::workflow::asset_kind_for_path(&source),
                        &source,
                    )?,
                },
            )?;
            let audio = crate::workflow::prepare_recording(&store, id)?;
            let chunks =
                crate::phone_runtime::recognize_wav(&model, &audio.normalized_path, Some(token))?;
            let report = Report {
                recording_id: id.to_string(),
                chunks,
            };
            std::fs::write(
                store.recording_dir(id).join("phones.json"),
                facet_json::to_string_pretty(&report)?,
            )?;
            eprintln!("Phone results and prepared audio retained in recording {id}.");
            Ok(CliOutput::custom(report))
        })();
        run.wrap_err_with(||format!("Phone recognition failed; recording {id} was retained if created. Retry the source file; find saved audio with recording list."))
    }
}
impl CliOutputValue for Report {
    fn default_format(&self) -> Option<OutputFormat> {
        Some(OutputFormat::Text)
    }
    fn render(&self, format: OutputFormat, _stdout_is_terminal: bool) -> Result<Option<String>> {
        Ok(Some(match format {
            OutputFormat::Text => self
                .chunks
                .iter()
                .map(|c| c.ipa.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            OutputFormat::Json => facet_json::to_string_pretty(self)?,
            OutputFormat::Csv => eyre::bail!("phone output supports text or JSON"),
        }))
    }
}
